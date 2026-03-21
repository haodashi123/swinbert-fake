import os
import torch
import pandas as pd
import numpy as np
import pickle
import argparse
import gc
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AutoImageProcessor, SwinModel, ViTModel, ViTImageProcessor
from torchvision import models, transforms
from tqdm import tqdm
import re
import cv2

# Configuration
BATCH_SIZE = 16  # Optimized for RTX 3050 (4GB/8GB VRAM)
NUM_WORKERS = 0  # Moderate workers to avoid overhead
IMAGE_DIR = "data/images"
TEXT_MODEL_NAME = "bert-base-uncased"
SWIN_MODEL_NAME = "microsoft/swin-base-patch4-window7-224"
VIT_MODEL_NAME = "google/vit-base-patch16-224"

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory Usage: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")

class FakedditDataset(Dataset):
    def __init__(self, csv_file, img_dir, tokenizer=None, image_processor=None, transform=None, limit=None):
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"{csv_file} not found. Please run tools/generate_captions.py first.")
            
        self.data = pd.read_csv(csv_file)
        
        # Optional: Limit for testing
        if limit:
            self.data = self.data.head(limit)
        
        # Verify images exist
        self.data['image_path'] = self.data['id'].apply(lambda x: os.path.join(img_dir, f"{x}.jpg"))
        
        # Filter out missing images
        initial_len = len(self.data)
        self.data = self.data[self.data['image_path'].apply(os.path.exists)].reset_index(drop=True)
        print(f"Dataset size: {len(self.data)} (Dropped {initial_len - len(self.data)} missing images)")
        
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Label
        try:
            label = int(row['2_way_label'])
        except:
            label = 0
            
        result = {'label': torch.tensor(label, dtype=torch.long)}

        # Text & Caption (if tokenizer provided)
        if self.tokenizer:
            # Text (Clean Title)
            text = str(row['clean_title']) if pd.notna(row['clean_title']) else ""
            
            # Text Tokenization
            text_encoding = self.tokenizer(
                text,
                return_tensors='pt',
                max_length=128,
                padding='max_length',
                truncation=True
            )
            result['text_input_ids'] = text_encoding['input_ids'].squeeze(0)
            result['text_attention_mask'] = text_encoding['attention_mask'].squeeze(0)
            
            # Caption
            caption = str(row['generated_caption']) if pd.notna(row['generated_caption']) else ""
            caption_encoding = self.tokenizer(
                caption,
                return_tensors='pt',
                max_length=128,
                padding='max_length',
                truncation=True
            )
            result['caption_input_ids'] = caption_encoding['input_ids'].squeeze(0)
            result['caption_attention_mask'] = caption_encoding['attention_mask'].squeeze(0)
            
        # Image (if processor provided)
        if self.image_processor:
            image_path = row['image_path']
            try:
                image = Image.open(image_path).convert("RGB")
                image_inputs = self.image_processor(images=image, return_tensors="pt")
                pixel_values = image_inputs['pixel_values'].squeeze(0)
            except Exception:
                # Fallback black image (224x224)
                pixel_values = torch.zeros((3, 224, 224))
            result['pixel_values'] = pixel_values
        
        # Image (if transform provided - for VGG)
        elif self.transform:
            image_path = row['image_path']
            try:
                image = Image.open(image_path).convert("RGB")
                pixel_values = self.transform(image)
            except Exception:
                # Fallback black image (224x224)
                pixel_values = torch.zeros((3, 224, 224))
            result['pixel_values'] = pixel_values
        
        return result

def process_split(csv_path, output_pkl, limit=None):
    print(f"\n==========================================")
    print(f"Processing {csv_path} -> {output_pkl}...")
    print(f"==========================================")
    
    # Final storage
    final_results = {
        'text_emb': [],
        'cap_emb': [],
        'swin_emb': [],
        'vit_emb': [],
        'vgg_emb': [],
        'labels': []
    }
    
    # ==========================================
    # Phase 1: Text & Caption (BERT)
    # ==========================================
    print("[Phase 1/4] Extracting Text Features (BERT)...")
    tokenizer = BertTokenizer.from_pretrained(TEXT_MODEL_NAME)
    try:
        bert_model = BertModel.from_pretrained(TEXT_MODEL_NAME, use_safetensors=True).to(device)
    except Exception as e:
        print(f"Warning: Failed to load safetensors for BERT ({e}). Trying default load...")
        bert_model = BertModel.from_pretrained(TEXT_MODEL_NAME).to(device)
    bert_model.eval()
    
    dataset_phase1 = FakedditDataset(csv_path, IMAGE_DIR, tokenizer=tokenizer, image_processor=None, limit=limit)
    dataloader_phase1 = DataLoader(dataset_phase1, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    with torch.no_grad():
        for batch in tqdm(dataloader_phase1, desc="BERT Extraction"):
            # Move inputs to device
            text_input_ids = batch['text_input_ids'].to(device)
            text_attention_mask = batch['text_attention_mask'].to(device)
            caption_input_ids = batch['caption_input_ids'].to(device)
            caption_attention_mask = batch['caption_attention_mask'].to(device)
            
            # Forward BERT (Text)
            text_out = bert_model(input_ids=text_input_ids, attention_mask=text_attention_mask)
            text_cls = text_out.last_hidden_state[:, 0, :]
            
            # Forward BERT (Caption)
            cap_out = bert_model(input_ids=caption_input_ids, attention_mask=caption_attention_mask)
            cap_cls = cap_out.last_hidden_state[:, 0, :]
            
            # Store
            final_results['text_emb'].append(text_cls.cpu().numpy())
            final_results['cap_emb'].append(cap_cls.cpu().numpy())
            final_results['labels'].append(batch['label'].numpy())
            
    # Cleanup Phase 1
    del bert_model
    del tokenizer
    del dataset_phase1
    del dataloader_phase1
    torch.cuda.empty_cache()
    gc.collect()
    
    # ==========================================
    # Phase 2: Visual A (Swin Transformer)
    # ==========================================
    print("[Phase 2/4] Extracting Visual Features (Swin Transformer)...")
    swin_processor = AutoImageProcessor.from_pretrained(SWIN_MODEL_NAME)
    try:
        swin_model = SwinModel.from_pretrained(SWIN_MODEL_NAME, use_safetensors=True).to(device)
    except:
        swin_model = SwinModel.from_pretrained(SWIN_MODEL_NAME).to(device)
    swin_model.eval()
    
    dataset_phase2 = FakedditDataset(csv_path, IMAGE_DIR, tokenizer=None, image_processor=swin_processor, limit=limit)
    dataloader_phase2 = DataLoader(dataset_phase2, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    with torch.no_grad():
        for batch in tqdm(dataloader_phase2, desc="Swin Extraction"):
            pixel_values = batch['pixel_values'].to(device)
            swin_out = swin_model(pixel_values=pixel_values)
            # Swin output is typically (batch, seq_len, dim), pooler_output is (batch, dim)
            img_pooler = swin_out.pooler_output
            final_results['swin_emb'].append(img_pooler.cpu().numpy())
            
    # Cleanup Phase 2
    del swin_model
    del swin_processor
    del dataset_phase2
    del dataloader_phase2
    torch.cuda.empty_cache()
    gc.collect()
    
    # ==========================================
    # Phase 3: Visual B (ViT)
    # ==========================================
    print("[Phase 3/4] Extracting Visual Features (ViT)...")
    # Use ViTImageProcessor explicitly as requested to avoid manual normalization issues
    vit_processor = ViTImageProcessor.from_pretrained(VIT_MODEL_NAME)
    try:
        vit_model = ViTModel.from_pretrained(VIT_MODEL_NAME, use_safetensors=True).to(device)
    except:
        vit_model = ViTModel.from_pretrained(VIT_MODEL_NAME).to(device)
    vit_model.eval()
    
    dataset_phase3 = FakedditDataset(csv_path, IMAGE_DIR, tokenizer=None, image_processor=vit_processor, limit=limit)
    dataloader_phase3 = DataLoader(dataset_phase3, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    with torch.no_grad():
        for batch in tqdm(dataloader_phase3, desc="ViT Extraction"):
            pixel_values = batch['pixel_values'].to(device)
            vit_out = vit_model(pixel_values=pixel_values)
            # Use CLS token (last_hidden_state[:, 0, :]) instead of pooler_output
            # This provides raw global semantic information better suited for downstream MLP
            img_cls = vit_out.last_hidden_state[:, 0, :]
            final_results['vit_emb'].append(img_cls.cpu().numpy())
            
    # Cleanup Phase 3
    del vit_model
    del vit_processor
    del dataset_phase3
    del dataloader_phase3
    torch.cuda.empty_cache()
    gc.collect()

    # ==========================================
    # Phase 4: Visual C (VGG-19)
    # ==========================================
    print("[Phase 4/4] Extracting Visual Features (VGG-19)...")
    vgg_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    try:
        # Use weights parameter if available (newer torchvision)
        vgg_model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
    except:
        # Fallback for older torchvision
        vgg_model = models.vgg19(pretrained=True)
        
    # Modify classifier to get 4096-dim output (up to first FC)
    vgg_model.classifier = vgg_model.classifier[:4]
    
    vgg_model.to(device)
    vgg_model.eval()
    
    dataset_phase4 = FakedditDataset(csv_path, IMAGE_DIR, tokenizer=None, image_processor=None, transform=vgg_transform, limit=limit)
    dataloader_phase4 = DataLoader(dataset_phase4, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    with torch.no_grad():
        for batch in tqdm(dataloader_phase4, desc="VGG-19 Extraction"):
            pixel_values = batch['pixel_values'].to(device)
            vgg_out = vgg_model(pixel_values)
            final_results['vgg_emb'].append(vgg_out.cpu().numpy())
            
    # Cleanup Phase 4
    del vgg_model
    del dataset_phase4
    del dataloader_phase4
    torch.cuda.empty_cache()
    gc.collect()
    
    # ==========================================
    # 5. Save
    # ==========================================
    print("\nConcatenating and saving...")
    if len(final_results['text_emb']) > 0:
        # Check consistency
        n_samples = len(np.concatenate(final_results['labels']))
        print(f"Total samples processed: {n_samples}")
        
        saved_data = {
            'text_emb': np.vstack(final_results['text_emb']),
            'cap_emb': np.vstack(final_results['cap_emb']),
            'swin_emb': np.vstack(final_results['swin_emb']),
            'vit_emb': np.vstack(final_results['vit_emb']),
            'vgg_emb': np.vstack(final_results['vgg_emb']),
            'labels': np.concatenate(final_results['labels'])
        }
        
        with open(output_pkl, 'wb') as f:
            pickle.dump(saved_data, f)
            
        print(f"Done! Saved to {output_pkl}")
        print("Feature Shapes:")
        for k, v in saved_data.items():
            print(f"  {k}: {v.shape}")
            
        # Clear memory explicitly
        del saved_data
        del final_results
        gc.collect()
        torch.cuda.empty_cache()
        
    else:
        print("No data extracted.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="Limit number of samples for testing")
    args = parser.parse_args()

    # Define tasks: (Input CSV, Output PKL)
    tasks = [
        ("data/dataset_train.csv", "data/features_train.pkl"),
        ("data/dataset_val.csv", "data/features_val.pkl"),
        ("data/dataset_test.csv", "data/features_test.pkl")
    ]
    
    print(f"Starting feature extraction for {len(tasks)} splits...")
    if args.limit:
        print(f"Running in TEST mode with limit={args.limit}")

    for csv_path, pkl_path in tasks:
        if os.path.exists(csv_path):
            process_split(csv_path, pkl_path, limit=args.limit)
        else:
            print(f"Skipping {csv_path}: File not found.")
            
    print("\nAll tasks completed.")

if __name__ == "__main__":
    main()
