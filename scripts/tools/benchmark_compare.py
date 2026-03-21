import torch
import torch.nn as nn
import numpy as np
import pickle
import os
import sys
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Network Definition
from train_model import DynamicFusionNet

# Configs
DATA_PATH = "data/features_test.pkl"
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# Dataset Class
class TestDataset(Dataset):
    def __init__(self, text_emb, cap_emb, labels, swin_emb=None, vit_emb=None, vgg_emb=None, visual_backbone=None):
        self.text_emb = torch.tensor(text_emb, dtype=torch.float32)
        self.cap_emb = torch.tensor(cap_emb, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.visual_backbone = visual_backbone
        
        self.img_emb = None
        if visual_backbone == 'swin' and swin_emb is not None:
            self.img_emb = torch.tensor(swin_emb, dtype=torch.float32)
        elif visual_backbone == 'vit' and vit_emb is not None:
            self.img_emb = torch.tensor(vit_emb, dtype=torch.float32)
        elif visual_backbone == 'vgg' and vgg_emb is not None:
            self.img_emb = torch.tensor(vgg_emb, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            'text_emb': self.text_emb[idx],
            'cap_emb': self.cap_emb[idx],
            'label': self.labels[idx]
        }
        if self.img_emb is not None:
            item['img_emb'] = self.img_emb[idx]
        else:
            item['img_emb'] = torch.zeros(1) 
        return item

def load_data():
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return None, None, None, None, None, None

    print(f"Loading features from {DATA_PATH}...")
    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)
    
    # Direct loading from test set
    y = data['labels']
    
    return (
        data['text_emb'], 
        data['cap_emb'], 
        data.get('swin_emb'), 
        data.get('vit_emb'), 
        data.get('vgg_emb'),
        y
    )

def load_model(model_path, config):
    if not os.path.exists(model_path):
        # Return None if model doesn't exist, we skip it
        return None
        
    try:
        checkpoint = torch.load(model_path, map_location=DEVICE)
        model = DynamicFusionNet(config).to(DEVICE)
        
        if 'state_dict' in checkpoint:
            sd = checkpoint['state_dict']
        else:
            sd = checkpoint
        
        model.load_state_dict(sd, strict=False)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading {model_path}: {e}")
        return None

def evaluate_model(model, text_emb, cap_emb, img_emb_data, labels, visual_backbone):
    dataset = TestDataset(text_emb, cap_emb, labels, 
                          swin_emb=img_emb_data if visual_backbone=='swin' else None,
                          vit_emb=img_emb_data if visual_backbone=='vit' else None,
                          vgg_emb=img_emb_data if visual_backbone=='vgg' else None,
                          visual_backbone=visual_backbone)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for batch in loader:
            t = batch['text_emb'].to(DEVICE)
            i = batch['img_emb'].to(DEVICE)
            c = batch['cap_emb'].to(DEVICE)
            l = batch['label'].to(DEVICE)
            
            logits = model(t, i, c)
            
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(l.cpu().numpy())
            
    # Calculate Metrics
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='weighted')
    prec = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    rec = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except:
        auc = 0.5
        
    return {
        'Accuracy': acc,
        'F1 Score': f1,
        'Precision': prec,
        'Recall': rec,
        'AUC': auc
    }

def main():
    # 1. Load Data
    text_emb, cap_emb, swin_emb, vit_emb, vgg_emb, labels = load_data()
    if text_emb is None:
        return

    # 2. Define Models to Benchmark
    models_to_test = [
        {
            'name': 'SpotFake (Baseline)',
            'path': 'models/model_spotfake.pth',
            'config': {'use_text': True, 'use_image': True, 'use_caption': False, 'visual_backbone': 'vgg'},
            'backbone': 'vgg',
            'img_data': vgg_emb
        },
        {
            'name': 'Swin (Ours - Ablation)',
            'path': 'models/model_text_image_swin.pth',
            'config': {'use_text': True, 'use_image': True, 'use_caption': False, 'visual_backbone': 'swin'},
            'backbone': 'swin',
            'img_data': swin_emb
        },
        {
            'name': 'Swin + Caption (Ours - Full)',
            'path': 'models/model_text_image_cap_swin.pth',
            'config': {'use_text': True, 'use_image': True, 'use_caption': True, 'visual_backbone': 'swin'},
            'backbone': 'swin',
            'img_data': swin_emb
        },
        {
            'name': 'ViT + Caption (SOTA)',
            'path': 'models/model_text_image_cap_vit.pth',
            'config': {'use_text': True, 'use_image': True, 'use_caption': True, 'visual_backbone': 'vit'},
            'backbone': 'vit',
            'img_data': vit_emb
        }
    ]
    
    results = []
    
    print("\nStarting Benchmark on Test Set...")
    print("-" * 60)
    
    for item in models_to_test:
        print(f"Evaluating: {item['name']}...")
        model = load_model(item['path'], item['config'])
        
        if model is None:
            print(f"  -> Model file not found ({item['path']}). Skipping.")
            continue
            
        metrics = evaluate_model(model, text_emb, cap_emb, item['img_data'], labels, item['backbone'])
        metrics['Model'] = item['name']
        results.append(metrics)
        
    # 3. Print Comparison Table
    if not results:
        print("No models evaluated.")
        return
        
    df_res = pd.DataFrame(results)
    # Reorder columns
    cols = ['Model', 'Accuracy', 'F1 Score', 'Precision', 'Recall', 'AUC']
    df_res = df_res[cols]
    
    print("\n" + "="*80)
    print("FINAL BENCHMARK RESULTS (Test Set)")
    print("="*80)
    print(df_res.to_string(index=False, float_format="%.4f"))
    print("="*80)
    
    # Save to CSV
    df_res.to_csv("outputs/benchmark_results.csv", index=False)
    print("Results saved to outputs/benchmark_results.csv")

if __name__ == "__main__":
    main()
