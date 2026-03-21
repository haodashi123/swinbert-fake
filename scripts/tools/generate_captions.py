import os
import pandas as pd
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from tqdm import tqdm

import argparse

def generate_captions():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="Limit number of images to process for testing")
    args = parser.parse_args()

    # Configuration
    BATCH_SIZE = 8
    INPUT_CSV = 'data/fakeddit_downloaded.csv'
    OUTPUT_CSV = 'data/fakeddit_with_captions.csv'
    IMAGE_DIR = 'data/images/'
    MODEL_ID = "Salesforce/blip-image-captioning-base"

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        return

    df = pd.read_csv(INPUT_CSV)
    if args.limit:
        df = df.head(args.limit)
    print(f"Loaded {len(df)} records from {INPUT_CSV}")

    # Load Model and Processor
    print("Loading BLIP model...")
    processor = BlipProcessor.from_pretrained(MODEL_ID)
    try:
        model = BlipForConditionalGeneration.from_pretrained(MODEL_ID, use_safetensors=True).to(device)
    except Exception as e:
        print(f"Failed to load safetensors: {e}. Trying default load...")
        model = BlipForConditionalGeneration.from_pretrained(MODEL_ID).to(device)
        
    model.eval()
    print("Model loaded successfully.")

    # Prepare for storage
    generated_captions = [""] * len(df)

    # Processing Loop
    print("Starting caption generation...")
    
    # We use a simple loop with manual batching to handle missing images gracefully
    for i in tqdm(range(0, len(df), BATCH_SIZE), desc="Generating Captions"):
        batch_df = df.iloc[i : i + BATCH_SIZE]
        batch_indices = batch_df.index.tolist()
        
        batch_images = []
        valid_indices_in_batch = [] # Indices relative to the current batch list (0 to BATCH_SIZE-1)
        
        # Load images for the current batch
        for idx, row in batch_df.iterrows():
            image_path = os.path.join(IMAGE_DIR, f"{row['id']}.jpg")
            
            if os.path.exists(image_path):
                try:
                    # Open and convert to RGB to ensure 3 channels
                    img = Image.open(image_path).convert('RGB')
                    batch_images.append(img)
                    valid_indices_in_batch.append(idx)
                except Exception as e:
                    # Corrupt image
                    generated_captions[idx] = "image unavailable"
            else:
                # Missing image
                generated_captions[idx] = "image unavailable"

        # If we have valid images, process them
        if batch_images:
            try:
                inputs = processor(images=batch_images, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    # Generate captions
                    # max_new_tokens=20 is usually enough for a short caption
                    generated_ids = model.generate(**inputs, max_new_tokens=50)
                    captions = processor.batch_decode(generated_ids, skip_special_tokens=True)
                
                # Assign back to results
                for local_idx, caption in enumerate(captions):
                    original_idx = valid_indices_in_batch[local_idx]
                    generated_captions[original_idx] = caption
                    
            except Exception as e:
                print(f"Error processing batch starting at index {i}: {e}")
                # Fallback for the whole batch if inference fails
                for idx in valid_indices_in_batch:
                    generated_captions[idx] = "generation failed"

    # Save results
    df['generated_caption'] = generated_captions
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved results to {OUTPUT_CSV}")

if __name__ == "__main__":
    generate_captions()
