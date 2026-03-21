import pickle
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split

# Configuration
DATA_PATH = "data/features_all.pkl"
OUTPUT_PATH = "core/shap_background.npy"

def save_background_data():
    """
    Extracts background data for SHAP analysis from training set.
    """
    print(f"Loading features from {DATA_PATH}...")
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return

    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)
    
    text_emb = data['text_emb']
    swin_emb = data['swin_emb']
    labels = data['labels']
    
    # Verify shapes
    print(f"Text embeddings: {text_emb.shape}")
    print(f"Swin embeddings: {swin_emb.shape}")
    
    # Split Data (Strictly matching user requirement: test_size=0.2, random_state=42)
    # Note: Previous scripts might have used random_state=1, but user requested 42 here.
    # We will follow the explicit instruction for this task.
    indices = np.arange(len(labels))
    train_idx, _ = train_test_split(indices, test_size=0.2, random_state=42)
    
    print(f"Training set size: {len(train_idx)}")
    
    # Select 100 random samples from training set
    np.random.seed(42) # Ensure reproducibility of the selection
    bg_indices = np.random.choice(train_idx, 100, replace=False)
    
    bg_text = text_emb[bg_indices]
    bg_swin = swin_emb[bg_indices]
    
    # Concatenate features: [Text (768), Swin (1024)]
    bg_data = np.concatenate([bg_text, bg_swin], axis=1)
    
    # Ensure output directory exists
    output_dir = os.path.dirname(OUTPUT_PATH)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Saving background data shape {bg_data.shape} to {OUTPUT_PATH}...")
    np.save(OUTPUT_PATH, bg_data)
    print("Success.")

if __name__ == "__main__":
    save_background_data()
