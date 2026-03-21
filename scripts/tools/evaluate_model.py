import torch
import torch.nn as nn
import numpy as np
import pickle
import os
import sys
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, auc
)
from tqdm import tqdm

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Network Definition & Dataset
from train_model import DynamicFusionNet, FusionDataset
from torch.utils.data import DataLoader

# Configuration
DATA_PATH = "data/features_test.pkl"
OUTPUT_DIR = "outputs"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_test_data():
    """Loads the test set features."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found. Please run tools/extract_features.py first.")
        
    print(f"Loading test data from {DATA_PATH}...")
    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)
        
    return data

def load_model(model_path):
    """Loads a model and its config from a checkpoint."""
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        return None, None

    try:
        print(f"Loading model: {model_path}")
        checkpoint = torch.load(model_path, map_location=DEVICE)
        
        # Extract config
        if isinstance(checkpoint, dict) and 'config' in checkpoint:
            config = checkpoint['config']
            state_dict = checkpoint['state_dict']
        else:
            # Fallback if config is missing (shouldn't happen with current train_model.py)
            print(f"Warning: Config not found in {model_path}. Using default config (risky).")
            config = {
                'use_text': True,
                'use_image': True,
                'use_caption': False,
                'visual_backbone': 'swin' 
            }
            state_dict = checkpoint

        model = DynamicFusionNet(config).to(DEVICE)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        return model, config
    except Exception as e:
        print(f"Failed to load {model_path}: {e}")
        return None, None

def evaluate_model(model, data, config, batch_size=32):
    """Evaluates a single model on the test data."""
    
    # Prepare Dataset
    # We need to handle optional features based on what's available in data and what model expects
    # The FusionDataset handles None inputs gracefully, but we should pass what we have.
    
    dataset = FusionDataset(
        data['text_emb'],
        data['cap_emb'],
        data['labels'],
        swin_emb=data.get('swin_emb'),
        vit_emb=data.get('vit_emb'),
        vgg_emb=data.get('vgg_emb'),
        visual_backbone=config.get('visual_backbone', 'swin')
    )
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_preds = []
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            text_emb = batch['text_emb'].to(DEVICE)
            cap_emb = batch['cap_emb'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            
            # Image embedding depends on config
            img_emb = batch['img_emb'].to(DEVICE)
            
            logits = model(text_emb, img_emb, cap_emb)
            probs = torch.softmax(logits, dim=1)[:, 1] # Probability of Class 1 (Fake)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
    return np.array(all_targets), np.array(all_preds), np.array(all_probs)

def plot_confusion_matrix(y_true, y_pred, model_name, output_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Confusion matrix saved to {output_path}")

def plot_roc_curves(roc_data_list, output_path):
    plt.figure(figsize=(8, 6))
    
    for model_name, y_true, y_probs in roc_data_list:
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.4f})')
        
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(output_path)
    plt.close()
    print(f"ROC comparison saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models on the test set.")
    parser.add_argument("--models", nargs='+', required=True, help="List of model paths to evaluate.")
    args = parser.parse_args()
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # 1. Load Data
    try:
        data = load_test_data()
    except Exception as e:
        print(e)
        return

    results = []
    roc_data = []
    
    print("\n" + "="*50)
    print("Starting Evaluation")
    print("="*50)

    for model_path in args.models:
        model_name = os.path.basename(model_path).replace('.pth', '')
        print(f"\nProcessing: {model_name}...")
        
        # 2. Load Model
        model, config = load_model(model_path)
        if model is None:
            continue
            
        # 3. Evaluate
        y_true, y_pred, y_probs = evaluate_model(model, data, config)
        
        # 4. Calculate Metrics
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        try:
            auc_score = roc_auc_score(y_true, y_probs)
        except:
            auc_score = 0.5
            
        results.append({
            'Model': model_name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'AUC': auc_score
        })
        
        # 5. Plot Confusion Matrix
        cm_path = os.path.join(OUTPUT_DIR, f"confusion_matrix_{model_name}.png")
        plot_confusion_matrix(y_true, y_pred, model_name, cm_path)
        
        # Store for ROC plot
        roc_data.append((model_name, y_true, y_probs))

    # 6. Summary Table
    if results:
        df_results = pd.DataFrame(results)
        print("\n" + "="*50)
        print("Evaluation Results Summary")
        print("="*50)
        print(df_results.to_string(index=False, float_format="%.4f"))
        
        # Save CSV
        csv_path = os.path.join(OUTPUT_DIR, "evaluation_metrics.csv")
        df_results.to_csv(csv_path, index=False)
        print(f"\nMetrics saved to {csv_path}")
        
        # 7. Plot ROC Comparison (if multiple models or just one)
        roc_path = os.path.join(OUTPUT_DIR, "roc_comparison.png")
        plot_roc_curves(roc_data, roc_path)

    else:
        print("No models were successfully evaluated.")

if __name__ == "__main__":
    main()
