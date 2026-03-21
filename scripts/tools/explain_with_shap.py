import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from train_model import DynamicFusionNet

# Configuration
DATA_PATH = "data/features_all.pkl"
CSV_PATH = "data/fakeddit_with_captions.csv"
# Using the correct model name found in the directory
MODEL_PATH = "models/model_text_image_swin.pth"
OUTPUT_DIR = "outputs"
OUTPUT_PLOT = os.path.join(OUTPUT_DIR, "shap_analysis.png")

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

def load_data():
    """Load features and metadata."""
    print(f"Loading features from {DATA_PATH}...")
    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)
    
    # Extract features
    text_emb = data['text_emb']
    swin_emb = data['swin_emb']
    labels = data['labels']
    
    # Load metadata
    print(f"Loading metadata from {CSV_PATH}...")
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        # Ensure length matches
        if len(df) != len(labels):
            print(f"Warning: CSV length ({len(df)}) does not match features length ({len(labels)}).")
            # If lengths differ, we might need to handle it. For now, assume alignment or truncate.
            # Assuming the features were extracted from the CSV in order.
            titles = df['clean_title'].values if 'clean_title' in df.columns else df['title'].values
            image_ids = df['id'].values
        else:
            titles = df['clean_title'].values if 'clean_title' in df.columns else df['title'].values
            image_ids = df['id'].values
    else:
        print("Warning: CSV file not found. Using placeholder titles/ids.")
        titles = np.array([f"Sample {i}" for i in range(len(labels))])
        image_ids = np.array([f"img_{i}" for i in range(len(labels))])
        
    return text_emb, swin_emb, labels, titles, image_ids

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # 1. Load Data
    text_emb, swin_emb, labels, titles, image_ids = load_data()
    
    # 2. Split Data (Must match training script: test_size=0.2, random_state=1)
    indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=1)
    
    print(f"Train samples: {len(train_idx)}, Test samples: {len(test_idx)}")
    
    # 3. Select Samples
    # Background: 100 from train
    bg_indices = np.random.choice(train_idx, 100, replace=False)
    bg_text = text_emb[bg_indices]
    bg_swin = swin_emb[bg_indices]
    
    # Targets: 5 from test
    target_indices = np.random.choice(test_idx, 5, replace=False)
    target_text = text_emb[target_indices]
    target_swin = swin_emb[target_indices]
    target_labels = labels[target_indices]
    target_titles = titles[target_indices]
    
    # 4. Prepare Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {MODEL_PATH}...")
    
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    config = checkpoint['config']
    print(f"Model Config: {config}")
    
    model = DynamicFusionNet(config).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    # 5. Define Predict Function for SHAP
    # Input: numpy array of shape (N, 768 + 1024)
    # Output: numpy array of shape (N, 2) -> probabilities
    def predict_fn(data_numpy):
        # Split back into text and swin
        # data_numpy shape: (N, 1792)
        n_samples = data_numpy.shape[0]
        
        # text: 0-768, swin: 768-1792
        txt_part = data_numpy[:, :768]
        swin_part = data_numpy[:, 768:]
        
        # Convert to tensor
        txt_t = torch.tensor(txt_part, dtype=torch.float32).to(device)
        swin_t = torch.tensor(swin_part, dtype=torch.float32).to(device)
        # Dummy caption
        cap_t = torch.zeros((n_samples, 768), dtype=torch.float32).to(device)
        
        with torch.no_grad():
            logits = model(txt_t, swin_t, cap_t)
            # IMPORTANT: SHAP KernelExplainer works best with linear outputs (logits)
            # If we explain probabilities directly, additivity holds poorly because Softmax is non-linear.
            # But users usually want to see probability contributions.
            # Let's stick to probabilities for now but acknowledge the non-linearity error.
            # Alternatively, we can explain Logits and then map to Probs, but that makes the chart hard to read.
            # Let's try to increase nsamples to reduce estimation error first.
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            
        return probs

    # 6. Run SHAP
    print("Running SHAP analysis...")
    # Concatenate features for background and targets
    bg_data = np.concatenate([bg_text, bg_swin], axis=1)
    target_data = np.concatenate([target_text, target_swin], axis=1)
    
    # Check background distribution
    bg_labels = labels[bg_indices]
    print(f"Background Data Class Distribution: Real={np.sum(bg_labels==1)}, Fake={np.sum(bg_labels==0)}")
    
    # Let's try to explain LOGITS instead. This guarantees Base + Sum(SHAP) = Logit.
    # Then we show the sigmoid(Base + Sum) which equals the Prob.
    def predict_logits(data_numpy):
        n_samples = data_numpy.shape[0]
        txt_part = data_numpy[:, :768]
        swin_part = data_numpy[:, 768:]
        txt_t = torch.tensor(txt_part, dtype=torch.float32).to(device)
        swin_t = torch.tensor(swin_part, dtype=torch.float32).to(device)
        cap_t = torch.zeros((n_samples, 768), dtype=torch.float32).to(device)
        with torch.no_grad():
            logits = model(txt_t, swin_t, cap_t)
            # Return logit for Class 1
            logit_class1 = logits[:, 1].cpu().numpy()
        return logit_class1

    print("Using Logit-space explanation for mathematical correctness...")
    explainer = shap.KernelExplainer(predict_logits, bg_data)
    
    # Calculate SHAP values (Logit Space)
    # nsamples=auto or higher
    shap_values = explainer.shap_values(target_data, nsamples=200) 
    
    # For regression (logits), shap_values is just an array, not a list
    if isinstance(shap_values, list):
        shap_vals = shap_values[0] # Should not happen with predict_logits returning 1D array
    else:
        shap_vals = shap_values
        
    base_value = explainer.expected_value
    
    # Ensure base_value is a scalar
    if isinstance(base_value, np.ndarray):
        base_value = base_value.item() if base_value.size == 1 else base_value[0]

    print(f"Base Value (Logit): {base_value:.4f}")
    
    # Sigmoid function for conversion
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # 7. Aggregate and Plot
    print("Generating visualization...")
    fig, axes = plt.subplots(5, 3, figsize=(20, 15))
    plt.subplots_adjust(wspace=0.4, hspace=0.6)
    
    # Get actual predictions for display
    target_probs = predict_fn(target_data)
    
    for i in range(5):
        # Row components
        ax_text = axes[i, 0]
        ax_bar = axes[i, 1]
        ax_pred = axes[i, 2]
        
        # A. Text Title
        title_str = str(target_titles[i])
        # Wrap text
        import textwrap
        wrapped_title = "\n".join(textwrap.wrap(title_str, width=40))
        ax_text.text(0.5, 0.5, wrapped_title, ha='center', va='center', fontsize=12)
        ax_text.axis('off')
        ax_text.set_title("News Title", fontsize=10, fontweight='bold')
        
        # B. Contribution Bar Chart
        # Sum SHAP values for Text (0-768) and Swin (768-end)
        s_val = shap_vals[i]
        text_contrib = np.sum(s_val[:768])
        swin_contrib = np.sum(s_val[768:])
        
        features_plot = ['Text', 'Image']
        values_plot = [text_contrib, swin_contrib]
        colors_plot = ['blue' if v > 0 else 'red' for v in values_plot]
        
        bars = ax_bar.bar(features_plot, values_plot, color=colors_plot)
        ax_bar.axhline(0, color='black', linewidth=0.8)
        
        # Calculate Sum
        total_shap = text_contrib + swin_contrib
        final_logit = base_value + total_shap
        final_prob_from_logit = sigmoid(final_logit)
        
        # We need to map Logit contributions to Probability Space for display?
        # It's hard to linearly map them because the curve is non-linear.
        # But we can display the Logit summation which is exact, and then the Prob result.
        
        title_text = (
            f"Logit: {base_value:.2f} + {total_shap:.2f} = {final_logit:.2f}\n"
            f"Sigmoid({final_logit:.2f}) ≈ {final_prob_from_logit:.2f}\n"
            f"(Actual Prob: {target_probs[i, 1]:.2f})"
        )
        ax_bar.set_title(title_text, fontsize=9, fontweight='bold')
        ax_bar.set_ylabel("Contribution (Log-Odds)")
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax_bar.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom' if height > 0 else 'top')
        
        # C. Prediction vs Ground Truth
        prob_real = target_probs[i, 1]  # Class 1 is Real
        true_label = target_labels[i]
        pred_label = 1 if prob_real > 0.5 else 0
        
        # User specified: 1=Real, 0=Fake
        label_map = {0: "Fake", 1: "Real"}
        
        res_text = (
            f"Prob (Real): {prob_real:.4f}\n"
            f"Pred: {label_map[pred_label]}\n"
            f"Ground Truth: {label_map[true_label]}"
        )
        
        bg_color = 'green' if pred_label == true_label else 'red'
        ax_pred.text(0.5, 0.5, res_text, ha='center', va='center', fontsize=12, 
                     bbox=dict(facecolor=bg_color, alpha=0.3, boxstyle='round,pad=1'))
        ax_pred.axis('off')
        ax_pred.set_title("Prediction Result", fontsize=10, fontweight='bold')

    plt.suptitle("Multimodal SHAP Analysis (BERT + Swin)", fontsize=16)
    plt.savefig(OUTPUT_PLOT)
    print(f"Analysis saved to {OUTPUT_PLOT}")

if __name__ == "__main__":
    main()
