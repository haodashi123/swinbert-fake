import torch
import torch.nn as nn
import numpy as np
import pickle
import os
import sys
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Network Definition
from train_model import DynamicFusionNet

# Configs
DATA_PATH = "data/features_test.pkl"
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# Dataset Class (Simplified for Inference)
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
    
    # Direct loading from test set (no split needed)
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
        print(f"!!! WARNING: {model_path} not found. Creating RANDOM model. RESULTS ARE INVALID !!!")
        model = DynamicFusionNet(config).to(DEVICE)
    else:
        try:
            checkpoint = torch.load(model_path, map_location=DEVICE)
            model = DynamicFusionNet(config).to(DEVICE)
            # Handle state dict structure
            if 'state_dict' in checkpoint:
                sd = checkpoint['state_dict']
            else:
                sd = checkpoint
            
            # Strict=False to be robust against minor mismatches if any
            model.load_state_dict(sd, strict=False)
        except Exception as e:
            print(f"Error loading {model_path}: {e}")
            model = DynamicFusionNet(config).to(DEVICE)
            
    model.eval()
    return model

def run_inference(model, text_emb, cap_emb, img_emb_data, labels, visual_backbone):
    dataset = TestDataset(text_emb, cap_emb, labels, 
                          swin_emb=img_emb_data if visual_backbone=='swin' else None,
                          vit_emb=img_emb_data if visual_backbone=='vit' else None,
                          vgg_emb=img_emb_data if visual_backbone=='vgg' else None,
                          visual_backbone=visual_backbone)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    results = {
        'preds': [],
        'probs': [],
        'correct': [],
        'losses': []
    }
    
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    with torch.no_grad():
        for batch in loader:
            t = batch['text_emb'].to(DEVICE)
            i = batch['img_emb'].to(DEVICE)
            c = batch['cap_emb'].to(DEVICE)
            l = batch['label'].to(DEVICE)
            
            logits = model(t, i, c)
            
            # Loss per sample
            loss = criterion(logits, l)
            results['losses'].extend(loss.cpu().numpy())
            
            # Predictions
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            
            results['probs'].extend(probs.cpu().numpy())
            results['preds'].extend(preds.cpu().numpy())
            results['correct'].extend((preds == l).cpu().numpy())
            
    return results

def perform_tests(res_A, res_B, name_A, name_B):
    print(f"\n=== Significance Test: {name_A} vs {name_B} ===")
    
    # 1. McNemar Test
    # Table:
    #          B_Corr  B_Wrong
    # A_Corr     n11     n10
    # A_Wrong    n01     n00
    
    correct_A = np.array(res_A['correct'])
    correct_B = np.array(res_B['correct'])
    
    n11 = np.sum(correct_A & correct_B)
    n10 = np.sum(correct_A & ~correct_B)
    n01 = np.sum(~correct_A & correct_B)
    n00 = np.sum(~correct_A & ~correct_B)
    
    table = [[n11, n10], [n01, n00]]
    
    # exact=True uses Binomial distribution, False uses Chi-Squared approximation
    # If numbers are small (<25), exact is better.
    try:
        mcnemar_res = mcnemar(table, exact=True)
        p_mcnemar = mcnemar_res.pvalue
        stat_mcnemar = mcnemar_res.statistic
    except:
        p_mcnemar = 1.0
        stat_mcnemar = 0.0

    print(f"McNemar Table: {table}")
    print(f"McNemar P-value: {p_mcnemar:.4f}")
    
    # Accuracy Report
    acc_A = np.mean(correct_A)
    acc_B = np.mean(correct_B)
    print(f"Accuracy: {name_A}={acc_A:.4f}, {name_B}={acc_B:.4f}")
    
    # 2. Wilcoxon Signed-Rank Test on Losses
    losses_A = np.array(res_A['losses'])
    losses_B = np.array(res_B['losses'])
    loss_mean_A = np.mean(losses_A)
    loss_mean_B = np.mean(losses_B)
    print(f"Avg Loss: {name_A}={loss_mean_A:.4f}, {name_B}={loss_mean_B:.4f}")
    
    # Test if distribution of (Loss_A - Loss_B) is symmetric around zero
    # alternative='two-sided' by default
    try:
        w_stat, p_wilcoxon = stats.wilcoxon(losses_A, losses_B)
    except Exception as e:
        print(f"Wilcoxon failed: {e}")
        p_wilcoxon = 1.0
        
    print(f"Wilcoxon P-value: {p_wilcoxon:.4f}")
    
    # 3. Paired t-test on Losses
    try:
        t_stat, p_ttest = stats.ttest_rel(losses_A, losses_B)
    except Exception as e:
        print(f"Paired t-test failed: {e}")
        p_ttest = 1.0
        
    print(f"Paired t-test P-value: {p_ttest:.4f}")
    
    # Conclusion
    alpha = 0.05
    
    # Determine winner based on Accuracy (for McNemar) and Loss (for Wilcoxon & t-test)
    # Note: Higher Accuracy is better, Lower Loss is better.
    
    print("-" * 30)
    # McNemar Conclusion
    if p_mcnemar < alpha:
        # If n01 > n10, B corrected more of A's errors -> B wins
        if n01 > n10:
            print(f"[McNemar] Significant: {name_B} is more accurate than {name_A} (p={p_mcnemar:.4f}).")
        else:
            print(f"[McNemar] Significant: {name_A} is more accurate than {name_B} (p={p_mcnemar:.4f}).")
    else:
        print(f"[McNemar] No significant difference in accuracy (p={p_mcnemar:.4f}).")
        
    # Wilcoxon Conclusion
    if p_wilcoxon < alpha:
        if loss_mean_B < loss_mean_A:
            print(f"[Wilcoxon] Significant: {name_B} has lower loss than {name_A} (p={p_wilcoxon:.4f}).")
        else:
            print(f"[Wilcoxon] Significant: {name_A} has lower loss than {name_B} (p={p_wilcoxon:.4f}).")
    else:
        print(f"[Wilcoxon] No significant difference in loss distribution (p={p_wilcoxon:.4f}).")
        
    # Paired t-test Conclusion
    if p_ttest < alpha:
        if loss_mean_B < loss_mean_A:
            print(f"[Paired t-test] Significant: {name_B} has lower avg loss than {name_A} (p={p_ttest:.4f}).")
        else:
            print(f"[Paired t-test] Significant: {name_A} has lower avg loss than {name_B} (p={p_ttest:.4f}).")
    else:
        print(f"[Paired t-test] No significant difference in avg loss (p={p_ttest:.4f}).")
    print("-" * 30)

def main():
    # 1. Load Data
    text_emb, cap_emb, swin_emb, vit_emb, vgg_emb, labels = load_data()
    if text_emb is None:
        return

    # 2. Define Configs & Load Models
    # Model A: SpotFake (Baseline: BERT + VGG)
    cfg_spotfake = {'use_text': True, 'use_image': True, 'use_caption': False, 'visual_backbone': 'vgg'}
    model_spotfake = load_model("models/model_spotfake.pth", cfg_spotfake)

    # Model B: Swin (Ours - Ablation Baseline)
    cfg_swin = {'use_text': True, 'use_image': True, 'use_caption': False, 'visual_backbone': 'swin'}
    model_swin = load_model("models/model_text_image_swin.pth", cfg_swin)
    
    # Model C: Swin + Blip (Ours - Full)
    cfg_swin_blip = {'use_text': True, 'use_image': True, 'use_caption': True, 'visual_backbone': 'swin'}
    model_swin_blip = load_model("models/model_text_image_cap_swin.pth", cfg_swin_blip)
    
    # Model D: ViT + Blip (SOTA Baseline)
    cfg_vit_blip = {'use_text': True, 'use_image': True, 'use_caption': True, 'visual_backbone': 'vit'}
    model_vit_blip = load_model("models/model_text_image_cap_vit.pth", cfg_vit_blip)
    
    # 3. Inference
    print("\nRunning Inference for SpotFake...")
    res_spotfake = run_inference(model_spotfake, text_emb, cap_emb, vgg_emb, labels, 'vgg')

    print("Running Inference for Swin...")
    res_swin = run_inference(model_swin, text_emb, cap_emb, swin_emb, labels, 'swin')
    
    print("Running Inference for Swin+Blip...")
    res_swin_blip = run_inference(model_swin_blip, text_emb, cap_emb, swin_emb, labels, 'swin')
    
    print("Running Inference for ViT+Blip...")
    res_vit_blip = run_inference(model_vit_blip, text_emb, cap_emb, vit_emb, labels, 'vit')
    
    # 4. Comparisons
    # Baseline vs Ours (Modified: SpotFake vs Swin)
    perform_tests(res_spotfake, res_swin, "SpotFake", "Swin")

    # Ablation: Swin vs Swin+Blip
    perform_tests(res_swin, res_swin_blip, "Swin", "Swin+Blip")
    
    # SOTA Compare: ViT+Blip vs Swin (Modified: ViT+Blip vs Swin)
    perform_tests(res_vit_blip, res_swin, "ViT+Blip", "Swin")

if __name__ == "__main__":
    main()
