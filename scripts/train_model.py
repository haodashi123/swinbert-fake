import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm
import os

def set_seed(seed=2026):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    print(f"Random seed set to {seed}")

# ==========================================
# 1. Configuration & Feature Selection
# ==========================================
FEATURE_CONFIG = {
    'use_text': True,      # BERT Features (768)
    'use_image': False,     # Enable Image Branch
    'use_caption': False,   # BLIP Caption Features (768)
    'visual_backbone': 'vit' # Options: 'swin' (1024), 'vit' (768), 'vgg' (4096)
}

# Training Hyperparameters
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-3  # Increased for stronger regularization
EARLY_STOP_PATIENCE = 5
TRAIN_DATA_PATH = "data/features_train.pkl"
VAL_DATA_PATH = "data/features_val.pkl"

# Determine Model Save Name automatically
def get_model_name(config):
    # Special case for SpotFake (BERT + VGG)
    if config['use_text'] and config['use_image'] and not config['use_caption'] and config.get('visual_backbone') == 'vgg':
        return "models/model_spotfake.pth"

    parts = []
    if config['use_text']: parts.append("text")
    if config['use_image']: parts.append("image")
    if config['use_caption']: parts.append("cap")
    
    backbone = config.get('visual_backbone', 'swin')
    parts.append(backbone)
    
    return f"models/model_{'_'.join(parts)}.pth"

MODEL_SAVE_PATH = get_model_name(FEATURE_CONFIG)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Active Features: {FEATURE_CONFIG}")
print(f"Model will be saved to: {MODEL_SAVE_PATH}")

# ==========================================
# 2. Dataset
# ==========================================
class FusionDataset(Dataset):
    def __init__(self, text_emb, cap_emb, labels, swin_emb=None, vit_emb=None, vgg_emb=None, visual_backbone=None):
        self.text_emb = torch.tensor(text_emb, dtype=torch.float32)
        self.cap_emb = torch.tensor(cap_emb, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.visual_backbone = visual_backbone
        
        # Store visual features based on config
        self.img_emb = None
        if visual_backbone == 'swin' and swin_emb is not None:
            self.img_emb = torch.tensor(swin_emb, dtype=torch.float32)
        elif visual_backbone == 'vit' and vit_emb is not None:
            self.img_emb = torch.tensor(vit_emb, dtype=torch.float32)
        elif visual_backbone == 'vgg' and vgg_emb is not None:
            self.img_emb = torch.tensor(vgg_emb, dtype=torch.float32)
        else:
            # Fallback placeholder
            pass 

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
            # Return dummy if not used
            item['img_emb'] = torch.zeros(1) 
            
        return item

# ==========================================
# 3. Dynamic Network Definition
# ==========================================
class DynamicFusionNet(nn.Module):
    def __init__(self, config):
        super(DynamicFusionNet, self).__init__()
        self.config = config
        
        # --- Deep Stream Dimensions ---
        self.deep_input_dim = 0
        if config['use_text']: self.deep_input_dim += 768
        
        # Determine image dimension based on backbone
        if config['use_image']:
            backbone = config.get('visual_backbone', 'swin') # Default to Swin if not set
            if backbone == 'swin':
                self.deep_input_dim += 1024
            elif backbone == 'vit':
                self.deep_input_dim += 768
            elif backbone == 'vgg':
                self.deep_input_dim += 4096
            else:
                pass

        if config['use_caption']: self.deep_input_dim += 768
        
        # --- Deep Stream MLP ---
        if self.deep_input_dim > 0:
            self.deep_mlp = nn.Sequential(
                nn.Linear(self.deep_input_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.5)
            )
            self.deep_out_dim = 512
        else:
            self.deep_mlp = None
            self.deep_out_dim = 0
            
        # --- Fusion Layer ---
        self.fusion_dim = self.deep_out_dim
        
        if self.fusion_dim == 0:
            raise ValueError("No features selected! Please enable at least one feature.")
            
        self.classifier = nn.Linear(self.fusion_dim, 2)
        
    def forward(self, text_emb, img_emb, cap_emb):
        deep_features = []
        
        # 1. Deep Stream Concatenation
        if self.config['use_text']:
            deep_features.append(text_emb)
        if self.config['use_image']:
            deep_features.append(img_emb)
        if self.config['use_caption']:
            deep_features.append(cap_emb)
            
        final_in = None
        
        # Process Deep Stream
        if deep_features:
            deep_in = torch.cat(deep_features, dim=1)
            final_in = self.deep_mlp(deep_in)
        else:
            # Fallback
            final_in = torch.zeros((text_emb.shape[0], self.fusion_dim)).to(text_emb.device)
            
        logits = self.classifier(final_in)
        return logits

# ==========================================
# 4. Training Loop
# ==========================================
def validate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            text_emb = batch['text_emb'].to(device)
            img_emb = batch['img_emb'].to(device)
            cap_emb = batch['cap_emb'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(text_emb, img_emb, cap_emb)
            loss = criterion(logits, labels)
            
            val_loss += loss.item()
            
            # Probabilities for AUC (Class 1)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            
            # Predictions
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.cpu().numpy())
    
    val_loss /= len(dataloader)
    
    # Metrics
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='weighted')
    precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except:
        auc = 0.5
        
    return val_loss, acc, f1, precision, recall, auc

def load_data(path):
    print(f"Loading features from {path}...")
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def train():
    # Load Data
    if not os.path.exists(TRAIN_DATA_PATH) or not os.path.exists(VAL_DATA_PATH):
        print(f"Error: Feature files not found. Please run tools/extract_features.py first.")
        return

    train_data = load_data(TRAIN_DATA_PATH)
    val_data = load_data(VAL_DATA_PATH)
    
    # Helper to create dataset
    def create_dataset(data):
        X_text = data['text_emb']
        X_cap = data['cap_emb']
        y = data['labels']
        
        # Optional Visual Features
        X_swin = data.get('swin_emb', None)
        X_vit = data.get('vit_emb', None)
        X_vgg = data.get('vgg_emb', None)
        
        return FusionDataset(
            X_text, 
            X_cap, 
            y,
            swin_emb=X_swin,
            vit_emb=X_vit,
            vgg_emb=X_vgg,
            visual_backbone=FEATURE_CONFIG.get('visual_backbone')
        )

    train_dataset = create_dataset(train_data)
    val_dataset = create_dataset(val_data)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Initialize Model
    model = DynamicFusionNet(FEATURE_CONFIG).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # LR Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.2, patience=3, verbose=True
    )
    
    # Tracking
    best_val_loss = float('inf')
    patience_counter = 0
    best_acc = 0.0
    best_f1 = 0.0
    best_prec = 0.0
    best_rec = 0.0
    best_auc = 0.0
    
    print("Starting training...")
    for epoch in range(EPOCHS):
        # --- Train ---
        model.train()
        train_loss = 0
        train_preds = []
        train_targets = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False):
            text_emb = batch['text_emb'].to(device)
            img_emb = batch['img_emb'].to(device)
            cap_emb = batch['cap_emb'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            logits = model(text_emb, img_emb, cap_emb)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            train_preds.extend(preds)
            train_targets.extend(labels.cpu().numpy())
            
        train_loss /= len(train_loader)
        train_acc = accuracy_score(train_targets, train_preds)
        train_f1 = f1_score(train_targets, train_preds, average='weighted')
        
        # --- Validate ---
        val_loss, val_acc, val_f1, val_prec, val_rec, val_auc = validate(model, val_loader, criterion, device)
        
        # Logging
        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f} | "
              f"Prec: {val_prec:.4f} | Rec: {val_rec:.4f} | AUC: {val_auc:.4f}")
        
        # Step Scheduler
        scheduler.step(val_loss)
        
        # Early Stopping & Saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_acc = val_acc
            best_f1 = val_f1
            best_prec = val_prec
            best_rec = val_rec
            best_auc = val_auc
            patience_counter = 0
            if not os.path.exists('models'):
                os.makedirs('models')
            
            # Save checkpoint with config
            checkpoint = {
                'state_dict': model.state_dict(),
                'config': FEATURE_CONFIG
            }
            torch.save(checkpoint, MODEL_SAVE_PATH)
            print(f"  --> Model Saved to {MODEL_SAVE_PATH} (New Best Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"Early stopping triggered after {patience_counter} epochs without improvement.")
                break

    # ==========================================
    # 5. Final Report
    # ==========================================
    print("\n" + "="*40)
    print("       TRAINING REPORT       ")
    print(f"Model: {MODEL_SAVE_PATH}")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Best Accuracy: {best_acc:.4f}")
    print(f"Best F1 Score: {best_f1:.4f}")
    print(f"Best Precision: {best_prec:.4f}")
    print(f"Best Recall: {best_rec:.4f}")
    print(f"Best AUC:      {best_auc:.4f}")
    print("="*40)

if __name__ == "__main__":
    set_seed()
    train()
