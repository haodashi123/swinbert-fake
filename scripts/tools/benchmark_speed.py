import torch
import torch.nn as nn
import time
import numpy as np
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from transformers import (
    BertTokenizer, BertModel, 
    AutoImageProcessor, SwinModel, 
    ViTImageProcessor, ViTModel, 
    BlipProcessor, BlipForConditionalGeneration
)

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_model import DynamicFusionNet

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "outputs"
WARMUP_ROUNDS = 20
TEST_ROUNDS = 200  # Increased for stability

# Model Paths
MODEL_SWIN_PATH = "models/model_text_image_swin.pth"
MODEL_VIT_PATH = "models/model_text_image_vit.pth"
MODEL_SWIN_BLIP_PATH = "models/model_text_image_cap_swin.pth"
MODEL_VIT_BLIP_PATH = "models/model_text_image_cap_vit.pth"

# Model Configs
CFG_SWIN = {'use_text': True, 'use_image': True, 'use_caption': False, 'visual_backbone': 'swin'}
CFG_VIT = {'use_text': True, 'use_image': True, 'use_caption': False, 'visual_backbone': 'vit'}
CFG_SWIN_BLIP = {'use_text': True, 'use_image': True, 'use_caption': True, 'visual_backbone': 'swin'}
CFG_VIT_BLIP = {'use_text': True, 'use_image': True, 'use_caption': True, 'visual_backbone': 'vit'}

def load_base_models():
    print("Loading Base Models & Processors...")
    models = {}
    
    # Text (BERT)
    models['tokenizer'] = BertTokenizer.from_pretrained("bert-base-uncased")
    models['bert'] = BertModel.from_pretrained("bert-base-uncased").to(DEVICE).eval()
    
    # Image (Swin)
    models['swin_processor'] = AutoImageProcessor.from_pretrained("microsoft/swin-base-patch4-window7-224")
    try:
        models['swin'] = SwinModel.from_pretrained("microsoft/swin-base-patch4-window7-224", use_safetensors=True).to(DEVICE).eval()
    except:
        models['swin'] = SwinModel.from_pretrained("microsoft/swin-base-patch4-window7-224").to(DEVICE).eval()
        
    # Image (ViT)
    models['vit_processor'] = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    try:
        models['vit'] = ViTModel.from_pretrained("google/vit-base-patch16-224", use_safetensors=True).to(DEVICE).eval()
    except:
        models['vit'] = ViTModel.from_pretrained("google/vit-base-patch16-224").to(DEVICE).eval()
        
    # Caption (BLIP)
    models['blip_processor'] = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    try:
        models['blip'] = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", use_safetensors=True).to(DEVICE).eval()
    except:
        models['blip'] = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(DEVICE).eval()
        
    return models

def load_classifier(path, config):
    if not os.path.exists(path):
        print(f"Warning: {path} not found. Creating random initialized model for benchmark.")
        model = DynamicFusionNet(config).to(DEVICE)
    else:
        try:
            checkpoint = torch.load(path, map_location=DEVICE)
            model = DynamicFusionNet(config).to(DEVICE)
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
        except:
            print(f"Error loading {path}, using random init.")
            model = DynamicFusionNet(config).to(DEVICE)
            
    model.eval()
    return model

# --- Inference Pipelines ---

def run_swin_pipeline(text, img, base_models, classifier):
    with torch.no_grad():
        # 1. Text Feature
        inputs = base_models['tokenizer'](text, return_tensors='pt', max_length=128, padding='max_length', truncation=True)
        input_ids = inputs['input_ids'].to(DEVICE)
        mask = inputs['attention_mask'].to(DEVICE)
        text_emb = base_models['bert'](input_ids, mask).last_hidden_state[:, 0, :]
        
        # 2. Image Feature (Swin)
        img_inputs = base_models['swin_processor'](images=img, return_tensors="pt").to(DEVICE)
        swin_out = base_models['swin'](**img_inputs)
        img_emb = swin_out.pooler_output
        
        # 3. Dummy Caption
        cap_emb = torch.zeros_like(text_emb).to(DEVICE)
        
        # 4. Classifier
        logits = classifier(text_emb, img_emb, cap_emb)
        return logits

def run_vit_pipeline(text, img, base_models, classifier):
    with torch.no_grad():
        # 1. Text Feature
        inputs = base_models['tokenizer'](text, return_tensors='pt', max_length=128, padding='max_length', truncation=True)
        input_ids = inputs['input_ids'].to(DEVICE)
        mask = inputs['attention_mask'].to(DEVICE)
        text_emb = base_models['bert'](input_ids, mask).last_hidden_state[:, 0, :]
        
        # 2. Image Feature (ViT)
        img_inputs = base_models['vit_processor'](images=img, return_tensors="pt").to(DEVICE)
        vit_out = base_models['vit'](**img_inputs)
        img_emb = vit_out.last_hidden_state[:, 0, :] # CLS token
        
        # 3. Dummy Caption
        cap_emb = torch.zeros_like(text_emb).to(DEVICE)
        
        # 4. Classifier
        logits = classifier(text_emb, img_emb, cap_emb)
        return logits

def run_swin_blip_pipeline(text, img, base_models, classifier):
    with torch.no_grad():
        # 1. BLIP Caption Generation
        blip_inputs = base_models['blip_processor'](images=img, return_tensors="pt").to(DEVICE)
        ids = base_models['blip'].generate(**blip_inputs, max_new_tokens=20) # Keep short for speed
        caption = base_models['blip_processor'].decode(ids[0], skip_special_tokens=True)
        
        # 2. Text Feature (Title)
        inputs = base_models['tokenizer'](text, return_tensors='pt', max_length=128, padding='max_length', truncation=True)
        input_ids = inputs['input_ids'].to(DEVICE)
        mask = inputs['attention_mask'].to(DEVICE)
        text_emb = base_models['bert'](input_ids, mask).last_hidden_state[:, 0, :]
        
        # 3. Text Feature (Caption)
        cap_inputs = base_models['tokenizer'](caption, return_tensors='pt', max_length=128, padding='max_length', truncation=True)
        cap_ids = cap_inputs['input_ids'].to(DEVICE)
        cap_mask = cap_inputs['attention_mask'].to(DEVICE)
        cap_emb = base_models['bert'](cap_ids, cap_mask).last_hidden_state[:, 0, :]
        
        # 4. Image Feature (Swin)
        img_inputs = base_models['swin_processor'](images=img, return_tensors="pt").to(DEVICE)
        swin_out = base_models['swin'](**img_inputs)
        img_emb = swin_out.pooler_output
        
        # 5. Classifier
        logits = classifier(text_emb, img_emb, cap_emb)
        return logits

def run_vit_blip_pipeline(text, img, base_models, classifier):
    with torch.no_grad():
        # 1. BLIP Caption Generation
        blip_inputs = base_models['blip_processor'](images=img, return_tensors="pt").to(DEVICE)
        ids = base_models['blip'].generate(**blip_inputs, max_new_tokens=20)
        caption = base_models['blip_processor'].decode(ids[0], skip_special_tokens=True)
        
        # 2. Text Feature (Title)
        inputs = base_models['tokenizer'](text, return_tensors='pt', max_length=128, padding='max_length', truncation=True)
        input_ids = inputs['input_ids'].to(DEVICE)
        mask = inputs['attention_mask'].to(DEVICE)
        text_emb = base_models['bert'](input_ids, mask).last_hidden_state[:, 0, :]
        
        # 3. Text Feature (Caption)
        cap_inputs = base_models['tokenizer'](caption, return_tensors='pt', max_length=128, padding='max_length', truncation=True)
        cap_ids = cap_inputs['input_ids'].to(DEVICE)
        cap_mask = cap_inputs['attention_mask'].to(DEVICE)
        cap_emb = base_models['bert'](cap_ids, cap_mask).last_hidden_state[:, 0, :]
        
        # 4. Image Feature (ViT)
        img_inputs = base_models['vit_processor'](images=img, return_tensors="pt").to(DEVICE)
        vit_out = base_models['vit'](**img_inputs)
        img_emb = vit_out.last_hidden_state[:, 0, :] # CLS token
        
        # 5. Classifier
        logits = classifier(text_emb, img_emb, cap_emb)
        return logits

def measure_latency(pipeline_fn, name, text_template, img_template, base_models, classifier):
    print(f"Benchmarking {name}...")
    
    # Force cleanup before starting to ensure consistent state
    import gc
    gc.collect()
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Warmup
    for _ in range(WARMUP_ROUNDS):
        pipeline_fn(text_template, img_template, base_models, classifier)
        
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    
    # Testing
    times = []
    
    # Pre-generate diverse random images to ensure robustness against caching
    test_images = []
    # Pre-generate diverse text to ensure Tokenizer robustness
    # BERT Tokenizer speed depends on subword splitting complexity
    base_text = "Breaking News: Major earthquake strikes the coast, causing tsunami warnings and widespread evacuation orders across the region."
    test_texts = []
    
    import random
    import string
    
    for _ in range(TEST_ROUNDS):
        rand_arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        test_images.append(Image.fromarray(rand_arr))
        
        # Inject random noise suffix to make each text unique but similar length
        # This prevents embedding caching while maintaining consistent workload
        random_suffix = ''.join(random.choices(string.ascii_letters, k=10))
        test_texts.append(f"{base_text} [{random_suffix}]")
    
    for i in range(TEST_ROUNDS):
        # Use a different random image and text for each iteration
        current_img = test_images[i]
        current_text = test_texts[i]
        
        start = time.time()
        pipeline_fn(current_text, current_img, base_models, classifier)
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()
        end = time.time()
        times.append((end - start) * 1000) # ms
    
    # Statistical processing
    times = np.array(times)
    
    # 1. Standard Outlier Removal (Tukey's Fences: 1.5 * IQR)
    # This is the standard statistical method for handling outliers.
    q1 = np.percentile(times, 25)
    q3 = np.percentile(times, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    filtered_times = times[(times >= lower_bound) & (times <= upper_bound)]

    avg_time = np.mean(filtered_times)
    std_time = np.std(filtered_times)
    
    # Standard 95% Confidence Interval Calculation
    # Formula: 1.96 * (Standard Deviation / sqrt(N))
    ci95 = 1.96 * (std_time / np.sqrt(len(filtered_times)))
    
    fps = 1000.0 / avg_time
    
    print(f"  -> Avg: {avg_time:.2f}ms | Std: {std_time:.2f} | 95% CI: ±{ci95:.2f} (N_eff={len(filtered_times)})")
    
    return avg_time, ci95, fps

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. Load Resources
    base_models = load_base_models()
    
    clf_swin = load_classifier(MODEL_SWIN_PATH, CFG_SWIN)
    clf_vit = load_classifier(MODEL_VIT_PATH, CFG_VIT)
    clf_swin_blip = load_classifier(MODEL_SWIN_BLIP_PATH, CFG_SWIN_BLIP)
    clf_vit_blip = load_classifier(MODEL_VIT_BLIP_PATH, CFG_VIT_BLIP)
    
    # 2. Random Real-world Simulation Input
    dummy_text = "Breaking News: Major earthquake strikes the coast, causing tsunami warnings and widespread evacuation orders across the region."
    dummy_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    
    # 3. Benchmark
    results = []
    
    # BERT + Swin
    t, ci, f = measure_latency(run_swin_pipeline, "BERT+Swin", dummy_text, dummy_img, base_models, clf_swin)
    results.append({'Model': 'SwinBERT', 'Latency (ms)': t, 'CI95': ci, 'FPS': f})
    
    # BERT + ViT
    t, ci, f = measure_latency(run_vit_pipeline, "BERT+ViT", dummy_text, dummy_img, base_models, clf_vit)
    results.append({'Model': 'BERT-ViT', 'Latency (ms)': t, 'CI95': ci, 'FPS': f})
    
    # BERT + Swin + BLIP
    t, ci, f = measure_latency(run_swin_blip_pipeline, "BERT+Swin+BLIP", dummy_text, dummy_img, base_models, clf_swin_blip)
    results.append({'Model': 'SwinBERT+BLIP', 'Latency (ms)': t, 'CI95': ci, 'FPS': f})
    
    # BERT + ViT + BLIP
    t, ci, f = measure_latency(run_vit_blip_pipeline, "BERT+ViT+BLIP", dummy_text, dummy_img, base_models, clf_vit_blip)
    results.append({'Model': 'BERT-ViT-BLIP', 'Latency (ms)': t, 'CI95': ci, 'FPS': f})
    
    # 4. Save & Visualize
    df = pd.DataFrame(results)
    print("\nBenchmark Results:")
    print(df.to_string(index=False))
    
    # Plot with Error Bars (Academic Style)
    # Configure Times New Roman font
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'stix' # For math expressions to match Times
    
    # 8.583 cm = 3.379 inches
    # Set global font size for 8-10pt requirement
    plt.rcParams['font.size'] = 9
    plt.rcParams['axes.labelsize'] = 9
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['legend.fontsize'] = 8
    
    plt.style.use('seaborn-v0_8-paper') 
    # Width: 3.38 inch, Height: 2.5 inch (Compact column width)
    fig, ax = plt.subplots(figsize=(3.38, 2.5)) 
    
    # Custom color palette (Colorblind friendly)
    colors = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2']
    
    # Create bars
    # Reduced linewidth for smaller figure
    bars = sns.barplot(x='Model', y='Latency (ms)', data=df, palette=colors, edgecolor='black', linewidth=0.8, ax=ax)
    
    # Add error bars (95% Confidence Interval)
    x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches]
    y_coords = [p.get_height() for p in ax.patches]
    
    yerr = df['CI95'].values
    # Reduced capsize and linewidth
    ax.errorbar(x=x_coords, y=y_coords, yerr=yerr, fmt='none', c='black', capsize=2, elinewidth=1.0, label='95% CI')
    
    # Formatting
    plt.ylabel('Inference Latency (ms)', fontsize=9)
    plt.xlabel(None) 
    plt.xticks(rotation=15, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    
    # Add Legend to explain Error Bars
    # Compact legend
    plt.legend(loc='upper left', fontsize=8, frameon=True, fancybox=False, edgecolor='black', shadow=False, handlelength=1.5)
    
    # Simple Annotations (No clutter)
    for i, row in df.iterrows():
        # Label text includes Mean and CI
        # Consistent decimal places: Mean (2 decimals), CI (2 decimals)
        label_text = f"{row['Latency (ms)']:.2f}±{row['CI95']:.2f}"
        y_pos = row['Latency (ms)'] + row['CI95'] + (df['Latency (ms)'].max() * 0.05)
        # Standard font size for annotation (min 8pt for academic papers)
        ax.text(i, y_pos, label_text, ha='center', va='bottom', fontsize=8, color='black')

    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    sns.despine()
    
    # Adjust Y-axis limit
    max_height = (df['Latency (ms)'] + df['CI95']).max()
    ax.set_ylim(0, max_height * 1.2)
    
    plt.tight_layout()
    
    # Save as PDF (Vector Graphics) for academic papers
    save_path_pdf = os.path.join(OUTPUT_DIR, "speed_benchmark.pdf")
    plt.savefig(save_path_pdf, format='pdf', bbox_inches='tight')

    # Save as SVG (Vector Graphics) - Good for web and editing
    save_path_svg = os.path.join(OUTPUT_DIR, "speed_benchmark.svg")
    plt.savefig(save_path_svg, format='svg', bbox_inches='tight')
    
    # Also save as PNG for preview
    plt.savefig(os.path.join(OUTPUT_DIR, "speed_benchmark.png"), dpi=300, bbox_inches='tight')
    print(f"\nCharts saved to:\n  - {save_path_pdf}\n  - {save_path_svg}")

if __name__ == "__main__":
    main()
