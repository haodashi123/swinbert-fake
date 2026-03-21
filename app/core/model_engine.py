import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from transformers import BertTokenizer, BertModel, AutoImageProcessor, SwinModel, BlipProcessor, BlipForConditionalGeneration, ViTModel
import requests
from io import BytesIO
import random
import os
import re
import nltk
from nltk.corpus import stopwords
import string
import cv2
import pickle
import shap
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from captum.attr import LayerIntegratedGradients
from utils.viz_helper import add_colorbar_to_image

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

# Helper for stopword filtering
def get_stopwords():
    try:
        stops = set(stopwords.words('english'))
    except:
        stops = set()
    # Add BERT special tokens and punctuation
    stops.update(['[cls]', '[sep]', '[pad]', '[unk]'])
    stops.update(list(string.punctuation))
    return stops

STOPWORDS = get_stopwords()

# ==========================================
# Helper Functions
# ==========================================
def reshape_transform_swin(tensor, height=7, width=7):
    # Handle tuple output from SwinBlock
    if isinstance(tensor, tuple):
        tensor = tensor[0]
        
    # Swin output is (B, 49, 1024) -> reshape to (B, 1024, 7, 7)
    result = tensor.transpose(1, 2)
    result = result.reshape(tensor.size(0), -1, height, width)
    return result

def reshape_transform_vit(tensor):
    # ViT output (B, 197, 768). Remove CLS token at index 0
    patches = tensor[:, 1:, :]
    h = w = int(patches.shape[1] ** 0.5) # 14 for base
    result = patches.reshape(tensor.size(0), h, w, tensor.size(2))
    result = result.permute(0, 3, 1, 2)
    return result

# ==========================================
# Dynamic Network Definition
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
            else:
                pass

        if config['use_caption']: self.deep_input_dim += 768
        
        # --- Explicit Stream Removed ---
        
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
            self.classifier = nn.Linear(1, 2) 
        else:
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
        if deep_features and self.deep_mlp:
            deep_in = torch.cat(deep_features, dim=1)
            final_in = self.deep_mlp(deep_in)
        else:
            # Fallback
            final_in = torch.zeros((text_emb.shape[0], self.fusion_dim)).to(text_emb.device)
            
        logits = self.classifier(final_in)
        return logits


class RealTimeDetector:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"RealTimeDetector initialized on {self.device}")
        
        # Load SHAP background data
        bg_path = os.path.join(os.path.dirname(__file__), "shap_background.npy")
        if not os.path.exists(bg_path):
            bg_path = "core/shap_background.npy"
        if os.path.exists(bg_path):
            self.background_data = np.load(bg_path)
            print(f"Loaded SHAP background data: {self.background_data.shape}")
            
            # Initialize Explainer lazily to avoid overhead if not used immediately
            # But prompt says "Initialize self.shap_explainer = shap.KernelExplainer..." in init.
            # However, KernelExplainer needs a prediction function which depends on models being loaded.
            # Models are loaded via _load_base_models (cached) and _load_fusion_model.
            # So we should probably initialize explainer when needed or ensure models are ready.
            # But the prompt says "In RealTimeDetector initialization... Initialize self.shap_explainer".
            # The predict_proba_fn needs access to the loaded model. 
            # We can define the function here, but it will need to load/access the model internally.
            self.shap_explainer = None 
        else:
            print(f"Warning: {bg_path} not found. SHAP analysis will be disabled.")
            self.background_data = None
            self.shap_explainer = None

    def _get_shap_explainer(self):
        """Lazy initialization of SHAP explainer to ensure models are ready."""
        if self.shap_explainer is None and self.background_data is not None:
            # We need a model for the prediction function. 
            # The requirement implies using the "Ours (Swin)" model for SHAP.
            # We'll ensure it's loaded inside the predict wrapper.
            print("Initializing SHAP KernelExplainer...")
            self.shap_explainer = shap.KernelExplainer(self.predict_proba_fn, self.background_data)
        return self.shap_explainer

    def predict_proba_fn(self, data_numpy):
        """
        Wrapper for SHAP. Input: (N, 1792) numpy array.
        Output: (N,) numpy array of LOGITS for class 1 (Real).
        Switching to Logits avoids saturation issues with Softmax where probabilities
        are extremely close to 0 or 1, making SHAP values tiny and indistinguishable.
        """
        # Ensure base models and fusion model are loaded
        base_models = self._load_base_models()
        
        # Load Swin Fusion Model (Model C)
        config_c = {'use_text': True, 'use_image': True, 'use_caption': False, 'use_explicit': False, 'visual_backbone': 'swin'}
        model_c = self._load_fusion_model("models/model_text_image_swin.pth", config_c)
        model_c.eval()
        
        n_samples = data_numpy.shape[0]
        # Split features
        txt_part = data_numpy[:, :768]
        swin_part = data_numpy[:, 768:]
        
        txt_t = torch.tensor(txt_part, dtype=torch.float32).to(self.device)
        swin_t = torch.tensor(swin_part, dtype=torch.float32).to(self.device)
        cap_t = torch.zeros((n_samples, 768), dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            logits = model_c(txt_t, swin_t, cap_t)
            # Return LOGIT for Class 1 directly
            # Shape (N,)
            return logits[:, 1].cpu().numpy()

    @st.cache_resource
    def _load_base_models(_self):
        """
        Loads all base feature extractors (BERT, ViT, Swin, BLIP) once.
        """
        print("Loading base models...")
        device = _self.device
        
        # 1. BERT
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        try:
            bert_model = BertModel.from_pretrained("bert-base-uncased", use_safetensors=True, low_cpu_mem_usage=True).to(device)
        except:
            print("Warning: Failed to load BERT with safetensors/low_mem. Fallback to default.")
            bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
        bert_model.eval()

        # 2. BLIP
        blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        try:
            blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", use_safetensors=True, low_cpu_mem_usage=True).to(device)
        except:
            print("Warning: Failed to load BLIP with safetensors/low_mem. Fallback to default.")
            blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
        blip_model.eval()

        # 3. ViT
        vit_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
        try:
            vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224", use_safetensors=True, low_cpu_mem_usage=True).to(device)
        except:
            print("Warning: Failed to load ViT with safetensors/low_mem. Fallback to default.")
            vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224").to(device)
        vit_model.eval()

        # 4. Swin
        swin_processor = AutoImageProcessor.from_pretrained("microsoft/swin-base-patch4-window7-224")
        try:
            swin_model = SwinModel.from_pretrained("microsoft/swin-base-patch4-window7-224", use_safetensors=True, low_cpu_mem_usage=True).to(device)
        except:
            print("Warning: Failed to load Swin with safetensors/low_mem. Fallback to default.")
            swin_model = SwinModel.from_pretrained("microsoft/swin-base-patch4-window7-224").to(device)
        swin_model.eval()
        
        return {
            'tokenizer': tokenizer,
            'bert': bert_model,
            'blip_processor': blip_processor,
            'blip': blip_model,
            'vit_processor': vit_processor,
            'vit': vit_model,
            'swin_processor': swin_processor,
            'swin': swin_model
        }

    def _load_fusion_model(self, model_path, config_override=None):
        """
        Loads a specific FusionNet checkpoint.
        """
        if not os.path.exists(model_path):
            # Create a dummy model if file missing (for demonstration/arena robustness)
            print(f"Warning: {model_path} not found. Creating random initialized model.")
            model = DynamicFusionNet(config_override).to(self.device)
            return model
            
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            # Check if config is in checkpoint, else use override
            if isinstance(checkpoint, dict) and 'config' in checkpoint:
                config = checkpoint['config']
                state_dict = checkpoint['state_dict']
            else:
                config = config_override
                state_dict = checkpoint
            
            model = DynamicFusionNet(config).to(self.device)
            # Filter out explicit weights if they exist in checkpoint but not in new model
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            model.eval()
            return model
        except Exception as e:
            print(f"Error loading {model_path}: {e}")
            return DynamicFusionNet(config_override).to(self.device)

    def predict_all(self, text, image_source=None):
        """
        Runs inference on 3 models:
        A: ViT Baseline
        B: ViT + Caption
        C: Swin + Caption (or Swin only, as per requirement 'Swin (config: use_image=True, backbone='swin', use_caption=False)')
        """
        base_models = self._load_base_models()
        
        # --- Pre-processing ---
        # Image
        image = None
        if image_source:
            try:
                if isinstance(image_source, str): # URL
                    if image_source.startswith("http"):
                        headers = {'User-Agent': 'Mozilla/5.0'}
                        response = requests.get(image_source, headers=headers, timeout=5)
                        image = Image.open(BytesIO(response.content)).convert("RGB")
                    else:
                        image = Image.open(image_source).convert("RGB")
                elif hasattr(image_source, 'read'): # File-like object (UploadedFile)
                     image = Image.open(image_source).convert("RGB")
                else: # PIL Image or compatible
                    image = image_source.convert("RGB")
            except Exception as e:
                print(f"Image load error: {e}")
        
        if image is None:
            image = Image.new('RGB', (224, 224), color='black')

        # 1. Text Embedding (BERT)
        inputs = base_models['tokenizer'](text, return_tensors='pt', max_length=128, padding='max_length', truncation=True)
        input_ids = inputs['input_ids'].to(self.device)
        mask = inputs['attention_mask'].to(self.device)
        with torch.no_grad():
            text_emb = base_models['bert'](input_ids, mask).last_hidden_state[:, 0, :]
            
        # 2. Image Embeddings
        # ViT
        vit_inputs = base_models['vit_processor'](images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            vit_out = base_models['vit'](**vit_inputs)
            vit_emb = vit_out.last_hidden_state[:, 0, :] # CLS token
            
        # Swin
        swin_inputs = base_models['swin_processor'](images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            swin_out = base_models['swin'](**swin_inputs)
            swin_emb = swin_out.pooler_output # Pooled output
            
        # 3. Caption Generation & Embedding
        generated_caption = ""
        try:
            blip_inputs = base_models['blip_processor'](images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                ids = base_models['blip'].generate(**blip_inputs, max_new_tokens=50)
                generated_caption = base_models['blip_processor'].decode(ids[0], skip_special_tokens=True)
        except:
            generated_caption = "caption unavailable"
            
        cap_inputs = base_models['tokenizer'](generated_caption, return_tensors='pt', max_length=128, padding='max_length', truncation=True)
        cap_ids = cap_inputs['input_ids'].to(self.device)
        cap_mask = cap_inputs['attention_mask'].to(self.device)
        with torch.no_grad():
            cap_emb = base_models['bert'](cap_ids, cap_mask).last_hidden_state[:, 0, :]

        # --- Load Fusion Models & Infer ---
        
        # Model A: ViT Baseline (Text + ViT)
        config_a = {'use_text': True, 'use_image': True, 'use_caption': False, 'use_explicit': False, 'visual_backbone': 'vit'}
        model_a = self._load_fusion_model("models/model_text_image_vit.pth", config_a)
        with torch.no_grad():
            logits_a = model_a(text_emb, vit_emb, torch.zeros_like(cap_emb))
            prob_a = torch.softmax(logits_a, dim=1)[0, 1].item()

        # Model B: ViT + Caption (Text + ViT + Caption)
        config_b = {'use_text': True, 'use_image': True, 'use_caption': True, 'use_explicit': False, 'visual_backbone': 'vit'}
        model_b = self._load_fusion_model("models/model_text_image_cap_vit.pth", config_b)
        with torch.no_grad():
            logits_b = model_b(text_emb, vit_emb, cap_emb)
            prob_b = torch.softmax(logits_b, dim=1)[0, 1].item()
            
        # Model C: Swin (Ours) (Text + Swin)
        # Requirement: Swin (config: use_image=True, backbone='swin', use_caption=False)
        config_c = {'use_text': True, 'use_image': True, 'use_caption': False, 'use_explicit': False, 'visual_backbone': 'swin'}
        model_c = self._load_fusion_model("models/model_text_image_swin.pth", config_c)
        with torch.no_grad():
            logits_c = model_c(text_emb, swin_emb, torch.zeros_like(cap_emb))
            prob_c = torch.softmax(logits_c, dim=1)[0, 1].item()
            
        # --- SHAP Analysis (New) ---
        shap_scores = None
        try:
            explainer = self._get_shap_explainer()
            if explainer:
                # Prepare input vector [Text(768), Swin(1024)]
                # Move tensors to cpu numpy
                t_np = text_emb.cpu().numpy()
                s_np = swin_emb.cpu().numpy()
                sample_data = np.concatenate([t_np, s_np], axis=1)
                
                # Calculate SHAP values (nsamples=500 for better stability)
                shap_values = explainer.shap_values(sample_data, nsamples=500) 
                
                # If prediction returns 1D array (logits), shap_values is an array (not list)
                if isinstance(shap_values, list):
                    shap_vals = shap_values[0] # Should not happen if predict_fn returns 1D
                else:
                    shap_vals = shap_values
                    
                # Aggregate
                # shap_vals shape (1, 1792)
                s_val = shap_vals[0] if len(shap_vals.shape) > 1 else shap_vals
                text_score = np.sum(s_val[:768])
                image_score = np.sum(s_val[768:])
                
                # Normalize to percentage (absolute sum normalization or softmax?)
                # Requirement: "Normalize (percentage)"
                # Let's use absolute contribution ratio
                total_abs = abs(text_score) + abs(image_score)
                if total_abs > 1e-9:
                    text_pct = abs(text_score) / total_abs
                    image_pct = abs(image_score) / total_abs
                else:
                    # If contribution is effectively zero, assume equal prior
                    text_pct = 0.5
                    image_pct = 0.5
                    
                shap_scores = {
                    "text_score": float(text_score),
                    "image_score": float(image_score),
                    "text_pct": float(text_pct),
                    "image_pct": float(image_pct)
                }
        except Exception as e:
            print(f"SHAP calculation error: {e}")

        return {
            "caption": generated_caption,
            "model_a": {"prob": prob_a, "label": "Real" if prob_a > 0.5 else "Fake"},
            "model_b": {"prob": prob_b, "label": "Real" if prob_b > 0.5 else "Fake"},
            "model_c": {"prob": prob_c, "label": "Real" if prob_c > 0.5 else "Fake"},
            "shap_scores": shap_scores
        }

    def explain(self, text, image_source=None):
        """
        Generates explanations for Swin (Ours) and ViT (Baseline).
        Returns:
            heatmap_swin: PIL Image (Grad-CAM overlay for Swin)
            heatmap_vit: PIL Image (Grad-CAM overlay for ViT)
            text_attributions: List of (token, score)
            caption_attributions: List of (token, score)
            generated_caption: str
        """
        print("Starting explanation generation...")
        base_models = self._load_base_models()
        device = self.device
        
        # Load Model C (Swin + Text + Caption)
        config_c = {'use_text': True, 'use_image': True, 'use_caption': True, 'use_explicit': False, 'visual_backbone': 'swin'}
        fusion_model_swin = self._load_fusion_model("models/model_text_image_cap_swin.pth", config_c)

        # Load Model A (ViT + Text) for comparison
        config_a = {'use_text': True, 'use_image': True, 'use_caption': False, 'visual_backbone': 'vit'}
        fusion_model_vit = self._load_fusion_model("models/model_text_image_vit.pth", config_a)
        
        # --- Prepare Inputs ---
        
        # 1. Image
        image = None
        if image_source:
            try:
                if isinstance(image_source, str): # URL
                    if image_source.startswith("http"):
                        headers = {'User-Agent': 'Mozilla/5.0'}
                        response = requests.get(image_source, headers=headers, timeout=5)
                        image = Image.open(BytesIO(response.content)).convert("RGB")
                    else:
                        image = Image.open(image_source).convert("RGB")
                elif hasattr(image_source, 'read'): # File-like object (UploadedFile)
                     image = Image.open(image_source).convert("RGB")
                else: # PIL Image or compatible
                    image = image_source.convert("RGB")
            except Exception as e:
                print(f"Image load error in explain: {e}")
        
        if image is None:
            image = Image.new('RGB', (224, 224), color='black')
            
        # 2. Text
        inputs = base_models['tokenizer'](text, return_tensors='pt', max_length=128, padding='max_length', truncation=True)
        input_ids = inputs['input_ids'].to(device)
        mask = inputs['attention_mask'].to(device)

        # 3. Caption
        generated_caption = ""
        try:
            blip_inputs = base_models['blip_processor'](images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                ids = base_models['blip'].generate(**blip_inputs, max_new_tokens=50)
                generated_caption = base_models['blip_processor'].decode(ids[0], skip_special_tokens=True)
        except:
            generated_caption = "caption unavailable"
            
        cap_inputs = base_models['tokenizer'](generated_caption, return_tensors='pt', max_length=128, padding='max_length', truncation=True)
        cap_ids = cap_inputs['input_ids'].to(device)
        
        # --- Pre-calculate Full Model Prediction (Swin) for Consistent Explanation Target ---
        pred_idx = 0
        try:
            with torch.no_grad():
                text_emb = base_models['bert'](input_ids, mask).last_hidden_state[:, 0, :]
                swin_inputs = base_models['swin_processor'](images=image, return_tensors="pt").to(device)
                swin_out = base_models['swin'](**swin_inputs)
                img_emb = swin_out.pooler_output
                cap_emb = base_models['bert'](cap_ids, cap_inputs['attention_mask'].to(device)).last_hidden_state[:, 0, :]
                logits = fusion_model_swin(text_emb, img_emb, cap_emb)
                pred_idx = torch.argmax(logits, dim=1).item()
                print(f"Explanation Target Class: {pred_idx}")
        except Exception as e:
            print(f"Error calculating target class: {e}")
            pred_idx = 0
        
        # ==========================================
        # 1. Swin Visual Explanation (EigenCAM)
        # ==========================================
        heatmap_swin = None
        try:
            class VisualWrapperSwin(nn.Module):
                def __init__(self, fusion, swin):
                    super().__init__()
                    self.fusion = fusion
                    self.swin = swin
                def forward(self, x):
                    swin_out = self.swin(x)
                    img_emb = swin_out.pooler_output
                    dummy_text = torch.zeros((x.size(0), 768)).to(device)
                    dummy_cap = torch.zeros((x.size(0), 768)).to(device)
                    logits = self.fusion(dummy_text, img_emb, dummy_cap)
                    return logits

            swin_backbone = base_models['swin']
            wrapper = VisualWrapperSwin(fusion_model_swin, swin_backbone)
            
            try:
                target_layers = [swin_backbone.encoder.layers[-1].blocks[-1]]
            except:
                target_layers = [swin_backbone.layernorm]
            
            cam = EigenCAM(model=wrapper, target_layers=target_layers, reshape_transform=reshape_transform_swin)
            swin_inputs = base_models['swin_processor'](images=image, return_tensors="pt").to(device)
            pixel_values = swin_inputs['pixel_values']
            targets = [ClassifierOutputTarget(pred_idx)]
            
            grayscale_cam = cam(input_tensor=pixel_values, targets=targets)
            grayscale_cam = grayscale_cam[0, :]
            
            # Robustness: Handle potential NaNs and normalization
            grayscale_cam = np.nan_to_num(grayscale_cam)
            if grayscale_cam.max() > grayscale_cam.min():
                grayscale_cam = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() - grayscale_cam.min())
            
            raw_width, raw_height = image.size
            grayscale_cam = cv2.resize(grayscale_cam, (raw_width, raw_height))
            
            # Ensure Image is Float32 [0, 1]
            rgb_img = np.float32(image) / 255
            rgb_img = np.clip(rgb_img, 0, 1)
            
            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True, image_weight=0.5)
            heatmap_swin = Image.fromarray(visualization)
            # Add colorbar
            heatmap_swin = add_colorbar_to_image(heatmap_swin)
            
        except Exception as e:
            print(f"Grad-CAM (Swin) failed: {e}")
            heatmap_swin = Image.new('RGB', (224, 224), color='gray')

        # ==========================================
        # 2. ViT Visual Explanation (EigenCAM)
        # ==========================================
        heatmap_vit = None
        try:
            class VisualWrapperViT(nn.Module):
                def __init__(self, fusion, vit):
                    super().__init__()
                    self.fusion = fusion
                    self.vit = vit
                def forward(self, x):
                    outputs = self.vit(pixel_values=x)
                    img_emb = outputs.pooler_output
                    dummy_text = torch.zeros((x.size(0), 768)).to(device)
                    dummy_cap = torch.zeros((x.size(0), 768)).to(device)
                    logits = self.fusion(dummy_text, img_emb, dummy_cap)
                    return logits

            vit_backbone = base_models['vit']
            wrapper_vit = VisualWrapperViT(fusion_model_vit, vit_backbone)
            
            # Target Layer: Final LayerNorm
            target_layers_vit = [vit_backbone.layernorm]
            
            cam_vit = EigenCAM(model=wrapper_vit, target_layers=target_layers_vit, reshape_transform=reshape_transform_vit)
            vit_inputs = base_models['vit_processor'](images=image, return_tensors="pt").to(device)
            pixel_values_vit = vit_inputs['pixel_values']
            
            # Use same target class
            grayscale_cam_vit = cam_vit(input_tensor=pixel_values_vit, targets=targets)
            grayscale_cam_vit = grayscale_cam_vit[0, :]
            
            # Robustness: Handle potential NaNs and normalization
            grayscale_cam_vit = np.nan_to_num(grayscale_cam_vit)
            if grayscale_cam_vit.max() > grayscale_cam_vit.min():
                grayscale_cam_vit = (grayscale_cam_vit - grayscale_cam_vit.min()) / (grayscale_cam_vit.max() - grayscale_cam_vit.min())
            
            grayscale_cam_vit = cv2.resize(grayscale_cam_vit, (raw_width, raw_height))
            
            # Ensure Image is Float32 [0, 1]
            rgb_img = np.float32(image) / 255
            rgb_img = np.clip(rgb_img, 0, 1)

            visualization_vit = show_cam_on_image(rgb_img, grayscale_cam_vit, use_rgb=True, image_weight=0.5)
            heatmap_vit = Image.fromarray(visualization_vit)
            # Add colorbar
            heatmap_vit = add_colorbar_to_image(heatmap_vit)
            
        except Exception as e:
            print(f"Grad-CAM (ViT) failed: {e}")
            heatmap_vit = Image.new('RGB', (224, 224), color='gray')

        # ==========================================
        # 3. Text Explanation (Integrated Gradients)
        # ==========================================
        text_attributions = []
        try:
            class TextWrapper(nn.Module):
                def __init__(self, fusion, bert):
                    super().__init__()
                    self.fusion = fusion
                    self.bert = bert
                def forward(self, input_embeds):
                    outputs = self.bert(inputs_embeds=input_embeds)
                    text_emb = outputs.last_hidden_state[:, 0, :]
                    dummy_img = torch.zeros((input_embeds.size(0), 1024)).to(device)
                    dummy_cap = torch.zeros((input_embeds.size(0), 768)).to(device)
                    logits = self.fusion(text_emb, dummy_img, dummy_cap)
                    return logits

            bert_backbone = base_models['bert']
            wrapper_txt = TextWrapper(fusion_model_swin, bert_backbone)
            embeddings = bert_backbone.embeddings(input_ids)
            lig = LayerIntegratedGradients(wrapper_txt, bert_backbone.embeddings)
            attributions, delta = lig.attribute(inputs=embeddings, target=pred_idx, n_steps=10, return_convergence_delta=True)
            attributions = attributions.sum(dim=2).squeeze(0)
            attributions = attributions / torch.norm(attributions)
            attr_scores = attributions.cpu().detach().numpy()
            tokens = base_models['tokenizer'].convert_ids_to_tokens(input_ids[0])
            text_attributions = self._aggregate_tokens(tokens, attr_scores)
                    
        except Exception as e:
            print(f"IG failed (Text): {e}")
            text_attributions = [("Error", 0.0)]
            
        # ==========================================
        # 4. Caption Explanation (Integrated Gradients)
        # ==========================================
        caption_attributions = []
        try:
            class CaptionWrapper(nn.Module):
                def __init__(self, fusion, bert):
                    super().__init__()
                    self.fusion = fusion
                    self.bert = bert
                def forward(self, input_embeds):
                    outputs = self.bert(inputs_embeds=input_embeds)
                    cap_emb = outputs.last_hidden_state[:, 0, :]
                    dummy_text = torch.zeros((input_embeds.size(0), 768)).to(device)
                    dummy_img = torch.zeros((input_embeds.size(0), 1024)).to(device)
                    logits = self.fusion(dummy_text, dummy_img, cap_emb)
                    return logits

            cap_wrapper = CaptionWrapper(fusion_model_swin, bert_backbone)
            cap_embeddings = bert_backbone.embeddings(cap_ids)
            lig_cap = LayerIntegratedGradients(cap_wrapper, bert_backbone.embeddings)
            attributions_cap, delta_cap = lig_cap.attribute(inputs=cap_embeddings, target=pred_idx, n_steps=10, return_convergence_delta=True)
            attributions_cap = attributions_cap.sum(dim=2).squeeze(0)
            attributions_cap = attributions_cap / torch.norm(attributions_cap)
            cap_attr_scores = attributions_cap.cpu().detach().numpy()
            cap_tokens = base_models['tokenizer'].convert_ids_to_tokens(cap_ids[0])
            caption_attributions = self._aggregate_tokens(cap_tokens, cap_attr_scores)
                    
        except Exception as e:
            print(f"IG failed (Caption): {e}")
            caption_attributions = [("Error", 0.0)]

        return heatmap_swin, heatmap_vit, text_attributions, caption_attributions, generated_caption

    def _aggregate_tokens(self, tokens, scores):
        """
        Aggregates sub-word tokens back to words and filters stopwords.
        """
        aggregated = []
        current_word = ""
        current_score = 0.0
        
        for token, score in zip(tokens, scores):
            # BERT subword token handling (##)
            if token.startswith("##"):
                current_word += token[2:]
                current_score += score # Sum scores for subwords
            else:
                if current_word:
                    aggregated.append((current_word, current_score))
                current_word = token
                current_score = score
                
        if current_word:
            aggregated.append((current_word, current_score))
            
        # --- Stopword Filtering & Renormalization ---
        filtered = []
        scores_only = []
        special_tokens = {'[cls]', '[sep]', '[pad]', '[unk]'}
        
        for word, score in aggregated:
            word_lower = word.lower()
            
            # Filter out BERT special tokens
            if word_lower in special_tokens:
                continue

            # Check if stopword or purely punctuation
            if word_lower in STOPWORDS or all(char in string.punctuation for char in word):
                filtered.append((word, 0.0)) # Force zero
            else:
                filtered.append((word, score))
                scores_only.append(abs(score))
                
        # Re-normalize only non-zero scores to stretch contrast
        if scores_only and max(scores_only) > 0:
            max_s = max(scores_only)
            min_s = min(scores_only)
            range_s = max_s - min_s if max_s > min_s else 1.0
            
            final_aggregated = []
            for word, score in filtered:
                if score == 0.0:
                    final_aggregated.append((word, 0.0))
                else:
                    # Min-Max Scaling to [0.1, 1.0] to ensure visibility
                    norm_score = 0.1 + 0.9 * ((abs(score) - min_s) / range_s)
                    # Restore sign if needed, though usually absolute importance is shown
                    final_aggregated.append((word, norm_score))
            return final_aggregated
        else:
            return filtered

    @staticmethod
    def simulate_prediction():
        """Fallback simulation for Arena Mode"""
        cap = "A group of people standing in the rain."
        return {
            "caption": cap,
            "model_a": {"prob": 0.45, "label": "Fake"}, # Low prob = Fake
            "model_b": {"prob": 0.60, "label": "Real"}, # High prob = Real
            "model_c": {"prob": 0.88, "label": "Real"}  # High prob = Real
        }
