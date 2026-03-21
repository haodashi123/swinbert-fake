import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import sys
import requests
from io import BytesIO
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from captum.attr import LayerIntegratedGradients
from transformers import BertTokenizer, BertModel, AutoImageProcessor, SwinModel

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Network Definition
from train_model import DynamicFusionNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path, config):
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return None
        
    try:
        checkpoint = torch.load(model_path, map_location=DEVICE)
        model = DynamicFusionNet(config).to(DEVICE)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def reshape_transform_swin(tensor, height=7, width=7):
    # Swin output is (B, 49, 1024) -> reshape to (B, 1024, 7, 7)
    # Removing CLS token if present (usually not in Swin basic output but careful)
    result = tensor.transpose(1, 2)
    result = result.reshape(tensor.size(0), -1, height, width)
    return result

def visualize_gradcam(model, image_path, target_layer, output_path):
    print("Generating Grad-CAM...")
    
    # Load Image
    try:
        if image_path.startswith("http"):
            response = requests.get(image_path, timeout=5)
            img = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            img = Image.open(image_path).convert("RGB")
    except:
        print("Failed to load image.")
        return

    # Preprocess
    processor = AutoImageProcessor.from_pretrained("microsoft/swin-base-patch4-window7-224")
    inputs = processor(images=img, return_tensors="pt")
    pixel_values = inputs['pixel_values'].to(DEVICE)
    
    # Swin Model Wrapper for GradCAM
    # We need to wrap the Swin backbone because GradCAM expects a model that takes image and returns logits
    # But here we are visualizing the backbone attention, not the full fusion model's decision on image only.
    # Actually, GradCAM usually works on the final classification. 
    # Let's target the Fusion Model's visual pipeline.
    
    class ModelWrapper(nn.Module):
        def __init__(self, fusion_model, swin_model):
            super().__init__()
            self.fusion_model = fusion_model
            self.swin_model = swin_model
            
        def forward(self, x):
            # 1. Swin Feature
            swin_out = self.swin_model(x)
            img_emb = swin_out.pooler_output
            
            # 2. Dummy Text & Caption
            dummy_text = torch.zeros((x.size(0), 768)).to(DEVICE)
            dummy_cap = torch.zeros((x.size(0), 768)).to(DEVICE)
            
            # 3. Fusion Forward
            logits = self.fusion_model(dummy_text, img_emb, dummy_cap)
            return logits

    # Load separate Swin backbone
    swin_backbone = SwinModel.from_pretrained("microsoft/swin-base-patch4-window7-224").to(DEVICE)
    swin_backbone.eval()
    
    wrapper = ModelWrapper(model, swin_backbone)
    
    # Target Layer: Last Norm layer of Swin
    target_layers = [swin_backbone.layernorm]

    cam = GradCAM(model=wrapper, target_layers=target_layers, reshape_transform=reshape_transform_swin)
    
    # Target Category: 1 (Real) or 0 (Fake). Let's maximize the predicted class.
    targets = [ClassifierOutputTarget(0)] # 0 = Fake

    grayscale_cam = cam(input_tensor=pixel_values, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    # Overlay
    img_resized = img.resize((224, 224))
    rgb_img = np.float32(img_resized) / 255
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    # Save
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    
    Image.fromarray(visualization).save(output_path)
    print(f"Grad-CAM saved to {output_path}")

def visualize_text_attr(model, text, output_path):
    print("Generating Text Attribution...")
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_backbone = BertModel.from_pretrained("bert-base-uncased").to(DEVICE)
    bert_backbone.eval()
    
    # Wrapper for Captum
    class TextWrapper(nn.Module):
        def __init__(self, fusion_model, bert_model):
            super().__init__()
            self.fusion_model = fusion_model
            self.bert_model = bert_model
            
        def forward(self, input_embeds):
            # input_embeds: (B, Seq, Hidden)
            # BERT forward with inputs_embeds
            outputs = self.bert_model(inputs_embeds=input_embeds)
            text_emb = outputs.last_hidden_state[:, 0, :] # CLS
            
            # Dummies
            dummy_img = torch.zeros((input_embeds.size(0), 1024)).to(DEVICE)
            dummy_cap = torch.zeros((input_embeds.size(0), 768)).to(DEVICE)
            
            logits = self.fusion_model(text_emb, dummy_img, dummy_cap)
            return logits

    wrapper = TextWrapper(model, bert_backbone)
    
    # Inputs
    encoded = tokenizer(text, return_tensors='pt', max_length=128, truncation=True, padding='max_length')
    input_ids = encoded['input_ids'].to(DEVICE)
    
    # Get Embeddings
    embeddings = bert_backbone.embeddings(input_ids)
    
    # Lig
    lig = LayerIntegratedGradients(wrapper, bert_backbone.embeddings)
    
    # Attribution
    # Target=0 (Fake)
    # n_steps=10 to save memory
    attributions, delta = lig.attribute(inputs=embeddings, target=0, n_steps=10, return_convergence_delta=True)
    
    # Sum over hidden dimension
    attributions = attributions.sum(dim=2).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    attr_scores = attributions.cpu().detach().numpy()
    
    # Decode tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    # Generate HTML
    html = "<h3>Text Attribution (Red = Contributes to Fake)</h3><p>"
    for token, score in zip(tokens, attr_scores):
        if token == "[PAD]": continue
        color = f"rgba(255, 0, 0, {abs(score)*5})" # Scale for visibility
        html += f"<span style='background-color:{color}; padding:2px; margin:1px'>{token}</span> "
    html += "</p>"
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Text attribution saved to {output_path}")

def generate_vis(text, image_path):
    # Load Best Model (Swin)
    cfg = {'use_text': True, 'use_image': True, 'use_caption': False, 'visual_backbone': 'swin'}
    model = load_model("models/model_text_image_swin.pth", cfg)
    
    if model:
        # 1. Grad-CAM
        visualize_gradcam(model, image_path, None, "outputs/gradcam_sample.jpg")
        
        # 2. Text Attribution
        visualize_text_attr(model, text, "outputs/text_attr.html")

if __name__ == "__main__":
    # Example Usage
    sample_text = "Breaking: Shocking video shows alien invasion in New York City! Government hiding the truth."
    sample_img = "https://via.placeholder.com/224" # Replace with real path if available
    
    generate_vis(sample_text, sample_img)
