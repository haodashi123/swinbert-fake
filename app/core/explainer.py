import shap
import torch
import numpy as np

class FusionExplainer:
    def __init__(self, model, background_data, device='cpu'):
        """
        Args:
            model: The trained DynamicFusionNet model.
            background_data: A numpy array (N, total_dim) for SHAP background.
            device: 'cpu' or 'cuda'.
        """
        self.model = model
        self.device = device
        self.background_data = background_data
        
        # Determine feature dimensions based on config
        self.config = model.config
        
        self.dim_text = 768 if self.config.get('use_text', True) else 0
        
        # Image dim depends on backbone
        self.dim_image = 0
        if self.config.get('use_image', True):
            backbone = self.config.get('visual_backbone', 'swin')
            if backbone == 'swin':
                self.dim_image = 1024
            elif backbone == 'vit':
                self.dim_image = 768
            else:
                self.dim_image = 1024 # Default fallback
                
        self.dim_caption = 768 if self.config.get('use_caption', True) else 0
        self.dim_explicit = 5 if self.config.get('use_explicit', True) else 0
        
        # Define slice indices
        current_idx = 0
        self.idx_text_end = current_idx + self.dim_text
        current_idx = self.idx_text_end
        
        self.idx_image_end = current_idx + self.dim_image
        current_idx = self.idx_image_end
        
        self.idx_caption_end = current_idx + self.dim_caption
        current_idx = self.idx_caption_end
        
        self.idx_explicit_end = current_idx + self.dim_explicit
        self.total_dim = current_idx

        # Initialize SHAP Explainer
        # We wrap the model to take a single concatenated vector
        # Using link='logit' to explain logits (default)
        self.explainer = shap.KernelExplainer(self._model_wrapper, self.background_data)

    def _model_wrapper(self, input_data_np):
        """
        Wraps the PyTorch model to accept numpy arrays of shape (batch, total_dim).
        Returns logits as numpy array (batch, 2).
        """
        # Convert to tensor
        input_tensor = torch.tensor(input_data_np, dtype=torch.float32).to(self.device)
        
        # Split into modalities
        # Note: We need to handle batches
        text_emb = input_tensor[:, 0 : self.idx_text_end]
        img_emb = input_tensor[:, self.idx_text_end : self.idx_image_end]
        cap_emb = input_tensor[:, self.idx_image_end : self.idx_caption_end]
        explicit_feat = input_tensor[:, self.idx_caption_end : self.idx_explicit_end]
        
        with torch.no_grad():
            logits, _, _ = self.model(text_emb, img_emb, cap_emb, explicit_feat)
            
        return logits.cpu().numpy()

    def explain(self, sample_vector):
        """
        Explain a single sample.
        Args:
            sample_vector: numpy array of shape (total_dim,)
        Returns:
            dict: { 'Text': %, 'Image': %, 'Caption': %, 'Explicit': % }
        """
        # Ensure 2D (1, D)
        if sample_vector.ndim == 1:
            sample_vector = sample_vector.reshape(1, -1)
            
        # Run SHAP (limit samples for speed if needed)
        # nsamples='auto' or 100. Lower is faster but less accurate.
        shap_values = self.explainer.shap_values(sample_vector, nsamples=100)
        
        # For binary classification, shap_values is a list [array_class0, array_class1]
        # We focus on the "Fake" class (index 1) contribution
        if isinstance(shap_values, list):
            vals = shap_values[1][0] # First sample, class 1
        else:
            vals = shap_values[0] # Regression or single output

        # Calculate absolute contribution sum per modality
        score_text = 0.0
        if self.dim_text > 0:
            score_text = np.sum(np.abs(vals[0 : self.idx_text_end]))
            
        score_image = 0.0
        if self.dim_image > 0:
            score_image = np.sum(np.abs(vals[self.idx_text_end : self.idx_image_end]))
            
        score_caption = 0.0
        if self.dim_caption > 0:
            score_caption = np.sum(np.abs(vals[self.idx_image_end : self.idx_caption_end]))
            
        score_explicit = 0.0
        if self.dim_explicit > 0:
            score_explicit = np.sum(np.abs(vals[self.idx_caption_end : self.idx_explicit_end]))
            
        # Normalize
        total_score = score_text + score_image + score_caption + score_explicit
        if total_score == 0:
            total_score = 1e-9
            
        contributions = {
            "Text": score_text / total_score,
            "Image": score_image / total_score,
            "Caption": score_caption / total_score,
            "Explicit": score_explicit / total_score
        }
        
        return contributions
