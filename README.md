# SwinBERT-Fake: Boosting Multimodal Fake News Detection via Hierarchical Visual Modeling and Explainable AI

## 📖 Project Overview
This repository contains the code and dataset split used in our IJCNN submission:

**SwinBERT-Fake: Boosting Multimodal Fake News Detection via Hierarchical Visual Modeling and Explainable AI**

It implements a multimodal fake news detector (Text + Image + Caption) with interpretability (SHAP + visual/text attributions) and includes fixed train/val/test CSV splits for reproducibility.

## 🚀 Core Features

### 1. Multi-Model Comparative Detection
The system runs three parallel inference streams to demonstrate architectural superiority:
*   **Baseline**: BERT (Text) + **ViT** (Image).
*   **Semantic+**: BERT (Text) + **ViT** (Image) + **BLIP** (Generated Caption).
*   **Ours (Swin)**: BERT (Text) + **Swin Transformer** (Image).
    *   *Highlights the advantage of Shifted Window Attention in capturing subtle visual manipulation traces.*

### 2. Advanced Interpretability System (XAI)
We implement a "Top-Down" explanation pipeline that answers *why* a decision was made:

*   **Global Modality Attribution (The "Brain")**:
    *   **Method**: **SHAP (Shapley Additive exPlanations)**.
    *   **Target**: The Trainable Fusion Layer (Fully Connected).
    *   **Function**: Quantifies the marginal contribution of Text vs. Image features to the final decision. It reveals whether the model relies more on semantic inconsistencies or visual artifacts.

*   **Fine-Grained Visual Attention (The "Eyes")**:
    *   **Method**: **EigenCAM (Principal Component Class Activation Mapping)**.
    *   **Target**: The last Block of Swin Transformer / Final LayerNorm of ViT.
    *   **Why EigenCAM?**: Unlike Grad-CAM, EigenCAM computes the principal components of feature maps without relying on noisy back-propagated gradients. This provides robust, object-centric heatmaps even for frozen backbones, clearly visualizing *where* the model is looking.

*   **Fine-Grained Textual Attribution**:
    *   **Method**: **Layer Integrated Gradients (LIG)**.
    *   **Target**: BERT Word Embeddings.
    *   **Function**: Computes the integral of gradients from the input embedding layer to the final output, assigning precise importance scores to every word (token) in the input text.

### 3. Dataset Splits (Reproducibility)
We provide fixed CSV split files containing sample `id`, `image_url`, text fields, and labels:
* `data/dataset_train.csv`
* `data/dataset_val.csv`
* `data/dataset_test.csv`

## ⚙️ Detection Logic & User Flow (`views/detection.py`)

The Detection page is the core interactive interface, designed with an "Academic Minimalist" aesthetic.

**1. Input Processing**
*   **Manual Mode**: Users can enter text and upload an image directly.
*   **Dataset Mode**: The Streamlit demo loads sample entries from Fakeddit **`multimodal_test_public.tsv`** (under `fakeddit_multimodal_only_samples/`) and randomly samples a small subset for interactive analysis. This demo dataset is only for UI demonstration and qualitative inspection, and is not the dataset split used to produce the reported experimental results.

**2. Inference Pipeline**
When "Execute Analysis" is clicked, the `RealTimeDetector` performs:
*   **Feature Extraction**: Uses BERT for text and Swin/ViT for images.
*   **Caption Generation**: Uses BLIP to generate a descriptive caption for the image.
*   **Parallel Prediction**: Runs inference on three models simultaneously (Baseline, Semantic+, Ours).
*   **SHAP Analysis**: Calculates the global contribution of text vs. image for the final decision.

**3. Visualization & Reporting**
*   **Result Cards**: Displays the prediction (Real/Fake) and confidence score for all three models side-by-side.
    *   *Note: Confidence Score is defined as the probability of the **predicted class**. If $P(Real) > 0.5$, Score = $P(Real)$; otherwise, Score = $1 - P(Real)$.*
*   **Insight Note**: Automatically generates a natural language summary of the result (e.g., "Architecture Divergence: Swin corrected the Baseline's error...").
*   **Visual Heatmaps**: Displays the Original Image, ViT Heatmap, and Swin Heatmap side-by-side to compare attention patterns.
*   **Text Heatmap**: Uses a custom Matplotlib-based renderer to highlight important words (based on LIG scores), filtering out special tokens like `[CLS]`.

## 🛠️ System Architecture & Experimental Setup

### 1. Feature Extraction (Offline)
To ensure training efficiency on limited hardware (e.g., RTX 3050 4GB), we adopt a "Frozen Backbone" strategy where features are pre-computed.

*   **Script**: `scripts/tools/extract_features.py`
*   **Batch Size**: 16 (Optimized for 4GB VRAM stability)
*   **Backbones (Frozen)**:
    *   **Text**: `bert-base-uncased` (Output: `[CLS]` token, 768-dim)
    *   **Image (Ours)**: `microsoft/swin-base-patch4-window7-224` (Output: Pooled, 1024-dim)
    *   **Image (Baseline)**: `google/vit-base-patch16-224` (Output: `[CLS]` token, 768-dim)
    *   **Caption**: `Salesforce/blip-image-captioning-base` (Output: `[CLS]` token, 768-dim)
*   **Output Storage**:
    *   `data/features_train.pkl`: Serialized dictionary of training tensors.
    *   `data/features_val.pkl`: Serialized dictionary of validation tensors.
    *   `data/features_test.pkl`: Serialized dictionary of testing tensors.

### 2. Fusion Network Training
We train a lightweight Multi-Layer Perceptron (MLP) on top of the frozen features.

*   **Script**: `scripts/train_model.py`
*   **Architecture**:
    *   **Input**: Concatenated vectors (e.g., Text 768 + Swin 1024 = 1792 dim).
    *   **Hidden Layer**: Linear(1792 -> 512) -> ReLU -> Dropout(0.5).
    *   **Output Layer**: Linear(512 -> 2) -> Logits.
*   **Hyperparameters**:
    *   **Optimizer**: AdamW
    *   **Learning Rate**: 1e-4
    *   **LR Scheduler**: `ReduceLROnPlateau` (mode='min', factor=0.2, patience=3).
    *   **Weight Decay**: 1e-3 (Strong regularization to prevent overfitting on frozen features).
    *   **Batch Size**: 32
    *   **Epochs**: 50 (with Early Stopping patience=5).
*   **Reproducibility**: Global random seed fixed at `2026`.

### 3. Model Evaluation
Comprehensive performance assessment beyond simple accuracy.

*   **Script**: `scripts/tools/evaluate_model.py`
*   **Metrics**:
    *   **Accuracy**: Overall correctness.
    *   **Precision/Recall/F1**: Weighted average to handle potential class imbalance.
    *   **AUC-ROC**: Area Under the Receiver Operating Characteristic curve.
*   **Visualization Output**:
    *   **Confusion Matrix**: `outputs/confusion_matrix.png`
    *   **ROC Curve**: `outputs/roc_curve.png`
*   **Batch Size**: 32 (Inference mode consumes less memory).

### 4. Computational Environment
All experiments were conducted on a workstation equipped with the following hardware and software specifications:

*   **Hardware**:
    *   **GPU**: NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM)
*   **Software Frameworks**:
    *   **Language**: Python 3.11.9
    *   **Deep Learning Framework**: PyTorch 2.6.0 with CUDA 12.4 support
    *   **Model Library**: HuggingFace Transformers 4.57.3
    *   **Visualization**: Streamlit 1.52.1, Plotly 6.3.1, Matplotlib 3.10.7
    *   **Explainability (XAI)**: SHAP 0.50.0, Captum 0.8.0, Grad-CAM 1.5.5

## 📂 Directory Structure

```
info_governance_platform/
├── app/                    # Streamlit Application (Front-end & Inference)
├── scripts/                # Developer & Research Tools
│   ├── train_model.py      # Training Script for MLP heads
│   ├── check_gpu.py        # CUDA/GPU Diagnostic Utility
│   └── tools/              # Benchmarking & Data Processing Scripts
├── data/                   # Dataset split CSVs (included)
├── README.md               # Documentation
└── requirements.txt        # Python Dependencies
```

## ⚡ Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run the Platform
```bash
streamlit run app/main.py
```
Access the dashboard at `http://localhost:8501`.

### 3. Notes
* Model checkpoints (`models/*.pth`) are not committed to this repository. If you want to run the full inference pipeline, place the corresponding checkpoints under `models/` or train them using the provided scripts.
* Image files are not bundled. The dataset split files include `image_url`, and images can be downloaded on demand.

### 3. (Optional) Reproduce Experiments
1.  **Extract Features**:
    ```bash
    python scripts/tools/extract_features.py
    ```
    *Extracts BERT/Swin/ViT features to PKL files.*
2.  **Train Models**:
    ```bash
    python scripts/train_model.py
    ```
    *Trains the MLP heads using the extracted features.*
3.  **Evaluate**:
    ```bash
    python scripts/tools/evaluate_model.py
    ```
    *Generates confusion matrices and ROC curves.*

## 📝 Citation & Reference
This project implements ideas from recent research in Multi-Modal Fake News Detection.
*   **Swin Transformer**: Liu et al. (ICCV 2021)
*   **BLIP**: Li et al. (ICML 2022)
*   **EigenCAM**: Muhammad et al. (IJCNN 2020)
*   **Integrated Gradients**: Sundararajan et al. (ICML 2017)

---
**Developer**: Zhao Yihao
**Version**: v3.0 (Academic Edition)
**Date**: 2026-03-21
