import streamlit as st
import pandas as pd
import time
from PIL import Image
from core.model_engine import RealTimeDetector
from core.data_factory import MockDataLoader
from utils.viz_helper import plot_text_heatmap
import urllib.request
from io import BytesIO
import os
import plotly.express as px
import plotly.graph_objects as go

# Initialize detector once
if 'detector' not in st.session_state:
    # Force fresh initialization
    st.session_state.detector = RealTimeDetector()

def render_detection():
    # --- Academic Minimalist Design System ---
    st.markdown("""
        <style>
        /* 1. Global Typography & Reset */
        @import url('https://fonts.googleapis.com/css2?family=Crimson+Pro:ital,wght@0,400;0,600;1,400&family=Inter:wght@300;400;500;600&display=swap');
        
        .main {
            background-color: #ffffff;
            font-family: 'Inter', sans-serif;
            color: #1a1a1a;
            line-height: 1.6;
        }
        
        /* 2. Headers - Serif for Authority */
        h1, h2, h3 {
            font-family: 'Crimson Pro', serif;
            color: #111;
            letter-spacing: -0.02em;
        }
        
        h1 { font-size: 2.2rem; font-weight: 600; margin-bottom: 0.5rem; }
        h3 { font-size: 1.1rem; font-weight: 600; margin-top: 1.5rem; text-transform: uppercase; letter-spacing: 0.05em; font-family: 'Inter', sans-serif; color: #555; border-bottom: 1px solid #eee; padding-bottom: 0.5rem; }
        h5 { font-family: 'Inter', sans-serif; font-weight: 500; font-size: 0.9rem; color: #666; margin-bottom: 0.5rem; }

        /* 3. Layout Containers - Strict Grid */
        .block-container {
            padding-top: 2rem;
            max-width: 1400px;
        }
        
        /* 4. Input Area - Minimalist Form */
        .stTextArea textarea {
            background-color: #fafafa;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            font-family: 'Crimson Pro', serif;
            font-size: 1.1rem;
            color: #333;
        }
        .stTextArea textarea:focus {
            border-color: #333;
            box-shadow: none;
        }
        
        /* 5. Result Cards - Bauhaus Functionalism */
        .metric-card {
            border: 1px solid #e0e0e0;
            padding: 1.5rem;
            background: #fff;
            transition: all 0.2s ease;
            height: 100%;
            position: relative;
        }
        .metric-card:hover {
            border-color: #333;
        }
        .metric-label {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: #888;
            margin-bottom: 0.5rem;
        }
        .metric-value {
            font-family: 'Inter', sans-serif;
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 0.25rem;
        }
        .metric-sub {
            font-family: 'Inter', sans-serif;
            font-size: 0.85rem;
            color: #666;
        }
        
        /* 6. Insight Box - Editorial Note Style */
        .insight-note {
            background-color: #f9f9f9;
            border-left: 3px solid #333;
            padding: 1rem 1.5rem;
            margin-top: 2rem;
            font-family: 'Crimson Pro', serif;
            font-size: 1.05rem;
            color: #333;
            font-style: italic;
        }
        
        /* 7. Buttons - Swiss Style */
        div.stButton > button {
            background-color: #111;
            color: #fff;
            border-radius: 0;
            border: none;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            font-size: 0.8rem;
            padding: 0.8rem 1.5rem;
            font-weight: 500;
        }
        div.stButton > button:hover {
            background-color: #333;
            color: #fff;
        }
        
        /* 8. Utility */
        .caption {
            font-family: 'Inter', sans-serif;
            font-size: 0.8rem;
            color: #888;
            margin-top: 0.5rem;
            text-align: center;
        }
        
        /* Hide Streamlit Branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)

    # --- Header Section ---
    st.markdown("# Multi-Modal Verification System")
    st.markdown("""
    <div style="font-family: 'Crimson Pro', serif; font-size: 1.1rem; color: #555; margin-bottom: 2rem;">
    Comparative analysis of tri-stream fusion architectures for digital information governance.
    </div>
    """, unsafe_allow_html=True)
    
    col_input, col_result = st.columns([1, 1.3], gap="large")
    
    # Initialize variables
    selected_sample = None
    text_input = ""
    uploaded_file = None
    image_url = None
    
    # Session State for Analysis Persistence
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None
    if 'explanation_result' not in st.session_state:
        st.session_state.explanation_result = None
    
    # Auto-clear stale explanation results from previous version
    if st.session_state.explanation_result and 'heatmap_swin' not in st.session_state.explanation_result:
        st.session_state.explanation_result = None

    # --- Left Column: Context Input ---
    with col_input:
        st.markdown("### 01. Context Input")
        
        input_mode = st.radio(
            "Input Mode",
            options=["Manual Entry", "Dataset Sample"],
            horizontal=True,
            label_visibility="collapsed",
            key="input_mode",
        )

        if input_mode == "Manual Entry":
            selected_sample = None
            image_url = None
            text_input = st.text_area("Textual Content", height=180, placeholder="Input text for verification...")
            uploaded_file = st.file_uploader("Visual Evidence (Optional)", type=["png", "jpg", "jpeg"])
            if uploaded_file:
                st.image(uploaded_file, use_container_width=True)
        else:
            # Load samples (Cached)
            if 'dataset_samples' not in st.session_state:
                with st.spinner("Accessing Fakeddit repository..."):
                    st.session_state.dataset_samples = MockDataLoader.load_fakeddit_sample()
            
            samples = st.session_state.dataset_samples
            if 'standard_idx' not in st.session_state:
                st.session_state.standard_idx = 0
            
            dataset_len = len(st.session_state.dataset_samples)
            if st.session_state.standard_idx >= dataset_len:
                st.session_state.standard_idx = 0

            sample_options = [f"Sample {i}: {s['clean_title'][:40]}..." for i, s in enumerate(st.session_state.dataset_samples)]
            if "std_selector" not in st.session_state:
                st.session_state.std_selector = sample_options[st.session_state.standard_idx]
            
            c_prev_std, c_info_std, c_next_std = st.columns([1, 2, 1])
            
            with c_prev_std:
                if st.button("Previous Sample"):
                    new_idx = max(0, st.session_state.standard_idx - 1)
                    st.session_state.standard_idx = new_idx
                    st.session_state.std_selector = sample_options[new_idx]
                    st.rerun()
                    
            with c_next_std:
                if st.button("Next Sample"):
                    new_idx = min(dataset_len - 1, st.session_state.standard_idx + 1)
                    st.session_state.standard_idx = new_idx
                    st.session_state.std_selector = sample_options[new_idx]
                    st.rerun()

            def on_std_select_change():
                val = st.session_state.std_selector
                try:
                    st.session_state.standard_idx = sample_options.index(val)
                except:
                    pass

            st.selectbox(
                "Select a sample to analyze:",
                options=sample_options,
                key="std_selector",
                on_change=on_std_select_change
            )
            
            sample_index = st.session_state.standard_idx
            selected_sample = st.session_state.dataset_samples[sample_index]
                
            if selected_sample:
                text_input = selected_sample['clean_title']
                image_url = selected_sample['image_url']
                
                # Minimalist Sample Display
                st.markdown(f"""
                <div style="margin-top: 1rem; padding: 1rem; border: 1px solid #eee; background: #fafafa;">
                    <div style="font-size: 0.8rem; color: #888; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;">Record Metadata</div>
                    <div style="font-family: 'Crimson Pro', serif; font-size: 1.1rem; margin-bottom: 1rem;">{text_input}</div>
                    <div style="font-size: 0.8rem; color: #555;">Ground Truth: <strong>{'Real' if selected_sample['label'] == 1 else 'Fake'}</strong></div>
                </div>
                """, unsafe_allow_html=True)
                
                
                if image_url:
                    st.image(image_url, use_container_width=True)

        st.markdown("---")
        predict_btn = st.button("Execute Analysis")
        
        if predict_btn and text_input:
            # Clear previous results first to avoid stale data mixing
            st.session_state.analysis_result = None
            st.session_state.explanation_result = None
            
            # Subtle Progress
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            progress_bar.empty()
            
            # Inference
            try:
                img_source = image_url if selected_sample else (uploaded_file if uploaded_file else None)
                result = st.session_state.detector.predict_all(text_input, img_source)
                st.session_state.analysis_result = result # Store result
                
                # Explanation (Run immediately to store)
                try:
                    with st.spinner("Visualizing attention mechanisms..."):
                        heatmap_swin, heatmap_vit, tokens, cap_tokens, cap_text = st.session_state.detector.explain(text_input, img_source)
                        st.session_state.explanation_result = {
                            'heatmap_swin': heatmap_swin,
                            'heatmap_vit': heatmap_vit,
                            'tokens': tokens,
                            'cap_tokens': cap_tokens,
                            'cap_text': cap_text
                        }
                except Exception as e:
                    st.session_state.explanation_result = {
                        'error': str(e),
                        'heatmap_swin': None,
                        'heatmap_vit': None,
                        'tokens': [],
                        'cap_tokens': [],
                        'cap_text': ""
                    }
                    
            except Exception as e:
                st.error(f"Computation Error: {e}")
                st.session_state.analysis_result = RealTimeDetector.simulate_prediction()
                st.session_state.explanation_result = None

    # --- Right Column: Analytical Output ---
    with col_result:
        st.markdown("### 02. Analytical Output")
        
        if st.session_state.analysis_result:
            result = st.session_state.analysis_result
            
            # --- Results Grid ---
            c1, c2, c3 = st.columns(3)
            
            def render_metric(col, title, subtitle, prob, label):
                conf = prob if label == "Real" else 1 - prob
                accent_color = "#1a1a1a" if label == "Real" else "#c0392b"
                
                with col:
                    st.markdown(f"""
                    <div class="metric-card" style="border-top: 2px solid {accent_color};">
                        <div class="metric-label">{title}</div>
                        <div style="font-size: 0.75rem; color: #999; margin-bottom: 0.5rem;">{subtitle}</div>
                        <div class="metric-value" style="color: {accent_color};">{label}</div>
                        <div class="metric-sub">Confidence: {conf:.1%}</div>
                    </div>
                    """, unsafe_allow_html=True)

            render_metric(c1, "Baseline", "BERT + ViT", result['model_a']['prob'], result['model_a']['label'])
            render_metric(c2, "Semantic+", "BERT + ViT + BLIP", result['model_b']['prob'], result['model_b']['label'])
            render_metric(c3, "Ours (Swin)", "BERT + Swin Trans.", result['model_c']['prob'], result['model_c']['label'])
            
            # --- Generated Caption Display ---
            if result.get('caption') and result['caption'] != "caption unavailable":
                st.markdown(f"""
                <div style="margin-top: 1rem; padding: 0.8rem; background-color: #fafafa; border: 1px dashed #ddd; font-family: 'Crimson Pro', serif; font-size: 0.95rem; color: #555;">
                    <strong style="color: #333; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.05em; margin-right: 0.5rem;">BLIP Generated Context:</strong> 
                    <em>"{result['caption']}"</em>
                </div>
                """, unsafe_allow_html=True)
            
            # --- Decision Attribution (New) ---
            if result.get('shap_scores'):
                st.markdown("""
                <div style="margin-top: 2rem; border-top: 1px solid #eee; padding-top: 1rem;">
                    <h5 style="margin-bottom: 0;">Modality Contribution Analysis</h5>
                    <div style="font-size: 0.8rem; color: #999; margin-bottom: 1rem;">Quantifying feature importance via SHAP (Text vs. Image)</div>
                </div>
                """, unsafe_allow_html=True)
                
                scores = result['shap_scores']
                
                # Plotly Horizontal Stacked Bar (Compact Percentage Strip)
                
                # Colors (Academic / Colorblind-friendly)
                # Text: Muted Blue (Professional, Calm) -> #4A90E2
                # Image: Warm Orange (Contrast, Active) -> #F5A623
                # Alternative (Nature/Science style):
                # Text: #2E86C1 (Strong Blue)
                # Image: #E67E22 (Carrot Orange) -> Keep similar but refined
                
                # Let's try a "Teal & Coral" modern academic look
                color_text = "#45B39D"
                color_image = "#EC7063"
                
                fig = go.Figure()

                # Text Contribution Trace
                fig.add_trace(go.Bar(
                    y=['Total Contribution'],
                    x=[scores['text_pct']],
                    name='Text Context',
                    orientation='h',
                    marker=dict(color=color_text, line=dict(color='black', width=1.5)),
                    text=[f"Text: {scores['text_pct']:.1%}"],
                    textposition='auto',
                    textfont=dict(color='white')
                ))

                # Image Contribution Trace
                fig.add_trace(go.Bar(
                    y=['Total Contribution'],
                    x=[scores['image_pct']],
                    name='Visual Content',
                    orientation='h',
                    marker=dict(color=color_image, line=dict(color='black', width=1.5)),
                    text=[f"Image: {scores['image_pct']:.1%}"],
                    textposition='auto',
                    textfont=dict(color='white')
                ))

                # Display Layout (Compact Strip)
                fig.update_layout(
                    barmode='stack',
                    showlegend=False,
                    margin=dict(t=20, b=20, l=20, r=20),
                    height=100, # Further reduced height since legend is gone
                    font=dict(family="Times New Roman", size=18, color='black'),
                    plot_bgcolor='white',
                    xaxis=dict(
                        visible=False,
                        range=[0, 1]
                    ),
                    yaxis=dict(
                        visible=False
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                
                # Download Button for Compact Chart
                try:
                    # High-res export configuration
                    fig.update_layout(
                         font=dict(family="Times New Roman", size=45), # Increased from 35
                         margin=dict(t=30, b=30, l=30, r=30), # Reduced margins slightly
                         height=150 # Reduced height from 200 to 150
                    )
                    fig.update_traces(textfont=dict(size=45)) # Increased from 35
                    
                    img_bytes = fig.to_image(format="png", width=1200, height=150, scale=2)
                    st.download_button(
                        label="Download Contribution Strip",
                        data=img_bytes,
                        file_name="modality_contribution_strip.png",
                        mime="image/png",
                        key="dl_contribution"
                    )
                except Exception as e:
                    # Fallback
                    st.warning(f"Chart download unavailable: {e}")

            # --- Synthesis / Insight ---
            label_a = result['model_a']['label']
            label_c = result['model_c']['label']
            gt_label = "Real" if (selected_sample and selected_sample['label'] == 1) else ("Fake" if selected_sample else None)
            
            insight_text = ""
            if label_a != label_c:
                if gt_label:
                    if label_c == gt_label:
                        insight_text = f"Architecture Divergence: The Swin Transformer backbone correctly identified this instance as {label_c}, correcting the Baseline's misclassification. This highlights the superior feature extraction capability for subtle visual-semantic discrepancies."
                    elif label_a == gt_label:
                        insight_text = f"Architecture Divergence: The Baseline model outperformed the Swin variant in this specific instance, correctly identifying it as {label_a}."
                    else:
                        insight_text = "Systemic Failure: All architectures failed to align with Ground Truth, indicating high-complexity features beyond current model capacity."
                else:
                    insight_text = f"Model Disagreement: Baseline ({label_a}) and Swin ({label_c}) predictions diverge. This ambiguity necessitates human expert review."
            elif label_a == label_c and label_c == "Fake":
                insight_text = "Consensus: Cross-architecture agreement indicates a high probability of fabrication."
            else:
                insight_text = "Consensus: Cross-architecture agreement indicates verification as authentic content."
                
            st.markdown(f"""
            <div class="insight-note">
                "{insight_text}"
            </div>
            """, unsafe_allow_html=True)
            
        elif not text_input:
            st.markdown("""
            <div style="color: #999; font-style: italic; margin-top: 2rem; font-family: 'Crimson Pro', serif;">
            Awaiting input for verification protocol...
            </div>
            """, unsafe_allow_html=True)
            
    # --- 03. Interpretability Section (Full Width) ---
    if st.session_state.analysis_result and st.session_state.explanation_result:
        st.markdown("---")
        
        # Container for Interpretability with distinct styling
        with st.container():
            st.markdown("""
            <div style="padding: 1rem 0;">
                <h3 style="border-bottom: none; margin-bottom: 1rem;">03. Interpretability Analysis</h3>
                <div style="font-family: 'Inter', sans-serif; font-size: 0.9rem; color: #666; margin-bottom: 2rem;">
                    Multi-modal feature attribution and cross-modal semantic consistency verification.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            exp = st.session_state.explanation_result or {}
            if exp.get("error"):
                st.warning(f"解释生成失败：{exp.get('error')}")
            heatmap_swin = exp.get('heatmap_swin')
            heatmap_vit = exp.get('heatmap_vit')
            tokens = exp.get('tokens') or []
            cap_tokens = exp.get('cap_tokens') or []
            
            # --- 1. Visual Analysis (Full Width Comparison) ---
            st.markdown("#### A. Visual Attention Analysis")
            st.caption("Comparison of Feature Activation Maps: Baseline (ViT) vs. Ours (Swin Transformer)")
            
            st.markdown("""
            <div style="background-color: #fafafa; padding: 1rem; border-radius: 4px; border: 1px solid #eee; margin-bottom: 2rem;">
            """, unsafe_allow_html=True)
            
            col_v1, col_v2, col_v3 = st.columns(3)
            
            with col_v1:
                st.markdown("**Original Input**")
                if image_url:
                    st.image(image_url, use_container_width=True)
                elif uploaded_file:
                    st.image(uploaded_file, use_container_width=True)
                else:
                    st.markdown("*No visual input provided*")
            
            with col_v2:
                st.markdown("**Baseline (ViT)**")
                if heatmap_vit:
                    st.image(heatmap_vit, use_container_width=True, caption="Layer: Final Norm")
                    
                    # Download Button for ViT Heatmap
                    buf_vit = BytesIO()
                    heatmap_vit.save(buf_vit, format="PNG")
                    byte_im_vit = buf_vit.getvalue()
                    st.download_button(
                        label="Download ViT Map",
                        data=byte_im_vit,
                        file_name="vit_heatmap.png",
                        mime="image/png",
                        key="dl_vit"
                    )
                else:
                    st.info("N/A")

            with col_v3:
                st.markdown("**Ours (Swin)**")
                if heatmap_swin:
                    st.image(heatmap_swin, use_container_width=True, caption="Layer: Final Block")
                    
                    # Download Button for Swin Heatmap
                    buf = BytesIO()
                    heatmap_swin.save(buf, format="PNG")
                    byte_im = buf.getvalue()
                    st.download_button(
                        label="Download Swin Map",
                        data=byte_im,
                        file_name="swin_heatmap.png",
                        mime="image/png",
                        key="dl_swin"
                    )
                else:
                    st.info("N/A")
            
            st.markdown("</div>", unsafe_allow_html=True)

            # --- 2. Textual Analysis ---
            st.markdown("#### B. Textual Attribution")
            st.caption("Method: Integrated Gradients | Model: BERT-Base")
            
            st.markdown("""
            <div style="background-color: #fafafa; padding: 1rem; border-radius: 4px; border: 1px solid #eee;">
            """, unsafe_allow_html=True)
            
            if tokens:
                # Use Matplotlib Visualization
                words, scores = zip(*tokens)
                # Full width plot
                fig = plot_text_heatmap(words, scores, width=20) 
                st.pyplot(fig, use_container_width=True)
                
                # Download Button
                buf_txt = BytesIO()
                fig.savefig(buf_txt, format="png", dpi=300, bbox_inches='tight', pad_inches=0.01)
                byte_txt = buf_txt.getvalue()
                
                st.download_button(
                    label="Download Attribution",
                    data=byte_txt,
                    file_name="text_attribution.png",
                    mime="image/png",
                    key="dl_text"
                )
            else:
                st.info("Textual modality unavailable.")
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Row 2: Caption (Full Width)
            st.markdown("#### C. Semantic Hallucination Analysis")
            st.caption("Method: Cross-Modal Consistency | Model: BLIP Image Captioning")
            
            st.markdown("""
            <div style="background-color: #fafafa; padding: 1rem; border-radius: 4px; border: 1px solid #eee; margin-top: 0.5rem;">
            """, unsafe_allow_html=True)
            
            if cap_tokens:
                 # Use Matplotlib Visualization with very wide setting
                 words_c, scores_c = zip(*cap_tokens)
                 fig_c = plot_text_heatmap(words_c, scores_c, width=20) 
                 st.pyplot(fig_c, use_container_width=True)
                 
                 # Download Button
                 buf_cap = BytesIO()
                 fig_c.savefig(buf_cap, format="png", dpi=300, bbox_inches='tight', pad_inches=0.01)
                 byte_cap = buf_cap.getvalue()
                 
                 col_dl, _ = st.columns([1, 4])
                 with col_dl:
                     st.download_button(
                         label="Download Caption Map",
                         data=byte_cap,
                         file_name="caption_attribution.png",
                         mime="image/png",
                         key="dl_cap"
                     )
            else:
                st.info("Caption attribution unavailable.")
            st.markdown("</div>", unsafe_allow_html=True)
