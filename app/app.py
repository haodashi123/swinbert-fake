import streamlit as st
from views import detection
from core.model_engine import RealTimeDetector

# Page Configuration
st.set_page_config(
    page_title="Multi-Modal Fake News Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize detector globally if not present
if 'detector' not in st.session_state:
    st.session_state.detector = RealTimeDetector()

# Sidebar
with st.sidebar:
    st.title("🛡️ Info Governance")
    st.markdown("### Multi-Modal Fake News Detection")
    st.markdown("---")
    st.info("Current Mode: Focused Detection")
    st.markdown("This system integrates Swin Transformer (Visual) and BERT (Textual) dual-stream architecture, incorporating BLIP for semantic augmentation.")
    
    st.markdown("---")
    st.caption("Powered by Zhao Yihao & PyTorch")
    st.caption("v2.1 (Detection Only)")

# Main Content - Direct Render
detection.render_detection()
