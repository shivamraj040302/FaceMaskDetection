import streamlit as st
import cv2
import numpy as np
from PIL import Image
import detect

# Page config
st.set_page_config(
    page_title="Face Mask Detector",
    page_icon="😷",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600&display=swap');

/* Dark theme background */
.stApp {
    background: linear-gradient(135deg, #0a0a0f 0%, #0d1117 50%, #0a0f1a 100%);
    color: #e0e0e0;
}

/* Hide default streamlit elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Main title */
.main-title {
    font-family: 'Orbitron', monospace;
    font-size: 3rem;
    font-weight: 900;
    text-align: center;
    background: linear-gradient(90deg, #00ff88, #00d4ff, #7b2fff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    padding: 1rem 0 0.2rem 0;
    letter-spacing: 2px;
}

.subtitle {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.1rem;
    text-align: center;
    color: #666;
    letter-spacing: 4px;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

/* Divider */
.neon-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #00ff88, #00d4ff, transparent);
    margin: 1rem 0 2rem 0;
}

/* Stat cards */
.stat-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(0,255,136,0.2);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    font-family: 'Rajdhani', sans-serif;
}

.stat-number {
    font-family: 'Orbitron', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #00ff88;
}

.stat-label {
    font-size: 0.8rem;
    color: #888;
    letter-spacing: 2px;
    text-transform: uppercase;
}

/* Upload area */
.upload-box {
    border: 2px dashed rgba(0,212,255,0.3);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    background: rgba(0,212,255,0.03);
    margin: 1rem 0;
}

/* Result badges */
.badge-mask {
    display: inline-block;
    background: rgba(0,255,136,0.15);
    border: 1px solid #00ff88;
    color: #00ff88;
    padding: 0.3rem 1rem;
    border-radius: 20px;
    font-family: 'Orbitron', monospace;
    font-size: 0.8rem;
    font-weight: 700;
}

.badge-nomask {
    display: inline-block;
    background: rgba(255,50,50,0.15);
    border: 1px solid #ff3232;
    color: #ff3232;
    padding: 0.3rem 1rem;
    border-radius: 20px;
    font-family: 'Orbitron', monospace;
    font-size: 0.8rem;
    font-weight: 700;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(0,0,0,0.5);
    border-right: 1px solid rgba(0,255,136,0.1);
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #00ff88, #00d4ff);
    color: #000;
    border: none;
    border-radius: 8px;
    font-family: 'Orbitron', monospace;
    font-weight: 700;
    letter-spacing: 1px;
    padding: 0.5rem 2rem;
    width: 100%;
    transition: all 0.3s;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 0 20px rgba(0,255,136,0.4);
}

/* Images */
.stImage {
    border-radius: 12px;
    overflow: hidden;
}

/* Info box */
.info-box {
    background: rgba(0,212,255,0.05);
    border-left: 3px solid #00d4ff;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.5rem;
    font-family: 'Rajdhani', sans-serif;
    margin: 1rem 0;
}

/* Section headers */
.section-header {
    font-family: 'Orbitron', monospace;
    font-size: 1rem;
    color: #00d4ff;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin: 1.5rem 0 1rem 0;
}

/* Selectbox */
.stSelectbox label {
    font-family: 'Rajdhani', sans-serif;
    color: #888;
    letter-spacing: 2px;
    text-transform: uppercase;
    font-size: 0.8rem;
}

/* Spinner */
.stSpinner > div {
    border-top-color: #00ff88 !important;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-title">😷 MASK DETECTOR</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Real-time Face Mask Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('<div class="section-header">⚙ Controls</div>', unsafe_allow_html=True)
    
    mode = st.selectbox("Detection Mode", ["📁 Upload Image", "📷 Live Webcam"])
    
    st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">ℹ System Info</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        <b>Face Detection:</b> YOLOv8<br>
        <b>Classification:</b> Custom CNN<br>
        <b>Input Size:</b> 64×64<br>
        <b>Classes:</b> Mask / No Mask
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">🎨 Legend</div>', unsafe_allow_html=True)
    st.markdown('<span class="badge-mask">✓ WITH MASK</span>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<span class="badge-nomask">✗ NO MASK</span>', unsafe_allow_html=True)

# Main content
if mode == "📁 Upload Image":
    st.markdown('<div class="section-header">📤 Upload Image</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Drop your image here",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown('<div class="section-header">🖼 Original</div>', unsafe_allow_html=True)
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)

        with col2:
            st.markdown('<div class="section-header">🔍 Detected</div>', unsafe_allow_html=True)
            with st.spinner("Analyzing faces..."):
                annotated = detect.detect_and_annotate(img)
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

        # Stats
        st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">📊 Detection Stats</div>', unsafe_allow_html=True)

        results = detect.yolo_face(img, verbose=False)
        total_faces = sum(len(r.boxes) for r in results)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{total_faces}</div>
                <div class="stat-label">Faces Detected</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number" style="color:#00d4ff">{img.shape[1]}×{img.shape[0]}</div>
                <div class="stat-label">Image Resolution</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number" style="color:#7b2fff">YOLOv8</div>
                <div class="stat-label">Detection Model</div>
            </div>""", unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="upload-box">
            <h3 style="font-family:'Orbitron',monospace; color:#00d4ff; font-size:1rem;">DROP IMAGE HERE</h3>
            <p style="font-family:'Rajdhani',sans-serif; color:#666;">Supports JPG, JPEG, PNG</p>
        </div>
        """, unsafe_allow_html=True)

elif mode == "📷 Live Webcam":
    st.markdown('<div class="section-header">📷 Live Detection</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        Click <b>Take photo</b> to capture from your device camera.
    </div>
    """, unsafe_allow_html=True)

    img_file = st.camera_input("Take a photo")

    if img_file is not None:
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        with st.spinner("Analyzing..."):
            annotated = detect.detect_and_annotate(frame)

        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)