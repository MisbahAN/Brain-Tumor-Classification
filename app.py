import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.metrics import Precision, Recall
import numpy as np
from PIL import Image
import os
import time

# Page config
st.set_page_config(
    page_title="Brain Tumor Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for clean white and purple theme
st.markdown("""
<style>
    /* Import Inter font (similar to Claude's) */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global reset and styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    /* Hide sidebar completely */
    section[data-testid="stSidebar"] {
        display: none;
    }
    
    /* Main app background */
    .stApp {
        background-color: #FAFAFA;
        min-height: 100vh;
        position: relative;
    }
    
    .main {
        background-color: #FAFAFA;
        padding: 2rem 1rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Center all content */
    .block-container {
        max-width: 1000px !important;
        padding-top: 3rem !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
        padding-bottom: 0 !important;
        margin: 0 auto !important;
    }
    
    /* Fix streamlit header overlap */
    .stApp > header {
        height: 3rem !important;
    }
    
    /* Remove empty containers */
    .element-container:has(.stMarkdown:empty) {
        display: none !important;
    }
    
    .element-container:empty {
        display: none !important;
    }
    
    div[data-testid="stVerticalBlock"] > div:has(div:empty):only-child {
        display: none !important;
    }
    
    /* All text in purple */
    h1, h2, h3, h4, h5, h6, p, span, div, label {
        color: #6B46C1 !important;
    }
    
    /* Main title */
    h1 {
        text-align: center;
        font-weight: 600;
        font-size: 2rem !important;
        margin-top: 0 !important;
        margin-bottom: 2rem !important;
        padding-top: 1rem !important;
        color: #6B46C1 !important;
        letter-spacing: -0.5px;
    }
    
    /* First h1 needs extra top margin to clear header */
    .main > div:first-child h1 {
        margin-top: 1rem !important;
    }
    
    /* Section headers */
    h3 {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        margin-bottom: 1rem !important;
        color: #6B46C1 !important;
        text-align: center;
    }
    
    h4 {
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        color: #6B46C1 !important;
    }
    
    /* Card container styling - REMOVED since we're not using containers */
    
    /* Input method buttons (radio buttons styled as buttons) */
    .stRadio > div {
        display: flex !important;
        gap: 0.75rem !important;
        background: transparent !important;
        padding: 0 !important;
        box-shadow: none !important;
        justify-content: center !important;
        margin-top: 1rem !important;
    }
    
    .stRadio > div > label {
        flex: 1 !important;
        background-color: #FFFFFF !important;
        border: 2px solid #E9D5FF !important;
        border-radius: 8px !important;
        padding: 0.75rem 1.5rem !important;
        text-align: center !important;
        cursor: pointer !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        transition: none !important;
        margin: 0 !important;
    }
    
    .stRadio > div > label > div {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    
    .stRadio > div > label:has(input:checked) {
        background-color: #F3E8FF !important;
        border-color: #6B46C1 !important;
    }
    
    /* Hide radio circles */
    .stRadio > div > label > div > input[type="radio"] {
        display: none !important;
    }
    
    /* Button styling - modern outline design */
    .stButton > button {
        background-color: #FFFFFF !important;
        color: #6B46C1 !important;
        border: 2px solid #6B46C1 !important;
        box-shadow: 0 1px 3px rgba(107, 70, 193, 0.12) !important;
        font-weight: 500 !important;
        padding: 0.6rem 1.5rem !important;
        border-radius: 8px !important;
        transition: none !important;
        width: 100% !important;
        font-size: 0.9rem !important;
        letter-spacing: 0.2px !important;
    }
    
    .stButton > button:hover {
        background-color: #FFFFFF !important;
        box-shadow: 0 1px 3px rgba(107, 70, 193, 0.12) !important;
        border: 2px solid #6B46C1 !important;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background-color: #FFFFFF !important;
        border: 2px solid #E9D5FF !important;
        border-radius: 8px !important;
        color: #6B46C1 !important;
        font-size: 0.9rem !important;
    }
    
    .stSelectbox label {
        color: #6B46C1 !important;
        font-weight: 500 !important;
        margin-bottom: 0.5rem !important;
        font-size: 0.9rem !important;
        text-align: center !important;
        display: block !important;
    }
    
    /* File uploader - clean white */
    [data-testid="stFileUploader"] {
        background-color: transparent !important;
    }
    
    [data-testid="stFileUploaderDropzone"] {
        background-color: #FFFFFF !important;
        border: 2px dashed #E9D5FF !important;
        border-radius: 8px !important;
        padding: 1.5rem !important;
    }
    
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #6B46C1 !important;
        background-color: #FFFFFF !important;
    }
    
    .uploadedFileName {
        color: #6B46C1 !important;
        font-size: 0.85rem !important;
    }
    
    /* Metrics - modern card style */
    [data-testid="metric-container"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E9D5FF !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        box-shadow: 0 1px 3px rgba(107, 70, 193, 0.08) !important;
    }
    
    [data-testid="metric-container"] label {
        color: #6B46C1 !important;
        font-weight: 500 !important;
        font-size: 0.8rem !important;
        text-align: center !important;
    }
    
    [data-testid="metric-container"] [data-testid="metric-value"] {
        color: #6B46C1 !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        text-align: center !important;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background-color: #6B46C1 !important;
    }
    
    .stProgress > div > div {
        background-color: #E9D5FF !important;
    }
    
    /* Success/Info/Error messages */
    .stAlert {
        background-color: #F3E8FF !important;
        color: #6B46C1 !important;
        border: 1px solid #E9D5FF !important;
        border-radius: 8px !important;
        font-size: 0.9rem !important;
    }
    
    /* Image container */
    .stImage {
        border-radius: 8px !important;
        overflow: hidden !important;
        box-shadow: 0 2px 6px rgba(107, 70, 193, 0.1) !important;
    }
    
    /* About section styling - REMOVED since we're not using containers */
    
    /* Footer - fixed at bottom */
    .footer {
        text-align: center;
        padding: 1.5rem 0;
        color: #6B46C1;
        font-size: 0.85rem;
        margin-top: 3rem;
        border-top: 1px solid #E9D5FF;
        background-color: #FAFAFA;
        position: relative;
        bottom: 0;
        width: 100%;
    }
    
    .footer a {
        color: #6B46C1;
        text-decoration: none;
        margin: 0 1rem;
        font-weight: 500;
    }
    
    .footer a:hover {
        text-decoration: underline;
    }
    
    /* Remove default Streamlit styling */
    .css-1d391kg {
        background-color: transparent !important;
    }
    
    /* Column gaps and padding */
    [data-testid="column"] {
        background-color: transparent !important;
        padding: 0 0.75rem !important;
    }
    
    /* Probability bars styling */
    .prob-container {
        background-color: #FAFAFA;
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1rem;
        border: 1px solid rgba(107, 70, 193, 0.05);
    }
    
    /* Hide streamlit menu and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Spinner text */
    .stSpinner > div {
        color: #6B46C1 !important;
        font-size: 0.9rem !important;
    }
    
    /* Remove extra padding and margins */
    .css-1y4p8pa {
        max-width: none !important;
        padding: 0 !important;
    }
    
    /* Ensure content is centered */
    .css-1aumxe8 {
        max-width: 1000px !important;
        margin: 0 auto !important;
    }
    
    /* Make success message more subtle */
    .stSuccess {
        background-color: #F3E8FF !important;
        color: #6B46C1 !important;
        border: 1px solid #E9D5FF !important;
        font-size: 0.85rem !important;
        padding: 0.5rem 1rem !important;
        text-align: center !important;
    }
    
    /* Style for all paragraphs */
    p {
        font-size: 0.9rem !important;
        text-align: center !important;
    }
    
    /* Remove any empty card containers */
    .stContainer > div:empty {
        display: none !important;
    }
    
    /* Ensure main container has proper spacing */
    .main .block-container {
        padding-top: 0 !important;
        padding-bottom: 2rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False

# Class dictionary
class_dict = {
    0: 'Glioma',
    1: 'Meningioma',
    2: 'No Tumor',
    3: 'Pituitary'
}

# Sample images with clean names
sample_images = {
    "Te-glTr_0000.jpg": "Glioma Sample 1",
    "Te-glTr_0001.jpg": "Glioma Sample 2", 
    "Te-glTr_0002.jpg": "Glioma Sample 3",
    "Te-meTr_0000.jpg": "Meningioma Sample 1",
    "Te-meTr_0001.jpg": "Meningioma Sample 2",
    "Te-meTr_0002.jpg": "Meningioma Sample 3",
    "Te-noTr_0000.jpg": "No Tumor Sample 1",
    "Te-noTr_0001.jpg": "No Tumor Sample 2",
    "Te-noTr_0002.jpg": "No Tumor Sample 3",
    "Te-piTr_0000.jpg": "Pituitary Sample 1",
    "Te-piTr_0001.jpg": "Pituitary Sample 2",
    "Te-piTr_0002.jpg": "Pituitary Sample 3"
}

@st.cache_resource
def load_model():
    """Load the pre-trained Xception model"""
    img_shape = (299, 299, 3)
    
    # Create base model
    base_model = tf.keras.applications.Xception(
        include_top=False,
        weights="imagenet",
        input_shape=img_shape,
        pooling='max'
    )
    
    # Build the model architecture
    model = Sequential([
        base_model,
        Flatten(),
        Dropout(rate=0.3),
        Dense(128, activation='relu'),
        Dropout(rate=0.25),
        Dense(4, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        Adamax(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', Precision(), Recall()]
    )
    
    # Load weights
    model.load_weights('models/xception_model.weights.h5')
    
    return model

def preprocess_image(image):
    """Preprocess image for model prediction"""
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to model input size
    image = image.resize((299, 299))
    
    # Convert to array and normalize
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    return img_array

def predict_tumor(model, img_array):
    """Make prediction on preprocessed image"""
    predictions = model.predict(img_array, verbose=0)
    probabilities = predictions[0]
    predicted_class_idx = np.argmax(probabilities)
    predicted_class = class_dict[predicted_class_idx]
    confidence = probabilities[predicted_class_idx]
    
    return predicted_class, confidence, probabilities

# Main app
st.markdown("<h1>Brain Tumor Classification</h1>", unsafe_allow_html=True)

# Create main layout
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    # Input method section
    st.markdown("<h3>Select Input Method</h3>", unsafe_allow_html=True)
    
    input_method = st.radio(
        "Input Method",
        ["Use Sample Image", "Upload Your Own"],
        label_visibility="collapsed"
    )
    
    image_to_analyze = None
    
    if input_method == "Use Sample Image":
        st.markdown("<p style='margin-top: 2rem; margin-bottom: 0.5rem;'>Choose a sample MRI scan:</p>", unsafe_allow_html=True)
        selected_sample = st.selectbox(
            "Sample Selection",
            list(sample_images.keys()),
            format_func=lambda x: sample_images[x],
            label_visibility="collapsed"
        )
        
        sample_path = f"sample_images/{selected_sample}"
        if os.path.exists(sample_path):
            image_to_analyze = Image.open(sample_path)
            # Display smaller image
            st.markdown("<p style='margin-top: 1.5rem; margin-bottom: 0.5rem; font-weight: 500;'>Selected MRI Image</p>", unsafe_allow_html=True)
            col_img1, col_img2, col_img3 = st.columns([1, 2, 1])
            with col_img2:
                st.image(image_to_analyze, use_container_width=True)
        else:
            st.error("Sample image not found. Please check the file path.")
    
    else:
        st.markdown("<p style='margin-top: 2rem; margin-bottom: 0.5rem;'>Upload a brain MRI scan:</p>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "File Upload",
            type=['jpg', 'jpeg', 'png'],
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            image_to_analyze = Image.open(uploaded_file)
            # Display smaller image
            st.markdown("<p style='margin-top: 1.5rem; margin-bottom: 0.5rem; font-weight: 500;'>Uploaded MRI Image</p>", unsafe_allow_html=True)
            col_img1, col_img2, col_img3 = st.columns([1, 2, 1])
            with col_img2:
                st.image(image_to_analyze, use_container_width=True)

with col2:
    # Results section
    st.markdown("<h3>Analysis Results</h3>", unsafe_allow_html=True)
    
    if image_to_analyze is not None:
        if st.button("Analyze Image", use_container_width=True):
            # Load model if not already loaded
            if not st.session_state.model_loaded:
                with st.spinner("Loading AI model..."):
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    st.session_state.model = load_model()
                    st.session_state.model_loaded = True
                    progress_bar.empty()
            
            # Make prediction
            with st.spinner("Analyzing MRI scan..."):
                progress_bar = st.progress(0)
                for i in range(50):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                # Preprocess and predict
                img_array = preprocess_image(image_to_analyze)
                predicted_class, confidence, probabilities = predict_tumor(st.session_state.model, img_array)
                
                for i in range(50, 100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                progress_bar.empty()
                
                st.session_state.analysis_done = True
                st.session_state.results = {
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'probabilities': probabilities
                }
        
        if st.session_state.analysis_done:
            results = st.session_state.results
            
            # Display results
            st.success("Analysis Complete")
            
            # Main prediction
            st.markdown(f"<h4 style='margin-top: 1rem;'>Detected: <strong>{results['predicted_class']}</strong></h4>", unsafe_allow_html=True)
            
            # Metrics in clean cards
            col_metric1, col_metric2 = st.columns(2)
            with col_metric1:
                st.metric("Confidence Score", f"{results['confidence']:.1%}")
            with col_metric2:
                st.metric("Model Accuracy", "98.48%")
            
            # Probability distribution
            st.markdown("<h4 style='margin-top: 2rem; margin-bottom: 1rem;'>Probability Distribution</h4>", unsafe_allow_html=True)
            
            for idx, (class_name, prob) in enumerate(zip(class_dict.values(), results['probabilities'])):
                prob_percent = prob * 100
                color = "#6B46C1" if idx == np.argmax(results['probabilities']) else "#E9D5FF"
                st.markdown(
                    f"""
                    <div style="margin-bottom: 12px;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                            <span style="color: #6B46C1; font-weight: 500; font-size: 0.85rem;">{class_name}</span>
                            <span style="color: #6B46C1; font-weight: 600; font-size: 0.85rem;">{prob_percent:.1f}%</span>
                        </div>
                        <div style="background-color: #F3E8FF; border-radius: 4px; height: 16px; overflow: hidden;">
                            <div style="background-color: {color}; width: {prob_percent}%; height: 100%; transition: width 0.3s ease;"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            # Analyze new image button
            st.markdown("<div style='margin-top: 2rem;'>", unsafe_allow_html=True)
            if st.button("Analyze New Image", use_container_width=True):
                st.session_state.analysis_done = False
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Please select or upload an MRI image to begin analysis")

# Model information
st.markdown("<h4 style='text-align: center; margin-top: 3rem; margin-bottom: 1rem;'>About the Model</h4>", unsafe_allow_html=True)
st.markdown("""
<div style='max-width: 800px; margin: 0 auto; text-align: left;'>

This brain tumor classification system uses a deep learning model based on the **Xception** architecture:

- **Model Architecture**: Xception (pre-trained on ImageNet) with custom classification layers
- **Training Accuracy**: 99.95%
- **Validation Accuracy**: 99.24%
- **Test Accuracy**: 98.48%
- **Classes**: Glioma, Meningioma, No Tumor, Pituitary
- **Input Size**: 299x299 RGB images

The model was trained on a comprehensive dataset of brain MRI scans and can classify four different conditions with high accuracy.

</div>
""", unsafe_allow_html=True)

# Footer
st.markdown(
    """
    <div class="footer">
        Made by Misbah | 
        <a href="https://MisbahAN.com" target="_blank">Portfolio</a> | 
        <a href="https://github.com/MisbahAN/Brain-Tumor-Classification" target="_blank">GitHub Repo</a>
    </div>
    """,
    unsafe_allow_html=True
)