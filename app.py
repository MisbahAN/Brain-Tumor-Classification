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
    /* Import clean font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global reset and styles */
    * {
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Hide sidebar completely */
    section[data-testid="stSidebar"] {
        display: none;
    }
    
    /* Main app background */
    .stApp {
        background-color: #FFFFFF;
    }
    
    .main {
        background-color: #FFFFFF;
    }
    
    /* All text in purple */
    h1, h2, h3, h4, h5, h6, p, span, div, label {
        color: #6B46C1 !important;
    }
    
    /* Main title */
    h1 {
        text-align: center;
        font-weight: 700;
        font-size: 2.5rem !important;
        margin-bottom: 3rem !important;
        color: #6B46C1 !important;
    }
    
    /* Section headers */
    h3 {
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        margin-bottom: 1.5rem !important;
        color: #6B46C1 !important;
    }
    
    h4 {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        color: #6B46C1 !important;
    }
    
    /* Remove all emojis from headers */
    h3:before, h4:before {
        content: none !important;
    }
    
    /* Card container styling */
    .card-container {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 4px 12px rgba(107, 70, 193, 0.1);
        margin-bottom: 1.5rem;
    }
    
    /* Button styling - clean and modern */
    .stButton > button {
        background-color: #FFFFFF !important;
        color: #6B46C1 !important;
        border: 2px solid #6B46C1 !important;
        box-shadow: 0 2px 8px rgba(107, 70, 193, 0.15) !important;
        font-weight: 600 !important;
        padding: 0.75rem 2rem !important;
        border-radius: 8px !important;
        transition: none !important;
        width: 100% !important;
        font-size: 1rem !important;
    }
    
    .stButton > button:hover {
        background-color: #FFFFFF !important;
        box-shadow: 0 2px 8px rgba(107, 70, 193, 0.15) !important;
        border: 2px solid #6B46C1 !important;
    }
    
    /* Radio button styling */
    .stRadio > div {
        background-color: #FFFFFF !important;
        padding: 1rem !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 8px rgba(107, 70, 193, 0.1) !important;
    }
    
    .stRadio label {
        color: #6B46C1 !important;
        font-weight: 500 !important;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background-color: #FFFFFF !important;
        border: 2px solid #E9D5FF !important;
        border-radius: 8px !important;
        color: #6B46C1 !important;
    }
    
    .stSelectbox label {
        color: #6B46C1 !important;
        font-weight: 500 !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* File uploader - clean white */
    [data-testid="stFileUploader"] {
        background-color: #FFFFFF !important;
    }
    
    [data-testid="stFileUploaderDropzone"] {
        background-color: #FFFFFF !important;
        border: 2px dashed #E9D5FF !important;
        border-radius: 8px !important;
    }
    
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #6B46C1 !important;
    }
    
    .uploadedFileName {
        color: #6B46C1 !important;
    }
    
    /* Metrics - modern card style */
    [data-testid="metric-container"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E9D5FF !important;
        border-radius: 8px !important;
        padding: 1.25rem !important;
        box-shadow: 0 2px 8px rgba(107, 70, 193, 0.08) !important;
    }
    
    [data-testid="metric-container"] label {
        color: #6B46C1 !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }
    
    [data-testid="metric-container"] [data-testid="metric-value"] {
        color: #6B46C1 !important;
        font-weight: 700 !important;
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
    }
    
    /* Image container */
    .stImage {
        border-radius: 8px !important;
        overflow: hidden !important;
        box-shadow: 0 2px 8px rgba(107, 70, 193, 0.1) !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #FFFFFF !important;
        color: #6B46C1 !important;
        border: 1px solid #E9D5FF !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
    }
    
    .streamlit-expanderContent {
        background-color: #FFFFFF !important;
        border: 1px solid #E9D5FF !important;
        border-top: none !important;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #6B46C1;
        font-size: 0.9rem;
        margin-top: 4rem;
        border-top: 1px solid #E9D5FF;
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
        background-color: #FFFFFF !important;
    }
    
    /* Column gaps and padding */
    [data-testid="column"] {
        background-color: #FFFFFF !important;
        padding: 0 1rem !important;
    }
    
    /* Results container */
    .results-container {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 4px 12px rgba(107, 70, 193, 0.1);
    }
    
    /* Probability bars container */
    .prob-container {
        background-color: #FFFFFF;
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    
    /* Hide streamlit menu and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Spinner text */
    .stSpinner > div {
        color: #6B46C1 !important;
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
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    st.markdown("### Select Input Method")
    
    input_method = st.radio(
        "",
        ["Use Sample Image", "Upload Your Own"],
        label_visibility="collapsed"
    )
    
    image_to_analyze = None
    
    if input_method == "Use Sample Image":
        st.markdown("<p style='margin-top: 1rem; margin-bottom: 0.5rem;'>Choose a sample MRI scan:</p>", unsafe_allow_html=True)
        selected_sample = st.selectbox(
            "",
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
        st.markdown("<p style='margin-top: 1rem; margin-bottom: 0.5rem;'>Upload a brain MRI scan:</p>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "",
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
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="results-container">', unsafe_allow_html=True)
    st.markdown("### Analysis Results")
    
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
            st.markdown(f"### Detected: **{results['predicted_class']}**")
            
            # Metrics in clean cards
            col_metric1, col_metric2 = st.columns(2)
            with col_metric1:
                st.metric("Confidence Score", f"{results['confidence']:.1%}")
            with col_metric2:
                st.metric("Model Accuracy", "98.48%")
            
            # Probability distribution
            st.markdown('<div class="prob-container">', unsafe_allow_html=True)
            st.markdown("#### Probability Distribution")
            
            for idx, (class_name, prob) in enumerate(zip(class_dict.values(), results['probabilities'])):
                prob_percent = prob * 100
                color = "#6B46C1" if idx == np.argmax(results['probabilities']) else "#E9D5FF"
                st.markdown(
                    f"""
                    <div style="margin-bottom: 12px;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                            <span style="color: #6B46C1; font-weight: 500;">{class_name}</span>
                            <span style="color: #6B46C1; font-weight: 600;">{prob_percent:.1f}%</span>
                        </div>
                        <div style="background-color: #F3E8FF; border-radius: 4px; height: 20px; overflow: hidden;">
                            <div style="background-color: {color}; width: {prob_percent}%; height: 100%; transition: width 0.3s ease;"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Analyze new image button
            if st.button("Analyze New Image", use_container_width=True):
                st.session_state.analysis_done = False
                st.rerun()
    else:
        st.info("Please select or upload an MRI image to begin analysis")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Model information
st.markdown("<div style='margin-top: 3rem;'>", unsafe_allow_html=True)
with st.expander("About the Model"):
    st.markdown("""
    This brain tumor classification system uses a deep learning model based on the **Xception** architecture:
    
    - **Model Architecture**: Xception (pre-trained on ImageNet) with custom classification layers
    - **Training Accuracy**: 99.95%
    - **Validation Accuracy**: 99.24%
    - **Test Accuracy**: 98.48%
    - **Classes**: Glioma, Meningioma, No Tumor, Pituitary
    - **Input Size**: 299x299 RGB images
    
    The model was trained on a comprehensive dataset of brain MRI scans and can classify four different conditions with high accuracy.
    """)
st.markdown("</div>", unsafe_allow_html=True)

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