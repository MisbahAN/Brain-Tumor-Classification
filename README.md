# 🧠 Brain Tumor Classification - Streamlit App

**Advanced AI-Powered Brain Tumor Detection System using Deep Learning**

This repository contains a **Streamlit web application** for an intelligent brain tumor classification system that leverages state-of-the-art deep learning techniques to analyze MRI scans and provide accurate tumor type predictions with AI-generated explanations.

## 🎯 Project Overview

Brain tumor classification is a critical medical imaging task where accuracy can significantly impact patient outcomes. This system addresses the inherent challenges in medical image analysis by providing consistent, precise tumor classification that supports healthcare professionals in their diagnostic process.

### 🧠 Why AI-Powered Brain Tumor Classification Matters

MRI scans exhibit significant variability in:

- **Shapes and sizes** - Tumors appear differently across patients
- **Orientations** - Scans taken from various angles (sagittal, coronal, axial)
- **Brightness levels** - Different imaging protocols and equipment settings
- **Image quality** - Variations in resolution and contrast

![Different Types of MRI Images](research/documentation/different_types_of_images.png)

This variability makes it challenging, even for experienced neurologists and neurosurgeons, to confidently distinguish between tumor types through visual examination alone. The subtle differences and inconsistencies can lead to misclassification by humans.

**Machine learning models provide:**

- ✅ **Consistency** - Eliminates human variability in interpretation
- ✅ **Precision** - Analyzes patterns across thousands of training examples
- ✅ **Speed** - Rapid analysis supporting faster diagnosis
- ✅ **Interpretability** - Visual insights into model decision-making process

## 🔬 Tumor Classification Categories

The system classifies brain MRI scans into **4 distinct categories**:

![Tumor Classification Types](research/documentation/tumor_classification.png)

| Tumor Type     | Description                       | Characteristics                                      |
| -------------- | --------------------------------- | ---------------------------------------------------- |
| **Glioma**     | Most common malignant brain tumor | Arises from glial cells, infiltrative growth pattern |
| **Meningioma** | Usually benign tumor              | Arises from meninges, well-defined borders           |
| **Pituitary**  | Tumor of pituitary gland          | Located in sella turcica, affects hormone production |
| **No Tumor**   | Normal brain tissue               | Healthy brain MRI without pathological findings      |

## 🏗️ Model Architecture

### Transfer Learning with Xception

The system utilizes **Google's Xception model** for transfer learning - a powerful convolutional neural network that excels at image classification tasks. Xception's depthwise separable convolutions efficiently analyze spatial and channel-wise features, making it ideal for medical imaging applications.

![Model Architecture](research/documentation/model_architecture.png)

**Why Xception?**

- 🔥 **Efficient Feature Extraction** - Breaks down images into meaningful components
- 🎯 **High Accuracy** - Proven performance on complex image classification
- ⚡ **Optimized Processing** - Faster inference compared to traditional CNNs
- 🧠 **Transfer Learning Ready** - Pre-trained on ImageNet, fine-tuned for medical data

### Model Pipeline Flow

![Model Flow](research/documentation/model_flow.png)

The prediction pipeline processes MRI images through:

1. **Image Preprocessing** - Resize to 299x299, normalization
2. **Feature Extraction** - Xception backbone extracts deep features
3. **Classification Head** - Dense layers with dropout for final prediction
4. **Probability Distribution** - Softmax output for all 4 classes
5. **Loss Calculation** - Categorical crossentropy for model optimization

## 📊 Model Performance

The trained Xception model achieves exceptional performance metrics:

![Training Metrics](research/documentation/model_training_metrics_over_epochs.png)

### Performance Metrics

- 🏆 **Test Accuracy: 98.48%**
- 📉 **Test Loss: 0.0626**
- 🎯 **Validation Accuracy: 99.24%**
- 📊 **Validation Loss: 0.0239**
- 🔥 **Training Accuracy: 99.95%**
- 📈 **Training Loss: 0.0032**

### Sample Predictions

<div align="center">
<table>
<tr>
<td align="center">
<strong>Glioma Detection Example</strong><br>
<img src="research/documentation/glioma_prediction_result.png" width="400" alt="Glioma Prediction">
</td>
<td align="center">
<strong>Meningioma Detection Example</strong><br>
<img src="research/documentation/meningioma_prediction_result.png" width="400" alt="Meningioma Prediction">
</td>
</tr>
</table>
</div>

## 🗂️ Project Structure

```
Brain-Tumor-Classification/
├── 🎯 app.py                               # Main Streamlit application
├── 📦 requirements.txt                     # Python dependencies
├── 🔧 utils/                               # Helper functions & utilities
│   ├── model_loader.py                     # Model loading & preprocessing
│   ├── image_processing.py                 # Image processing functions
│   └── prediction_utils.py                 # Prediction & post-processing
├── 📄 pages/                               # Streamlit multi-page components
│   ├── 🏠 Home.py                          # Main prediction page
│   ├── 📊 Model_Performance.py             # Model metrics & visualizations
│   ├── 🖼️  Sample_Images.py                # Pre-made sample testing
│   └── ℹ️  About.py                         # Project information
├── 🤖 models/                              # Trained Model Weights
│   └── xception_model.weights.h5           # Trained Xception model (253MB)
├── 🖼️  sample_images/                      # Pre-made sample images
│   ├── Te-glTr_0000.jpg                    # Glioma samples (3 images)
│   ├── Te-meTr_0000.jpg                    # Meningioma samples (3 images)
│   ├── Te-noTr_0000.jpg                    # No tumor samples (3 images)
│   └── Te-piTr_0000.jpg                    # Pituitary samples (3 images)
├── 🔬 research/                            # Research & Documentation
│   ├── 📓 notebooks/                       # Jupyter notebooks
│   │   └── BrainTumorClassification.ipynb
│   └── 📚 documentation/                   # Project insights & visualizations
│       ├── findings.txt                    # Research findings & insights
│       ├── model_architecture.png          # Model architecture diagram
│       ├── model_training_metrics_over_epochs.png
│       ├── tumor_classification.png        # Tumor types visualization
│       ├── different_types_of_images.png
│       ├── glioma_prediction_result.png
│       └── meningioma_prediction_result.png
├── 🗃️  Testing/                            # Original dataset (preserved)
├── 🗃️  Training/                           # Original dataset (preserved)
└── 📖 README.md                            # Project documentation
```

## 🔮 Future Enhancements

### Planned Features

- 🔄 **Production Pipeline** - Convert Jupyter notebook to production-ready Python modules
- 🤖 **CNN Model Integration** - Implement custom CNN architecture alongside Xception
- 🤝 **Gemini Integration** - Add Google Gemini 2.5 Flash for AI-powered explanations
- 📊 **Model Interpretability** - Implement GRAD-CAM for visual explanations
- 🎨 **Enhanced UI/UX** - Improved Streamlit interface with custom styling
- 📈 **Model Comparison** - Side-by-side comparison of different model architectures
- 📱 **Mobile Optimization** - Responsive design for mobile devices

### Technical Roadmap

- [ ] **Streamlit App Development** - Interactive web interface for brain tumor classification
- [ ] **Model Pipeline** - Production-ready inference pipeline
- [ ] **Image Processing** - Advanced preprocessing and augmentation
- [ ] **Multi-page App** - Organized sections for prediction, performance, and samples
- [ ] **Error Handling** - Comprehensive validation and user feedback
- [ ] **Performance Optimization** - Caching and efficient model loading

## 🔬 Dataset Information

- **Total Images**: 7,023 brain MRI scans
- **Classes**: 4 (Glioma, Meningioma, Pituitary, No Tumor)
- **Format**: JPG images with varying dimensions
- **Source**: [Kaggle Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **Preprocessing**: Resize to 299x299, normalization, data augmentation

## 🌐 Deployment Architecture

This Streamlit application is designed for **easy deployment** on various platforms:

- **Streamlit Cloud**: Direct GitHub integration, free hosting for public repos
- **Heroku**: Container-based deployment with model file support
- **Railway**: Simple deployment with generous resource limits
- **Google Cloud Run**: Scalable serverless deployment
- **AWS EC2**: Full control over compute resources

## 🛠️ Installation & Setup

_Coming Soon - Streamlit app implementation in progress_

## 🚀 Usage

_Coming Soon - Streamlit app interface and features_

## 📱 App Features

_Coming Soon - Interactive components and user interface_

## 🧪 Testing

_Coming Soon - Model validation and testing procedures_

---

## 👨‍💻 Author

**Misbah Ahmed Nauman**  
Portfolio: [MisbahAN.com](https://MisbahAN.com)

---

_This project represents a complete Streamlit web application for AI-powered brain tumor classification with an intuitive interface for medical professionals and researchers._
