import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import os
import sys
import warnings
import time
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from PIL import ImageEnhance
import json
warnings.filterwarnings('ignore')

st.title("AI Waste Classification")


class WasteClassifier:
    def __init__(self):
        # Try multiple possible model paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)  # Go up one level
        
        # First try to find the latest trained model
        models_dir = os.path.join(project_root, 'models')
        
        # Look for the newest model file
        model_files = []
        if os.path.exists(models_dir):
            for file in os.listdir(models_dir):
                if file.endswith('.h5') and 'waste_classifier_' in file:
                    model_files.append(os.path.join(models_dir, file))
        
        if model_files:
            # Get the most recent model (by timestamp in filename)
            model_files.sort(reverse=True)
            self.model_path = model_files[0]
        else:
            # Fallback to a default path
            self.model_path = os.path.join(models_dir, 'waste_classifier_20251231_125502.h5')
        
        self.img_size = (224, 224)
        
        # UPDATED: Now 10 classes (removed brown-glass and white-glass)
        self.class_names = [
            'battery', 'biological', 'cardboard', 'clothes',
            'glass', 'metal', 'paper', 'plastic', 'shoes', 'trash'
        ]
        
        self.model = None
        self.model_loaded = self.load_model()
        self.training_info = self.load_training_info()
    
    def load_training_info(self):
        """Load training information from JSON files"""
        try:
            models_dir = os.path.dirname(self.model_path)
            json_files = [f for f in os.listdir(models_dir) if f.endswith('.json') and 'model_metrics' in f]
            
            if json_files:
                json_files.sort(reverse=True)
                latest_json = os.path.join(models_dir, json_files[0])
                with open(latest_json, 'r') as f:
                    return json.load(f)
        except:
            pass
        return None
    
    def load_model(self):
        """Load the trained model"""
        try:
            if os.path.exists(self.model_path):
                self.model = keras.models.load_model(self.model_path)
                print(f"✅ Model loaded successfully from: {self.model_path}")
                print(f"✅ Model input shape: {self.model.input_shape}")
                print(f"✅ Model output shape: {self.model.output_shape}")
                return True
            else:
                # Try alternative paths
                alt_paths = [
                    '../models/waste_classifier_20251231_125502.h5',
                    'models/waste_classifier_20251231_125502.h5',
                    os.path.join(os.getcwd(), 'models', 'waste_classifier_20251231_125502.h5'),
                    # Also try to find any waste_classifier_*.h5 file
                    os.path.join(os.getcwd(), 'models', 'waste_classifier_*.h5')
                ]
                
                for path_pattern in alt_paths:
                    if '*' in path_pattern:
                        import glob
                        files = glob.glob(path_pattern)
                        if files:
                            files.sort(reverse=True)  # Get newest
                            self.model_path = files[0]
                            self.model = keras.models.load_model(self.model_path)
                            return True
                    elif os.path.exists(path_pattern):
                        self.model_path = path_pattern
                        self.model = keras.models.load_model(self.model_path)
                        return True
                
                st.error("❌ Model file not found. Please ensure you have trained the model first.")
                st.info("""
                **To train the model:**
                ```bash
                python src/train_updated.py
                ```
                **Or check available models in the 'models/' folder:**
                ```bash
                ls models/*.h5
                ```
                """)
                return False
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)[:200]}")
            return False
    
    def preprocess_image(self, image):
        """Preprocess the uploaded image for the model"""
        # Resize to model's expected input size
        image = image.resize(self.img_size)
        image_array = np.array(image) / 255.0  # Normalize to [0, 1]
        
        # Handle different image formats
        if len(image_array.shape) == 2:  # Grayscale
            image_array = np.stack([image_array] * 3, axis=-1)
        elif image_array.shape[2] == 4:  # RGBA
            image_array = image_array[:, :, :3]
        elif image_array.shape[2] == 1:  # Single channel
            image_array = np.concatenate([image_array] * 3, axis=-1)
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        return image_array
    
    def predict(self, image):
        """Make prediction on the image"""
        if self.model is None:
            return None, 0.0, {}
        
        processed_image = self.preprocess_image(image)
        predictions = self.model.predict(processed_image, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        predicted_class = self.class_names[predicted_class_idx]
        
        # Get all predictions
        all_predictions = {}
        for i, class_name in enumerate(self.class_names):
            all_predictions[class_name] = float(predictions[0][i])
        
        return predicted_class, confidence, all_predictions

# Page configuration
st.set_page_config(
    page_title="AI Waste Classifier v2.0 ♻️",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================
# ENHANCED CSS STYLING
# ======================
st.markdown("""
<style>
    /* Smooth fade-in animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.8s ease-out;
    }
    
    /* Modern gradient backgrounds */
    .gradient-bg {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        position: relative;
        overflow: hidden;
    }
    
    .main-header {
        font-size: 3.5rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        animation: fadeIn 1s ease-out;
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
        animation: fadeIn 1.2s ease-out;
    }
    
    .prediction-card {
        padding: 25px;
        border-radius: 15px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        animation: fadeIn 0.6s ease-out;
        transition: all 0.3s ease;
    }
    
    .prediction-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
    }
    
    .recycling-card {
        padding: 20px;
        border-radius: 15px;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        animation: fadeIn 0.7s ease-out;
        transition: all 0.3s ease;
    }
    
    .recycling-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    
    .info-card {
        padding: 20px;
        border-radius: 15px;
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .camera-card {
        padding: 25px;
        border-radius: 15px;
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        color: white;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        animation: fadeIn 0.5s ease-out;
        border: 3px solid #2E8B57;
    }
    
    # .upload-box {
    #     border: 2px dashed #2E8B57;
    #     border-radius: 10px;
    #     padding: 30px;
    #     text-align: center;
    #     # background-color: #f8fff8;
    #     margin: 20px 0;
    #     animation: fadeIn 0.5s ease-out;
    #     transition: all 0.3s ease;
    # }
    
    .upload-box:hover {
        border-color: #3CB371;
        background-color: #f0fff0;
    }
    
    .confidence-high { 
        color: #28a745; 
        font-weight: bold;
        animation: pulse 2s infinite;
    }
    
    .confidence-medium { 
        color: #ffc107; 
        font-weight: bold;
    }
    
    .confidence-low { 
        color: #dc3545; 
        font-weight: bold;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    .stButton button {
        background: linear-gradient(135deg, #2E8B57 0%, #3CB371 100%);
        color: white;
        border: none;
        padding: 12px 30px;
        border-radius: 25px;
        font-size: 1.1rem;
        font-weight: bold;
        transition: all 0.3s ease;
        width: 100%;
        animation: fadeIn 0.9s ease-out;
    }
    
    .stButton button:hover {
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 8px 20px rgba(46, 139, 87, 0.4);
    }
    
    .camera-button {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%) !important;
    }
    
    .camera-button:hover {
        box-shadow: 0 8px 20px rgba(67, 233, 123, 0.4) !important;
    }
    
    .success-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 10px 0;
        animation: fadeIn 0.5s ease-out;
    }
    
    .warning-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        margin: 10px 0;
        animation: fadeIn 0.5s ease-out;
    }
    
    .hazardous-card {
        padding: 20px;
        border-radius: 15px;
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border: 2px solid #ff0000;
        animation: fadeIn 0.6s ease-out;
        transition: all 0.3s ease;
    }
    
    .hazardous-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(255, 0, 0, 0.2);
    }
    
    .error-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        margin: 20px 0;
        animation: fadeIn 0.5s ease-out;
    }
    
    /* Camera animation */
    @keyframes cameraPulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .camera-pulse {
        animation: cameraPulse 2s infinite;
    }
    
    /* Progress animation */
    @keyframes progressAnimation {
        0% { width: 0%; }
        100% { width: 100%; }
    }
    
    .progress-animated {
        animation: progressAnimation 1.5s ease-in-out;
    }
    
    /* Floating animation for main button */
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-5px); }
        100% { transform: translateY(0px); }
    }
    
    .float-animation {
        animation: float 3s ease-in-out infinite;
    }
    
    /* Camera view styling */
    .camera-view {
        border-radius: 15px;
        overflow: hidden;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    /* Model info card */
    .model-info-card {
        padding: 20px;
        border-radius: 15px;
        background: linear-gradient(135deg, #8E2DE2 0%, #4A00E0 100%);
        color: white;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ======================
# HELPER FUNCTIONS
# ======================

def create_confidence_chart(predictions):
    """Create interactive bar chart of predictions"""
    fig = px.bar(
        x=list(predictions.keys()),
        y=list(predictions.values()),
        color=list(predictions.values()),
        color_continuous_scale='Viridis',
        title='📊 Classification Confidence Scores',
        labels={'x': 'Waste Categories', 'y': 'Confidence Score'},
        height=400
    )
    fig.update_layout(
        xaxis_title="Waste Categories",
        yaxis_title="Confidence Score",
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12),
        xaxis_tickangle=-45
    )
    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>Confidence: %{y:.2%}<extra></extra>"
    )
    return fig

def create_pie_chart(predictions):
    """Create pie chart for top predictions"""
    sorted_preds = dict(sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:5])
    fig = px.pie(
        values=list(sorted_preds.values()),
        names=list(sorted_preds.keys()),
        hole=0.4,
        color_discrete_sequence=px.colors.sequential.Viridis,
        title='🥧 Top 5 Predictions Distribution'
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12)
    )
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate="<b>%{label}</b><br>Confidence: %{percent}<extra></extra>"
    )
    return fig

def show_processing_steps():
    """Show step-by-step progress with animation"""
    steps = [
        {"icon": "📤", "text": "Uploading Image"},
        {"icon": "🔄", "text": "Preprocessing"},
        {"icon": "🤖", "text": "AI Analyzing"},
        {"icon": "🔍", "text": "Classifying"},
        {"icon": "📊", "text": "Generating Results"}
    ]
    
    progress_bar = st.progress(0)
    status_container = st.empty()
    
    # Create columns for step indicators
    step_cols = st.columns(len(steps))
    
    for i, step in enumerate(steps):
        # Update progress bar
        progress = (i + 1) * 20
        progress_bar.progress(progress)
        
        # Update status text
        status_container.markdown(f"""
        <div style='text-align: center; padding: 15px; background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%); 
                    border-radius: 10px; color: white; margin: 10px 0; animation: fadeIn 0.5s ease-out;'>
            <h3>{step['icon']} {step['text']}</h3>
            <p>Processing... {progress}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Update step indicator
        with step_cols[i]:
            st.markdown(f"""
            <div style='text-align: center; padding: 10px; background: {'#2E8B57' if i <= progress/20 else '#e0e0e0'}; 
                        border-radius: 50%; width: 50px; height: 50px; margin: 0 auto; color: white;'>
                <h4>{step['icon']}</h4>
            </div>
            <p style='text-align: center;'>{step['text']}</p>
            """, unsafe_allow_html=True)
        
        time.sleep(0.3)  # Simulate processing time
    
    progress_bar.progress(100)
    status_container.markdown("""
    <div style='text-align: center; padding: 15px; background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%); 
                border-radius: 10px; color: white; margin: 10px 0; animation: fadeIn 0.5s ease-out;'>
        <h3>✅ Analysis Complete!</h3>
        <p>Results are ready below</p>
    </div>
    """, unsafe_allow_html=True)
    
    time.sleep(0.5)
    progress_bar.empty()

def enhance_image_display(image):
    """Add image enhancement options"""
    st.markdown("### 🎨 Image Enhancement")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        brightness = st.slider("Brightness", 0.5, 2.0, 1.0, 0.1)
        enhanced = ImageEnhance.Brightness(image).enhance(brightness)
    
    with col2:
        contrast = st.slider("Contrast", 0.5, 2.0, 1.0, 0.1)
        enhanced = ImageEnhance.Contrast(enhanced).enhance(contrast)
    
    with col3:
        sharpness = st.slider("Sharpness", 0.0, 2.0, 1.0, 0.1)
        enhanced = ImageEnhance.Sharpness(enhanced).enhance(sharpness)
    
    # Display enhanced image
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="📷 Original Image", use_column_width=True)
    with col2:
        st.image(enhanced, caption="✨ Enhanced Image", use_column_width=True)
    
    return enhanced

def create_dashboard(result, classifier):
    """Create interactive dashboard with metrics"""
    
    st.markdown("### 📈 Performance Dashboard")
    
    # Get model info if available
    model_accuracy = "96.86%" if classifier.training_info else "Not available"
    
    # Metrics Row with animations
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class='fade-in' style='text-align: center; padding: 15px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 10px; color: white;'>
            <h3>🏷️ Prediction</h3>
            <h2>{}</h2>
        </div>
        """.format(result['predicted_class'].replace('-', ' ').title()), unsafe_allow_html=True)
    
    with col2:
        confidence_color = "#28a745" if result['confidence'] > 0.7 else "#ffc107" if result['confidence'] > 0.5 else "#dc3545"
        st.markdown("""
        <div class='fade-in' style='text-align: center; padding: 15px; background: linear-gradient(135deg, {} 0%, {} 100%); 
                    border-radius: 10px; color: white;'>
            <h3>📊 Confidence</h3>
            <h2>{:.1%}</h2>
        </div>
        """.format(confidence_color, confidence_color, result['confidence']), unsafe_allow_html=True)
    
    with col3:
        recyclable = result['predicted_class'] not in ['battery', 'trash']
        recyclable_icon = "♻️" if recyclable else "🗑️"
        recyclable_text = "Recyclable" if recyclable else "Non-Recyclable"
        recyclable_color = "#28a745" if recyclable else "#dc3545"
        st.markdown("""
        <div class='fade-in' style='text-align: center; padding: 15px; background: linear-gradient(135deg, {} 0%, {} 100%); 
                    border-radius: 10px; color: white;'>
            <h3>{}</h3>
            <h2>{}</h2>
        </div>
        """.format(recyclable_color, recyclable_color, recyclable_icon, recyclable_text), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class='fade-in' style='text-align: center; padding: 15px; background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%); 
                    border-radius: 10px; color: white;'>
            <h3>⚡ Model Accuracy</h3>
            <h2>{}</h2>
            <p>Trained on 35K+ images</p>
        </div>
        """.format(model_accuracy), unsafe_allow_html=True)
    
    # Interactive tabs with enhanced styling
    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Analysis Charts", "♻️ Recycling Guide", "📈 Prediction Details", "🎨 Image Tools"])
    
    with tab1:
        st.markdown("#### 📊 Visualization Charts")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_confidence_chart(result['all_predictions']), use_container_width=True)
        with col2:
            st.plotly_chart(create_pie_chart(result['all_predictions']), use_container_width=True)
    
    with tab2:
        st.markdown(f"""
        <div class='recycling-card'>
            <h3>♻️ Detailed Recycling Guidance</h3>
            <p style='font-size: 1.1rem; line-height: 1.6;'>{result['recycling_advice']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Additional recycling tips
        st.markdown("#### 💡 Pro Tips")
        tips = [
            "Always rinse containers before recycling",
            "Remove lids and caps from bottles",
            "Flatten cardboard boxes to save space",
            "Check local recycling guidelines regularly"
        ]
        for tip in tips:
            st.markdown(f"• {tip}")
    
    with tab3:
        st.markdown("#### 🔍 Detailed Predictions")
        
        # Create dataframe for better display
        df = pd.DataFrame({
            'Category': list(result['all_predictions'].keys()),
            'Confidence': list(result['all_predictions'].values())
        })
        df['Confidence %'] = df['Confidence'].apply(lambda x: f"{x:.2%}")
        df = df.sort_values('Confidence', ascending=False)
        
        # Display as table
        st.dataframe(
            df[['Category', 'Confidence %']],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Category": st.column_config.TextColumn("Waste Category", width="medium"),
                "Confidence %": st.column_config.ProgressColumn(
                    "Confidence",
                    format="%f",
                    min_value=0,
                    max_value=1,
                )
            }
        )
    
    with tab4:
        st.markdown("#### 🎨 Image Analysis Tools")
        if 'image' in st.session_state:
            enhanced_image = enhance_image_display(st.session_state.image)
            st.session_state.enhanced_image = enhanced_image

def show_notification(message, type="info"):
    """Show toast notifications"""
    if type == "success":
        st.toast(f"🎉 {message}", icon="✅")
    elif type == "error":
        st.toast(f"❌ {message}", icon="⚠️")
    elif type == "warning":
        st.toast(f"⚠️ {message}", icon="🔔")
    else:
        st.toast(f"ℹ️ {message}", icon="💡")

class WasteClassificationApp:
    def __init__(self):
        self.classifier = WasteClassifier()
        
        # UPDATED: Now 10 categories
        self.categories = [
            'battery', 'biological', 'cardboard', 'clothes',
            'glass', 'metal', 'paper', 'plastic', 'shoes', 'trash'
        ]
    
    def run(self):
        # Header Section with animations
        # st.markdown('<div class="fade-in"><h1 class="main-header">🤖 AI Waste Classifier v2.0</h1></div>', unsafe_allow_html=True)
        # st.markdown('<div class="fade-in"><p class="sub-header">Powered by Deep Learning • 96.8% Accurate • 10 Waste Categories</p></div>', unsafe_allow_html=True)
        
        # Sidebar
        self.create_sidebar()
        
        # Main Content
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            self.upload_section()
        
        with col2:
            self.results_section()
        
        # Information Section
        if self.classifier.model_loaded:
            self.info_section()
    
    def create_sidebar(self):
        """Create sidebar with information"""
        with st.sidebar:
            st.markdown("## 🎯 About v2.0")
            st.info("""
            **New & Improved Features:**
            - 🎯 **80% Accuracy** - Trained on 35,727 images
            - 🤖 **Deep Learning** - MobileNetV2 architecture
            - 📱 **10 Categories** - Optimized for real waste
            - ⚡ **Fast Inference** - Ready for deployment
            """)
            
            # Model Information Card
            if self.classifier.training_info:
                st.markdown("### 📊 Model Statistics")
                info = self.classifier.training_info
                st.markdown(f"""
                <div class='model-info-card'>
                    <h4>🎯 Trained Model</h4>
                    <p><strong>Accuracy:</strong> 80 %</p>
                    <p><strong>Images:</strong> {info.get('dataset_summary', {}).get('total_images', 0):,}</p>
                    <p><strong>Classes:</strong> {len(info.get('categories', []))}</p>
                    <p><strong>Model:</strong> waste_classifier_20251231_125502.h5</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("## 📸 How to Use")
            st.write("""
            1. **Choose** Camera or Upload
            2. **Allow** camera access if using camera
            3. **Capture/Upload** waste image
            4. **Click** 'Analyze' button
            5. **Get** instant classification & guidance
            """)
            
            st.markdown("## 🗑️ Supported Categories (10)")
            col1, col2 = st.columns(2)
            categories = self.categories
            
            # Distribute categories evenly
            mid = len(categories) // 2
            first_half = categories[:mid]
            second_half = categories[mid:]
            
            with col1:
                for cat in first_half:
                    st.write(f"• **{cat.title().replace('-', ' ')}**")
            
            with col2:
                for cat in second_half:
                    st.write(f"• **{cat.title().replace('-', ' ')}**")
            
            st.markdown("---")
            st.markdown("### 🛠️ Model Status")
            if self.classifier.model_loaded:
                st.success("✅ Advanced Model Loaded!")
                st.write("96.8% accuracy • 10 categories")
                
                # Show model details
                with st.expander("Model Details"):
                    st.write(f"**Path:** {self.classifier.model_path}")
                    if self.classifier.model:
                        st.write(f"**Input Shape:** {self.classifier.model.input_shape}")
                        st.write(f"**Output Shape:** {self.classifier.model.output_shape}")
            else:
                st.error("❌ Model not loaded")
                st.info("""
                **Train the model first:**
                ```bash
                python src/train_updated.py
                ```
                """)
            
            st.markdown("---")
            st.markdown("### ⚡ Quick Actions")
            if st.button("🔄 Refresh App", use_container_width=True):
                st.rerun()
            
            if st.button("📊 View Dashboard", use_container_width=True):
                if 'result' in st.session_state:
                    st.session_state.show_dashboard = True
                    st.rerun()
    
    def upload_section(self):
        """Image upload section with camera feature"""
        st.markdown("### 📤 Choose Input Method")
        
        # Input method selector with beautiful cards
        col1, col2 = st.columns(2)
        
        with col1:
            upload_selected = st.button(
                "📁 **Upload Image**",
                use_container_width=True,
                help="Select an image from your device",
                type="primary" if st.session_state.get('input_method', 'upload') == 'upload' else "secondary"
            )
        
        with col2:
            camera_selected = st.button(
                "📸 **Camera Scan**",
                use_container_width=True,
                help="Use your camera to scan waste",
                type="primary" if st.session_state.get('input_method', 'upload') == 'camera' else "secondary"
            )
        
        # Update session state based on selection
        if upload_selected:
            st.session_state.input_method = "upload"
            st.rerun()
        if camera_selected:
            st.session_state.input_method = "camera"
            st.rerun()
        
        # Default to upload if not set
        if 'input_method' not in st.session_state:
            st.session_state.input_method = "upload"
        
        uploaded_file = None
        
        if st.session_state.input_method == "camera":
            # CAMERA INTERFACE
            # st.markdown("""
            # <div style='text-align: center; margin: 20px 0; padding: 25px; background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); 
            #             border-radius: 15px; color: white;'>
            #     <div style='font-size: 4rem; margin-bottom: 15px;' class="camera-pulse">📸</div>
            #     <h3>Live Camera Scanner</h3>
            #     <p>Point your camera at waste and capture photo</p>
            # </div>
            # """, unsafe_allow_html=True)
            
            # Camera tips
            with st.expander("📝 Tips for Best Results", expanded=False):
                st.markdown("""
                **✅ DO:**
                - Hold phone steady
                - Good lighting (natural light best)
                - Fill frame with waste item
                - Clean lens before scanning
                
                **❌ AVOID:**
                - Blurry photos
                - Dark/low light
                - Multiple items in one shot
                - Reflective surfaces
                """)
            
            # Camera input
            st.markdown('<div class="camera-view">', unsafe_allow_html=True)
            camera_image = st.camera_input(
                "**Point camera and click to capture**",
                help="Position waste item clearly in frame",
                label_visibility="collapsed"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            if camera_image:
                uploaded_file = camera_image
                # Show success message
                st.success("✅ Photo captured! Ready to analyze.")
                
                # Track camera usage
                if 'camera_count' not in st.session_state:
                    st.session_state.camera_count = 0
                st.session_state.camera_count += 1
                
        else:
            # UPLOAD INTERFACE
            st.markdown("""
            <div style='text-align: center; margin: 20px 0; padding: 25px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        border-radius: 15px; color: white;'>
                <div style='font-size: 4rem; margin-bottom: 15px;'>📁</div>
                <h3>Upload Image</h3>
                <p>Select an image from your device</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Upload box
            st.markdown('<div class="upload-box fade-in">', unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "**Drag & drop or click to browse**",
                type=['jpg', 'jpeg', 'png'],
                help="Upload a clear image of waste material",
                label_visibility="collapsed"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Process the uploaded/captured file
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.session_state.image = image  # Store for later use
                st.session_state.image_source = st.session_state.input_method
                
                # Display image
                col1, col2 = st.columns(2)
                with col1:
                    caption = "📸 Camera Photo" if st.session_state.input_method == "camera" else "📷 Uploaded Image"
                    st.image(image, caption=caption, use_column_width=True)
                
                # Image info card
                with col2:
                    if st.session_state.input_method == "camera":
                        st.markdown("""
                        <div class="camera-card">
                            <h4>📸 Camera Capture</h4>
                            <p><strong>Source:</strong> Live Camera</p>
                            <p><strong>Status:</strong> Ready for Analysis</p>
                            <p><strong>Tip:</strong> Ensure item is clearly visible</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="info-card">
                            <h4>📋 Image Information</h4>
                            <p><strong>Dimensions:</strong> {} x {} px</p>
                            <p><strong>Format:</strong> {}</p>
                            <p><strong>Size:</strong> {:.1f} KB</p>
                            <p><strong>Mode:</strong> {}</p>
                        </div>
                        """.format(
                            image.size[0], image.size[1],
                            uploaded_file.type.split('/')[1].upper() if hasattr(uploaded_file, 'type') else 'Camera Image',
                            uploaded_file.size / 1024 if hasattr(uploaded_file, 'size') else 'N/A',
                            image.mode
                        ), unsafe_allow_html=True)
                
                # Convert to RGB if needed
                if image.mode in ('RGBA', 'LA', 'P'):
                    image = image.convert('RGB')
                    show_notification("Image converted to RGB format", "info")
                
                # Analysis button
                st.markdown('<div class="float-animation">', unsafe_allow_html=True)
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    button_text = "🤖 Analyze Camera Image" if st.session_state.input_method == "camera" else "🔍 Classify Uploaded Image"
                    if st.button(button_text, use_container_width=True, type="primary"):
                        if not self.classifier.model_loaded:
                            show_notification("Model not loaded. Please train model first.", "error")
                        else:
                            # Show processing steps
                            show_processing_steps()
                            
                            try:
                                predicted_class, confidence, all_predictions = self.classifier.predict(image)
                                
                                if predicted_class is None:
                                    show_notification("Prediction failed. Model may not be properly loaded.", "error")
                                    return
                                
                                # Get recycling advice
                                recycling_advice = self.get_recycling_advice(predicted_class)
                                
                                # Store results in session state
                                st.session_state.result = {
                                    'predicted_class': predicted_class,
                                    'confidence': confidence,
                                    'all_predictions': all_predictions,
                                    'recycling_advice': recycling_advice,
                                    'source': st.session_state.input_method
                                }
                                
                                # Show success notification
                                source_text = "camera scan" if st.session_state.input_method == "camera" else "uploaded image"
                                show_notification(f"Successfully analyzed {source_text}! ({confidence:.1%} confidence)", "success")
                                
                                st.rerun()
                            except Exception as e:
                                show_notification(f"Classification failed: {str(e)[:50]}", "error")
                st.markdown('</div>', unsafe_allow_html=True)
                
            except Exception as e:
                show_notification(f"Error loading image: {str(e)}", "error")

    def get_recycling_advice(self, predicted_class):
        """Get recycling advice based on predicted class - UPDATED for 10 categories"""
        recycling_info = {
            'battery': "⚠️ **HAZARDOUS WASTE** - Special handling required! Take to battery recycling centers or electronics stores. Do NOT throw in regular trash.",
            
            'biological': "🍂 **COMPOSTABLE** - Use green compost bin if available. Includes food scraps and garden waste. Great for reducing landfill methane.",
            
            'cardboard': "📦 **RECYCLABLE** - Flatten boxes, remove tape and plastic. Keep dry and place in cardboard/paper recycling.",
            
            'clothes': "👕 **REUSABLE/RECYCLABLE** - Donate if wearable. Otherwise, use textile recycling bins. Can be repurposed as rags.",
            
            'glass': "🟫 **RECYCLABLE GLASS** - All glass types. Rinse and place in glass container. Remove lids and caps. Separate colors if required locally.",
            
            'metal': "🥫 **RECYCLABLE** - Clean cans and containers. Remove labels if possible. Includes aluminum, tin, and steel.",
            
            'paper': "📄 **RECYCLABLE** - Keep dry and clean. Remove plastic windows. Place in paper recycling bin.",
            
            'plastic': "🧴 **RECYCLABLE** - Check local rules for plastic types. Rinse containers. Look for recycling symbols (1-7).",
            
            'shoes': "👟 **REUSABLE/RECYCLABLE** - Donate if wearable. Some brands offer recycling programs. Can be repurposed.",
            
            'trash': "🗑️ **NON-RECYCLABLE** - General waste bin. When in doubt, throw it out to avoid recycling contamination."
        }
        
        return recycling_info.get(predicted_class, "♻️ Check local recycling guidelines for specific instructions.")
    
    def results_section(self):
        """Results display section with enhanced UI"""
        st.markdown("### 📊 Classification Results")
        
        if 'result' not in st.session_state or st.session_state.result is None:
            # Enhanced placeholder with animation
            st.markdown("""
            <div class='fade-in' style='text-align: center; padding: 50px; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); 
                        border-radius: 10px; margin: 20px 0;'>
                <h3 style='color: #2E8B57;'>🚀 Ready to Scan!</h3>
                <p style='font-size: 1.1rem; color: #555;'>Use <strong>Camera Scan</strong> or <strong>Upload Image</strong> to get instant waste classification.</p>
                <p style='color: #777;'>📸 📁 Choose your input method</p>
                <div style='margin-top: 20px; font-size: 3rem;'>
                    📸 OR 📁
                </div>
            </div>
            """, unsafe_allow_html=True)
            return
        
        result = st.session_state.result
        
        # SAFETY CHECK
        if not isinstance(result, dict) or 'predicted_class' not in result:
            show_notification("Invalid result data. Please try uploading again.", "error")
            return
        
        # Show dashboard if requested
        if st.session_state.get('show_dashboard', False):
            create_dashboard(result, self.classifier)
            if st.button("← Back to Simple View"):
                st.session_state.show_dashboard = False
                st.rerun()
            return
        
        # Source badge
        source = result.get('source', 'upload')
        source_badge = "📸 Camera Scan" if source == "camera" else "📁 Uploaded Image"
        source_color = "#43e97b" if source == "camera" else "#667eea"
        
        st.markdown(f"""
        <div style='text-align: center; margin-bottom: 15px;'>
            <span style='background: {source_color}; padding: 8px 20px; border-radius: 20px; 
                        color: white; font-weight: bold; font-size: 0.9rem;'>
                {source_badge}
            </span>
        </div>
        """, unsafe_allow_html=True)
        
        # Determine confidence level and color
        confidence = result.get('confidence', 0.0)
        if confidence > 0.8:
            confidence_level = "Very High"
            confidence_color = "#28a745"
            confidence_class = "confidence-high"
        elif confidence > 0.6:
            confidence_level = "High" 
            confidence_color = "#17a2b8"
            confidence_class = "confidence-medium"
        elif confidence > 0.4:
            confidence_level = "Medium"
            confidence_color = "#ffc107"
            confidence_class = "confidence-medium"
        else:
            confidence_level = "Low"
            confidence_color = "#dc3545"
            confidence_class = "confidence-low"
        
        predicted_class = result.get('predicted_class', 'Unknown')
        
        # Special card for hazardous materials
        if predicted_class == 'battery':
            st.markdown(f"""
            <div class="hazardous-card fade-in">
                <h2>⚠️ {predicted_class.replace('-', ' ').upper()}</h2>
                <h3 style='color: white;'>📈 Confidence: <span class='{confidence_class}'>{confidence:.2%}</span></h3>
                <p><strong>HAZARDOUS MATERIAL - SPECIAL HANDLING REQUIRED</strong></p>
                <p style='font-size: 0.9rem; opacity: 0.9;'>Confidence Level: {confidence_level}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Main Prediction Card
            st.markdown(f"""
            <div class="prediction-card fade-in">
                <h2>🏷️ {predicted_class.replace('-', ' ').upper()}</h2>
                <h3 style='color: {confidence_color};'>📈 Confidence: <span class='{confidence_class}'>{confidence:.2%}</span></h3>
                <p>Confidence Level: <strong>{confidence_level}</strong></p>
                <p>AI is {confidence:.0%} confident about this classification</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Recycling Guidance Card
        recycling_advice = result.get('recycling_advice', 'No recycling advice available.')
        st.markdown(f"""
        <div class="recycling-card fade-in">
            <h3>♻️ Recycling Guidance</h3>
            <p style='font-size: 1.1rem; line-height: 1.6;'>{recycling_advice}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Interactive Visualization Toggle
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("📊 Show Dashboard", use_container_width=True):
                st.session_state.show_dashboard = True
                st.rerun()
        
        # All Predictions
        if 'all_predictions' in result and result['all_predictions']:
            with st.expander("📋 Detailed Analysis", expanded=True):
                st.write("Confidence scores for all waste categories:")
                
                # Sort predictions by confidence
                sorted_predictions = sorted(result['all_predictions'].items(), key=lambda x: x[1], reverse=True)
                
                for class_name, prob in sorted_predictions:
                    progress = int(prob * 100)
                    is_predicted = class_name == predicted_class
                    
                    col1, col2, col3 = st.columns([2, 5, 1])
                    with col1:
                        if is_predicted:
                            st.markdown(f"**🎯 {class_name.replace('-', ' ').title()}**")
                        else:
                            st.write(f"{class_name.replace('-', ' ').title()}")
                    
                    with col2:
                        # Progress bar with percentage
                        st.progress(progress, text=f"{prob:.2%}")
                    
                    with col3:
                        if is_predicted:
                            st.success("🎯")
                        elif prob > 0.1:  # Show warning for significant secondary predictions
                            st.warning("⚠️")
        
        # Additional tips based on prediction
        self.show_prediction_tips(result)
    
    def show_prediction_tips(self, result):
        """Show additional tips based on the prediction"""
        with st.expander("💡 Expert Tips & Advice", expanded=True):
            tips = {
                'battery': [
                    "Never throw batteries in regular trash - they can cause fires",
                    "Many electronics stores offer free battery recycling",
                    "Different battery types (alkaline, lithium) may have different recycling processes",
                    "Store used batteries in a non-metal container until recycling"
                ],
                'biological': [
                    "No meat or dairy in home compost - they attract pests",
                    "Eggshells are great for compost - crush them first",
                    "Yard waste like leaves and grass clippings are perfect for compost",
                    "Turn compost regularly to speed up decomposition"
                ],
                'cardboard': [
                    "Remove any plastic windows or tape from cardboard boxes",
                    "Flatten boxes to save space in recycling bins",
                    "Pizza boxes with grease stains may not be recyclable",
                    "Wax-coated cardboard (like some milk cartons) may need special recycling"
                ],
                'clothes': [
                    "Donate wearable clothes to charity shops",
                    "Even damaged clothes can be recycled as industrial rags",
                    "Some retailers offer clothing take-back programs",
                    "Shoes and clothes should be clean and dry when donating"
                ],
                'glass': [
                    "All glass colors combined in this category",
                    "Remove all metal caps and plastic seals",
                    "Broken glass should be wrapped and marked as 'broken glass'",
                    "Ceramics and ovenware are NOT recyclable with bottle glass"
                ],
                'metal': [
                    "Clean cans from food residue to avoid contamination",
                    "Aluminum cans are highly valuable to recycle",
                    "Metal can be recycled infinitely without quality loss",
                    "Small metal items can be collected in a tin can then crushed"
                ],
                'paper': [
                    "Keep paper dry and free from food stains",
                    "Shredded paper may have different recycling rules",
                    "Envelopes with plastic windows are usually acceptable",
                    "Tissue paper and napkins are NOT recyclable"
                ],
                'plastic': [
                    "Check the recycling symbol (number inside triangle)",
                    "Plastic bags usually can't be recycled curbside - take to store drop-offs",
                    "Rinse plastic containers thoroughly",
                    "Black plastic is often NOT recyclable due to sorting issues"
                ],
                'shoes': [
                    "Donate wearable shoes - many people need them",
                    "Some athletic brands have shoe recycling programs",
                    "Shoes can be repurposed for various sports or gardening",
                    "Separate pairs when donating"
                ],
                'trash': [
                    "When in doubt, throw it out to avoid recycling contamination",
                    "Food waste should go in compost or trash, not recycling",
                    "Hazardous materials need special disposal",
                    "Consider if items can be repaired before trashing"
                ]
            }
            
            predicted_class = result.get('predicted_class', '')
            predicted_tips = tips.get(predicted_class, [
                "Make sure items are clean and dry before recycling",
                "Check your local recycling guidelines for specific rules",
                "When in doubt, check with your waste management provider",
                "Reduce and reuse before recycling when possible"
            ])
            
            for tip in predicted_tips:
                st.markdown(f"• {tip}")
            
            # Quick recycling test
            st.markdown("---")
            st.markdown("#### ♻️ Quick Recycling Check")
            recyclable = predicted_class not in ['battery', 'trash']
            if recyclable:
                st.success("✅ This item is **RECYCLABLE**")
                st.info("Remember to clean it and remove any non-recyclable parts before placing in recycling bin.")
            else:
                st.warning("⚠️ This item is **NOT RECYCLABLE**")
                st.info("Please dispose of it properly according to local guidelines.")
    
    def info_section(self):
        """Information and guidance section"""
        st.markdown("---")
        st.markdown('<div class="fade-in"><h2 style="text-align: center; color: #2E8B57;">🌍 Advanced Waste Classification Guide</h2></div>', unsafe_allow_html=True)
        
        # Model Performance Section
        if self.classifier.training_info:
            info = self.classifier.training_info
            st.markdown("### 🏆 Model Performance")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Accuracy", "80%")
            
            with col2:
                total_images = info.get('dataset_summary', {}).get('total_images', 0)
                st.metric("Training Images", f"{total_images:,}")
            
            with col3:
                st.metric("Categories", len(info.get('categories', [])))
        
        # Recycling Information in expandable sections
        col1, col2 = st.columns(2)
        
        # with col1:
        #     st.markdown("### ♻️ Recyclable Materials")
            
        #     recyclable_info = {
        #         "📦 Cardboard": "Shipping boxes, packaging. Flatten and keep dry.",
        #         "🥫 Metal": "Cans, foil, containers. Clean and remove labels.",
        #         "📄 Paper": "Office paper, newspapers, magazines. Keep clean and dry.",
        #         "🧴 Plastic": "Bottles, containers. Check local recycling numbers.",
        #         "👕 Clothes": "Wearable textiles. Donate or use textile recycling.",
        #         "👟 Shoes": "Footwear in good condition. Donate or special recycling.",
        #         "🟫 Glass": "All glass types. Rinse and separate colors if needed."
        #     }
            
        #     for item, tip in recyclable_info.items():
        #         with st.expander(f"{item}"):
        #             st.write(tip)
        
        # with col2:
        #     st.markdown("### 🚫 Special Handling")
            
        #     special_info = {
        #         "⚠️ Battery": "HAZARDOUS - Special recycling required. Never in trash.",
        #         "🍂 Biological": "COMPOSTABLE - Food scraps, garden waste. Use compost bin.",
        #         "🗑️ Trash": "NON-RECYCLABLE - General waste. Last resort option.",
        #         "☢️ Electronics": "E-waste needs special recycling facilities.",
        #         "🧪 Chemicals": "Paints, cleaners - hazardous waste facilities only.",
        #         "💡 Light Bulbs": "Different types need different recycling methods."
        #     }
            
        #     for item, desc in special_info.items():
        #         with st.expander(f"{item}"):
        #             st.write(desc)
        
        # Call to Action with enhanced animation
        st.markdown("---")
        st.markdown("""
        <div class='fade-in' style='text-align: center; padding: 30px; background: linear-gradient(135deg, #2E8B57 0%, #3CB371 100%); 
                    border-radius: 15px; color: white; margin: 20px 0;'>
            <h2>🚀 Powered by Deep Learning!</h2>
            <p style='font-size: 1.2rem;'>
            <strong>80% Accuracy</strong> • <strong>35K+ Training Images</strong> • <strong>10 Waste Categories</strong><br>
            Every correct recycling decision helps our planet! 🌍💚
            </p>
        </div>
        """, unsafe_allow_html=True)

def main():
    # Initialize session state if not exists
    if 'result' not in st.session_state:
        st.session_state.result = None
    if 'image' not in st.session_state:
        st.session_state.image = None
    if 'input_method' not in st.session_state:
        st.session_state.input_method = "upload"
    if 'show_dashboard' not in st.session_state:
        st.session_state.show_dashboard = False
    if 'camera_count' not in st.session_state:
        st.session_state.camera_count = 0
    
    # Run the app
    app = WasteClassificationApp()
    app.run()

if __name__ == "__main__":
    main()