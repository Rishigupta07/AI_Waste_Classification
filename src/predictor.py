import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

class WastePredictor:
    def __init__(self, model_path='models/waste_model.h5'):
        self.model_path = model_path
        self.model = None
        self.class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
        self.load_model()
        
        # Recycling information
        self.recycling_info = {
            'cardboard': '♻️ RECYCLABLE - Flatten boxes and remove any tape or labels',
            'glass': '♻️ RECYCLABLE - Rinse containers and remove lids', 
            'metal': '♻️ RECYCLABLE - Clean cans and remove labels if possible',
            'paper': '♻️ RECYCLABLE - Keep dry and clean, no food contamination',
            'plastic': '♻️ CHECK SYMBOL - Usually recyclable if clean. Check the number inside the triangle.',
            'trash': '🚫 NOT RECYCLABLE - Dispose in general waste bin. Includes food waste and contaminated items.'
        }
    
    def load_model(self):
        """Load the trained model"""
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"⚠️ Could not load model: {e}")
            print("🔧 Using mock predictions for testing")
            self.model = None
    
    def convert_to_3_channel(self, img_array):
        """
        Convert 4-channel RGBA images to 3-channel RGB
        """
        if img_array.shape[-1] == 4:  # RGBA image
            # Convert RGBA to RGB by removing alpha channel
            img_array = img_array[:, :, :3]
            print("🔄 Converted RGBA to RGB")
        
        return img_array
    
    def preprocess_image(self, img_array):
        """
        Preprocess image for model prediction
        """
        # Convert to 3 channels if needed
        img_array = self.convert_to_3_channel(img_array)
        
        # Resize to 224x224
        img = tf.image.resize(img_array, [224, 224])
        
        # Normalize to [0, 1]
        img = img / 255.0
        
        # Add batch dimension
        img = tf.expand_dims(img, axis=0)
        
        return img
    
    def predict_image(self, img_array):
        """
        Predict waste category from image array
        """
        if self.model is None:
            # Mock prediction for testing
            return self.mock_prediction()
        
        try:
            # Preprocess image
            img = self.preprocess_image(img_array)
            
            # Make prediction
            predictions = self.model.predict(img, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class_idx]
            
            predicted_class = self.class_names[predicted_class_idx]
            recycling_advice = self.recycling_info[predicted_class]
            
            # Get all probabilities
            all_predictions = {
                self.class_names[i]: float(predictions[0][i]) 
                for i in range(len(self.class_names))
            }
            
            return {
                'predicted_class': predicted_class,
                'confidence': float(confidence),
                'recycling_advice': recycling_advice,
                'all_predictions': all_predictions
            }
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return self.mock_prediction()
    
    def mock_prediction(self):
        """Return mock prediction for testing"""
        return {
            'predicted_class': 'plastic',
            'confidence': 0.85,
            'recycling_advice': '♻️ CHECK SYMBOL - Usually recyclable if clean. Check the number inside the triangle.',
            'all_predictions': {
                'cardboard': 0.05,
                'glass': 0.03, 
                'metal': 0.02,
                'paper': 0.01,
                'plastic': 0.85,
                'trash': 0.04
            }
        }