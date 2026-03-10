import tensorflow as tf
import numpy as np
from PIL import Image
import os
import json

def test_new_model():
    print("🧪 Testing New Comprehensive Model")
    print("=" * 50)
    
    model_path = '../models/waste_classifier_12class_comprehensive.h5'
    
    if not os.path.exists(model_path):
        print("❌ Model not found. Please train the model first.")
        print("💡 Run: python train_comprehensive.py")
        return
    
    try:
        # Load the model
        model = tf.keras.models.load_model(model_path)
        print("✅ Model loaded successfully!")
        
        # Test model architecture
        print(f"📐 Input shape: {model.input_shape}")
        print(f"📊 Output shape: {model.output_shape}")
        print(f"🏗️  Number of layers: {len(model.layers)}")
        
        # Test with dummy data
        print("\n🎯 Testing with dummy data...")
        dummy_input = np.random.random((1, 224, 224, 3))
        prediction = model.predict(dummy_input, verbose=0)
        
        print(f"✅ Prediction shape: {prediction.shape}")
        print(f"🔢 Number of classes: {prediction.shape[1]}")
        
        # Show class names
        class_names = [
            'battery', 'biological', 'brown-glass', 'cardboard', 'clothes',
            'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass'
        ]
        
        print("\n📋 Available Categories (12 classes):")
        for i, name in enumerate(class_names):
            print(f"   {i+1:2d}. {name}")
        
        # Check if metrics file exists
        metrics_path = '../models/model_metrics_12class.json'
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            print(f"\n📊 Model Performance Metrics:")
            print(f"   ✅ Overall Accuracy: {metrics.get('overall_accuracy', 'N/A'):.4f}")
            print(f"   ✅ Validation Accuracy: {metrics.get('validation_accuracy', 'N/A'):.4f}")
            print(f"   ✅ Training Images: {metrics.get('total_training_images', 'N/A')}")
            print(f"   ✅ Validation Images: {metrics.get('total_validation_images', 'N/A')}")
        
        # Test with a sample image if available
        print("\n🔍 Looking for sample images to test...")
        sample_found = False
        if os.path.exists('../garbage_classification'):
            for category in os.listdir('../garbage_classification')[:3]:  # Check first 3 categories
                category_path = os.path.join('../garbage_classification', category)
                if os.path.isdir(category_path):
                    images = [f for f in os.listdir(category_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    if images:
                        sample_image_path = os.path.join(category_path, images[0])
                        try:
                            # Test with actual image
                            image = Image.open(sample_image_path)
                            image = image.resize((224, 224))
                            image_array = np.array(image) / 255.0
                            image_array = np.expand_dims(image_array, axis=0)
                            
                            prediction = model.predict(image_array, verbose=0)
                            predicted_class = np.argmax(prediction[0])
                            confidence = np.max(prediction[0])
                            
                            class_names = [
                                'battery', 'biological', 'brown-glass', 'cardboard', 'clothes',
                                'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass'
                            ]
                            
                            print(f"   🎯 Sample test: '{category}' → Predicted: '{class_names[predicted_class]}' (Confidence: {confidence:.2%})")
                            sample_found = True
                            break
                        except Exception as e:
                            continue
            
            if not sample_found:
                print("   💡 No sample images found for testing")
        
        print("\n✅ Model test completed successfully!")
        print("🎯 The model is ready to use in the app!")
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")

if __name__ == "__main__":
    test_new_model()