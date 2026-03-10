import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import json
from datetime import datetime

def analyze_dataset(data_dir):
    """Analyze the dataset structure and return information"""
    print("\n[INFO] Analyzing dataset structure...")
    
    # Get all categories (folders)
    categories = []
    for d in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, d)):
            categories.append(d)
    categories.sort()
    
    if not categories:
        print("[ERROR] No categories found! Please check your dataset structure.")
        return None
    
    category_info = {}
    total_images = 0
    
    print(f"[INFO] Found {len(categories)} categories:")
    print("-" * 40)
    
    for category in categories:
        category_path = os.path.join(data_dir, category)
        image_count = 0
        for f in os.listdir(category_path):
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')):
                image_count += 1
        total_images += image_count
        category_info[category] = image_count
        
        status = "[OK]" if image_count >= 10 else "[LOW]"
        print(f"{status} {category:20} - {image_count:4} images")
        
        if image_count < 10:
            print(f"   [WARNING] Low sample count! Consider adding more images.")
    
    print("-" * 40)
    print(f"[INFO] Total images: {total_images}")
    print(f"[INFO] Average per class: {total_images/len(categories):.1f}")
    
    if category_info:
        min_images = min(category_info.values())
        max_images = max(category_info.values())
        
        if max_images > 5 * min_images:
            print("[WARNING] Significant class imbalance detected!")
            print(f"   Smallest class: {min_images} images")
            print(f"   Largest class: {max_images} images")
    
    return categories, category_info, total_images

def train_updated_model():
    """Train model with automatically detected dataset"""
    print("\n" + "=" * 70)
    print("AUTO-DETECT WASTE CLASSIFICATION MODEL TRAINING")
    print("=" * 70)
    
    # FIXED PATH: Changed from 'app/garbage_classification' to 'garbage_classification'
    DATA_DIR = 'garbage_classification'  # IMPORTANT: This is the correct path for your structure
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 30
    
    # Create timestamp for model versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    MODEL_NAME = f'waste_classifier_{timestamp}.h5'
    MODEL_SAVE_PATH = f'models/{MODEL_NAME}'
    
    # Create directories if they don't exist
    os.makedirs('models', exist_ok=True)
    
    # Check if dataset exists
    if not os.path.exists(DATA_DIR):
        print(f"[ERROR] Dataset not found at: {DATA_DIR}")
        print("\nCurrent directory structure:")
        for item in os.listdir('.'):
            if os.path.isdir(item):
                print(f"   {item}/")
        return None, None
    
    # Analyze dataset
    result = analyze_dataset(DATA_DIR)
    if result is None:
        return None, None
    
    categories, category_info, total_images = result
    
    if total_images == 0:
        print("[ERROR] No images found in the dataset!")
        return None, None
    
    # Data generators with augmentation
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )
    
    print("\n[INFO] Loading and preprocessing images...")
    
    # Training data
    train_generator = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Validation data
    validation_generator = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    print(f"[OK] Training samples: {train_generator.samples}")
    print(f"[OK] Validation samples: {validation_generator.samples}")
    print(f"[OK] Classes: {list(train_generator.class_indices.keys())}")
    
    # Model building
    print(f"\n[INFO] Building model for {len(categories)} classes...")
    print("[INFO] Using transfer learning with MobileNetV2")
    
    base_model = tf.keras.applications.MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    base_model.trainable = True
    
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(len(categories), activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            MODEL_SAVE_PATH,
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # Train the model
    print("\n[INFO] Starting training...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save training history
    history_path = f'models/training_history_{timestamp}.json'
    with open(history_path, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'model_name': MODEL_NAME,
            'categories': categories,
            'category_counts': category_info,
            'total_images': total_images,
            'training_samples': train_generator.samples,
            'validation_samples': validation_generator.samples,
            'class_indices': train_generator.class_indices,
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']],
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']]
        }, f, indent=2)
    
    # Evaluate the model
    print("\n[INFO] Evaluating model performance...")
    validation_generator.reset()
    predictions = model.predict(validation_generator, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = validation_generator.classes
    class_labels = list(validation_generator.class_indices.keys())
    
    # Generate classification report
    report = classification_report(true_classes, predicted_classes, 
                                 target_names=class_labels, output_dict=True)
    
    # Save evaluation metrics
    metrics_path = f'models/model_metrics_{timestamp}.json'
    with open(metrics_path, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'model_name': MODEL_NAME,
            'overall_accuracy': float(report['accuracy']),
            'validation_accuracy': float(history.history['val_accuracy'][-1]),
            'training_accuracy': float(history.history['accuracy'][-1]),
            'categories': categories,
            'dataset_summary': {
                'total_images': total_images,
                'training_images': train_generator.samples,
                'validation_images': validation_generator.samples,
                'categories_count': len(categories)
            }
        }, f, indent=2)
    
    # Create training plots
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = f'models/training_plots_{timestamp}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Training plots saved: {plot_path}")
    
    # Create README file
    create_model_readme(MODEL_NAME, categories, category_info, timestamp, 
                       float(report['accuracy']))
    
    # Print results
    print("\n" + "=" * 70)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Overall Accuracy: {report['accuracy']:.4f}")
    print(f"Model saved: {MODEL_SAVE_PATH}")
    print(f"Training history: {history_path}")
    print(f"Performance metrics: {metrics_path}")
    
    print("\nNext steps:")
    print("1. Test the model: python src/test_new_model.py")
    print("2. Update your app.py to use the new model")
    print("3. Test with sample images")
    print("4. Deploy the updated application")
    
    return model, categories

def create_model_readme(model_name, categories, category_info, timestamp, accuracy):
    """Create a README file with model information"""
    categories_list = ""
    for cat in categories:
        categories_list += f"- {cat}: {category_info[cat]} images\n"
    
    total_imgs = sum(category_info.values())
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Build the README content safely
    readme_content = f"# Waste Classification Model - {timestamp}\n\n"
    readme_content += f"## Model Information\n"
    readme_content += f"- **Model Name**: {model_name}\n"
    readme_content += f"- **Training Date**: {timestamp}\n"
    readme_content += f"- **Accuracy**: {accuracy:.4f}\n"
    readme_content += f"- **Number of Classes**: {len(categories)}\n\n"
    readme_content += f"## Classes\n{categories_list}\n"
    readme_content += f"## Dataset Summary\n"
    readme_content += f"- **Total Images**: {total_imgs}\n"
    readme_content += f"- **Training Split**: 80%\n"
    readme_content += f"- **Validation Split**: 20%\n"
    readme_content += f"- **Image Size**: 224x224 pixels\n\n"
    readme_content += f"## Training Parameters\n"
    readme_content += f"- **Base Model**: MobileNetV2 (Transfer Learning)\n"
    readme_content += f"- **Epochs**: 30\n"
    readme_content += f"- **Batch Size**: 32\n"
    readme_content += f"- **Learning Rate**: 0.0001\n\n"
    readme_content += f"## Files Generated\n"
    readme_content += f"1. `{model_name}` - Trained model weights\n"
    readme_content += f"2. `training_history_{timestamp}.json` - Training history\n"
    readme_content += f"3. `model_metrics_{timestamp}.json` - Performance metrics\n"
    readme_content += f"4. `training_plots_{timestamp}.png` - Training visualizations\n\n"
    readme_content += f"## Usage\n```python\nfrom tensorflow import keras\nmodel = keras.models.load_model('models/{model_name}')\n```\n\n"
    readme_content += f"## Performance\n"
    readme_content += f"- Validation Accuracy: {accuracy:.2%}\n"
    readme_content += f"- Classes: {len(categories)}\n"
    readme_content += f"- Last updated: {current_time}\n"
    
    readme_path = f'models/README_{timestamp}.md'
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"[OK] Model documentation saved: {readme_path}")

def check_gpu():
    """Check GPU availability"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"[OK] GPU Available: {len(gpus)} device(s)")
        return True
    else:
        print("[INFO] No GPU detected - training will be slower on CPU")
        return False

if __name__ == "__main__":
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"Python Version: {os.sys.version}")
    
    # Check GPU
    check_gpu()
    
    # Start training
    try:
        model, categories = train_updated_model()
    except Exception as e:
        print(f"\n[ERROR] Error during training: {e}")
        print("\nTroubleshooting steps:")
        print("1. Check if dataset exists at: garbage_classification/")
        print("2. Make sure each category has image files")
        print("3. Check TensorFlow installation")
        print("4. Verify Python dependencies are installed")
        import traceback
        traceback.print_exc()