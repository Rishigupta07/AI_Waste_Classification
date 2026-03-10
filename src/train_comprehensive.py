import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import json

def train_comprehensive_model():
    print("🚀 Training Comprehensive Waste Classification Model")
    print("=" * 60)
    print("📁 Using dataset: garbage_classification/ (12 categories)")
    print("🎯 Categories: battery, biological, brown-glass, cardboard, clothes, green-glass, metal, paper, plastic, shoes, trash, white-glass")
    print("=" * 60)
    
    # Configuration
    DATA_DIR = '../garbage_classification'
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 30
    MODEL_SAVE_PATH = '../models/waste_classifier_12class_comprehensive.h5'
    
    # Create models directory if it doesn't exist
    os.makedirs('../models', exist_ok=True)
    
    # Check if dataset exists
    if not os.path.exists(DATA_DIR):
        print(f"❌ Dataset not found: {DATA_DIR}")
        print("📂 Available directories:")
        for item in os.listdir('..'):
            if os.path.isdir(os.path.join('..', item)):
                print(f"   📁 {item}")
        return
    
    # Get categories
    categories = sorted([d for d in os.listdir(DATA_DIR) 
                        if os.path.isdir(os.path.join(DATA_DIR, d))])
    
    print(f"📊 Found {len(categories)} categories: {categories}")
    
    # Count images per category
    total_images = 0
    for category in categories:
        category_path = os.path.join(DATA_DIR, category)
        images = [f for f in os.listdir(category_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        image_count = len(images)
        total_images += image_count
        print(f"   📁 {category:15} - {image_count:4} images")
    
    print(f"🖼️  Total images in dataset: {total_images}")
    
    if total_images == 0:
        print("❌ No images found in the dataset!")
        return
    
    # Data generators with augmentation
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2  # 80% training, 20% validation
    )
    
    print("\n📦 Loading and preprocessing images...")
    
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
    
    print(f"✅ Training samples: {train_generator.samples}")
    print(f"✅ Validation samples: {validation_generator.samples}")
    print(f"✅ Class indices: {train_generator.class_indices}")
    
    # Model architecture (using transfer learning)
    print("\n🏗️ Building model architecture...")
    
    # Use pretrained MobileNetV2
    base_model = tf.keras.applications.MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Fine-tune the base model
    base_model.trainable = True
    
    # Add custom classification head
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
    
    print("✅ Model compiled successfully!")
    print(f"📐 Total layers: {len(model.layers)}")
    print(f"🎯 Output classes: {len(categories)}")
    
    # Callbacks for better training
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
    print("\n🎯 Starting training... This may take a while...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save training history
    history_path = '../models/training_history_12class.json'
    with open(history_path, 'w') as f:
        json.dump({
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']],
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']],
            'class_indices': train_generator.class_indices,
            'categories': categories
        }, f, indent=2)
    
    print(f"✅ Training completed! Model saved: {MODEL_SAVE_PATH}")
    print(f"✅ Training history saved: {history_path}")
    
    # Evaluate the model
    print("\n📊 Evaluating model performance...")
    validation_generator.reset()
    predictions = model.predict(validation_generator, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Get true classes
    true_classes = validation_generator.classes
    class_labels = list(validation_generator.class_indices.keys())
    
    # Generate classification report
    report = classification_report(true_classes, predicted_classes, 
                                 target_names=class_labels, output_dict=True)
    
    print(f"🎯 Overall Accuracy: {report['accuracy']:.4f}")
    print(f"📈 Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    
    # Save evaluation metrics
    metrics_path = '../models/model_metrics_12class.json'
    with open(metrics_path, 'w') as f:
        json.dump({
            'overall_accuracy': float(report['accuracy']),
            'validation_accuracy': float(history.history['val_accuracy'][-1]),
            'training_accuracy': float(history.history['accuracy'][-1]),
            'class_wise_accuracy': {class_labels[i]: float(report[str(i)]['precision']) 
                                  for i in range(len(class_labels))},
            'categories': categories,
            'total_training_images': train_generator.samples,
            'total_validation_images': validation_generator.samples
        }, f, indent=2)
    
    print(f"✅ Evaluation metrics saved: {metrics_path}")
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    plt.title('Model Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('Model Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    # Class distribution
    class_counts = [sum(true_classes == i) for i in range(len(class_labels))]
    plt.barh(class_labels, class_counts)
    plt.title('Validation Set Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Images', fontsize=12)
    plt.tight_layout()
    
    plt.savefig('../models/training_history_12class.png', dpi=300, bbox_inches='tight')
    print("✅ Training plots saved: ../models/training_history_12class.png")
    
    # Final message
    print("\n" + "=" * 60)
    print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"📊 Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"📁 Model saved: {MODEL_SAVE_PATH}")
    print(f"📈 Training history: ../models/training_history_12class.json")
    print(f"📋 Performance metrics: ../models/model_metrics_12class.json")
    
    print("\n🎯 Next steps:")
    print("1. Run: python test_new_model.py (to verify the model)")
    print("2. Update your app.py to use the new model")
    print("3. Test with sample images")
    print("4. Deploy the updated application")
    
    return model, categories

if __name__ == "__main__":
    # Check for TensorFlow and GPU
    print(f"🔧 TensorFlow Version: {tf.__version__}")
    print(f"🏃‍♂️ GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    
    if len(tf.config.list_physical_devices('GPU')) > 0:
        print("✅ Using GPU for faster training!")
    else:
        print("⚠️  No GPU detected - training will be slower")
    
    model, categories = train_comprehensive_model()