import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

class WasteModelTrainer:
    def __init__(self, data_dir="data", img_size=224):
        self.data_dir = data_dir
        self.img_size = img_size
        self.model = None
        
    def create_model(self, num_classes=6):
        """
        Create CNN model using transfer learning
        """
        # Use MobileNetV2 as base
        base_model = keras.applications.MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(self.img_size, self.img_size, 3)
        )
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Build custom classifier
        self.model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        return self.model
    
    def train(self, epochs=10, batch_size=32):
        """
        Train the waste classification model
        """
        from .data_loader import DataLoader
        
        # Create data generators
        loader = DataLoader(self.data_dir)
        train_gen, val_gen = loader.create_data_generators(batch_size=batch_size)
        
        # Create model
        self.create_model(num_classes=6)
        
        # Compile model
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(
                'models/waste_model.h5',
                save_best_only=True,
                monitor='val_accuracy'
            )
        ]
        
        print("🔄 Starting training...")
        # Train model
        history = self.model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        return history