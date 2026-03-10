import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2

class DataLoader:
    def __init__(self, data_dir="data", img_size=(224, 224)):
        self.data_dir = data_dir
        self.img_size = img_size
        self.categories = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    
    def create_data_generators(self, batch_size=32, validation_split=0.2):
        """
        Create data generators for training and validation
        """
        datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            validation_split=validation_split
        )
        
        train_generator = datagen.flow_from_directory(
            self.data_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        validation_generator = datagen.flow_from_directory(
            self.data_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=True
        )
        
        print(f"✅ Training samples: {train_generator.samples}")
        print(f"✅ Validation samples: {validation_generator.samples}")
        
        return train_generator, validation_generator
    
    def get_class_names(self):
        """Get sorted class names"""
        return sorted(self.categories)