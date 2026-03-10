#!/usr/bin/env python3
"""
Training script for waste classification model
"""

import os
import tensorflow as tf
from src.model_trainer import WasteModelTrainer

def main():
    print("🎯 Training Waste Classification Model")
    print("=" * 40)
    
    # Check if data exists
    data_dir = "data"
    categories = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    
    # Verify data exists
    data_exists = True
    total_images = 0
    
    print("📊 Checking dataset...")
    for category in categories:
        category_path = os.path.join(data_dir, category)
        if os.path.exists(category_path):
            images = [f for f in os.listdir(category_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            count = len(images)
            print(f"   {category}: {count} images")
            total_images += count
            if count == 0:
                data_exists = False
        else:
            print(f"   {category}: FOLDER MISSING")
            data_exists = False
    
    if not data_exists or total_images == 0:
        print(f"\n❌ No training data found!")
        print("Please organize data first using: python organize_data.py")
        return
    
    print(f"\n✅ Found {total_images} total images")
    print("🚀 Starting model training...")
    
    # Initialize trainer
    trainer = WasteModelTrainer(data_dir=data_dir)
    
    # Train model
    history = trainer.train(epochs=15)
    
    print("✅ Training completed!")
    print("📊 Model saved to: models/waste_model.h5")

if __name__ == "__main__":
    main()