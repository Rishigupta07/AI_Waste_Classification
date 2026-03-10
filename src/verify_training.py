import os
import glob
import json

def verify_current_training():
    print("🔍 Verifying Current Training Status...")
    print("=" * 50)
    
    # Check model files
    model_files = glob.glob('../models/*.h5') + glob.glob('../models/*.hdf5') + glob.glob('../models/*.keras')
    
    if model_files:
        print("✅ Found trained model files:")
        for model_file in model_files:
            file_size = os.path.getsize(model_file) / (1024 * 1024)
            print(f"   📁 {os.path.basename(model_file)} ({file_size:.1f} MB)")
            
            # Try to detect which dataset was used
            filename = os.path.basename(model_file).lower()
            if '12class' in filename or 'comprehensive' in filename:
                print("   🎯 Likely trained on: garbage_classification (12 categories)")
            elif '6class' in filename or 'basic' in filename:
                print("   🎯 Likely trained on: data (6 categories)")
            else:
                print("   🤔 Unknown dataset - need to verify")
    else:
        print("❌ No trained models found in ../models/")
    
    # Check dataset sizes for comparison
    print("\n📊 Dataset Comparison:")
    
    # Check data/ directory
    if os.path.exists('../data'):
        data_categories = [d for d in os.listdir('../data') if os.path.isdir(os.path.join('../data', d))]
        data_images = 0
        for category in data_categories:
            cat_path = os.path.join('../data', category)
            images = [f for f in os.listdir(cat_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            data_images += len(images)
        print(f"   📁 data/ - {len(data_categories)} categories, {data_images} total images")
    else:
        print("   📁 data/ - Not found")
    
    # Check garbage_classification/ directory
    if os.path.exists('../garbage_classification'):
        garbage_categories = [d for d in os.listdir('../garbage_classification') if os.path.isdir(os.path.join('../garbage_classification', d))]
        garbage_images = 0
        for category in garbage_categories:
            cat_path = os.path.join('../garbage_classification', category)
            images = [f for f in os.listdir(cat_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            garbage_images += len(images)
        print(f"   📁 garbage_classification/ - {len(garbage_categories)} categories, {garbage_images} total images")
        print(f"   📋 Categories: {', '.join(sorted(garbage_categories))}")
    else:
        print("   📁 garbage_classification/ - Not found")
    
    # Check for training history files
    print("\n📈 Training Artifacts:")
    history_files = glob.glob('../models/*history*.json')
    if history_files:
        for history_file in history_files:
            print(f"   ✅ {os.path.basename(history_file)}")
            try:
                with open(history_file, 'r') as f:
                    history = json.load(f)
                if 'accuracy' in history:
                    epochs = len(history['accuracy'])
                    final_acc = history['accuracy'][-1] if history['accuracy'] else 'N/A'
                    final_val_acc = history['val_accuracy'][-1] if history['val_accuracy'] else 'N/A'
                    print(f"      📊 Epochs: {epochs}, Final Acc: {final_acc:.4f}, Val Acc: {final_val_acc:.4f}")
            except:
                print("      ⚠️ Could not read history file")
    else:
        print("   ❌ No training history files found")

def main():
    verify_current_training()
    
    print("\n" + "=" * 50)
    print("🎯 RECOMMENDATION")
    print("=" * 50)
    
    if os.path.exists('../garbage_classification'):
        garbage_categories = [d for d in os.listdir('../garbage_classification') if os.path.isdir(os.path.join('../garbage_classification', d))]
        if len(garbage_categories) >= 12:
            print("💡 Train the COMPREHENSIVE dataset for better results!")
            print("   Run: python train_comprehensive.py")
        else:
            print("💡 Use the existing models or train with available data")
    else:
        print("💡 Use the existing models in ../models/")

if __name__ == "__main__":
    main()