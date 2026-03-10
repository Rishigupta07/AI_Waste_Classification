import os

def check_folder_structure():
    print("🔍 Checking your dataset structure...")
    print("=" * 50)
    
    if not os.path.exists('garbage_classification'):
        print("❌ garbage_classification folder not found!")
        print("📂 Current folders:")
        for item in os.listdir('.'):
            if os.path.isdir(item):
                print(f"   📁 {item}")
        return
    
    print("✅ Found garbage_classification folder!")
    print("\n📁 Contents of garbage_classification:")
    print("=" * 30)
    
    # Check what's inside
    for item in os.listdir('garbage_classification'):
        item_path = os.path.join('garbage_classification', item)
        if os.path.isdir(item_path):
            # Count images in this folder
            images = [f for f in os.listdir(item_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"📁 {item}/ - {len(images)} images")
            
            # Show first 3 image names
            if images:
                print(f"   Sample: {', '.join(images[:3])}")
        else:
            print(f"📄 {item}")

if __name__ == "__main__":
    check_folder_structure()
    