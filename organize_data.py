import os
import shutil
import glob

def organize_dataset():
    """
    Main function to organize dataset into proper structure
    """
    print("🚀 Organizing Waste Classification Dataset")
    print("=" * 50)
    
    # Create data directory structure
    categories = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    for category in categories:
        os.makedirs(f'data/{category}', exist_ok=True)
    
    total_copied = 0
    
    # Method 1: Check for garbage_classification folder
    if os.path.exists('garbage_classification'):
        print("📁 Found garbage_classification folder")
        total_copied = organize_from_garbage_classification()
    
    # Method 2: Check for downloaded_dataset
    elif os.path.exists('downloaded_dataset'):
        print("📁 Found downloaded_dataset folder")
        total_copied = organize_from_downloaded()
    
    # Method 3: Check for DATASET folder
    elif os.path.exists('DATASET'):
        print("📁 Found DATASET folder")
        total_copied = organize_from_dataset()
    
    else:
        print("🔍 Searching for dataset folders...")
        total_copied = search_and_organize()
    
    if total_copied == 0:
        print("❌ No dataset found or organized!")
        print("\n📋 Please make sure you have one of these folders:")
        print("   - garbage_classification/")
        print("   - downloaded_dataset/") 
        print("   - DATASET/")
        create_sample_structure()
    else:
        print(f"\n✅ Successfully organized {total_copied} images!")
        show_final_counts()

def organize_from_garbage_classification():
    """Organize from garbage_classification folder"""
    total_copied = 0
    categories = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    
    print("🔄 Organizing from garbage_classification...")
    
    # Check if categories are directly inside garbage_classification
    for category in categories:
        source_path = os.path.join('garbage_classification', category)
        if os.path.exists(source_path):
            images = [f for f in os.listdir(source_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if images:
                print(f"   ✅ {category}: {len(images)} images")
                for image in images:
                    copy_image(source_path, image, category)
                    total_copied += 1
        else:
            print(f"   ❌ {category}: folder not found")
    
    # If no direct categories, check for train/test split
    if total_copied == 0:
        print("   🔍 Checking for train/test structure...")
        total_copied = organize_from_train_test()
    
    return total_copied

def organize_from_train_test():
    """Organize from train/test split structure"""
    total_copied = 0
    categories = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    
    # Check train folder
    train_path = os.path.join('garbage_classification', 'train')
    if os.path.exists(train_path):
        print("   📁 Processing train/ folder")
        for category in categories:
            category_path = os.path.join(train_path, category)
            if os.path.exists(category_path):
                images = [f for f in os.listdir(category_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if images:
                    print(f"      ✅ {category}: {len(images)} images")
                    for image in images:
                        copy_image(category_path, image, category)
                        total_copied += 1
    
    # Check test folder
    test_path = os.path.join('garbage_classification', 'test')
    if os.path.exists(test_path):
        print("   📁 Processing test/ folder")
        for category in categories:
            category_path = os.path.join(test_path, category)
            if os.path.exists(category_path):
                images = [f for f in os.listdir(category_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if images:
                    print(f"      ✅ {category}: {len(images)} images")
                    for image in images:
                        copy_image(category_path, image, category)
                        total_copied += 1
    
    return total_copied

def organize_from_downloaded():
    """Organize from downloaded_dataset"""
    total_copied = 0
    categories = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    
    for category in categories:
        source_path = os.path.join('downloaded_dataset', category)
        if os.path.exists(source_path):
            images = [f for f in os.listdir(source_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"   {category}: {len(images)} images")
            for image in images:
                copy_image(source_path, image, category)
                total_copied += 1
    
    return total_copied

def organize_from_dataset():
    """Organize from DATASET/TEST structure"""
    total_copied = 0
    categories = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    
    test_path = os.path.join('DATASET', 'TEST')
    if os.path.exists(test_path):
        print("📁 Processing TEST folder...")
        
        # Process TEST/O -> trash
        test_o_path = os.path.join(test_path, 'O')
        if os.path.exists(test_o_path):
            images = [f for f in os.listdir(test_o_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            print(f"   Organic waste: {len(images)} images -> trash")
            for image in images:
                copy_image(test_o_path, image, 'trash')
                total_copied += 1
        
        # Process TEST/R subfolders
        test_r_path = os.path.join(test_path, 'R')
        if os.path.exists(test_r_path):
            for category in categories:
                if category != 'trash':
                    category_path = os.path.join(test_r_path, category)
                    if os.path.exists(category_path):
                        images = [f for f in os.listdir(category_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
                        print(f"   {category}: {len(images)} images")
                        for image in images:
                            copy_image(category_path, image, category)
                            total_copied += 1
    
    return total_copied

def search_and_organize():
    """Search for any folder with category names"""
    total_copied = 0
    categories = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    
    for category in categories:
        # Look for folders with category names anywhere
        for root, dirs, files in os.walk('.'):
            if os.path.basename(root).lower() == category.lower():
                images = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if images:
                    print(f"   🔍 Found {category} folder with {len(images)} images")
                    for image in images:
                        copy_image(root, image, category)
                        total_copied += 1
                    break
    
    return total_copied

def copy_image(source_dir, filename, target_category):
    """Copy image to target category folder"""
    src_path = os.path.join(source_dir, filename)
    dest_path = f'data/{target_category}/{filename}'
    
    # Handle duplicate filenames
    counter = 1
    while os.path.exists(dest_path):
        name, ext = os.path.splitext(filename)
        dest_path = f'data/{target_category}/{name}_{counter}{ext}'
        counter += 1
    
    shutil.copy2(src_path, dest_path)

def create_sample_structure():
    """Create sample structure with instructions"""
    print("\n📝 Creating sample folder structure with instructions...")
    categories = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    
    for category in categories:
        readme_path = f'data/{category}/INSTRUCTIONS.txt'
        with open(readme_path, 'w') as f:
            f.write(f"=== {category.upper()} IMAGES ===\n\n")
            f.write(f"Add your {category} waste images to this folder.\n\n")
            f.write("SUPPORTED FORMATS:\n")
            f.write("- .jpg, .jpeg, .png\n\n")
            f.write("RECOMMENDED:\n")
            f.write("- 50-200 images per category\n")
            f.write("- Clear, well-lit photos\n")
            f.write("- Various angles and backgrounds\n\n")
            f.write("DATASET DOWNLOAD:\n")
            f.write("1. Go to: https://www.kaggle.com/datasets/mostafaabla/garbage-classification\n")
            f.write("2. Download and extract to 'garbage_classification' folder\n")
            f.write("3. Run: python organize_data.py\n")
    
    print("✅ Created sample structure in data/ folder")
    print("📋 Download the dataset and extract as 'garbage_classification'")

def show_final_counts():
    """Show final image counts"""
    print("\n📊 FINAL DATASET COUNTS:")
    print("=" * 30)
    categories = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    total = 0
    
    for category in categories:
        if os.path.exists(f'data/{category}'):
            images = [f for f in os.listdir(f'data/{category}') if f.endswith(('.jpg', '.jpeg', '.png'))]
            count = len(images)
            print(f"   {category}: {count} images")
            total += count
        else:
            print(f"   {category}: 0 images")
    
    print(f"   {'='*20}")
    print(f"   TOTAL: {total} images")
    
    if total > 0:
        print(f"\n🎯 Ready for training! Run: python train_model.py")
    else:
        print(f"\n❌ No images found. Please check dataset folder.")

def main():
    organize_dataset()

if __name__ == "__main__":
    main()