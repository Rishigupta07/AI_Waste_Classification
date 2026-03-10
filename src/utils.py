import os
import numpy as np
from PIL import Image

def check_data_balance(data_dir):
    """
    Check if data is balanced across categories
    """
    categories = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    counts = {}
    
    for category in categories:
        category_path = os.path.join(data_dir, category)
        if os.path.exists(category_path):
            images = [f for f in os.listdir(category_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            counts[category] = len(images)
        else:
            counts[category] = 0
    
    return counts

def validate_image(file_path):
    """
    Validate if file is a proper image
    """
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except:
        return False