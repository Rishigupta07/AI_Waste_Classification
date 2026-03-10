#!/usr/bin/env python3
"""
Main entry point for AI Waste Classification Project
"""

import os
import sys

def main():
    print("♻️ AI Waste Classification Project")
    print("=" * 40)
    
    print("1. Train model")
    print("2. Run web app") 
    print("3. Check data balance")
    print("4. Exit")
    
    choice = input("\nChoose an option (1-4): ")
    
    if choice == "1":
        from train_model import main as train_main
        train_main()
    elif choice == "2":
        os.system("streamlit run app/app.py")
    elif choice == "3":
        from src.utils import check_data_balance
        counts = check_data_balance("data")
        print("\n📊 Data Balance:")
        for category, count in counts.items():
            print(f"   {category}: {count} images")
    elif choice == "4":
        print("Goodbye!")
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()