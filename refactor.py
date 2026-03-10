import os
import re

app_path = r"c:\Users\ritik\OneDrive\Desktop\NewOne\AI_Waste_Classification\app\app.py"
css_path = r"c:\Users\ritik\OneDrive\Desktop\NewOne\AI_Waste_Classification\assets\style.css"

with open(app_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Match the CSS block exactly
css_pattern = r'st\.markdown\("""\n<style>(.*?)</style>\n""", unsafe_allow_html=True\)'
match = re.search(css_pattern, content, re.DOTALL)

if match:
    css_content = match.group(1).strip()
    
    # Save to style.css
    os.makedirs(os.path.dirname(css_path), exist_ok=True)
    with open(css_path, 'w', encoding='utf-8') as f:
        f.write(css_content)
        
    print("CSS extracted successfully.")

    # Replace in app.py
    new_css_loader = """def load_css():
    css_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'style.css')
    try:
        with open(css_path, 'r') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except Exception as e:
        print(f"Could not load CSS: {e}")

load_css()"""
    
    content = content.replace(match.group(0), new_css_loader)
    
    with open(app_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("app.py updated with CSS loader.")
else:
    print("Could not find CSS block.")
