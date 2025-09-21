"""
Quick setup guide for the Rain Streak Detection System
"""

import os
import subprocess
import sys

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = [
        ('streamlit', 'streamlit'),
        ('numpy', 'numpy'), 
        ('opencv-python', 'cv2'),
        ('scipy', 'scipy'),
        ('scikit-image', 'skimage'),
        ('matplotlib', 'matplotlib'),
        ('Pillow', 'PIL'),
        ('plotly', 'plotly')
    ]
    
    missing_packages = []
    
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    return missing_packages

def main():
    print("🌧️ Rain Streak Detection System - Setup Check")
    print("=" * 50)
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return
    else:
        print("✅ Python version is compatible")
    
    # Check dependencies
    print("\n🔍 Checking dependencies...")
    missing = check_dependencies()
    
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print("\n📦 To install missing packages, run:")
        print("pip install -r requirements.txt")
        return
    else:
        print("✅ All dependencies are installed")
    
    # Check if app.py exists
    if os.path.exists('app.py'):
        print("✅ Streamlit app found")
    else:
        print("❌ app.py not found")
        return
    
    print("\n🚀 Setup complete! To start the application:")
    print("1. Run: streamlit run app.py")
    print("2. Or use: ./launch.sh")
    print("3. Open browser at: http://localhost:8501")
    
    print("\n💡 Usage Tips:")
    print("• Upload both ground truth and rainy images")
    print("• Images must have the same dimensions")
    print("• Adjust parameters in the sidebar for different scenarios")
    print("• Download results using the buttons at the bottom")

if __name__ == "__main__":
    main()
