#!/usr/bin/env python3
"""
Quick install script for missing packages
"""

import subprocess
import sys

def install_package(package):
    """Install package using pip"""
    print(f"Installing {package}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False

def main():
    """Install essential packages"""
    print("📦 Installing essential packages for harassment detection...")
    
    packages = [
        "ultralytics",  # YOLOv8
        "opencv-python",
        "numpy",
        "pillow",
        "pyyaml",
        "fastapi",
        "uvicorn[standard]",
        "python-multipart",
        "requests"  # For testing
    ]
    
    success_count = 0
    for package in packages:
        if install_package(package):
            success_count += 1
    
    print(f"\n📊 Installation complete: {success_count}/{len(packages)} packages installed")
    
    if success_count == len(packages):
        print("🎉 All packages installed! You can now test the system.")
        print("\nNext steps:")
        print("1. Restart your FastAPI server")
        print("2. Run: python debug_system.py")
        print("3. Check the demo at http://localhost:8000/demo")
    else:
        print("⚠️ Some packages failed to install. Check the errors above.")

if __name__ == "__main__":
    main()