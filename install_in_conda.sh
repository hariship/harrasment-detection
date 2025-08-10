#!/bin/bash

echo "📦 Installing packages in conda environment..."
echo ""

# Activate conda environment and install packages
echo "Make sure you run this in your terminal where conda works:"
echo ""
echo "conda activate harassment-detection"
echo "pip install ultralytics opencv-python numpy pillow pyyaml fastapi uvicorn[standard] python-multipart"
echo ""
echo "OR if you're using poetry:"
echo ""
echo "conda activate harassment-detection"
echo "poetry install"
echo "poetry add ultralytics"
echo ""
echo "Then restart your FastAPI server in the same terminal!"
echo ""
echo "After installation, test with:"
echo "python debug_system.py"