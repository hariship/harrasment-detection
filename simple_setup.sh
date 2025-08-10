#!/bin/bash

echo "🚀 Simple Setup for Harassment Detection System"
echo ""
echo "This script will guide you through the setup process."
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed or not in PATH"
    echo "Please install Anaconda or Miniconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "Step 1: Creating conda environment..."
echo "Running: conda create -n harassment-detection python=3.10 -y"
echo ""
conda create -n harassment-detection python=3.10 -y

echo ""
echo "Step 2: Environment created! Now you need to:"
echo ""
echo "1. Activate the environment:"
echo "   conda activate harassment-detection"
echo ""
echo "2. Install Poetry in the environment:"
echo "   pip install poetry"
echo ""
echo "3. Install project dependencies:"
echo "   poetry install"
echo ""
echo "4. (Optional) For M1/M2 Macs, update PyTorch:"
echo "   poetry run pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu"
echo ""
echo "5. Test the setup:"
echo "   poetry run python test_webcam.py"