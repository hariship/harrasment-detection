#!/bin/bash

echo "🚀 Setting up Harassment Detection with venv + Poetry"
echo ""

# Use Python 3.10 if available, otherwise use system Python
if command -v python3.10 &> /dev/null; then
    PYTHON_CMD="python3.10"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    echo "❌ Python 3 not found. Please install Python 3.10+"
    exit 1
fi

echo "Using Python: $($PYTHON_CMD --version)"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
$PYTHON_CMD -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install poetry
echo "Installing Poetry..."
pip install poetry

# Install dependencies
echo "Installing project dependencies..."
poetry install

# For Apple Silicon Macs
if [[ $(uname -m) == 'arm64' ]]; then
    echo ""
    echo "Detected Apple Silicon Mac, installing PyTorch with MPS support..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "Then run the system with:"
echo "  python app/main.py"
echo ""
echo "Or test components with:"
echo "  python test_webcam.py"