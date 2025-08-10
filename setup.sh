#!/bin/bash

# Setup script for harassment detection with conda + poetry

echo "🚀 Setting up Harassment Detection Environment..."
echo ""

# Step 1: Create conda environment
echo "Step 1: Creating conda environment with Python 3.10..."
conda env create -f environment.yml

echo ""
echo "Step 2: Activating environment and installing dependencies with Poetry..."
echo ""

# Create activation script for user to run
cat << 'EOF' > activate_and_install.sh
#!/bin/bash

# Activate conda environment
conda activate harassment-detection

# Install poetry dependencies
echo "Installing dependencies with Poetry..."
poetry install

# For M1/M2 Macs - special handling for PyTorch with MPS support
if [[ $(uname -m) == 'arm64' ]]; then
    echo ""
    echo "Detected Apple Silicon Mac, installing PyTorch with MPS support..."
    poetry run pip uninstall torch torchvision -y
    poetry run pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "To test the installation:"
echo "  poetry run python test_webcam.py"
echo ""
echo "To run the system:"
echo "  poetry run python app/main.py"
echo ""
echo "Or use the shortcuts:"
echo "  poetry run harassment-detection"
EOF

chmod +x activate_and_install.sh

echo "✨ Conda environment created!"
echo ""
echo "Now run these commands to complete setup:"
echo ""
echo "  conda activate harassment-detection"
echo "  ./activate_and_install.sh"
echo ""
echo "After setup, always activate with:"
echo "  conda activate harassment-detection"