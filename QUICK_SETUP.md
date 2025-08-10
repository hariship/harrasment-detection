# Quick Setup Instructions

## Manual Setup (Recommended)

Run these commands one by one:

```bash
# 1. Create conda environment with Python 3.10
conda create -n harassment-detection python=3.10 -y

# 2. Activate the environment
conda activate harassment-detection

# 3. Install Poetry
pip install poetry

# 4. Install all project dependencies
poetry install

# 5. (For Apple Silicon Macs) Install PyTorch with MPS support
poetry run pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 6. Verify installation
poetry run python -c "import cv2, torch, ultralytics; print('✅ All packages installed!')"
```

## Test Your Setup

```bash
# Make sure environment is activated
conda activate harassment-detection

# Test webcam and basic components
poetry run python test_webcam.py

# Run the full system
poetry run python app/main.py
```

## If You Get Errors

### Conda not found
- Make sure conda is installed: https://docs.conda.io/en/latest/miniconda.html
- Try: `source ~/miniconda3/etc/profile.d/conda.sh` (adjust path as needed)

### Poetry not found after installation
```bash
# Make sure you're in the conda environment
conda activate harassment-detection
# Reinstall poetry
pip install --upgrade poetry
```

### Package conflicts
```bash
# Clean install
conda deactivate
conda env remove -n harassment-detection -y
# Start over from step 1
```

## Daily Usage

Always remember to activate the environment before working:
```bash
conda activate harassment-detection
poetry run python app/main.py
```