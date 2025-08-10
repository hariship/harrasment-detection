# Setup Guide - Conda + Poetry Environment

## Quick Setup

```bash
# 1. Create conda environment (Python only)
conda env create -f environment.yml

# 2. Activate the environment
conda activate harassment-detection

# 3. Install dependencies with Poetry
poetry install

# 4. (Optional) For M1/M2 Macs - Install PyTorch with MPS support
poetry run pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Alternative: Use Setup Script

```bash
# Make setup script executable
chmod +x setup.sh

# Run setup
./setup.sh

# Follow the instructions printed
```

## Daily Usage

```bash
# Always activate conda environment first
conda activate harassment-detection

# Run with Poetry
poetry run python app/main.py

# Or use the shortcut
poetry run harassment-detection

# Run tests
poetry run python test_webcam.py
```

## Why Conda + Poetry?

- **Conda**: Manages Python version and system-level dependencies
- **Poetry**: Manages Python packages with lock file for reproducibility
- **Separation**: Keeps project isolated from system Python

## Managing Dependencies

```bash
# Add new dependency
poetry add package-name

# Add dev dependency
poetry add --group dev package-name

# Update dependencies
poetry update

# Show installed packages
poetry show
```

## Troubleshooting

### If Poetry is not found after activating conda:
```bash
conda install -c conda-forge poetry
```

### If PyTorch doesn't detect MPS (M1/M2 Macs):
```bash
poetry run pip uninstall torch torchvision -y
poetry run pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Reset environment:
```bash
conda deactivate
conda env remove -n harassment-detection
# Then run setup again
```