# Harassment Detection System

Real-time harassment detection system for surveillance cameras using YOLOv8 for person detection and pluggable action recognition models.

## Architecture

**Modular monorepo design** - Each component can be easily separated into its own package:
- **Video Module**: Handles multiple input sources (webcam, RTSP, files)
- **Detection Module**: Person detection and tracking (currently YOLOv8)
- **Action Module**: Pluggable action recognition (MoViNet, custom models)
- **Pipeline**: Orchestrates all components

## Features

- Real-time person detection and tracking
- Modular action detection (easily swap between MoViNet and custom models)
- Support for webcam, RTSP streams, and video files
- Visual alerts for harassment detection
- Per-person action tracking

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# For macOS with M1/M2 (use MPS acceleration)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Quick Start

### Using Make Commands (Recommended)

```bash
# Install dependencies
make install

# Run ASGI server (FastAPI with web interface)
make run

# Run in development mode with auto-reload
make dev

# Run CLI version
make run-cli

# Test webcam
make run-webcam

# See all available commands
make help
```

### Manual Commands

```bash
# Run ASGI server
poetry run uvicorn app.api:app --reload

# Run CLI version
poetry run python app/main.py

# Run with custom config
poetry run python app/main.py --config config/custom.yaml

# Override model type
poetry run python app/main.py --model custom_harassment
```

### Access the Web Interface

After running `make run`, open your browser to:
- **Web Interface**: http://localhost:8000/demo
- **API Documentation**: http://localhost:8000/docs
- **Video Stream**: http://localhost:8000/stream

## Adding Your Custom Harassment Model

1. **Create your model class** in `core/action/your_model.py`:

```python
from core.action.base import ActionDetector
from core.action.registry import register_action_model

@register_action_model("your_model_name")
class YourHarassmentModel(ActionDetector):
    def load_model(self):
        # Load your trained model
        pass
    
    def preprocess_frames(self, frames, bbox=None):
        # Your preprocessing
        pass
    
    def predict(self, preprocessed_input):
        # Your inference logic
        pass
```

2. **Update config** to use your model:

```yaml
action_detection:
  model_type: "your_model_name"
  custom:
    model_path: "models/your_model.pt"
```

3. **Run with your model**:
```bash
python app/main.py --model your_model_name
```

## Project Structure

```
harassment-detection/
├── core/               # Core modules (can be separated)
│   ├── video/         # Video capture implementations
│   ├── detection/     # Person detection
│   └── action/        # Action recognition (pluggable)
├── app/               # Main application
├── models/            # Model weights
├── config/            # Configuration files
└── tests/             # Unit tests
```

## Configuration

Edit `config/default.yaml` to customize:
- Video source settings
- Detection thresholds
- Action model selection
- Display preferences

## Extending the System

### Add New Video Source
Implement `VideoSource` interface in `core/video/`

### Add New Person Detector
Implement `PersonDetector` interface in `core/detection/`

### Add New Action Model
Implement `ActionDetector` interface in `core/action/`

## Controls

- `q` - Quit application
- `s` - Save screenshot

## Notes

- MoViNet requires internet connection for first download
- Custom models should be placed in `models/custom/`
- Tracking IDs persist across frames for consistent person identification