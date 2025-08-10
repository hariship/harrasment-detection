import numpy as np
from typing import List, Dict, Any, Optional
import cv2
from .base import ActionDetector
from .registry import register_action_model

# Make PyTorch optional
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    print("Warning: PyTorch not available for custom harassment model")
    TORCH_AVAILABLE = False


@register_action_model("custom_harassment")
class CustomHarassmentDetector(ActionDetector):
    """
    Template for custom harassment detection model.
    Replace the model loading and prediction logic with your trained model.
    """
    
    # Define harassment action classes your model predicts
    ACTION_CLASSES = [
        'normal',
        'pushing',
        'hitting',
        'grabbing',
        'cornering',
        'threatening',
        'stalking',
        'physical_harassment',
        'aggressive_gesture'
    ]
    
    def __init__(self, config: dict):
        """
        Args:
            config: Custom model configuration
        """
        super().__init__(config)
        custom_config = config.get('custom', {})
        self.model_path = custom_config.get('model_path', 'models/custom/harassment_detector.pt')
        self.input_size = tuple(custom_config.get('input_size', [224, 224]))
        self.num_frames = custom_config.get('num_frames', 16)
        self.preprocessing = custom_config.get('preprocessing', 'standard')
        self.device = None
        self.model = None
        
        # Set device only if torch is available
        if TORCH_AVAILABLE:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_model(self) -> bool:
        """Load your custom trained harassment detection model."""
        if not TORCH_AVAILABLE:
            print("PyTorch not available - custom model cannot be loaded")
            return False
            
        try:
            # Example: Load a PyTorch model
            # Replace this with your actual model loading logic
            print(f"Loading custom model from: {self.model_path}")
            
            # Option 1: Load entire model
            # self.model = torch.load(self.model_path, map_location=self.device)
            
            # Option 2: Load just state dict (recommended)
            # self.model = YourModelArchitecture()  # Define your model architecture
            # self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            
            # Placeholder - implement your model loading
            # self.model = self._create_dummy_model()
            # self.model.to(self.device)
            # self.model.eval()
            
            print("Custom model loaded successfully")
            return True
            
        except Exception as e:
            print(f"Failed to load custom model: {e}")
            return False
    
    def _create_dummy_model(self):
        """
        Placeholder model architecture.
        Replace with your actual model.
        """
        class DummyHarassmentModel(nn.Module):
            def __init__(self, num_frames=16, num_classes=9):
                super().__init__()
                # Example: 3D CNN for video classification
                self.conv3d_1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=1)
                self.pool3d = nn.MaxPool3d(kernel_size=(2, 2, 2))
                self.conv3d_2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1)
                self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
                self.fc = nn.Linear(128, num_classes)
                self.relu = nn.ReLU()
                
            def forward(self, x):
                x = self.relu(self.conv3d_1(x))
                x = self.pool3d(x)
                x = self.relu(self.conv3d_2(x))
                x = self.global_pool(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x
        
        return DummyHarassmentModel()
    
    def preprocess_frames(self, frames: List[np.ndarray], bbox: Optional[List[float]] = None) -> torch.Tensor:
        """
        Preprocess frames for your custom model.
        Implement your specific preprocessing pipeline here.
        """
        processed_frames = []
        
        # Sample frames uniformly
        indices = np.linspace(0, len(frames) - 1, self.num_frames, dtype=int)
        sampled_frames = [frames[i] for i in indices]
        
        for frame in sampled_frames:
            # Crop to person if bbox provided
            if bbox is not None:
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                # Add padding
                pad = 30
                y1 = max(0, y1 - pad)
                x1 = max(0, x1 - pad)
                y2 = min(frame.shape[0], y2 + pad)
                x2 = min(frame.shape[1], x2 + pad)
                cropped = frame[y1:y2, x1:x2]
            else:
                cropped = frame
            
            # Resize
            resized = cv2.resize(cropped, self.input_size)
            
            # Apply preprocessing based on config
            if self.preprocessing == 'standard':
                # Standard normalization
                normalized = resized.astype(np.float32) / 255.0
                # ImageNet normalization (optional)
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                normalized = (normalized - mean) / std
            else:
                # Custom preprocessing
                normalized = resized.astype(np.float32) / 255.0
            
            processed_frames.append(normalized)
        
        # Convert to tensor [B, C, T, H, W] for 3D CNN
        video_array = np.stack(processed_frames, axis=0)  # [T, H, W, C]
        video_array = video_array.transpose(3, 0, 1, 2)  # [C, T, H, W]
        video_tensor = torch.from_numpy(video_array).float().unsqueeze(0)  # [1, C, T, H, W]
        
        return video_tensor.to(self.device)
    
    def predict(self, preprocessed_input: torch.Tensor) -> Dict[str, Any]:
        """
        Run your custom model prediction.
        """
        if self.model is None:
            # Return empty if model not loaded
            return {'actions': [], 'confidences': []}
        
        try:
            with torch.no_grad():
                # Run inference
                outputs = self.model(preprocessed_input)
                
                # Apply softmax to get probabilities
                probs = torch.nn.functional.softmax(outputs, dim=1)
                probs_np = probs.cpu().numpy()[0]
                
                # Get predictions above threshold
                detected_actions = []
                confidences = []
                
                for idx, prob in enumerate(probs_np):
                    if prob > self.confidence_threshold and idx > 0:  # Skip 'normal' class
                        action = self.ACTION_CLASSES[idx]
                        detected_actions.append(action)
                        confidences.append(float(prob))
                
                # Sort by confidence
                if detected_actions:
                    sorted_pairs = sorted(zip(detected_actions, confidences), 
                                        key=lambda x: x[1], reverse=True)
                    detected_actions, confidences = zip(*sorted_pairs)
                
                return {
                    'actions': list(detected_actions),
                    'confidences': list(confidences),
                    'raw_output': {
                        'all_probs': probs_np.tolist(),
                        'classes': self.ACTION_CLASSES
                    }
                }
                
        except Exception as e:
            print(f"Custom model prediction error: {e}")
            return {'actions': [], 'confidences': []}
    
    def cleanup(self):
        """Release model resources."""
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()