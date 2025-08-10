from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np


class ActionDetector(ABC):
    """
    Abstract base class for action detection models.
    Designed to be swappable - implement this interface for any custom model.
    """
    
    def __init__(self, config: dict):
        """
        Args:
            config: Model-specific configuration
        """
        self.config = config
        self.confidence_threshold = config.get('confidence_threshold', 0.6)
        self.frame_buffer = []  # Store frames for temporal models
        
    @abstractmethod
    def load_model(self) -> bool:
        """Load action recognition model."""
        pass
    
    @abstractmethod
    def preprocess_frames(self, frames: List[np.ndarray], bbox: Optional[List[float]] = None) -> Any:
        """
        Preprocess frames for model input.
        
        Args:
            frames: List of frames for temporal analysis
            bbox: Optional person bounding box to crop [x1, y1, x2, y2]
        
        Returns:
            Preprocessed input ready for model
        """
        pass
    
    @abstractmethod
    def predict(self, preprocessed_input: Any) -> Dict[str, Any]:
        """
        Run action prediction on preprocessed input.
        
        Returns:
            Dict containing:
                - actions: List of detected actions
                - confidences: Confidence scores for each action
                - raw_output: Optional raw model output for debugging
        """
        pass
    
    def detect_actions(self, frame: np.ndarray, person_bbox: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Main entry point for action detection.
        Handles frame buffering and calls model-specific methods.
        
        Args:
            frame: Current frame
            person_bbox: Bounding box of person to analyze
            
        Returns:
            Detection results from predict()
        """
        # Add frame to buffer
        self.frame_buffer.append(frame)
        
        # Keep only required number of frames
        buffer_size = self.config.get('num_frames', 8)
        if len(self.frame_buffer) > buffer_size:
            self.frame_buffer = self.frame_buffer[-buffer_size:]
        
        # Need enough frames for prediction
        if len(self.frame_buffer) < buffer_size:
            return {'actions': [], 'confidences': []}
        
        # Preprocess and predict
        preprocessed = self.preprocess_frames(self.frame_buffer, person_bbox)
        return self.predict(preprocessed)
    
    @abstractmethod
    def cleanup(self):
        """Release model resources."""
        pass
    
    def reset_buffer(self):
        """Clear frame buffer (useful when switching persons or scenes)."""
        self.frame_buffer = []