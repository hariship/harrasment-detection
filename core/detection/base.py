from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np


class PersonDetector(ABC):
    """Abstract base class for person detection models."""
    
    def __init__(self, config: dict):
        """
        Args:
            config: Model configuration including thresholds and device settings
        """
        self.config = config
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        
    @abstractmethod
    def load_model(self) -> bool:
        """Load detection model weights and initialize."""
        pass
    
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect persons in frame.
        
        Returns:
            List of detections, each containing:
                - bbox: [x1, y1, x2, y2]
                - confidence: float
                - track_id: Optional tracking ID
        """
        pass
    
    @abstractmethod
    def cleanup(self):
        """Release model resources."""
        pass