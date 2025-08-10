"""
Dummy person detector for testing purposes.
Returns fake person detections without requiring YOLO.
"""

import random
import numpy as np
from typing import List, Dict, Any
from .base import PersonDetector


class DummyPersonDetector(PersonDetector):
    """Dummy detector that generates fake person detections for testing."""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.detection_probability = 0.8  # High chance to detect fake persons
        
    def load_model(self) -> bool:
        """No model to load."""
        print("Dummy person detector loaded (no real detection)")
        return True
    
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Generate fake person detections."""
        detections = []
        
        if random.random() < self.detection_probability:
            # Generate 1-3 fake persons
            num_persons = random.randint(1, 3)
            
            for i in range(num_persons):
                # Random bounding box
                h, w = frame.shape[:2]
                x1 = random.randint(50, w//2)
                y1 = random.randint(50, h//2)
                x2 = x1 + random.randint(100, 200)
                y2 = y1 + random.randint(200, 300)
                
                # Keep within bounds
                x2 = min(x2, w - 50)
                y2 = min(y2, h - 50)
                
                detection = {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': random.uniform(0.7, 0.95),
                    'track_id': i + 1
                }
                detections.append(detection)
        
        return detections
    
    def cleanup(self):
        """Nothing to clean up."""
        pass