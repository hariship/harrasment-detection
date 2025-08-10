import cv2
from typing import Optional, Tuple
import numpy as np
from .base import VideoSource


class WebcamSource(VideoSource):
    """Webcam video source implementation for local camera access."""
    
    def __init__(self, source_config: dict):
        """
        Args:
            source_config: Dict with webcam_id, fps, resolution settings
        """
        super().__init__(source_config)
        self.camera_id = source_config.get('webcam_id', 0)
        self.cap = None
        
    def initialize(self) -> bool:
        """Initialize webcam connection and configure capture settings."""
        self.cap = cv2.VideoCapture(self.camera_id)
        if self.cap.isOpened():
            # Configure camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            return True
        return False
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read single frame from webcam."""
        if self.cap is None:
            return False, None
        return self.cap.read()
    
    def release(self):
        """Release webcam resources."""
        if self.cap:
            self.cap.release()
            self.cap = None
    
    @property
    def is_opened(self) -> bool:
        """Check if webcam is active and ready."""
        return self.cap is not None and self.cap.isOpened()