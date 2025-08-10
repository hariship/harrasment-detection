from typing import List, Dict, Any
import numpy as np
from ultralytics import YOLO
from .base import PersonDetector


class YOLODetector(PersonDetector):
    """YOLOv8 person detection implementation with tracking support."""
    
    def __init__(self, config: dict):
        """
        Args:
            config: YOLO configuration with model path, device, thresholds
        """
        super().__init__(config)
        self.model_path = config.get('model', 'yolov8n.pt')
        self.device = config.get('device', 'cpu')
        self.iou_threshold = config.get('iou_threshold', 0.45)
        self.model = None
        self.tracker_enabled = config.get('enable_tracking', True)
        
    def load_model(self) -> bool:
        """Load YOLOv8 model with specified weights."""
        try:
            self.model = YOLO(self.model_path)
            # Warm up model with dummy inference
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model(dummy, device=self.device, verbose=False)
            return True
        except Exception as e:
            print(f"Failed to load YOLO model: {e}")
            return False
    
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect persons in frame using YOLOv8.
        
        Returns:
            List of person detections with bounding boxes and tracking IDs
        """
        if self.model is None:
            return []
        
        detections = []
        
        # Run inference with or without tracking
        if self.tracker_enabled:
            results = self.model.track(
                frame, 
                persist=True,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                classes=[0],  # Person class only
                device=self.device,
                verbose=False,
                tracker="bytetrack.yaml",  # Use ByteTrack for better tracking
                imgsz=640  # Smaller image size for speed
            )
        else:
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                classes=[0],  # Person class only
                device=self.device,
                verbose=False,
                imgsz=640  # Smaller image size for speed
            )
        
        # Process results
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                for i, box in enumerate(result.boxes):
                    detection = {
                        'bbox': box.xyxy[0].cpu().numpy().tolist(),  # [x1, y1, x2, y2]
                        'confidence': float(box.conf[0]),
                        'track_id': None
                    }
                    
                    # Add tracking ID if available
                    if self.tracker_enabled and hasattr(box, 'id') and box.id is not None:
                        detection['track_id'] = int(box.id[0])
                    
                    detections.append(detection)
        
        return detections
    
    def cleanup(self):
        """Release model resources."""
        self.model = None