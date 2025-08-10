import cv2
import numpy as np
from typing import Dict, Any, Optional, List
from collections import defaultdict
import time

# Import core modules
from core.video.webcam import WebcamSource
from core.detection.yolo_detector import YOLODetector
from core.action.registry import ActionModelRegistry


class HarassmentDetectionPipeline:
    """
    Main pipeline orchestrating video capture, person detection, and action recognition.
    Modular design allows easy swapping of components.
    """
    
    def __init__(self, config: dict):
        """
        Initialize pipeline with configuration.
        
        Args:
            config: Full system configuration dict
        """
        self.config = config
        self.video_source = None
        self.person_detector = None
        self.action_detector = None
        
        # Tracking data per person
        self.person_buffers = defaultdict(list)  # Store frames per person ID
        self.person_actions = defaultdict(list)  # Store detected actions per person
        
        # Performance metrics
        self.fps_counter = {'frames': 0, 'start_time': time.time()}
        
    def initialize(self) -> bool:
        """Initialize all pipeline components."""
        print("Initializing Harassment Detection Pipeline...")
        
        # 1. Initialize video source
        video_config = self.config.get('video', {})
        source_type = video_config.get('source', 'webcam')
        
        if source_type == 'webcam':
            self.video_source = WebcamSource(video_config)
        # Add other sources (RTSP, file) as needed
        
        if not self.video_source.initialize():
            print("Failed to initialize video source")
            return False
        print(f"✓ Video source initialized: {source_type}")
        
        # 2. Initialize person detector
        yolo_config = self.config.get('yolo', {})
        self.person_detector = YOLODetector(yolo_config)
        
        if not self.person_detector.load_model():
            print("Failed to load person detector")
            return False
        print("✓ YOLOv8 person detector loaded")
        
        # 3. Initialize action detector (pluggable)
        action_config = self.config.get('action_detection', {})
        model_type = action_config.get('model_type', 'movinet')
        
        self.action_detector = ActionModelRegistry.get_model(model_type, action_config)
        if self.action_detector and self.action_detector.load_model():
            print(f"✓ Action detector loaded: {model_type}")
        else:
            print(f"Warning: Action detector '{model_type}' not available")
            
        print("Pipeline initialization complete!\n")
        return True
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process single frame through the pipeline.
        
        Returns:
            Dict with detections, actions, and metadata
        """
        results = {
            'persons': [],
            'harassment_detected': False,
            'alerts': [],
            'fps': self.calculate_fps()
        }
        
        # 1. Detect persons
        person_detections = self.person_detector.detect(frame)
        
        # 2. Process each detected person
        for detection in person_detections:
            person_id = detection.get('track_id', -1)
            bbox = detection['bbox']
            
            person_result = {
                'id': person_id,
                'bbox': bbox,
                'confidence': detection['confidence'],
                'actions': [],
                'action_confidences': []
            }
            
            # 3. Run action detection if available
            if self.action_detector and person_id != -1:
                # Use person-specific buffer for better tracking
                # Check if detector supports person_id parameter
                try:
                    action_results = self.action_detector.detect_actions(frame, bbox, person_id)
                except TypeError:
                    # Fallback for detectors that don't support person_id
                    action_results = self.action_detector.detect_actions(frame, bbox)
                
                if action_results['actions']:
                    person_result['actions'] = action_results['actions']
                    person_result['action_confidences'] = action_results['confidences']
                    
                    # Check for harassment actions
                    harassment_actions = self.config['detection']['harassment_actions']
                    detected_harassment = [
                        action for action in action_results['actions'] 
                        if any(h in action.lower() for h in harassment_actions)
                    ]
                    
                    if detected_harassment:
                        results['harassment_detected'] = True
                        results['alerts'].append({
                            'person_id': person_id,
                            'actions': detected_harassment,
                            'timestamp': time.time(),
                            'bbox': bbox
                        })
            
            results['persons'].append(person_result)
        
        return results
    
    def visualize_results(self, frame: np.ndarray, results: Dict[str, Any]) -> np.ndarray:
        """
        Draw detection results on frame.
        
        Args:
            frame: Original frame
            results: Processing results from process_frame()
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        display_config = self.config.get('display', {})
        
        # Draw FPS
        if display_config.get('show_fps', True):
            fps_text = f"FPS: {results['fps']:.1f}"
            cv2.putText(annotated, fps_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw person detections
        for person in results['persons']:
            x1, y1, x2, y2 = [int(c) for c in person['bbox']]
            
            # Color based on gesture/harassment detection
            harassment_detected = False
            gesture_detected = False
            
            if person['actions']:
                harassment_actions = self.config['detection']['harassment_actions']
                for action in person['actions']:
                    if any(h in action.lower() for h in [h.lower() for h in harassment_actions]):
                        harassment_detected = True
                        break
                    if 'gesture' in action.lower() or 'fist' in action.lower() or 'pointing' in action.lower():
                        gesture_detected = True
            
            if harassment_detected:
                color = (0, 0, 255)  # Red for harassment gestures
            elif gesture_detected:
                color = (0, 165, 255)  # Orange for normal gestures
            else:
                color = tuple(display_config.get('bbox_color', [0, 255, 0]))  # Green for normal
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw person ID
            if person['id'] != -1:
                id_text = f"ID: {person['id']}"
                cv2.putText(annotated, id_text, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw detected actions with emphasis on gestures
            if display_config.get('show_actions', True) and person['actions']:
                y_offset = y1 - 30
                for action, conf in zip(person['actions'], person['action_confidences']):
                    # Make gesture text more prominent
                    if any(keyword in action.lower() for keyword in ['fist', 'pointing', 'gesture', 'throat', 'middle']):
                        # Larger, bold text for gestures
                        action_text = f"👆 {action.upper()}: {conf:.2f}"
                        text_color = (0, 255, 255) if harassment_detected else (255, 255, 0)
                        thickness = 3 if harassment_detected else 2
                        scale = 0.7 if harassment_detected else 0.6
                    else:
                        action_text = f"{action}: {conf:.2f}"
                        text_color = (255, 255, 255)
                        thickness = 2
                        scale = 0.5
                    
                    cv2.putText(annotated, action_text, (x1, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, scale, text_color, thickness)
                    y_offset -= 25
        
        # Draw harassment alert with gesture focus
        if results['harassment_detected']:
            alert_text = "🚨 THREATENING GESTURE DETECTED"
            cv2.putText(annotated, alert_text, (frame.shape[1] // 2 - 200, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            
            # Add blinking effect for serious alerts
            import time
            if int(time.time() * 2) % 2:  # Blink every 0.5 seconds
                cv2.putText(annotated, "GESTURE ALERT", (frame.shape[1] // 2 - 100, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        return annotated
    
    def calculate_fps(self) -> float:
        """Calculate current FPS."""
        self.fps_counter['frames'] += 1
        elapsed = time.time() - self.fps_counter['start_time']
        
        if elapsed > 1.0:
            fps = self.fps_counter['frames'] / elapsed
            self.fps_counter = {'frames': 0, 'start_time': time.time()}
            return fps
        
        return self.fps_counter.get('last_fps', 0.0)
    
    def run(self):
        """
        Main pipeline loop for continuous processing.
        Press 'q' to quit, 's' to save screenshot.
        """
        print("Starting harassment detection...")
        print("Controls: 'q' to quit, 's' to save screenshot")
        
        while True:
            # Read frame
            ret, frame = self.video_source.read_frame()
            if not ret:
                print("Failed to read frame")
                break
            
            # Process frame
            results = self.process_frame(frame)
            
            # Visualize results
            display_frame = self.visualize_results(frame, results)
            
            # Show frame
            cv2.imshow('Harassment Detection System', display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"screenshot_{timestamp}.jpg"
                cv2.imwrite(filename, display_frame)
                print(f"Screenshot saved: {filename}")
        
        self.cleanup()
    
    def cleanup(self):
        """Release all resources."""
        print("\nCleaning up resources...")
        
        if self.video_source:
            self.video_source.release()
        
        if self.person_detector:
            self.person_detector.cleanup()
        
        if self.action_detector:
            self.action_detector.cleanup()
        
        cv2.destroyAllWindows()
        print("Cleanup complete")