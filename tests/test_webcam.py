#!/usr/bin/env python3
"""
Quick test script to verify webcam and basic detection works.
Run this before the full pipeline to debug issues.
"""

import cv2
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.video.webcam import WebcamSource
from core.detection.yolo_detector import YOLODetector


def test_webcam():
    """Test webcam capture."""
    print("Testing webcam...")
    config = {
        'webcam_id': 0,
        'fps': 30,
        'resolution': [1280, 720]
    }
    
    webcam = WebcamSource(config)
    if not webcam.initialize():
        print("Failed to initialize webcam")
        return False
    
    print("Webcam initialized. Press 'q' to continue to next test...")
    while True:
        ret, frame = webcam.read_frame()
        if not ret:
            print("Failed to read frame")
            break
        
        cv2.imshow('Webcam Test', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    webcam.release()
    cv2.destroyAllWindows()
    return True


def test_yolo():
    """Test YOLO person detection."""
    print("\nTesting YOLO person detection...")
    
    # Initialize webcam
    webcam_config = {
        'webcam_id': 0,
        'fps': 30,
        'resolution': [1280, 720]
    }
    webcam = WebcamSource(webcam_config)
    
    if not webcam.initialize():
        print("Failed to initialize webcam")
        return False
    
    # Initialize YOLO
    yolo_config = {
        'model': 'yolov8n.pt',
        'confidence_threshold': 0.5,
        'device': 'cpu',
        'enable_tracking': True
    }
    
    detector = YOLODetector(yolo_config)
    if not detector.load_model():
        print("Failed to load YOLO model")
        return False
    
    print("YOLO loaded. Detecting persons. Press 'q' to quit...")
    
    while True:
        ret, frame = webcam.read_frame()
        if not ret:
            break
        
        # Detect persons
        detections = detector.detect(frame)
        
        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = [int(c) for c in det['bbox']]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Show track ID if available
            if det.get('track_id'):
                text = f"Person {det['track_id']}"
                cv2.putText(frame, text, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Show count
        count_text = f"Persons detected: {len(detections)}"
        cv2.putText(frame, count_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('YOLO Person Detection Test', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    webcam.release()
    detector.cleanup()
    cv2.destroyAllWindows()
    return True


if __name__ == '__main__':
    print("=== Harassment Detection System - Component Test ===\n")
    
    # Test webcam
    if test_webcam():
        print("✓ Webcam test passed")
    else:
        print("✗ Webcam test failed")
        sys.exit(1)
    
    # Test YOLO
    if test_yolo():
        print("✓ YOLO test passed")
    else:
        print("✗ YOLO test failed")
        sys.exit(1)
    
    print("\n=== All tests passed! ===")
    print("You can now run the full system with: python app/main.py")