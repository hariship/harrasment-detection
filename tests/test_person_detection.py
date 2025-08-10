#!/usr/bin/env python3
"""
Test person detection with live webcam feed
"""

import cv2
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.detection.yolo_detector import YOLODetector
import yaml

def test_live_person_detection():
    """Test YOLO person detection on live webcam"""
    print("🔍 Testing live person detection...")
    print("Position yourself in front of the camera and press 'q' to quit")
    
    # Load config
    with open('config/default.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot access webcam")
        return
    
    # Initialize YOLO detector
    detector = YOLODetector(config['yolo'])
    if not detector.load_model():
        print("❌ Failed to load YOLO")
        return
    
    print("✅ YOLO loaded, starting detection...")
    print("Make sure you're visible in the frame!")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect persons
        detections = detector.detect(frame)
        
        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = [int(c) for c in det['bbox']]
            confidence = det['confidence']
            track_id = det.get('track_id', 'N/A')
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"Person {track_id}: {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Show detection count
        count_text = f"Persons detected: {len(detections)}"
        cv2.putText(frame, count_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Show frame
        cv2.imshow('Person Detection Test', frame)
        
        # Print detection info
        if detections:
            print(f"🎯 Found {len(detections)} person(s) - Confidences: {[f'{d['confidence']:.2f}' for d in detections]}")
        else:
            print("👻 No persons detected in this frame")
        
        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    detector.cleanup()
    cv2.destroyAllWindows()
    print("✅ Test complete!")

if __name__ == "__main__":
    test_live_person_detection()