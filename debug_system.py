#!/usr/bin/env python3
"""
Debug script to test system components individually
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import yaml
import numpy as np

def test_webcam():
    """Test basic webcam access"""
    print("📹 Testing webcam access...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot access webcam")
        return False
    
    ret, frame = cap.read()
    if not ret:
        print("❌ Cannot read from webcam")
        cap.release()
        return False
    
    print(f"✅ Webcam working - Frame shape: {frame.shape}")
    cap.release()
    return True

def test_yolo():
    """Test YOLO loading"""
    print("\n🎯 Testing YOLO detector...")
    try:
        from ultralytics import YOLO
        print("✅ Ultralytics imported successfully")
        
        # Try loading YOLO model
        model = YOLO('yolov8n.pt')
        print("✅ YOLOv8 model loaded successfully")
        
        # Test inference on dummy image
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        results = model(dummy_img, verbose=False)
        print("✅ YOLO inference working")
        
        return True
    except Exception as e:
        print(f"❌ YOLO test failed: {e}")
        return False

def test_config():
    """Test config loading"""
    print("\n⚙️ Testing configuration...")
    try:
        with open('config/default.yaml', 'r') as f:
            config = yaml.safe_load(f)
        print("✅ Config loaded successfully")
        print(f"   Video source: {config['video']['source']}")
        print(f"   Action model: {config['action_detection']['model_type']}")
        return config
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return None

def test_pipeline_components():
    """Test individual pipeline components"""
    print("\n🔧 Testing pipeline components...")
    
    config = test_config()
    if not config:
        return False
    
    # Test video source
    try:
        from core.video.webcam import WebcamSource
        video_source = WebcamSource(config['video'])
        if video_source.initialize():
            print("✅ Video source initialized")
            ret, frame = video_source.read_frame()
            if ret and frame is not None:
                print(f"✅ Frame capture working - Shape: {frame.shape}")
            else:
                print("❌ Frame capture failed")
            video_source.release()
        else:
            print("❌ Video source failed to initialize")
    except Exception as e:
        print(f"❌ Video source test failed: {e}")
    
    # Test YOLO detector
    try:
        from core.detection.yolo_detector import YOLODetector
        detector = YOLODetector(config['yolo'])
        if detector.load_model():
            print("✅ YOLO detector loaded")
            
            # Test detection on dummy frame
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            detections = detector.detect(dummy_frame)
            print(f"✅ YOLO detection working - Found {len(detections)} objects")
        else:
            print("❌ YOLO detector failed to load")
    except Exception as e:
        print(f"❌ YOLO detector test failed: {e}")
    
    # Test action detector
    try:
        from core.action.registry import ActionModelRegistry
        action_detector = ActionModelRegistry.get_model(
            config['action_detection']['model_type'], 
            config['action_detection']
        )
        if action_detector and action_detector.load_model():
            print(f"✅ Action detector loaded: {config['action_detection']['model_type']}")
        else:
            print(f"❌ Action detector failed to load: {config['action_detection']['model_type']}")
    except Exception as e:
        print(f"❌ Action detector test failed: {e}")

def test_full_pipeline():
    """Test complete pipeline"""
    print("\n🚀 Testing full pipeline...")
    
    try:
        from app.pipeline import HarassmentDetectionPipeline
        
        config = test_config()
        if not config:
            return False
        
        pipeline = HarassmentDetectionPipeline(config)
        if pipeline.initialize():
            print("✅ Pipeline initialized successfully")
            
            # Test processing a frame
            ret, frame = pipeline.video_source.read_frame()
            if ret and frame is not None:
                results = pipeline.process_frame(frame)
                print(f"✅ Frame processing working")
                print(f"   Persons detected: {len(results['persons'])}")
                print(f"   Harassment detected: {results['harassment_detected']}")
                return True
            else:
                print("❌ Could not read frame from pipeline")
        else:
            print("❌ Pipeline initialization failed")
            
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
    
    return False

def main():
    """Run all debug tests"""
    print("🔍 Harassment Detection System - Debug Tests")
    print("=" * 50)
    
    # Run tests
    webcam_ok = test_webcam()
    yolo_ok = test_yolo()
    
    if not webcam_ok:
        print("\n❌ Webcam issues detected. Check:")
        print("   - Is webcam connected?")
        print("   - Is another app using it?")
        print("   - Try different webcam_id in config")
    
    if not yolo_ok:
        print("\n❌ YOLO issues detected. Try:")
        print("   - poetry install (reinstall packages)")
        print("   - Check internet connection (YOLO downloads weights)")
    
    if webcam_ok and yolo_ok:
        test_pipeline_components()
        test_full_pipeline()
    
    print("\n" + "=" * 50)
    print("Debug complete!")

if __name__ == "__main__":
    main()