#!/usr/bin/env python3
"""
Test script for the Harassment Detection API
Run this to test your running FastAPI server
"""

import requests
import json
import base64
import cv2
import numpy as np
from io import BytesIO
import time


BASE_URL = "http://localhost:8000"


def test_health_check():
    """Test the health endpoint"""
    print("🏥 Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data['status']}")
            print(f"   Pipeline initialized: {data['pipeline_initialized']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False


def test_root_endpoint():
    """Test the root endpoint"""
    print("\n📝 Testing root endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Root endpoint working")
            print(f"   API: {data['name']} v{data['version']}")
            return True
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Root endpoint error: {e}")
        return False


def create_test_image():
    """Create a simple test image"""
    # Create a simple test image with some shapes
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add some colored rectangles to simulate people
    cv2.rectangle(img, (100, 100), (200, 400), (0, 255, 0), -1)  # Green person
    cv2.rectangle(img, (300, 150), (400, 350), (0, 0, 255), -1)  # Red person
    
    # Add some text
    cv2.putText(img, "Test Image", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return img


def test_image_upload():
    """Test image upload endpoint"""
    print("\n📸 Testing image upload...")
    try:
        # Create test image
        img = create_test_image()
        
        # Encode image
        _, img_encoded = cv2.imencode('.jpg', img)
        img_bytes = img_encoded.tobytes()
        
        # Upload image
        files = {'file': ('test.jpg', img_bytes, 'image/jpeg')}
        response = requests.post(f"{BASE_URL}/detect/image", files=files)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Image upload successful")
            print(f"   Detections: {len(data['detections'])}")
            print(f"   Harassment detected: {data['harassment_detected']}")
            if data['detections']:
                for i, det in enumerate(data['detections']):
                    print(f"   Person {i+1}: confidence={det['confidence']:.2f}, actions={det['actions']}")
            return True
        else:
            print(f"❌ Image upload failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Image upload error: {e}")
        return False


def test_base64_detection():
    """Test base64 image detection"""
    print("\n🔢 Testing base64 detection...")
    try:
        # Create test image
        img = create_test_image()
        
        # Convert to base64
        _, img_encoded = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')
        
        # Send request
        payload = {'image': img_base64}
        response = requests.post(f"{BASE_URL}/detect/base64", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Base64 detection successful")
            print(f"   Detections: {len(data['detections'])}")
            print(f"   Harassment detected: {data['harassment_detected']}")
            return True
        else:
            print(f"❌ Base64 detection failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Base64 detection error: {e}")
        return False


def test_stream_endpoint():
    """Test if stream endpoint is accessible"""
    print("\n📹 Testing stream endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/stream", stream=True, timeout=5)
        if response.status_code == 200:
            print(f"✅ Stream endpoint accessible")
            print(f"   Content-Type: {response.headers.get('content-type')}")
            print(f"   Open http://localhost:8000/stream in browser to view")
            return True
        else:
            print(f"❌ Stream endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Stream endpoint error: {e}")
        return False


def test_demo_page():
    """Test if demo page is accessible"""
    print("\n🖥️ Testing demo page...")
    try:
        response = requests.get(f"{BASE_URL}/demo")
        if response.status_code == 200:
            print(f"✅ Demo page accessible")
            print(f"   Open http://localhost:8000/demo in browser")
            return True
        else:
            print(f"❌ Demo page failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Demo page error: {e}")
        return False


def run_all_tests():
    """Run all API tests"""
    print("🧪 Starting API Tests for Harassment Detection System")
    print("=" * 60)
    
    tests = [
        test_health_check,
        test_root_endpoint,
        test_image_upload,
        test_base64_detection,
        test_stream_endpoint,
        test_demo_page
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        time.sleep(0.5)  # Brief pause between tests
    
    print("\n" + "=" * 60)
    print(f"🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your API is working correctly.")
        print("\n📱 Next steps:")
        print("   1. Open http://localhost:8000/demo in your browser")
        print("   2. View API docs at http://localhost:8000/docs")
        print("   3. Stream video at http://localhost:8000/stream")
    else:
        print("⚠️  Some tests failed. Check your server setup.")
        
    return passed == total


if __name__ == "__main__":
    print("Make sure your FastAPI server is running on http://localhost:8000")
    print("Run: poetry run uvicorn app.api:app --reload")
    print()
    input("Press Enter when server is ready...")
    print()
    
    run_all_tests()