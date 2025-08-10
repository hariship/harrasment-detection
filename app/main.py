#!/usr/bin/env python3
"""
Harassment Detection System - Main Entry Point
Modular architecture allows swapping detection models easily.
"""

import argparse
import yaml
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.pipeline import HarassmentDetectionPipeline


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Main entry point for harassment detection system."""
    parser = argparse.ArgumentParser(description='Harassment Detection System')
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/default.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--source',
        type=str,
        choices=['webcam', 'rtsp', 'file'],
        help='Override video source type'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['movinet', 'custom_harassment'],
        help='Override action detection model'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda', 'mps'],
        help='Override compute device'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Config file not found: {args.config}")
        print("Creating default configuration...")
        # Use embedded default config
        config = create_default_config()
    
    # Override config with command line arguments
    if args.source:
        config['video']['source'] = args.source
    if args.model:
        config['action_detection']['model_type'] = args.model
    if args.device:
        config['yolo']['device'] = args.device
    
    # Initialize and run pipeline
    pipeline = HarassmentDetectionPipeline(config)
    
    if pipeline.initialize():
        try:
            pipeline.run()
        except KeyboardInterrupt:
            print("\nStopping harassment detection...")
        except Exception as e:
            print(f"Error during execution: {e}")
            import traceback
            traceback.print_exc()
        finally:
            pipeline.cleanup()
    else:
        print("Failed to initialize pipeline")
        sys.exit(1)


def create_default_config():
    """Create default configuration if file doesn't exist."""
    return {
        'video': {
            'source': 'webcam',
            'webcam_id': 0,
            'fps': 30,
            'resolution': [1280, 720]
        },
        'yolo': {
            'model': 'yolov8n.pt',
            'confidence_threshold': 0.5,
            'iou_threshold': 0.45,
            'device': 'cpu',
            'enable_tracking': True
        },
        'action_detection': {
            'model_type': 'movinet',
            'confidence_threshold': 0.6,
            'movinet': {
                'model_name': 'movinet_a0_stream',
                'num_frames': 8,
                'frame_step': 2
            },
            'custom': {
                'model_path': 'models/custom/harassment_detector.pt',
                'input_size': [224, 224],
                'num_frames': 16
            }
        },
        'detection': {
            'harassment_actions': [
                'pushing', 'hitting', 'grabbing', 
                'fighting', 'aggressive_gesture'
            ],
            'buffer_size': 30,
            'min_detection_frames': 5
        },
        'display': {
            'show_fps': True,
            'show_detections': True,
            'show_actions': True,
            'bbox_color': [0, 255, 0],
            'text_color': [255, 255, 255],
            'font_scale': 0.6
        }
    }


if __name__ == '__main__':
    main()