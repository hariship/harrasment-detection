"""
Dummy action detector for testing purposes.
Returns random/mock detections without requiring any ML libraries.
"""

import numpy as np
import random
from typing import List, Dict, Any, Optional
from .base import ActionDetector
from .registry import register_action_model


@register_action_model("dummy")
class DummyActionDetector(ActionDetector):
    """
    Simple dummy action detector for testing the pipeline.
    Randomly generates harassment actions for demonstration purposes.
    """
    
    # Focus on gesture-based actions for harassment detection
    NEUTRAL_GESTURES = [
        'normal_gesture',       # Regular gestures
        'neutral_stance',       # Standing normally
        'arms_crossed',         # Neutral blocking gesture
        'hands_on_hips',        # Neutral stance
        'waving_hands'          # Normal waving
    ]
    
    AGGRESSIVE_GESTURES = [
        'raised_fist',           # Threatening gesture
        'pointing_aggressively', # Aggressive pointing
        'finger_pointing',      # Accusatory gesture
        'shoving_motion',       # Push-like gesture
        'throat_slash',         # Threatening gesture
        'middle_finger',        # Offensive gesture
        'fist_shake',          # Angry gesture
        'hitting_motion',       # Hitting movement
        'stop_hand'            # Aggressive blocking
    ]
    
    HARASSMENT_GESTURES = [
        'raised_fist', 'pointing_aggressively', 'throat_slash', 
        'middle_finger', 'fist_shake', 'shoving_motion', 'hitting_motion'
    ]
    
    def __init__(self, config: dict):
        """Initialize dummy detector."""
        super().__init__(config)
        self.detection_probability = config.get('dummy', {}).get('detection_probability', 0.1)
        self.harassment_probability = config.get('dummy', {}).get('harassment_probability', 0.05)
        self.person_states = {}  # Track state per person to avoid rapid changes
        self.last_gestures = {}  # Track last gesture per person for context
        
    def load_model(self) -> bool:
        """No model to load for dummy detector."""
        print("Dummy action detector loaded (no ML model required)")
        return True
    
    def preprocess_frames(self, frames: List[np.ndarray], bbox: Optional[List[float]] = None) -> np.ndarray:
        """
        Mock preprocessing - just return frame count.
        
        Args:
            frames: Input frames
            bbox: Person bounding box
            
        Returns:
            Simple array representing frame data
        """
        return np.array([len(frames)])
    
    def predict(self, preprocessed_input: np.ndarray, person_id: int = None) -> Dict[str, Any]:
        """
        Mock prediction - generate stable actions per person.
        
        Returns:
            Fake detection results for testing
        """
        detected_actions = []
        confidences = []
        
        # Use person_id for consistent behavior, fallback to random
        if person_id is None:
            person_id = 1
        
        # Generate gesture detection every 2 frames for high responsiveness
        frame_count = len(self.frame_buffer)
        should_detect_gesture = (frame_count + person_id) % 2 == 0  # High frequency
        
        # Simulate movement detection - slightly reduced for better balance
        sudden_movement = random.randint(1, 6) == 1  # ~17% chance of "sudden movement"
        # Additional random trigger for extra responsiveness  
        extra_trigger = random.randint(1, 10) == 1  # 10% extra chance
        
        if should_detect_gesture or extra_trigger:
            # Only detect aggressive gestures during significant movement
            if sudden_movement:
                # Different gestures based on movement type simulation
                movement_type = random.randint(1, 4)
                
                if movement_type == 1:
                    # Hitting-related movements
                    gesture = random.choice(['hitting_motion', 'shoving_motion'])
                elif movement_type == 2:
                    # Hand/fist movements (less random)
                    gesture = random.choice(['fist_shake', 'pointing_aggressively'])
                elif movement_type == 3:
                    # Raised hand movements - only if previous context makes sense
                    last_gesture = self.last_gestures.get(person_id, None)
                    if last_gesture in ['fist_shake', 'pointing_aggressively'] or random.randint(1, 3) == 1:
                        gesture = 'raised_fist'
                    else:
                        gesture = 'fist_shake'  # More contextual alternative
                else:
                    # General aggressive motion
                    gesture = 'shoving_motion'
                
                confidence = random.uniform(0.85, 0.99)
                print(f"💥 MOVEMENT: '{gesture}' for person {person_id}")
                detected_actions.append(gesture)
                confidences.append(confidence)
                # Track this gesture for context
                self.last_gestures[person_id] = gesture
            elif extra_trigger and random.randint(1, 3) == 1:  # Reduced from 50% to 33%
                # Less aggressive detection for minor activity
                minor_gestures = ['pointing_aggressively', 'normal_gesture']
                gesture = random.choice(minor_gestures)
                confidence = random.uniform(0.70, 0.85)
                print(f"👁️ MINOR ACTIVITY: '{gesture}' for person {person_id}")
                detected_actions.append(gesture)
                confidences.append(confidence)
            else:
                # Sometimes show neutral behavior, balanced approach
                behavior_type = random.randint(1, 5)  # 1-5 range, more silence time
                
                if behavior_type == 1:
                    # Occasionally show normal gestures (non-threatening)
                    gesture = random.choice(self.NEUTRAL_GESTURES)
                    confidence = random.uniform(0.70, 0.85)
                    detected_actions.append(gesture)
                    confidences.append(confidence)
                elif behavior_type == 2:
                    # Show minimal postural changes
                    minimal_gestures = ['neutral_stance', 'arms_crossed']
                    gesture = random.choice(minimal_gestures)
                    confidence = random.uniform(0.75, 0.88)
                    detected_actions.append(gesture)
                    confidences.append(confidence)
                # behavior_type 3-5: No detection (silent/stationary)
        
        return {
            'actions': detected_actions,
            'confidences': confidences,
            'raw_output': {
                'person_id': person_id,
                'frame_count': frame_count,
                'detection_triggered': should_detect_gesture
            }
        }
    
    def cleanup(self):
        """Nothing to clean up."""
        pass