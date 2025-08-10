import numpy as np
from typing import List, Dict, Any, Optional
import cv2
from .base import ActionDetector
from .registry import register_action_model

# Make TensorFlow imports optional
try:
    import tensorflow as tf
    import tensorflow_hub as hub
    TF_AVAILABLE = True
except ImportError as e:
    print(f"Warning: TensorFlow/TensorFlow Hub not available: {e}")
    print("MoViNet action detection will be disabled")
    TF_AVAILABLE = False


@register_action_model("movinet")
class MoViNetDetector(ActionDetector):
    """
    MoViNet action detection implementation.
    Uses Google's Mobile Video Networks for efficient action recognition.
    """
    
    # MoViNet action labels (subset relevant to harassment detection)
    RELEVANT_ACTIONS = {
        'pushing': ['pushing', 'shoving'],
        'hitting': ['punching', 'slapping', 'hitting'],
        'grabbing': ['grabbing', 'pulling'],
        'fighting': ['fighting', 'wrestling'],
        'threatening': ['threatening gesture', 'aggressive posture']
    }
    
    def __init__(self, config: dict):
        """
        Args:
            config: MoViNet configuration including model variant and frame settings
        """
        super().__init__(config)
        self.model_name = config.get('movinet', {}).get('model_name', 'movinet_a0_stream')
        self.num_frames = config.get('movinet', {}).get('num_frames', 8)
        self.frame_step = config.get('movinet', {}).get('frame_step', 2)
        self.model = None
        self.input_size = (172, 172)  # MoViNet input size
        
    def load_model(self) -> bool:
        """Load MoViNet model from TensorFlow Hub."""
        if not TF_AVAILABLE:
            print("TensorFlow not available - MoViNet cannot be loaded")
            return False
            
        try:
            # MoViNet models available on TF Hub
            model_urls = {
                'movinet_a0_stream': 'https://tfhub.dev/tensorflow/movinet/a0/stream/kinetics-600/classification/3',
                'movinet_a1_stream': 'https://tfhub.dev/tensorflow/movinet/a1/stream/kinetics-600/classification/3',
                'movinet_a2_stream': 'https://tfhub.dev/tensorflow/movinet/a2/stream/kinetics-600/classification/3',
            }
            
            if self.model_name not in model_urls:
                print(f"Unknown MoViNet model: {self.model_name}")
                return False
            
            print(f"Loading MoViNet model: {self.model_name}")
            self.model = hub.load(model_urls[self.model_name])
            
            # Initialize state for streaming mode
            self.init_states = self.model.init_states
            self.states = self.init_states(1)  # Batch size 1
            
            return True
        except Exception as e:
            print(f"Failed to load MoViNet model: {e}")
            return False
    
    def preprocess_frames(self, frames: List[np.ndarray], bbox: Optional[List[float]] = None) -> np.ndarray:
        """
        Preprocess frames for MoViNet input.
        
        Args:
            frames: List of frames
            bbox: Optional person bounding box [x1, y1, x2, y2]
            
        Returns:
            Preprocessed tensor ready for model
        """
        processed_frames = []
        
        # Sample frames based on frame_step
        sampled_frames = frames[::self.frame_step][:self.num_frames]
        
        for frame in sampled_frames:
            # Crop to person if bbox provided
            if bbox is not None:
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                # Add padding around bbox
                pad = 20
                y1 = max(0, y1 - pad)
                x1 = max(0, x1 - pad)
                y2 = min(frame.shape[0], y2 + pad)
                x2 = min(frame.shape[1], x2 + pad)
                cropped = frame[y1:y2, x1:x2]
            else:
                cropped = frame
            
            # Resize to model input size
            resized = cv2.resize(cropped, self.input_size)
            
            # Normalize to [0, 1]
            normalized = resized.astype(np.float32) / 255.0
            processed_frames.append(normalized)
        
        # Pad if not enough frames
        while len(processed_frames) < self.num_frames:
            processed_frames.append(processed_frames[-1])
        
        # Stack and add batch dimension
        video_tensor = np.stack(processed_frames, axis=0)  # [T, H, W, C]
        video_tensor = np.expand_dims(video_tensor, axis=0)  # [1, T, H, W, C]
        
        return tf.constant(video_tensor, dtype=tf.float32)
    
    def predict(self, preprocessed_input: tf.Tensor) -> Dict[str, Any]:
        """
        Run MoViNet prediction on preprocessed frames.
        
        Returns:
            Detected actions and confidence scores
        """
        if self.model is None:
            return {'actions': [], 'confidences': []}
        
        try:
            # Run inference in streaming mode
            logits, self.states = self.model({
                'image': preprocessed_input,
                'states': self.states
            })
            
            # Get probabilities
            probs = tf.nn.softmax(logits, axis=-1)
            probs_np = probs.numpy()[0]
            
            # Get top predictions
            top_k = 5
            top_indices = np.argsort(probs_np)[-top_k:][::-1]
            
            # Map to action labels (simplified - in real implementation, load Kinetics labels)
            detected_actions = []
            confidences = []
            
            for idx in top_indices:
                confidence = float(probs_np[idx])
                if confidence > self.confidence_threshold:
                    # Map index to action (placeholder - need actual Kinetics-600 labels)
                    action = self._map_kinetics_to_harassment(idx)
                    if action:
                        detected_actions.append(action)
                        confidences.append(confidence)
            
            return {
                'actions': detected_actions,
                'confidences': confidences,
                'raw_output': {'logits': logits.numpy(), 'top_indices': top_indices.tolist()}
            }
            
        except Exception as e:
            print(f"MoViNet prediction error: {e}")
            return {'actions': [], 'confidences': []}
    
    def _map_kinetics_to_harassment(self, kinetics_idx: int) -> Optional[str]:
        """
        Map Kinetics-600 class index to harassment-related action.
        This is a placeholder - implement with actual Kinetics label mapping.
        """
        # Placeholder mapping - replace with actual Kinetics-600 labels
        harassment_indices = {
            100: 'pushing',
            150: 'hitting',
            200: 'fighting',
            250: 'grabbing'
        }
        
        return harassment_indices.get(kinetics_idx)
    
    def cleanup(self):
        """Release model resources."""
        self.model = None
        self.states = None
    
    def reset_buffer(self):
        """Reset frame buffer and model states."""
        super().reset_buffer()
        if self.model and self.init_states:
            self.states = self.init_states(1)