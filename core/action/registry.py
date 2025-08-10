from typing import Dict, Type, Optional
from .base import ActionDetector


class ActionModelRegistry:
    """
    Registry for action detection models.
    Allows dynamic model loading and easy swapping between implementations.
    """
    
    _models: Dict[str, Type[ActionDetector]] = {}
    
    @classmethod
    def register(cls, name: str, model_class: Type[ActionDetector]):
        """
        Register a new action detection model.
        
        Args:
            name: Model identifier (e.g., 'movinet', 'custom_harassment')
            model_class: Class implementing ActionDetector interface
        """
        cls._models[name] = model_class
        print(f"Registered action model: {name}")
    
    @classmethod
    def get_model(cls, name: str, config: dict) -> Optional[ActionDetector]:
        """
        Get an instance of registered model.
        
        Args:
            name: Model identifier
            config: Model configuration
            
        Returns:
            Initialized model instance or None if not found
        """
        if name not in cls._models:
            print(f"Model '{name}' not found in registry. Available: {list(cls._models.keys())}")
            return None
        
        model_class = cls._models[name]
        return model_class(config)
    
    @classmethod
    def list_models(cls) -> list:
        """Get list of all registered models."""
        return list(cls._models.keys())


# Decorator for easy model registration
def register_action_model(name: str):
    """
    Decorator to register action detection models.
    
    Usage:
        @register_action_model("my_model")
        class MyActionDetector(ActionDetector):
            ...
    """
    def decorator(cls):
        ActionModelRegistry.register(name, cls)
        return cls
    return decorator