from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np


class VideoSource(ABC):
    def __init__(self, source_config: dict):
        self.config = source_config
        self.fps = source_config.get('fps', 30)
        self.resolution = source_config.get('resolution', (1280, 720))
        
    @abstractmethod
    def initialize(self) -> bool:
        pass
    
    @abstractmethod
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        pass
    
    @abstractmethod
    def release(self):
        pass
    
    @property
    @abstractmethod
    def is_opened(self) -> bool:
        pass