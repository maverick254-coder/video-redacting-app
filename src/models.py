"""
Model loading module for Ultralytics YOLO.

This module provides utilities to load and configure YOLO object detection models.
It supports both CPU and GPU (CUDA) inference devices.

Environment variables:
  - YOLO_MODEL_PATH: Path to a custom model weights file (optional, defaults to yolov8n.pt)
  - YOLO_DEFAULT_DEVICE: Default device for inference ('cpu' or 'cuda', defaults to 'cpu')

Typical usage:
  from models import load_models
  model = load_models(device='cpu')
  results = model(frame)  # Run inference on a frame
"""

import logging
from typing import Optional
from ultralytics import YOLO

logger = logging.getLogger(__name__)

def load_models(model_path: Optional[str] = None, device: str = 'cpu') -> YOLO:
    """Load and return an Ultralytics YOLO model instance.

    YOLO is a real-time object detection model that detects multiple classes
    (person, car, face, etc.) in images/video frames. This function:
    1. Loads the model weights (either custom or default yolov8n.pt)
    2. Sets the inference device (CPU or CUDA/GPU)
    3. Returns the model ready for inference

    Args:
        model_path (Optional[str]): Path to custom model weights file.
                                   If None, uses default 'yolov8n.pt' (nano model).
                                   For face detection, use a face-specific model.
        device (str): Inference device, 'cpu' or 'cuda'. Defaults to 'cpu'.
                     Use 'cuda' for GPU acceleration on NVIDIA GPUs.

    Returns:
        YOLO: Loaded model instance ready for inference.

    Raises:
        RuntimeError: If model fails to load due to file not found, 
                     network issues, or incompatible weights format.

    Example:
        >>> model = load_models(device='cpu')  # Load default model on CPU
        >>> model = load_models(model_path='/path/to/yolov8-face.pt', device='cuda')  # Custom model on GPU
    """
    try:
        if model_path:
            logger.info(f'Loading YOLO model from {model_path}')
            model = YOLO(model_path)
        else:
            logger.info(f'Loading default YOLO model (yolov8n.pt) for {device}')
            # yolov8n.pt is the nano model (~3.2MB), lightweight and fast
            # Ultralytics will auto-download if not cached locally
            model = YOLO('yolov8n.pt')
        
        # Move model to specified device (CPU or GPU)
        try:
            model.to(device)
            logger.info(f'Model loaded and set to device: {device}')
        except Exception as e:
            # If device setting fails, model will still work on default device
            logger.warning(f'Could not set device to {device}: {e}')
        
        return model
    except Exception as e:
        logger.error(f'Failed to load YOLO model: {e}', exc_info=True)
        raise RuntimeError(f"Failed to load YOLO model: {e}")
