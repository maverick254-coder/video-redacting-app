"""
Video redaction processor module.

This module handles the core redaction pipeline:
1. Read video frame-by-frame using OpenCV
2. Run YOLO model inference on each frame to detect objects
3. Extract bounding boxes for detected objects (filtered by class)
4. Apply redaction (blur, pixelate, or black box) to each box
5. Write redacted frames to output video file

The redaction methods are:
  - blur: Gaussian blur (smooth, natural looking)
  - pixelate: Downscale then upscale (blocky effect)
  - block: Solid black rectangle (complete obscuring)

Usage example:
  from processor import redact_video
  from models import load_models
  
  model = load_models(device='cpu')
  meta = redact_video(
      input_path='video.mp4',
      output_path='redacted.mp4',
      model=model,
      classes_to_redact=['face', 'person'],
      method='blur'
  )
  print(f"Processed {meta['frames']} frames, redacted {meta['boxes_redacted']} boxes")
"""

import cv2
import numpy as np
import logging
import os
from pathlib import Path
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


def _redact_region(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, method: str = 'blur') -> None:
    """Apply redaction to a rectangular region in a frame.
    
    This function modifies the frame in-place by applying one of three redaction methods
    to the specified bounding box region.

    Args:
        frame (np.ndarray): The video frame (BGR image array from OpenCV).
        x1, y1, x2, y2 (int): Bounding box coordinates (top-left and bottom-right).
        method (str): Redaction method:
                     - 'blur': Gaussian blur (smooth)
                     - 'pixelate': Pixelated effect (downscale + upscale)
                     - 'block': Solid black rectangle (most obscuring)

    Returns:
        None (modifies frame in-place)

    Note:
        - Bounding box is automatically clipped to frame boundaries
        - All coordinates are converted to int for indexing
    """
    # Calculate region dimensions, ensure at least 1 pixel
    h = max(1, y2 - y1)
    w = max(1, x2 - x1)
    
    # Extract the region of interest (ROI)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return  # Skip if region is empty
    
    if method == 'blur':
        # Apply Gaussian blur with kernel size proportional to region size
        # Kernel must be odd, so we use bitwise OR with 1 to ensure odd number
        k = max(3, (min(w, h) // 7) | 1)
        blurred = cv2.GaussianBlur(roi, (k, k), 0)
        frame[y1:y2, x1:x2] = blurred
    
    elif method == 'pixelate':
        # Create pixelated effect: downsample then upsample to create blocky appearance
        small_w = max(1, w // 16)  # Reduce to 1/16 width
        small_h = max(1, h // 16)  # Reduce to 1/16 height
        small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        # Upscale back to original size with nearest neighbor (creates blocky pixels)
        up = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        frame[y1:y2, x1:x2] = up
    
    else:  # 'block' - solid black rectangle
        # Draw filled rectangle with black color (0, 0, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), thickness=-1)


def _extract_boxes_from_results(results, class_names: Optional[List[str]] = None, min_confidence: float = 0.25) -> List[Tuple[int, int, int, int]]:
    """Extract and filter bounding boxes from YOLO inference results.
    
    YOLO returns detection results with bounding boxes and class predictions.
    This function extracts the boxes and optionally filters by class names.

    Args:
        results: YOLO inference results from model(frame).
        class_names (Optional[List[str]]): If provided, only return boxes for these classes.
                                          If None, return all detected boxes.
                                          Example: ['face', 'person']

    Returns:
        List[Tuple[int, int, int, int]]: List of bounding boxes as (x1, y1, x2, y2) tuples.
                                         Coordinates are integers suitable for indexing.

    Note:
        - Handles both CPU and GPU tensor outputs
        - Gracefully skips frames with detection errors
        - Empty results return empty list (no boxes)
    """
    boxes = []
    try:
        if not results or len(results) == 0:
            return boxes
        
        # YOLO results are returned as a list, access first frame's detections
        r = results[0]
        
        # Check if boxes were detected in this frame
        if hasattr(r, 'boxes') and r.boxes is not None:
            # Extract bounding box coordinates (xyxy format: x1, y1, x2, y2)
            xyxy = None
            try:
                # Try GPU tensor first (has .cpu() method)
                xyxy = r.boxes.xyxy.cpu().numpy()
            except Exception:
                try:
                    # Fall back to CPU tensor
                    xyxy = r.boxes.xyxy.numpy()
                except Exception:
                    pass
            
            # Extract class indices
            cls_idx = None
            try:
                cls_idx = r.boxes.cls.cpu().numpy()
            except Exception:
                try:
                    cls_idx = r.boxes.cls.numpy()
                except Exception:
                    cls_idx = None

            # Process each detected box and respect confidence threshold
            confs = None
            try:
                confs = r.boxes.conf.cpu().numpy()
            except Exception:
                try:
                    confs = r.boxes.conf.numpy()
                except Exception:
                    confs = None

            for i, b in enumerate(xyxy if xyxy is not None else []):
                x1, y1, x2, y2 = map(int, b[:4])

                # Respect confidence threshold if available
                if confs is not None:
                    try:
                        if float(confs[i]) < float(min_confidence):
                            continue
                    except Exception:
                        pass

                # Filter by class name if specified
                if cls_idx is not None and hasattr(r, 'names') and class_names:
                    c = int(cls_idx[i])
                    # Get class name from YOLO's class mapping
                    name = r.names.get(c, str(c)) if isinstance(r.names, dict) else r.names[c]
                    # Skip if class not in requested list
                    if name not in class_names:
                        continue

                boxes.append((x1, y1, x2, y2))
    
    except Exception as e:
        logger.warning(f'Error extracting boxes from results: {e}')
    
    return boxes


def redact_video(input_path: str | Path, output_path: str | Path, model, classes_to_redact: Optional[List[str]] = None, method: str = 'blur',
                 progress_callback=None, cancel_check=None) -> dict:
    """Redact objects in a video using YOLO model inference.
    
    This is the main redaction pipeline. It:
    1. Opens the input video and extracts metadata (resolution, fps, frame count)
    2. Creates an output video writer with same properties
    3. For each frame:
       - Runs YOLO inference to detect objects
       - Extracts boxes for specified classes
       - Applies redaction method to each box
       - Writes redacted frame to output
    4. Releases resources and returns processing metadata

    Args:
        input_path (str | Path): Path to input video file (MP4, AVI, MOV, etc).
        output_path (str | Path): Path to save redacted video (will be MP4).
        model: YOLO model instance from load_models().
        classes_to_redact (Optional[List[str]]): Classes to redact. If None, redacts all detections.
                                                Example: ['face', 'person', 'license_plate']
        method (str): Redaction method ('blur', 'pixelate', or 'block'). Defaults to 'blur'.

    Returns:
        dict: Processing metadata including:
              - width, height: Output video resolution
              - fps: Frames per second
              - frames: Total frames processed
              - boxes_redacted: Total bounding boxes redacted
              - output_path: Path to output video

    Raises:
        RuntimeError: If input video cannot be opened or output writer fails.

    Example:
        >>> model = load_models(device='cpu')
        >>> meta = redact_video('input.mp4', 'output.mp4', model, 
        ...                     classes_to_redact=['face'], method='blur')
        >>> print(f"Processed {meta['frames']} frames, redacted {meta['boxes_redacted']} boxes")
    """
    input_path = str(input_path)
    output_path = str(output_path)
    logger.info(f'Starting video redaction: {input_path} -> {output_path}, method={method}, classes={classes_to_redact}')
    
    # Check file existence and try to open with OpenCV
    if not os.path.isfile(input_path):
        raise RuntimeError(f'Input video file does not exist: {input_path}')
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f'Could not open input video (unsupported format or corrupted file): {input_path}')

    # Extract video metadata
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0  # Default to 25 fps if not detected
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)  # Total frame count
    logger.info(f'Video properties: {width}x{height}, {fps} fps, {total} frames')

    # Create video writer for output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4V codec
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f'Could not create video writer for {output_path}')
    
    frame_idx = 0
    boxes_found = 0

    try:
        # Process video frame by frame
        while True:
            # Support cancellation check from external caller
            if callable(cancel_check) and cancel_check():
                logger.info('Processing cancelled by request')
                break

            ret, frame = cap.read()
            if not ret:
                break  # End of video

            # Run YOLO inference on frame
            try:
                results = model(frame)
                boxes = _extract_boxes_from_results(results, class_names=classes_to_redact)
                boxes_found += len(boxes)
            except Exception as e:
                logger.warning(f'Error running inference on frame {frame_idx}: {e}')
                boxes = []  # Skip redaction for this frame if inference fails

            # If user asked to redact faces but YOLO didn't find any, use
            # a lightweight Haar-cascade face detector as a fallback. This
            # helps when the YOLO weights are not face-specialized.
            try:
                need_face = False
                if classes_to_redact:
                    # normalize to simple names
                    need_face = any(str(c).lower() == 'face' or str(c).lower() == 'faces' for c in classes_to_redact)
                if need_face and len(boxes) == 0:
                    try:
                        # Convert frame to grayscale for cascade
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                        face_cascade = cv2.CascadeClassifier(cascade_path)
                        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                        fb = []
                        for (x, y, w, h) in faces:
                            x1 = int(x)
                            y1 = int(y)
                            x2 = int(x + w)
                            y2 = int(y + h)
                            fb.append((x1, y1, x2, y2))
                        if fb:
                            logger.debug(f'Face fallback detected {len(fb)} faces on frame {frame_idx}')
                            boxes = fb
                            boxes_found += len(boxes)
                    except Exception as e:
                        logger.debug(f'Face cascade fallback failed: {e}')
            except Exception:
                pass

            # Apply redaction to all detected boxes
            for (x1, y1, x2, y2) in boxes:
                _redact_region(frame, x1, y1, x2, y2, method=method)

            # Write redacted frame to output video
            writer.write(frame)
            frame_idx += 1

            # Report progress via callback (if provided) every 5 frames
            if callable(progress_callback) and (frame_idx % 5 == 0 or frame_idx == total):
                try:
                    progress_callback(frame_idx, total)
                except Exception:
                    pass

            # Log progress every 30 frames for long videos
            if frame_idx % 30 == 0:
                logger.info(f'Processed {frame_idx}/{total} frames, {boxes_found} boxes redacted')
    
    finally:
        # Always release resources, even if error occurs
        cap.release()
        writer.release()
    
    logger.info(f'Video redaction complete: {frame_idx} frames, {boxes_found} boxes redacted')
    return {
        'width': width,
        'height': height,
        'fps': fps,
        'frames': frame_idx,
        'total_frames_reported': total,
        'boxes_redacted': boxes_found,
        'output_path': output_path,
        # If cancel_check signaled cancellation, mark cancelled
        'cancelled': bool(callable(cancel_check) and cancel_check()),
    }
