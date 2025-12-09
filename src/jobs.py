"""
Background job manager for asynchronous video redaction.

This module manages long-running redaction jobs in background threads.
It provides:
  - Job queuing and execution tracking
  - Status queries while jobs run
  - Thumbnail and preview generation after completion
  - In-memory job storage (note: not persistent across restarts)

Job lifecycle:
  1. create_job() -> job_id returned to client immediately
  2. Job queued in _JOBS dictionary
  3. Worker thread starts asynchronously
  4. Job status: 'queued' -> 'running' -> 'done' or 'failed'
  5. Client polls GET /jobs/{job_id} to check status
  6. When done, client downloads output via /jobs/{job_id}/download
  7. Thumbnails and preview available at /jobs/{job_id}/thumbnail/{i} and /preview

Typical usage:
  from jobs import create_job, get_job
  
  job_id = create_job(input_path, method='blur', classes=['face'], device='cpu')
  
  # Poll status
  info = get_job(job_id)
  if info['status'] == 'done':
      output = info['output_path']
      thumbnails = info.get('thumbnails', [])
"""

import threading
import time
import logging
from typing import Dict, Optional
from uuid import uuid4

from pathlib import Path
import shutil
import os

import cv2

from src.models import load_models
from src.processor import redact_video

logger = logging.getLogger(__name__)


# Global dictionary storing all job information
# Key: job_id (UUID string)
# Value: dict with status, paths, timing, and results
_JOBS: Dict[str, Dict] = {}


def create_job(input_path: str, method: str = 'blur', classes: list | None = None, device: str = 'cpu', tmpdir: Optional[str] = None) -> str:
    """
    Create and queue a new background redaction job.
    
    Args:
        input_path: Path to input video file
        method: Redaction method ('blur', 'pixelate', or 'block'). Default: 'blur'
        classes: List of YOLO class names to redact. Default: None (all classes)
        device: Processing device ('cpu' or 'cuda'). Default: 'cpu'
    
    Returns:
        job_id: UUID string identifying this job
        
    Side effects:
        - Stores job info in _JOBS dictionary
        - Starts background worker thread (daemon)
        - Worker will process video, generate thumbnails, and create preview
        
    Example:
        job_id = create_job('video.mp4', method='blur', classes=['face'], device='cpu')
        # Returns immediately, processing happens in background
    """
    job_id = str(uuid4())
    # Initialize job state with all tracking metadata
    info = {
        'id': job_id,
        'status': 'queued',
        'input_path': input_path,
        'tmpdir': tmpdir,
        'output_path': None,  # Set when processing completes
        'method': method,
        'classes': classes,
        'device': device,
        'error': None,  # Set if processing fails
        'created_at': time.time(),  # Job creation timestamp
        'started_at': None,  # Set when worker starts
        'finished_at': None,  # Set when worker completes
        'thumbnails': [],  # List of thumbnail image paths
        'preview': None,  # Path to preview video clip
        'progress': 0,  # 0-100 integer percent
        'frames_processed': 0,
        'total_frames': None,
        'cancel_requested': False,
    }
    _JOBS[job_id] = info

    def _worker():
        """
        Worker function running in background thread.
        
        Orchestrates the complete redaction pipeline:
          1. Load YOLO model on specified device
          2. Run redact_video() to process the input video
          3. Generate thumbnail images from output video
          4. Create a preview video clip (first 3 seconds)
          5. Update job status and store artifact paths
          
        On failure, catches exception and stores error message in job info.
        """
        info['status'] = 'running'
        info['started_at'] = time.time()
        logger.info(f'Job {job_id} started: method={method}, device={device}, classes={classes}')
        try:
            # Load YOLO model on the specified device (CPU or CUDA)
            logger.info(f'Job {job_id}: loading model for {device}')
            model = load_models(device=device)
            
            # Prepare output path in same directory as input
            out_path = str(Path(input_path).parent / f"output_{job_id}.mp4")
            logger.info(f'Job {job_id}: starting redaction to {out_path}')
            
            # Run the main redaction pipeline
            # Define progress callback and cancel check for the worker
            def _progress_cb(processed, total):
                try:
                    info['frames_processed'] = int(processed)
                    info['total_frames'] = int(total) if total else None
                    if total and total > 0:
                        info['progress'] = int(processed * 100 / total)
                    else:
                        info['progress'] = 0
                except Exception:
                    pass

            def _cancel_check():
                return bool(info.get('cancel_requested'))

            meta = redact_video(input_path, out_path, model, classes_to_redact=classes, method=method, progress_callback=_progress_cb, cancel_check=_cancel_check)
            # ensure latest progress updated
            _progress_cb(meta.get('frames', 0), meta.get('total_frames_reported', 0))
            if meta.get('cancelled'):
                info['status'] = 'cancelled'
                info['finished_at'] = time.time()
                info['output_path'] = meta.get('output_path')
                logger.info(f'Job {job_id} cancelled after processing {info.get("frames_processed")} frames')
            else:
                info['output_path'] = meta.get('output_path', out_path)
                info['status'] = 'done'
                info['finished_at'] = time.time()
                elapsed = info['finished_at'] - info['started_at']
                logger.info(f'Job {job_id} completed successfully in {elapsed:.2f}s')
        except Exception as e:
            # Mark job as failed and store error message
            info['status'] = 'failed'
            info['error'] = str(e)
            info['finished_at'] = time.time()
            logger.error(f'Job {job_id} failed: {e}', exc_info=True)

        # Generate thumbnails and a short preview clip if output exists and job completed
        if info['status'] == 'done':
            try:
                out = info.get('output_path')
                if out and Path(out).exists():
                    logger.info(f'Job {job_id}: generating thumbnails and preview')
                    
                    # Open output video to extract frames
                    cap = cv2.VideoCapture(out)
                    fps = cap.get(cv2.CAP_PROP_FPS) or 25
                    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                    
                    # Generate 3 evenly-spaced thumbnail frames
                    thumbs = []
                    sample_count = min(3, max(1, total))
                    indices = [int(i * total / (sample_count + 1)) for i in range(1, sample_count + 1)]
                    
                    for idx_i, fi in enumerate(indices):
                        # Seek to frame index and read
                        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                        ret, frm = cap.read()
                        if not ret:
                            continue
                        
                        # Resize to 320px width (maintaining aspect ratio) and save as JPEG
                        thumb_path = str(Path(out).parent / f"thumb_{job_id}_{idx_i}.jpg")
                        small = cv2.resize(frm, (320, int(frm.shape[0] * 320 / frm.shape[1])), interpolation=cv2.INTER_AREA)
                        cv2.imwrite(thumb_path, small)
                        thumbs.append(thumb_path)
                    
                    info['thumbnails'] = thumbs
                    logger.info(f'Job {job_id}: generated {len(thumbs)} thumbnails')

                    # Generate preview video: first 3 seconds at max 10 fps
                    try:
                        preview_path = str(Path(out).parent / f"preview_{job_id}.mp4")
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        
                        # Seek back to start and get video dimensions
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        
                        # Use lower fps for preview to reduce file size
                        preview_fps = min(10, int(fps) if fps >= 1 else 10)
                        writer = cv2.VideoWriter(preview_path, fourcc, preview_fps, (w, h))
                        
                        # Write frames for 3 seconds
                        max_frames = int(preview_fps * 3)
                        written = 0
                        while written < max_frames:
                            ret, frm = cap.read()
                            if not ret:
                                break
                            writer.write(frm)
                            written += 1
                        
                        writer.release()
                        info['preview'] = preview_path
                        logger.info(f'Job {job_id}: preview generated ({written} frames)')
                    except Exception as e:
                        logger.warning(f'Job {job_id}: failed to generate preview: {e}')
                        info['preview'] = None

                    cap.release()
                    # Move artifacts (output, thumbnails, preview) to a persistent
                    # artifacts directory outside the upload tmpdir so they remain
                    # available after we clean up the tmpdir.
                    try:
                        base_artifacts = Path(__file__).resolve().parent.parent / 'artifacts' / job_id
                        base_artifacts.mkdir(parents=True, exist_ok=True)

                        # Move output video if exists
                        out_path = Path(out)
                        if out_path.exists():
                            new_out = base_artifacts / out_path.name
                            shutil.move(str(out_path), str(new_out))
                            info['output_path'] = str(new_out)

                        # Move thumbnails
                        new_thumbs = []
                        for t in info.get('thumbnails', []):
                            tp = Path(t)
                            if tp.exists():
                                nt = base_artifacts / tp.name
                                shutil.move(str(tp), str(nt))
                                new_thumbs.append(str(nt))
                        if new_thumbs:
                            info['thumbnails'] = new_thumbs

                        # Move preview
                        pv = info.get('preview')
                        if pv:
                            pvp = Path(pv)
                            if pvp.exists():
                                new_pv = base_artifacts / pvp.name
                                shutil.move(str(pvp), str(new_pv))
                                info['preview'] = str(new_pv)

                        logger.info(f'Job {job_id}: moved artifacts to {base_artifacts}')
                    except Exception as e:
                        logger.warning(f'Job {job_id}: failed to move artifacts out of tmpdir: {e}')
            except Exception as e:
                logger.warning(f'Job {job_id}: failed to generate thumbnails: {e}')

        # Clean up temporary upload directory if provided. This ensures we don't
        # leave user uploads lying around after processing completes (or fails).
        try:
            td = info.get('tmpdir')
            if td:
                # Only remove if path exists and appears to be a directory
                if os.path.isdir(td):
                    shutil.rmtree(td)
                    logger.info(f'Job {job_id}: cleaned up temporary dir {td}')
        except Exception as e:
            logger.warning(f'Job {job_id}: failed to cleanup tmpdir {td}: {e}')

    # Start worker thread as daemon (won't block shutdown)
    thread = threading.Thread(target=_worker, name=f"job-{job_id}", daemon=True)
    thread.start()

    return job_id


def get_job(job_id: str) -> Optional[Dict]:
    """
    Retrieve job information by ID.
    
    Args:
        job_id: UUID string identifying the job
        
    Returns:
        Job info dict (see create_job() for structure) or None if job not found.
        
    Job status values:
        'queued' - waiting to run
        'running' - currently processing
        'done' - completed successfully (output_path set)
        'failed' - encountered error (error field set)
    """
    return _JOBS.get(job_id)

