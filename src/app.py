"""
FastAPI web service for video redaction using YOLO object detection.

This module provides:
  - Web UI form for submitting redaction jobs (GET /)
  - REST API endpoints for video upload and processing (POST /redact-video)
  - Job status querying and artifact download (GET /jobs/*)
  - Both synchronous and asynchronous (background) processing modes

Typical flow:
  1. User opens browser to http://localhost:8000
  2. Selects video file, redaction method, categories, and processing mode
  3. Submits form via POST /redact-video
  4. For sync: receives MP4 file download immediately
  5. For async: receives job_id + status_url, polls GET /jobs/{job_id} for status
  6. When done, downloads via /jobs/{job_id}/download, previews, and thumbnails

Key features:
  - CATEGORY_MAP: Friendly category names → YOLO class names
  - Support for blur, pixelate, and block redaction methods
  - CPU or CUDA device selection
  - Background job tracking with in-memory storage
  - Thumbnail and preview generation after processing
  - Comprehensive HTTP status codes and error handling
"""

import os
import shutil
import tempfile
import logging
import os
import shutil
import tempfile
import logging
import time
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse


from src.models import load_models
from src.processor import redact_video
from src.jobs import create_job, get_job

logger = logging.getLogger("video_redaction_app")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

APP_DIR = Path(__file__).resolve().parent

app = FastAPI(title='Video Redaction Service')


# Mapping from user-friendly category names to YOLO class names
# This allows the form to show human-readable options while correctly
# filtering the model's detected classes for redaction.
# Note: Class names may vary by YOLO model version; adjust as needed.
CATEGORY_MAP = {
    'faces': ['face'],  # Face detection (requires face-enabled YOLO model)
    'heads': ['person'],  # Map heads to person class
    'people': ['person'],  # Person/human detection
    'license_plates': ['license_plate', 'license plate'],  # Vehicle license plates
    'vehicles': ['car', 'truck', 'bus', 'motorbike', 'bicycle', 'motorcycle'],  # Various vehicle types
    'screens': ['tv', 'tvmonitor', 'laptop', 'monitor', 'screen'],  # Electronic screens/displays
    'logos': ['logo'],  # Brand logos (requires logo-trained model)
    'signatures': ['signature'],  # Handwritten signatures (requires signature-trained model)
}


@app.on_event('startup')
def startup_event():
    """
    FastAPI startup event: pre-load YOLO model on default device.
    
    Configurable via environment variables:
      YOLO_MODEL_PATH: Path to custom YOLO weights file (defaults to yolov8n.pt)
      YOLO_DEFAULT_DEVICE: Device for startup model (cpu or cuda, defaults to cpu)
    
    If model loading fails, warning is logged but server still starts.
    """
    model_path = os.environ.get('YOLO_MODEL_PATH')
    # Optionally pre-load default device model for faster first request
    try:
        default_device = os.environ.get('YOLO_DEFAULT_DEVICE', 'cpu')
        app.state.model = load_models(model_path=model_path, device=default_device)
    except Exception as e:
        app.state.model = None
        print(f"Warning: failed to load model at startup: {e}")


@app.get('/', response_class=HTMLResponse)
def index():
    """
    Serve the web form interface for video redaction.
    
    Returns HTML with:
      - Video file upload input
      - Category checkboxes (faces, vehicles, etc.)
      - Redaction method selector (blur, pixelate, block)
      - Device selector (CPU or CUDA)
      - Background/sync processing toggle
    """
    tpl = APP_DIR.parent / 'templates' / 'index.html'
    if tpl.exists():
        return HTMLResponse(tpl.read_text(encoding='utf-8'))
    # Fallback if template not found
    return HTMLResponse('<html><body><h3>Upload endpoint: POST /redact-video</h3></body></html>')


def _cleanup_files(*paths: str):
    """
    Helper to clean up temporary files and directories.
    
    Used in background tasks to remove uploaded files and job artifacts
    after sending response to client (e.g., after FileResponse for sync mode).
    
    Args:
        *paths: Variable number of file or directory paths to remove
        
    Note: Silently ignores errors if files don't exist or can't be deleted.
    """
    for p in paths:
        try:
            if os.path.isfile(p):
                os.remove(p)
            elif os.path.isdir(p):
                shutil.rmtree(p)
        except Exception:
            pass


def _expand_categories(categories: Optional[List[str]]) -> Optional[List[str]]:
    """
    Expand user-friendly category names to YOLO class names.
    
    Args:
        categories: List of category names from form (e.g., ['faces', 'vehicles'])
        
    Returns:
        List of YOLO class names to filter detections by, or None if empty/None input.
        
    Example:
        Input: ['faces', 'people']
        Output: ['face', 'person']  (no duplicates)
        
    Note: Unknown category names are passed through as-is (for custom models).
    """
    if not categories or len(categories) == 0:
        return None
    classes = []
    for c in categories:
        if not c:
            continue
        # Normalize to lowercase for consistent lookup
        key = c.strip().lower()
        # Look up in CATEGORY_MAP, or use raw key if not found
        mapped = CATEGORY_MAP.get(key)
        if mapped:
            classes.extend(mapped)
        else:
            classes.append(key)
    # Remove duplicates while preserving order
    return list(dict.fromkeys(classes)) if classes else None


@app.post('/redact-video')
async def redact_video_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    method: str = Form('blur'),
    categories: Optional[List[str]] = Form(None),
    background: bool = Form(True),
    device: str = Form('cpu'),
):
    """
    Upload video and redact specified categories.
    
    Accepts multipart form data with:
      - file: Video file to process
      - method: Redaction method (blur, pixelate, or block)
      - categories: List of category names to redact (empty = all objects)
      - background: True for async job, False for synchronous
      - device: Processing device (cpu or cuda)
    
    Returns:
      - Sync mode (background=False): FileResponse with MP4 file (200)
      - Async mode (background=True): JSON with job_id and status_url (200)
      - Errors: HTTPException with 400, 404, or 500 status code
      
    Form validation:
      - File required (400 if missing)
      - Method must be blur/pixelate/block (400 if invalid)
      - Device must be cpu/cuda (400 if invalid)
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail='No file uploaded')
    if method not in ('blur', 'pixelate', 'block'):
        raise HTTPException(status_code=400, detail=f'Invalid redaction method: {method}')
    if device not in ('cpu', 'cuda'):
        raise HTTPException(status_code=400, detail=f'Invalid device: {device}')

    # Create temporary directory for this upload
    tmpdir = tempfile.mkdtemp(prefix='redact_')
    logger.info(f'Received upload: {file.filename}, background={background}, device={device}, categories={categories}')
    try:
        # Save uploaded file to temporary directory
        upload_path = Path(tmpdir) / file.filename
        with open(upload_path, 'wb') as f:
            shutil.copyfileobj(file.file, f)
        logger.info(f'Saved upload to {upload_path}')

        # Expand friendly category names to YOLO class names
        class_filter = _expand_categories(categories)
        logger.info(f'Expanded categories to: {class_filter}')

        if background:
            # Background processing: return immediately with job_id
            job_id = create_job(str(upload_path), method=method, classes=class_filter, device=device, tmpdir=tmpdir)
            logger.info(f'Created background job: {job_id}')
            # NOTE: Do NOT schedule cleanup here for background jobs ---
            # the background worker needs access to the uploaded file while
            # it runs. Cleanup of temporary directories for background jobs
            # should be handled by the worker or a later maintenance task.
            return JSONResponse({'job_id': job_id, 'status_url': f'/jobs/{job_id}'})

        # Synchronous processing: block until done, return file
        model = None
        try:
            # Try to reuse pre-loaded model if device matches
            if getattr(app.state, 'model', None) is not None and device == os.environ.get('YOLO_DEFAULT_DEVICE', 'cpu'):
                model = app.state.model
                logger.info('Using pre-loaded model')
        except Exception:
            model = None

        # Load model if not reusing pre-loaded one
        if model is None:
            logger.info(f'Loading model for device: {device}')
            model = load_models(device=device)

        # Run redaction pipeline on input video
        output_path = Path(tmpdir) / 'output_redacted.mp4'
        logger.info(f'Starting synchronous redaction')
        meta = redact_video(str(upload_path), str(output_path), model, classes_to_redact=class_filter, method=method)
        logger.info(f'Redaction complete: {meta}')

        # Schedule cleanup of temporary directory after response sent
        background_tasks.add_task(_cleanup_files, tmpdir)
        # Send redacted video file to client
        return FileResponse(path=str(output_path), filename='output_redacted.mp4', media_type='video/mp4')
    except Exception as e:
        logger.error(f'Error during redaction: {e}', exc_info=True)
        # Clean up on error
        _cleanup_files(tmpdir)
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/jobs/{job_id}')
def job_status(job_id: str):
    """
    Get the status of a background redaction job.
    
    Args:
        job_id: UUID of the job (returned from POST /redact-video)
        
    Returns:
        JSON object with:
          - id, status, error: Job tracking info
          - method, device: Processing parameters used
          - created_at, started_at, finished_at: Timestamps
          - elapsed_seconds: Processing duration (once running)
          - download_url: URL to fetch redacted video (only when status='done')
          - thumbnails: List of thumbnail image URLs (only when done)
          - preview_url: URL to fetch preview video clip (only when done)
          
    Status values:
      - queued: Waiting to run
      - running: Currently processing
      - done: Completed successfully (output available)
      - failed: Encountered error (see error field)
      
    HTTP codes:
      - 200: Job found and info returned
      - 404: Job ID not found
    """
    info = get_job(job_id)
    if not info:
        raise HTTPException(status_code=404, detail='Job not found')
    
    # Calculate elapsed time if job has started
    elapsed = None
    if info.get('started_at'):
        end = info.get('finished_at') or time.time()
        elapsed = end - info['started_at']
    
    # Build response with job metadata
    resp = {
        'id': info['id'],
        'status': info['status'],
        'error': info.get('error'),
        'method': info.get('method'),
        'device': info.get('device'),
        'created_at': info.get('created_at'),
        'started_at': info.get('started_at'),
        'finished_at': info.get('finished_at'),
        'elapsed_seconds': elapsed,
    }
    # Always include progress-related fields so clients can show live updates
    resp['progress'] = info.get('progress', 0)
    resp['frames_processed'] = info.get('frames_processed', 0)
    resp['total_frames'] = info.get('total_frames')
    resp['cancel_requested'] = bool(info.get('cancel_requested'))
    
    # Add download/preview URLs when job is complete
    if info.get('status') == 'done' and info.get('output_path'):
        resp['download_url'] = f"/jobs/{job_id}/download"
        # Add thumbnail URLs if available
        if info.get('thumbnails'):
            resp['thumbnails'] = [f"/jobs/{job_id}/thumbnail/{i}" for i in range(len(info.get('thumbnails', [])))]
        # Add preview URL if available
        if info.get('preview'):
            resp['preview_url'] = f"/jobs/{job_id}/preview"
    return JSONResponse(resp)



@app.post('/jobs/{job_id}/cancel')
def cancel_job(job_id: str):
    """
    Request cancellation for a background job.

    Sets the job's `cancel_requested` flag which the worker checks periodically.
    Returns 200 if the cancel request was accepted, 400 if the job is already finished,
    and 404 if the job id is unknown.
    """
    info = get_job(job_id)
    if not info:
        raise HTTPException(status_code=404, detail='Job not found')

    # If job already finished, cannot cancel
    if info.get('status') in ('done', 'failed', 'cancelled'):
        raise HTTPException(status_code=400, detail=f"Job already in terminal state: {info.get('status')}")

    # Mark cancel_requested -- worker will observe this and stop as soon as possible
    info['cancel_requested'] = True
    logger.info(f"Cancel requested for job {job_id}")
    return JSONResponse({'id': job_id, 'cancel_requested': True, 'status': info.get('status')})


@app.get('/jobs/{job_id}/download')
def job_download(job_id: str):
    """
    Download the redacted video output from a completed job.
    
    Args:
        job_id: UUID of the job
        
    Returns:
        FileResponse with MP4 video file (200)
        
    Errors:
      - 404: Job not found
      - 400: Job not complete yet (status != 'done')
      - 410: Output file no longer available (deleted after processing)
    """
    info = get_job(job_id)
    if not info:
        raise HTTPException(status_code=404, detail='Job not found')
    # Only allow download when job is complete
    if info.get('status') != 'done':
        raise HTTPException(status_code=400, detail=f"Job status is '{info.get('status')}', not 'done'")
    # Verify output file still exists
    if not info.get('output_path') or not Path(info['output_path']).exists():
        raise HTTPException(status_code=410, detail='Output file no longer available')
    logger.info(f'Downloading job {job_id}')
    return FileResponse(path=str(info['output_path']), filename='output_redacted.mp4', media_type='video/mp4')


@app.get('/jobs/{job_id}/thumbnail/{index}')
def job_thumbnail(job_id: str, index: int):
    """
    Download a thumbnail image from a completed job.
    
    Args:
        job_id: UUID of the job
        index: Thumbnail index (0, 1, 2 for the 3 generated thumbnails)
        
    Returns:
        FileResponse with JPEG image (200)
        
    Errors:
      - 404: Job not found, or thumbnail index out of range
      - 410: Thumbnail file no longer available (deleted after processing)
    """
    info = get_job(job_id)
    if not info:
        raise HTTPException(status_code=404, detail='Job not found')
    # Get list of thumbnail paths from job info
    thumbs = info.get('thumbnails') or []
    if index < 0 or index >= len(thumbs):
        raise HTTPException(status_code=404, detail=f'Thumbnail {index} not found')
    # Verify thumbnail file still exists
    thumb_path = str(thumbs[index])
    if not Path(thumb_path).exists():
        raise HTTPException(status_code=410, detail='Thumbnail no longer available')
    return FileResponse(path=thumb_path, filename=Path(thumb_path).name, media_type='image/jpeg')


@app.get('/jobs/{job_id}/preview')
def job_preview(job_id: str):
    """
    Download a short preview video clip from a completed job.
    
    Preview is the first 3 seconds of the redacted video at reduced fps
    to minimize file size while showing the redaction in action.
    
    Args:
        job_id: UUID of the job
        
    Returns:
        FileResponse with MP4 video file (200)
        
    Errors:
      - 404: Job not found, or preview not available
      - 410: Preview file no longer available (deleted after processing)
    """
    info = get_job(job_id)
    if not info:
        raise HTTPException(status_code=404, detail='Job not found')
    # Get preview path from job info
    preview = info.get('preview')
    if not preview:
        raise HTTPException(status_code=404, detail='Preview not available or still processing')
    # Verify preview file still exists
    preview_path = str(preview)
    if not Path(preview_path).exists():
        raise HTTPException(status_code=410, detail='Preview no longer available')
    return FileResponse(path=preview_path, filename='preview.mp4', media_type='video/mp4')


@app.get('/api/categories')
def list_categories():
    """
    Return list of available redaction categories.
    
    Returns:
        JSON object with 'categories' key containing list of friendly category names.
        
    Example response:
        {"categories": ["faces", "people", "vehicles", "license_plates", ...]}
    """
    return JSONResponse({'categories': list(CATEGORY_MAP.keys())})


if __name__ == '__main__':
    import uvicorn
    logger.info('Starting Video Redaction Service on http://0.0.0.0:8000')
    uvicorn.run('src.app:app', host='0.0.0.0', port=8000, reload=False)

