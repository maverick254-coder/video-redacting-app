# Video Redaction Tool

A production-ready FastAPI application for automated video redaction using YOLO object detection. Detects and redacts sensitive objects (faces, people, vehicles, license plates, etc.) from video files with multiple redaction methods (blur, pixelate, or block).

## Features

- **YOLO Object Detection**: Uses Ultralytics YOLOv8 for real-time object detection per-frame
- **Multiple Redaction Methods**: 
  - Blur (Gaussian blur with 31px kernel)
  - Pixelate (16x downscaling then upscaling)
  - Block (solid black rectangle)
- **Flexible Category Selection**: Redact specific object types via web form checkboxes
- **Sync & Async Processing**:
  - Synchronous: Returns redacted MP4 immediately
  - Asynchronous: Queue job, poll status, download when done
- **GPU Support**: CPU or CUDA device selection for inference
- **Preview & Thumbnails**: Auto-generated preview clips and thumbnail images
- **Comprehensive Logging**: INFO-level logging with timestamps for all operations
- **RESTful API**: REST endpoints for upload, status, download, and metadata
- **Web Interface**: Simple HTML form for video upload and configuration

## Installation

### Prerequisites

- Python 3.8+
- (Optional) NVIDIA GPU with CUDA support for faster processing

### Setup Steps

**On Windows (PowerShell):**

```powershell
# Clone or download the repository
cd "m:\video reduction tool"

# Create virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

**On Linux/Mac (Bash):**

```bash
cd video-redaction-tool
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Running the Application

### Quick Start

```powershell
uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
```

Then open your browser to: **http://localhost:8000**

### Configuration via Environment Variables

```powershell
# Use a custom YOLO model
$env:YOLO_MODEL_PATH = "C:\path\to\custom_model.pt"

# Pre-load a specific device on startup
$env:YOLO_DEFAULT_DEVICE = "cuda"  # or "cpu"

# Then start the app
uvicorn src.app:app --host 0.0.0.0 --port 8000
```

## How the Redaction Process Works

### Overall Pipeline

1. **Video Input**: User uploads MP4 file via web form
2. **Frame Extraction**: OpenCV reads video frame-by-frame, preserving original FPS and resolution
3. **YOLO Inference**: For each frame, YOLOv8 detects objects and returns bounding boxes
4. **Category Filtering**: Only boxes matching selected redaction categories are processed
5. **Redaction Application**: Selected method (blur/pixelate/block) applied to each matching region
6. **Video Output**: Processed frames written to new MP4 file with mp4v codec
7. **Artifacts**: Thumbnails and preview clip generated from output video

### Frame-by-Frame Redaction

- Each video frame is processed independently
- YOLO inference runs on each frame (GPU or CPU)
- All detected objects of selected types are redacted before writing to output
- No frame interpolation or temporal smoothing—each redaction is per-frame

### Redaction Methods

| Method | How It Works | Use Case |
|--------|-------------|----------|
| **Blur** | Applies Gaussian blur (31px kernel) to region | Smooth, natural-looking redaction |
| **Pixelate** | Downscales region 16x then upscales back | Mosaic/pixelated look |
| **Block** | Fills region with solid black rectangle | Maximum opacity |

### Processing Modes

#### Synchronous (Blocking)

- Form submission with `background = false`
- Server processes immediately, client waits
- Returns MP4 file when complete
- Suitable for short videos or when instant feedback needed

#### Asynchronous (Background Job)

- Form submission with `background = true` (default)
- Server returns immediately with `job_id`
- Client polls `GET /jobs/{job_id}` for status
- When `status = 'done'`, download via provided URLs
- Suitable for long videos and non-blocking UX

### Job Lifecycle

```
submitted (background=true)
    ↓
create_job() → returns job_id
    ↓
worker thread starts: status = 'queued'
    ↓
load YOLO model: status = 'running'
    ↓
redact_video() runs frame-by-frame
    ↓
generate thumbnails (3 JPEG images)
    ↓
generate preview (first 3 seconds at ≤10 fps)
    ↓
status = 'done' (or 'failed' on error)
    ↓
client downloads via /jobs/{job_id}/download
client views preview via /jobs/{job_id}/preview
client views thumbnails via /jobs/{job_id}/thumbnail/{0,1,2}
```

## Web Interface

### Form Fields

- **Video Upload**: Select MP4 file from disk
- **Categories to Redact**: Checkboxes for selecting object types
  - Faces
  - Heads
  - People (persons)
  - License Plates
  - Vehicles (cars, trucks, buses, etc.)
  - Screens (TV, monitor, laptop displays)
  - Logos (brand logos)
  - Signatures
- **Redaction Method**: Dropdown (blur, pixelate, block)
- **Processing Device**: Radio buttons (CPU or CUDA/GPU)
- **Background Processing**: Checkbox for async mode (default: enabled)

### Response Handling

**Sync Mode (background unchecked):**
- Browser automatically downloads `output_redacted.mp4`

**Async Mode (background checked):**
- Page shows `job_id` and `status_url`
- Copy status URL to check job progress
- Navigate to `/jobs/{job_id}` for live status updates with download/preview links

## API Endpoints

### Web Interface

**GET /**
- Returns HTML form for video upload
- Includes category checkboxes, method/device selectors, background toggle

### Upload & Processing

**POST /redact-video**
- Accepts: multipart form data
  - `file`: Video file (required)
  - `method`: blur | pixelate | block (default: blur)
  - `categories`: List of category names (optional, default: all)
  - `background`: true | false (default: true)
  - `device`: cpu | cuda (default: cpu)
- Returns (sync): FileResponse with MP4 (status 200)
- Returns (async): JSON `{"job_id": "...", "status_url": "/jobs/..."}` (status 200)
- Errors: 400 (bad request), 500 (server error)

### Job Status & Downloads

**GET /jobs/{job_id}**
- Returns: JSON with job status, timing, and artifact URLs
- Status values: `queued`, `running`, `done`, `failed`
- When done, includes: `download_url`, `thumbnails[]`, `preview_url`
- Errors: 404 (job not found)

**GET /jobs/{job_id}/download**
- Returns: FileResponse with redacted MP4 video
- Errors: 404 (job not found), 400 (not done yet), 410 (file deleted)

**GET /jobs/{job_id}/thumbnail/{index}**
- Returns: FileResponse with JPEG thumbnail (index 0, 1, or 2)
- Errors: 404 (job/thumbnail not found), 410 (file deleted)

**GET /jobs/{job_id}/preview**
- Returns: FileResponse with MP4 preview clip (first ~3 seconds)
- Errors: 404 (job/preview not found), 410 (file deleted)

### Metadata

**GET /api/categories**
- Returns: JSON `{"categories": ["faces", "people", ...]}`
- Lists available redaction category options

## Project Structure

```
video reduction tool/
├── src/
│   ├── app.py              # FastAPI application and endpoints
│   ├── models.py           # YOLO model loading with device management
│   ├── processor.py        # Core redaction pipeline (frame loop, inference, redaction)
│   └── jobs.py             # Background job manager with threading
├── templates/
│   └── index.html          # Web form UI with category checkboxes and controls
├── requirements.txt        # Python package dependencies
└── README.md              # This file
```

### Module Documentation

#### src/models.py
- `load_models(model_path, device)`: Load YOLO model on specified device
- Supports custom model paths via `YOLO_MODEL_PATH` env var
- Handles CUDA fallback if GPU unavailable
- Caches model in memory

#### src/processor.py
- `redact_video(input_path, output_path, model, classes_to_redact, method)`: Main redaction pipeline
- Reads video frames, runs inference, applies redaction, writes output
- Returns metadata: frames processed, boxes redacted, output path
- Supports blur, pixelate, and block redaction methods
- Logs progress every 30 frames

#### src/jobs.py
- `create_job(input_path, method, classes, device)`: Queue background redaction job
- `get_job(job_id)`: Retrieve job status and artifact paths
- Daemon threads for non-blocking background processing
- Auto-generates thumbnails and preview after completion
- In-memory job storage (not persistent across restarts)

#### src/app.py
- FastAPI application with all REST endpoints
- Form handler with multipart file upload
- Sync and async processing modes
- Category name mapping (friendly names → YOLO classes)
- Comprehensive error handling and validation
- Logging of all requests and operations

## Example Usage

### Using the Web Interface

1. Open http://localhost:8000 in your browser
2. Click "Choose File" and select a video
3. Check "People" and "License Plates" boxes
4. Select "Blur" method
5. Leave "CPU" and "Background Processing" enabled
6. Click "Upload & Redact"
7. Page shows job ID; bookmark the status URL
8. Poll the status URL until `status = 'done'`
9. Click "Download Redacted Video" to download output

### Using curl (Sync Mode)

```bash
curl -X POST http://localhost:8000/redact-video \
  -F "file=@input.mp4" \
  -F "method=blur" \
  -F "categories=people" \
  -F "categories=license_plates" \
  -F "background=false" \
  -F "device=cpu" \
  -o output_redacted.mp4
```

### Using curl (Async Mode)

```bash
# Submit job
curl -X POST http://localhost:8000/redact-video \
  -F "file=@input.mp4" \
  -F "method=blur" \
  -F "categories=people" \
  -F "background=true" \
  -F "device=cuda"

# Response: {"job_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx", "status_url": "/jobs/..."}

# Check status
curl http://localhost:8000/jobs/xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx

# Download when done
curl http://localhost:8000/jobs/xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx/download -o output.mp4
```

## Performance Notes

- **GPU (CUDA)**: YOLOv8 nano on GPU ~30-50 fps depending on resolution
- **CPU**: YOLOv8 nano on CPU ~5-15 fps depending on hardware
- **First Run**: Model weights downloaded (~7 MB for nano) on first inference
- **Video I/O**: OpenCV VideoWriter is CPU-bound; encode time depends on video codec and resolution
- **Memory**: In-memory job storage; jobs kept until server restart (consider cleanup for production)

## Limitations & Future Improvements

- **Job Persistence**: Jobs stored in memory; lost on server restart. Production should use database (SQLite, PostgreSQL).
- **File Cleanup**: Output videos and artifacts not auto-deleted; consider scheduled cleanup.
- **Authentication**: No user authentication; open access. Production should add OAuth/JWT.
- **Rate Limiting**: No rate limiting; consider implementing to prevent abuse.
- **Custom Models**: Only default YOLOv8 classes supported without code changes; user-uploaded models not yet supported.
- **Resume Failed Jobs**: Failed jobs cannot be retried; would require job queueing system.

## Troubleshooting

### Import Error: No module named 'ultralytics'
- Ensure virtual environment is activated
- Run `pip install -r requirements.txt`

### CUDA Error: No such file or directory
- Check GPU drivers and CUDA toolkit installation
- Verify `device='cuda'` in form or set `YOLO_DEFAULT_DEVICE=cpu`

### Video Output Not Generated
- Check server logs for error messages
- Verify input video is readable MP4 format
- Check disk space in temp directory

### Model Download Fails
- Ensure internet connection; YOLO auto-downloads model on first run
- Or pre-download model and set `YOLO_MODEL_PATH` env var

### Slow Processing
- Consider using GPU (CUDA) for faster inference
- Use smaller YOLO model (nano vs small/medium)
- Reduce video resolution (use ffmpeg to downscale)

## License

[Your License Here]

## Contributing

[Contribution guidelines here]

