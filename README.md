# Real-Time Face & Emotion Detection System

## Overview
A command-line computer vision tool that detects faces in a webcam feed, video file, or folder of images, classifies the emotion expressed on each detected face using a pretrained deep learning model, and produces analytics — annotated output, a CSV log of every detection, and a summary chart of emotion distribution.

Built for the Computer Vision course project (VITyarthi flipped-course evaluation).

## Features
- Face detection on live webcam feed, video files, or static images
- Emotion classification into 7 categories: angry, disgust, fear, happy, sad, surprise, neutral
- Bounding-box + emotion-label annotation drawn on output frames/video
- Per-frame CSV logging (timestamp, face id, bounding box, predicted emotion, confidence)
- End-of-session summary report: emotion frequency bar chart (matplotlib) + text summary
- Configurable detection/classification confidence thresholds via `src/config.py`
- Structured logging to `outputs/run.log`
- Graceful handling of frames with no detected faces, corrupted files, or camera errors

## Technologies / Tools Used
| Purpose | Library |
|---|---|
| Face detection | OpenCV DNN (`res10_300x300_ssd`) |
| Emotion classification | `fer` / `deepface` (pretrained on FER2013) |
| Video/image I/O | OpenCV (`cv2`) |
| Charts | `matplotlib` |
| CLI | `argparse` |
| Logging | Python `logging` |
| Testing | `pytest` |

## Project Structure
```
face-emotion-detector/
├── src/
│   ├── main.py                # CLI entry point
│   ├── face_detector.py       # Module 1: face detection
│   ├── emotion_classifier.py  # Module 2: emotion classification
│   ├── analytics.py           # Module 3: aggregation & reporting
│   ├── video_stream.py        # webcam/video/image input handling
│   ├── logger_config.py       # logging setup
│   ├── config.py              # thresholds, paths, constants
│   └── utils.py               # drawing boxes, shared helpers
├── tests/
│   ├── test_face_detector.py
│   └── test_emotion_classifier.py
├── data/samples/               # put sample test images/videos here
├── outputs/                    # generated logs, annotated frames, charts
├── requirements.txt
├── README.md
└── statement.md
```

## Prerequisites
- Python 3.9–3.11
- A webcam (only needed for `--mode webcam`)
- ~2 GB free disk space for pretrained model weights (downloaded automatically on first run by `fer`/`deepface`)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/face-emotion-detector.git
   cd face-emotion-detector
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate        # macOS/Linux
   venv\Scripts\activate           # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration
Open `src/config.py` and adjust if needed:
- `FACE_CONFIDENCE_THRESHOLD` — minimum confidence to accept a face detection (default `0.6`)
- `EMOTION_CONFIDENCE_THRESHOLD` — minimum confidence to label an emotion (default `0.4`)
- `OUTPUT_DIR` — where annotated output, logs, and charts are saved (default `outputs/`)
- `FRAME_SKIP` — process every Nth frame for performance on slower machines (default `1`)

## Running the Project

All usage is via the command line — no GUI required.

**Live webcam:**
```bash
python src/main.py --mode webcam
```

**Video file:**
```bash
python src/main.py --mode video --input data/samples/sample_video.mp4
```

**Folder of images:**
```bash
python src/main.py --mode image --input data/samples/
```

Optional flags:
```bash
python src/main.py --mode video --input data/samples/sample_video.mp4 \
    --output outputs/ --frame-skip 2 --no-display
```
| Flag | Description |
|---|---|
| `--mode` | `webcam`, `video`, or `image` (required) |
| `--input` | path to video file or image folder (required for `video`/`image`) |
| `--output` | output directory (default: `outputs/`) |
| `--frame-skip` | process every Nth frame (default: `1`) |
| `--no-display` | run headless, do not open a preview window (useful on servers) |

Press `q` to quit a live/video session early (if a display window is open).

## Output
After a run, check the `outputs/` folder for:
- `annotated/` — frames/video with bounding boxes + emotion labels drawn
- `detections_log.csv` — one row per detected face: `timestamp, frame_id, face_id, x, y, w, h, emotion, confidence`
- `emotion_summary.png` — bar chart of emotion frequency for the session
- `run.log` — structured run log (info/warnings/errors)
- Console prints a final text summary, e.g.:
  ```
  Session summary: 342 faces analyzed
  happy: 41%, neutral: 33%, sad: 12%, surprise: 9%, angry: 5%
  ```

## Testing
Run the test suite:
```bash
pytest tests/ -v
```
Tests validate: face detector returns expected bounding-box format on a known sample image; emotion classifier returns a valid label from the defined emotion set; no-face-detected frames are handled without crashing.

## Screenshots
_(Add screenshots of annotated output and the summary chart here after running the project.)_

## Future Enhancements
- Multi-face tracking across frames (avoid re-identifying the same face as new each frame)
- Fine-tune the emotion model on a domain-specific dataset
- REST API wrapper for remote inference
- Export session report as PDF
