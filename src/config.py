"""Central configuration: thresholds, paths, and constants used across modules."""

import os

# Detection / classification thresholds
FACE_CONFIDENCE_THRESHOLD = 0.6
EMOTION_CONFIDENCE_THRESHOLD = 0.4

# Performance
FRAME_SKIP_DEFAULT = 1

# Emotion label set (order matches common FER2013-based model output)
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
HAAR_CASCADE_PATH = os.path.join(BASE_DIR, "models", "haarcascade_frontalface_default.xml")
ANNOTATED_SUBDIR = "annotated"
LOG_FILE_NAME = "run.log"
CSV_LOG_NAME = "detections_log.csv"
CHART_NAME = "emotion_summary.png"

# Drawing
BOX_COLOR = (0, 200, 0)  # BGR
BOX_THICKNESS = 2
FONT = "FONT_HERSHEY_SIMPLEX"
FONT_SCALE = 0.6
