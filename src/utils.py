"""Shared helper functions — drawing annotations, saving frames, etc."""

import os

import cv2
import numpy as np

from config import BOX_COLOR, BOX_THICKNESS, FONT_SCALE
from face_detector import BoundingBox
from emotion_classifier import EmotionResult


def draw_annotation(frame: np.ndarray, box: BoundingBox, emotion: EmotionResult = None) -> np.ndarray:
    """Draw a bounding box and optional emotion label onto the frame in place."""
    cv2.rectangle(frame, (box.x, box.y), (box.x + box.w, box.y + box.h), BOX_COLOR, BOX_THICKNESS)

    label = "unknown"
    if emotion is not None:
        label = f"{emotion.label} ({emotion.confidence:.0%})"

    text_y = max(box.y - 10, 15)
    cv2.putText(frame, label, (box.x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SCALE, BOX_COLOR, 2)
    return frame


def save_frame(frame: np.ndarray, output_dir: str, frame_id: int) -> str:
    """Save an annotated frame to disk, returning the path."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"frame_{frame_id:05d}.jpg")
    cv2.imwrite(path, frame)
    return path


def crop_face(frame: np.ndarray, box: BoundingBox) -> np.ndarray:
    """Return the cropped face region from a frame given a bounding box."""
    return frame[box.y:box.y + box.h, box.x:box.x + box.w]
