"""Input handling for webcam, video file, or a folder of images.

Provides a single VideoStream abstraction so main.py doesn't need to
care which input mode it's dealing with.
"""

import glob
import os
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np

from logger_config import setup_logger

logger = setup_logger(__name__)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


class VideoStream:
    """Yields (frame_id, frame) tuples from webcam, video file, or image folder."""

    def __init__(self, mode: str, source: Optional[str] = None):
        """
        Args:
            mode: one of "webcam", "video", "image".
            source: path to video file or image folder. Ignored for webcam.
        """
        self.mode = mode
        self.source = source
        self.cap = None
        self.image_paths = []

        if mode == "webcam":
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise RuntimeError("Could not open webcam. Check camera permissions/connection.")
        elif mode == "video":
            if not source or not os.path.isfile(source):
                raise FileNotFoundError(f"Video file not found: {source}")
            self.cap = cv2.VideoCapture(source)
            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open video file: {source}")
        elif mode == "image":
            if not source or not os.path.isdir(source):
                raise NotADirectoryError(f"Image folder not found: {source}")
            self.image_paths = sorted([
                p for p in glob.glob(os.path.join(source, "*"))
                if p.lower().endswith(IMAGE_EXTENSIONS)
            ])
            if not self.image_paths:
                logger.warning("No images found in %s", source)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def frames(self) -> Iterator[Tuple[int, np.ndarray]]:
        """Yield (frame_id, frame) pairs from the configured source."""
        if self.mode in ("webcam", "video"):
            frame_id = 0
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                yield frame_id, frame
                frame_id += 1
        else:  # image mode
            for frame_id, path in enumerate(self.image_paths):
                frame = cv2.imread(path)
                if frame is None:
                    logger.warning("Could not read image: %s; skipping.", path)
                    continue
                yield frame_id, frame

    def release(self):
        if self.cap is not None:
            self.cap.release()
