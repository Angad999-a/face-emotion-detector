"""Module 1: Face Detection.

Wraps a pretrained OpenCV DNN face detector. Given a frame, returns a list
of bounding boxes for every face found above the confidence threshold.
"""

import os
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

from config import FACE_CONFIDENCE_THRESHOLD, HAAR_CASCADE_PATH
from logger_config import setup_logger

logger = setup_logger(__name__)


@dataclass
class BoundingBox:
    x: int
    y: int
    w: int
    h: int
    confidence: float


class FaceDetector:
    """Detects faces in a frame using OpenCV's pretrained DNN SSD face detector.

    NOTE: Requires the model files (deploy.prototxt + res10_300x300_ssd_iter_140000.caffemodel).
    Download once and place under a `models/` folder, or swap this implementation
    for `cv2.CascadeClassifier` (Haar cascade, ships with OpenCV, no download needed)
    if you want a zero-setup alternative — see `_load_haar_fallback`.
    """

    def __init__(self, confidence_threshold: float = FACE_CONFIDENCE_THRESHOLD,
                 prototxt_path: str = None, model_path: str = None):
        self.confidence_threshold = confidence_threshold
        self.net = None
        self.use_haar = False

        if prototxt_path and model_path:
            try:
                self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
                logger.info("Loaded DNN face detector from %s", model_path)
            except cv2.error as e:
                logger.warning("Failed to load DNN model (%s); falling back to Haar cascade.", e)
                self._load_haar_fallback()
        else:
            logger.info("No DNN model paths provided; using Haar cascade fallback.")
            self._load_haar_fallback()

    def _load_haar_fallback(self):
        # Prefer the cascade bundled in this repo (models/) over cv2.data.haarcascades,
        # since some opencv-python installs (especially when opencv-python-headless
        # is also present) ship without the packaged data files, which throws a
        # "Can't open file" error from cv2.FileStorage.
        cascade_path = HAAR_CASCADE_PATH
        if not os.path.isfile(cascade_path):
            fallback_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            logger.warning("Bundled cascade not found at %s; trying cv2.data path: %s",
                            cascade_path, fallback_path)
            cascade_path = fallback_path

        self.haar_cascade = cv2.CascadeClassifier(cascade_path)
        if self.haar_cascade.empty():
            raise RuntimeError(
                f"Could not load Haar cascade from '{cascade_path}'. "
                "Ensure models/haarcascade_frontalface_default.xml exists in the repo."
            )
        self.use_haar = True

    def detect(self, frame: np.ndarray) -> List[BoundingBox]:
        """Detect faces in a single BGR frame.

        Args:
            frame: image as a numpy array (H, W, 3) in BGR order.

        Returns:
            List of BoundingBox objects for faces above the confidence threshold.
        """
        if frame is None or frame.size == 0:
            logger.warning("Received empty frame; skipping detection.")
            return []

        if self.use_haar:
            return self._detect_haar(frame)
        return self._detect_dnn(frame)

    def _detect_haar(self, frame: np.ndarray) -> List[BoundingBox]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return [BoundingBox(x=int(x), y=int(y), w=int(w), h=int(h), confidence=1.0)
                for (x, y, w, h) in faces]

    def _detect_dnn(self, frame: np.ndarray) -> List[BoundingBox]:
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                      (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()

        boxes = []
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < self.confidence_threshold:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (start_x, start_y, end_x, end_y) = box.astype("int")
            start_x, start_y = max(0, start_x), max(0, start_y)
            end_x, end_y = min(w, end_x), min(h, end_y)
            boxes.append(BoundingBox(x=int(start_x), y=int(start_y),
                                      w=int(end_x - start_x), h=int(end_y - start_y),
                                      confidence=confidence))
        return boxes
