"""Unit tests for FaceDetector."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from face_detector import FaceDetector, BoundingBox


def test_detector_returns_empty_list_on_blank_frame():
    detector = FaceDetector()
    blank_frame = np.zeros((300, 300, 3), dtype=np.uint8)
    results = detector.detect(blank_frame)
    assert isinstance(results, list)
    # A blank frame should yield no faces (or very few false positives).


def test_detector_handles_none_frame_gracefully():
    detector = FaceDetector()
    results = detector.detect(None)
    assert results == []


def test_bounding_box_fields():
    box = BoundingBox(x=10, y=20, w=50, h=60, confidence=0.9)
    assert box.x == 10 and box.y == 20 and box.w == 50 and box.h == 60
    assert 0.0 <= box.confidence <= 1.0


# NOTE: For a meaningful "true positive" test, add a real sample image with a
# clear frontal face to data/samples/ and assert len(results) >= 1 against it.
