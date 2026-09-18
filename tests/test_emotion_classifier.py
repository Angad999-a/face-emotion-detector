"""Unit tests for EmotionClassifier."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import EMOTION_LABELS
from emotion_classifier import EmotionClassifier


def test_classifier_handles_empty_crop_gracefully():
    classifier = EmotionClassifier()
    result = classifier.classify(None)
    assert result is None


def test_classifier_returns_valid_label_or_none():
    classifier = EmotionClassifier()
    dummy_face = np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8)
    result = classifier.classify(dummy_face)
    if result is not None:
        assert result.label in EMOTION_LABELS
        assert 0.0 <= result.confidence <= 1.0


# NOTE: For a meaningful test, add a real cropped face image with a known
# expression to data/samples/ and assert the predicted label matches.
