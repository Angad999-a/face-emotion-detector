"""Module 2: Emotion Classification.

Wraps a pretrained emotion recognition model (via the `fer` package, which
uses a CNN trained on FER2013) to classify a cropped face image into one
of the standard emotion categories.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from config import EMOTION_CONFIDENCE_THRESHOLD, EMOTION_LABELS
from logger_config import setup_logger

logger = setup_logger(__name__)


@dataclass
class EmotionResult:
    label: str
    confidence: float


class EmotionClassifier:
    """Classifies emotion on a cropped face image using a pretrained CNN.

    Uses the `fer` package's FER model (pretrained on FER2013) under the hood.
    Swap `_predict` for `deepface.DeepFace.analyze` if you prefer that library —
    the public interface (`classify`) stays the same either way.
    """

    def __init__(self, confidence_threshold: float = EMOTION_CONFIDENCE_THRESHOLD):
        self.confidence_threshold = confidence_threshold
        self._model = None
        self._load_model()

    def _load_model(self):
        try:
            from fer import FER
            self._model = FER(mtcnn=False)
            logger.info("Loaded pretrained FER emotion model.")
        except ImportError as e:
            logger.error("Could not import 'fer' package: %s. Install via requirements.txt.", e)
            raise

    def classify(self, face_crop: np.ndarray) -> Optional[EmotionResult]:
        """Classify the dominant emotion in a cropped face image.

        Args:
            face_crop: cropped BGR face image as a numpy array.

        Returns:
            EmotionResult with the top label and confidence, or None if the
            face crop was invalid or confidence fell below threshold.
        """
        if face_crop is None or face_crop.size == 0:
            logger.warning("Empty face crop passed to classifier; skipping.")
            return None

        try:
            predictions = self._model.detect_emotions(face_crop)
        except Exception as e:
            logger.error("Emotion classification failed: %s", e)
            return None

        if not predictions:
            return None

        emotions = predictions[0]["emotions"]
        top_label = max(emotions, key=emotions.get)
        top_confidence = emotions[top_label]

        if top_confidence < self.confidence_threshold:
            return None

        return EmotionResult(label=top_label, confidence=top_confidence)
