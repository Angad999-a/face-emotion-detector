"""CLI entry point for the Real-Time Face & Emotion Detection System.

Usage examples:
    python main.py --mode webcam
    python main.py --mode video --input data/samples/sample_video.mp4
    python main.py --mode image --input data/samples/
"""

import argparse
import os
import sys

import cv2

from analytics import AnalyticsEngine
from config import DEFAULT_OUTPUT_DIR, ANNOTATED_SUBDIR, FRAME_SKIP_DEFAULT
from emotion_classifier import EmotionClassifier
from face_detector import FaceDetector
from logger_config import setup_logger
from utils import draw_annotation, save_frame, crop_face
from video_stream import VideoStream

logger = setup_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Real-Time Face & Emotion Detection System")
    parser.add_argument("--mode", required=True, choices=["webcam", "video", "image"],
                         help="Input source type.")
    parser.add_argument("--input", default=None,
                         help="Path to video file or image folder (required for video/image mode).")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR,
                         help="Directory to write annotated frames, logs, and charts.")
    parser.add_argument("--frame-skip", type=int, default=FRAME_SKIP_DEFAULT,
                         help="Process every Nth frame (for performance on slower machines).")
    parser.add_argument("--no-display", action="store_true",
                         help="Run headless; do not open a live preview window.")
    return parser.parse_args()


def run(args):
    if args.mode in ("video", "image") and not args.input:
        logger.error("--input is required for mode=%s", args.mode)
        sys.exit(1)

    annotated_dir = os.path.join(args.output, ANNOTATED_SUBDIR)
    analytics = AnalyticsEngine(output_dir=args.output)

    logger.info("Initializing face detector and emotion classifier (pretrained models)...")
    face_detector = FaceDetector()
    emotion_classifier = EmotionClassifier()

    stream = VideoStream(mode=args.mode, source=args.input)

    try:
        for frame_id, frame in stream.frames():
            if args.frame_skip > 1 and frame_id % args.frame_skip != 0:
                continue

            faces = face_detector.detect(frame)
            if not faces:
                logger.debug("Frame %d: no faces detected.", frame_id)

            for face_id, box in enumerate(faces):
                face_crop = crop_face(frame, box)
                emotion = emotion_classifier.classify(face_crop)

                draw_annotation(frame, box, emotion)

                analytics.add_record(
                    frame_id=frame_id, face_id=face_id,
                    x=box.x, y=box.y, w=box.w, h=box.h,
                    emotion=emotion.label if emotion else "unclassified",
                    confidence=emotion.confidence if emotion else 0.0,
                )

            save_frame(frame, annotated_dir, frame_id)

            if not args.no_display:
                cv2.imshow("Face & Emotion Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Quit signal received; stopping.")
                    break
    finally:
        stream.release()
        cv2.destroyAllWindows()

    analytics.write_csv()
    analytics.generate_chart()
    analytics.print_summary()


if __name__ == "__main__":
    run(parse_args())
