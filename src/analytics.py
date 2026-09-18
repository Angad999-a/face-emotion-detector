"""Module 3: Analytics & Reporting.

Aggregates per-face detection records across a session and produces:
- a CSV log of every detection
- a bar chart of emotion frequency
- a console text summary
"""

import csv
import os
from collections import Counter
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List

import matplotlib
matplotlib.use("Agg")  # headless-safe backend for CLI/server use
import matplotlib.pyplot as plt

from config import CSV_LOG_NAME, CHART_NAME
from logger_config import setup_logger

logger = setup_logger(__name__)


@dataclass
class DetectionRecord:
    timestamp: str
    frame_id: int
    face_id: int
    x: int
    y: int
    w: int
    h: int
    emotion: str
    confidence: float


class AnalyticsEngine:
    """Collects detection records for a session and generates reports."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.records: List[DetectionRecord] = []

    def add_record(self, frame_id: int, face_id: int, x: int, y: int, w: int, h: int,
                    emotion: str, confidence: float):
        record = DetectionRecord(
            timestamp=datetime.now().isoformat(timespec="seconds"),
            frame_id=frame_id, face_id=face_id,
            x=x, y=y, w=w, h=h,
            emotion=emotion, confidence=round(confidence, 4),
        )
        self.records.append(record)

    def write_csv(self) -> str:
        path = os.path.join(self.output_dir, CSV_LOG_NAME)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(self.records[0]).keys())
                                     if self.records else
                                     ["timestamp", "frame_id", "face_id", "x", "y", "w", "h",
                                      "emotion", "confidence"])
            writer.writeheader()
            for r in self.records:
                writer.writerow(asdict(r))
        logger.info("Wrote %d detection records to %s", len(self.records), path)
        return path

    def generate_chart(self) -> str:
        counts = Counter(r.emotion for r in self.records)
        path = os.path.join(self.output_dir, CHART_NAME)

        if not counts:
            logger.warning("No records to chart; skipping chart generation.")
            return ""

        labels, values = zip(*sorted(counts.items(), key=lambda kv: -kv[1]))
        plt.figure(figsize=(8, 5))
        plt.bar(labels, values, color="#4C72B0")
        plt.title("Emotion Distribution — Session Summary")
        plt.xlabel("Emotion")
        plt.ylabel("Number of Detections")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        logger.info("Saved emotion summary chart to %s", path)
        return path

    def print_summary(self):
        total = len(self.records)
        if total == 0:
            print("Session summary: no faces detected.")
            return

        counts = Counter(r.emotion for r in self.records)
        print(f"Session summary: {total} faces analyzed")
        parts = [f"{label}: {count / total:.0%}" for label, count in
                  sorted(counts.items(), key=lambda kv: -kv[1])]
        print(", ".join(parts))
