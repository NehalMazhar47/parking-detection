"""
logger.py
---------
Detection event logger.
Writes each detection event to a CSV file for later analysis.
"""

import csv
import os
import time
from pathlib import Path


class DetectionLogger:
    """
    Appends detection events to a CSV file.

    CSV columns:
        timestamp, frame_id, vehicle_type, status, confidence
    """

    HEADERS = ["timestamp", "frame_id", "vehicle_type", "status", "confidence"]

    def __init__(self, log_file: str = "logs/detection_log.csv"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self._init_file()
        print(f"[Logger] Logging to: {self.log_file.resolve()}")

    def _init_file(self):
        """Create the CSV with headers if it doesn't already exist."""
        if not self.log_file.exists():
            with open(self.log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self.HEADERS)

    def log_frame(self, frame_id: int, evaluated: list[dict]):
        """
        Log all detections from a single frame.

        Args:
            frame_id:  Sequential frame number.
            evaluated: List of evaluated detection dicts from ParkingRuleEngine.
        """
        if not evaluated:
            return

        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        rows = []
        for det in evaluated:
            rows.append([
                ts,
                frame_id,
                det["label"],
                det["status"],
                f"{det['confidence']:.4f}",
            ])

        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
