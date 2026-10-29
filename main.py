"""
main.py
-------
Entry point for the Vehicle Detection & Parking Restriction System.

Usage:
    python main.py                          # uses settings from config.yaml
    python main.py --source 0               # override camera source (webcam)
    python main.py --source video.mp4       # override with a video file
    python main.py --allowed motorcycle     # override allowed vehicle type
    python main.py --model yolov8s.pt       # use a bigger/more accurate model

Keyboard controls (while the window is open):
    q   → quit
    s   → save a screenshot to screenshots/
    p   → pause / resume
    d   → toggle dashboard panel on/off
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import yaml

from detector import VehicleDetector
from parking_rules import ParkingRuleEngine
from ui_overlay import (
    draw_detections,
    draw_alert_banner,
    draw_dashboard,
    draw_timestamp,
)
from logger import DetectionLogger


# ── Config helpers ────────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Vehicle Detection & Parking Restriction System"
    )
    parser.add_argument("--source",  type=str, default=None,
                        help="Camera index, video file path, or RTSP URL")
    parser.add_argument("--allowed", type=str, default=None,
                        help="Override allowed vehicle type from config.yaml")
    parser.add_argument("--model",   type=str, default=None,
                        help="Override YOLO model path (e.g. yolov8s.pt)")
    parser.add_argument("--config",  type=str, default="config.yaml",
                        help="Path to config file (default: config.yaml)")
    return parser.parse_args()


def resolve_source(source_val):
    """Convert source string to int if it's a digit (camera index)."""
    if isinstance(source_val, int):
        return source_val
    s = str(source_val)
    return int(s) if s.isdigit() else s


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Load config
    cfg = load_config(args.config)

    # CLI overrides
    source          = resolve_source(args.source  or cfg["camera_source"])
    allowed_vehicle = (args.allowed or cfg["allowed_vehicle"]).lower().strip()
    model_path      = args.model    or cfg["model"]
    conf_thresh     = float(cfg["confidence_threshold"])
    iou_thresh      = float(cfg["iou_threshold"])
    show_dashboard  = bool(cfg["show_dashboard"])
    show_confidence = bool(cfg["show_confidence"])
    display_scale   = float(cfg["display_scale"])
    enable_logging  = bool(cfg["enable_logging"])
    log_file        = cfg["log_file"]
    alert_duration  = int(cfg["alert_duration_frames"])

    print("=" * 55)
    print("  Vehicle Detection & Parking Restriction System")
    print("=" * 55)
    print(f"  Source:          {source}")
    print(f"  Allowed vehicle: {allowed_vehicle.upper()}")
    print(f"  Model:           {model_path}")
    print(f"  Confidence:      {conf_thresh}")
    print("=" * 55)

    # Initialise components
    detector = VehicleDetector(model_path, conf_thresh, iou_thresh)
    rules    = ParkingRuleEngine(allowed_vehicle)
    logger   = DetectionLogger(log_file) if enable_logging else None

    # Open video source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        sys.exit(1)

    print("\n[System] Running — press 'q' to quit, 's' to screenshot, 'p' to pause.\n")

    # State
    frame_id        = 0
    alert_counter   = 0
    unauth_labels   = []
    paused          = False
    fps             = 0.0
    fps_timer       = time.time()
    fps_frame_count = 0

    screenshots_dir = Path("screenshots")
    screenshots_dir.mkdir(exist_ok=True)

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("[Info] End of stream or cannot read frame. Exiting.")
                break

            frame_id += 1

            # ── Scale frame if needed ─────────────────────────────────────────
            if display_scale != 1.0:
                h, w = frame.shape[:2]
                frame = cv2.resize(
                    frame,
                    (int(w * display_scale), int(h * display_scale)),
                    interpolation=cv2.INTER_AREA,
                )

            # ── Detection & rule evaluation ───────────────────────────────────
            raw_detections = detector.detect(frame)
            evaluated      = rules.evaluate_frame(raw_detections)

            # ── Alert state ───────────────────────────────────────────────────
            if rules.has_unauthorized(evaluated):
                alert_counter = alert_duration
                unauth_labels = [d["label"] for d in evaluated if not d["allowed"]]
            else:
                alert_counter = max(0, alert_counter - 1)

            # ── Logging ───────────────────────────────────────────────────────
            if logger and evaluated:
                logger.log_frame(frame_id, evaluated)

            # ── FPS calculation ───────────────────────────────────────────────
            fps_frame_count += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                fps = fps_frame_count / elapsed
                fps_frame_count = 0
                fps_timer = time.time()

        # ── Drawing ───────────────────────────────────────────────────────────
        draw_alert_banner(frame, alert_counter, alert_duration, unauth_labels)
        draw_detections(frame, evaluated if not paused else [], show_confidence)
        draw_dashboard(frame, fps, allowed_vehicle, evaluated if not paused else [],
                       show_dashboard)
        draw_timestamp(frame)

        # Pause indicator
        if paused:
            h, w = frame.shape[:2]
            cv2.putText(frame, "  ⏸  PAUSED — press P to resume",
                        (w // 2 - 200, h // 2),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 220, 255), 2, cv2.LINE_AA)

        cv2.imshow("Parking Restriction Monitor", frame)

        # ── Keyboard controls ─────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("[System] Quit.")
            break
        elif key == ord("p"):
            paused = not paused
            print(f"[System] {'Paused' if paused else 'Resumed'}.")
        elif key == ord("d"):
            show_dashboard = not show_dashboard
            print(f"[System] Dashboard {'on' if show_dashboard else 'off'}.")
        elif key == ord("s"):
            ts   = time.strftime("%Y%m%d_%H%M%S")
            path = screenshots_dir / f"screenshot_{ts}.jpg"
            cv2.imwrite(str(path), frame)
            print(f"[System] Screenshot saved: {path}")

    cap.release()
    cv2.destroyAllWindows()
    print("[System] Session ended.")


if __name__ == "__main__":
    main()
