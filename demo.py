w"""
demo.py
-------
Demonstration script — runs vehicle detection on a downloaded traffic image
and saves the annotated result as demo_result.jpg.

Usage:
    python demo.py
    python demo.py --allowed motorcycle
"""

import argparse
import urllib.request
import sys
import os
import cv2
import yaml
import numpy as np

from detector import VehicleDetector, VEHICLE_CLASSES
from parking_rules import ParkingRuleEngine
from ui_overlay import draw_detections, draw_alert_banner, draw_dashboard, draw_timestamp

# ── Sample traffic images (public domain / CC0) ───────────────────────────────
SAMPLE_IMAGES = [
    # Busy street with cars, motorcycles, buses
    "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Morning_Rush_Hour_Traffic_in_Dhaka.jpg/1280px-Morning_Rush_Hour_Traffic_in_Dhaka.jpg",
    # Fallback: mixed traffic
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png",
]


def download_image(url: str, dest: str) -> bool:
    """Download an image from a URL. Returns True on success."""
    try:
        print(f"[Demo] Downloading sample image...")
        headers = {"User-Agent": "Mozilla/5.0"}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
        with open(dest, "wb") as f:
            f.write(data)
        print(f"[Demo] Saved to: {dest}")
        return True
    except Exception as e:
        print(f"[Demo] Download failed: {e}")
        return False


def parse_args():
    parser = argparse.ArgumentParser(description="Vehicle Detection Demo")
    parser.add_argument("--allowed", type=str, default=None,
                        help="Override allowed vehicle type (car/motorcycle/bus/truck/bicycle)")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to a local image file to use instead of downloading")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    cfg = yaml.safe_load(open("config.yaml"))
    allowed_vehicle = (args.allowed or cfg["allowed_vehicle"]).lower().strip()
    model_path      = cfg["model"]
    conf_thresh     = float(cfg["confidence_threshold"])
    iou_thresh      = float(cfg["iou_threshold"])

    print("=" * 55)
    print("  Vehicle Detection Demo")
    print("=" * 55)
    print(f"  Allowed vehicle: {allowed_vehicle.upper()}")
    print(f"  Model:           {model_path}")
    print("=" * 55)

    # ── Get image ─────────────────────────────────────────────────────────────
    if args.image and os.path.exists(args.image):
        img_path = args.image
        print(f"[Demo] Using local image: {img_path}")
    else:
        img_path = "demo_input.jpg"
        success = False
        for url in SAMPLE_IMAGES:
            if download_image(url, img_path):
                success = True
                break
        if not success:
            print("[Demo] Could not download sample image.")
            sys.exit(1)

    # ── Load frame ───────────────────────────────────────────────────────────
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"[Demo] Failed to read image: {img_path}")
        sys.exit(1)

    # Resize to a reasonable display size
    h, w = frame.shape[:2]
    max_w = 1280
    if w > max_w:
        scale = max_w / w
        frame = cv2.resize(frame, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)
    h, w = frame.shape[:2]
    print(f"[Demo] Image size: {w}×{h}")

    # ── Detection ─────────────────────────────────────────────────────────────
    detector = VehicleDetector(model_path, conf_thresh, iou_thresh)
    rules    = ParkingRuleEngine(allowed_vehicle)

    print("\n[Demo] Running detection...")
    raw_detections = detector.detect(frame)
    evaluated      = rules.evaluate_frame(raw_detections)

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\n[Demo] Found {len(evaluated)} vehicle(s):\n")
    allowed_count   = 0
    unauth_count    = 0
    unauth_labels   = []

    for det in evaluated:
        status_icon = "✅ ALLOWED     " if det["allowed"] else "🚫 UNAUTHORIZED"
        print(f"  {status_icon}  {det['label'].upper():12s}  conf={det['confidence']:.2%}")
        if det["allowed"]:
            allowed_count += 1
        else:
            unauth_count += 1
            unauth_labels.append(det["label"])

    print(f"\n  Total: {len(evaluated)}  |  Allowed: {allowed_count}  |  Unauthorized: {unauth_count}")

    # ── Draw overlays ─────────────────────────────────────────────────────────
    alert_counter = 30 if unauth_count > 0 else 0

    draw_alert_banner(frame, alert_counter, 30, unauth_labels)
    draw_detections(frame, evaluated, show_confidence=True)
    draw_dashboard(frame, fps=0.0, allowed_vehicle=allowed_vehicle,
                   evaluated=evaluated, show_dashboard=True)
    draw_timestamp(frame)

    # ── Save result ───────────────────────────────────────────────────────────
    out_path = "demo_result.jpg"
    cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"\n[Demo] Annotated result saved → {out_path}")
    print("[Demo] Done! ✓")


if __name__ == "__main__":
    main()
