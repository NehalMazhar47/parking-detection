"""
detect_image.py
---------------
Dedicated Image Inspector & Parking Violation Analyzer.
Given an image, it detects all vehicles and clearly explains:
  ❌ WHAT IS WRONG (Unauthorized vehicles / violations)
  ✅ WHAT IS RIGHT (Permitted vehicles)

Usage:
    python detect_image.py
    python detect_image.py --image path/to/image.jpg
    python detect_image.py --image path/to/image.jpg --allowed motorcycle
"""

import argparse
import os
import sys
import cv2
import yaml
import numpy as np

from detector import VehicleDetector, VEHICLE_CLASSES
from parking_rules import ParkingRuleEngine
from ui_overlay import draw_detections, draw_alert_banner, draw_dashboard, draw_timestamp


def get_violation_explanation(label: str, allowed_vehicle: str) -> str:
    """Generate a clear, human-readable reason for why a vehicle is unauthorized."""
    reasons = {
        "motorcycle": f"Motorcycles are not permitted in {allowed_vehicle.upper()}-only zones.",
        "truck": f"Trucks/heavy vehicles are not permitted in {allowed_vehicle.upper()}-only zones.",
        "bus": f"Buses are not permitted in {allowed_vehicle.upper()}-only zones.",
        "car": f"Passenger cars are not permitted in {allowed_vehicle.upper()}-only zones.",
        "bicycle": f"Bicycles are not permitted in {allowed_vehicle.upper()}-only zones.",
    }
    return reasons.get(label.lower(), f"'{label.capitalize()}' does not match allowed vehicle type '{allowed_vehicle.upper()}'.")


def analyze_image(image_path: str, allowed_vehicle: str = "car",
                  model_path: str = "yolov8n.pt",
                  conf_thresh: float = 0.45,
                  iou_thresh: float = 0.45,
                  output_path: str = None) -> dict:
    """
    Run detection on an image and generate a complete 'What is Wrong vs Right' report.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Resize if very large for clean display
    h, w = frame.shape[:2]
    max_w = 1280
    if w > max_w:
        scale = max_w / w
        frame = cv2.resize(frame, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)

    detector = VehicleDetector(model_path, conf_thresh, iou_thresh)
    rules = ParkingRuleEngine(allowed_vehicle)

    raw_detections = detector.detect(frame)
    evaluated = rules.evaluate_frame(raw_detections)

    wrong_items = []
    right_items = []

    for i, det in enumerate(evaluated, start=1):
        label = det["label"]
        conf = det["confidence"]
        bbox = det["bbox"]

        if not det["allowed"]:
            wrong_items.append({
                "id": i,
                "label": label,
                "confidence": conf,
                "bbox": bbox,
                "reason": get_violation_explanation(label, allowed_vehicle),
                "action": "Immediate removal / parking citation required"
            })
        else:
            right_items.append({
                "id": i,
                "label": label,
                "confidence": conf,
                "bbox": bbox,
                "reason": f"Authorized vehicle ({label.capitalize()}) matches designated rule ({allowed_vehicle.upper()}).",
                "action": "Compliant — No action needed"
            })

    # Overlay annotations on frame
    annotated = frame.copy()
    alert_counter = 30 if len(wrong_items) > 0 else 0
    draw_alert_banner(annotated, alert_counter, 30, [item["label"] for item in wrong_items])
    draw_detections(annotated, evaluated, show_confidence=True)
    draw_dashboard(annotated, fps=0.0, allowed_vehicle=allowed_vehicle,
                   evaluated=evaluated, show_dashboard=True)
    draw_timestamp(annotated)

    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_analyzed{ext or '.jpg'}"

    cv2.imwrite(output_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])

    verdict = "VIOLATIONS_FOUND" if len(wrong_items) > 0 else ("ALL_COMPLIANT" if len(right_items) > 0 else "NO_VEHICLES_DETECTED")

    return {
        "verdict": verdict,
        "image_path": image_path,
        "output_path": output_path,
        "allowed_rule": allowed_vehicle,
        "total_vehicles": len(evaluated),
        "wrong_count": len(wrong_items),
        "right_count": len(right_items),
        "wrong_items": wrong_items,
        "right_items": right_items,
        "all_detections": evaluated,
    }


def print_report(result: dict):
    """Print an attractive, crystal-clear terminal report."""
    print("\n" + "=" * 64)
    print("       🚗 VEHICLE DETECTION & PARKING VIOLATION REPORT")
    print("=" * 64)
    print(f"  📷 Input Image:    {result['image_path']}")
    print(f"  🅿️  Designated Rule: Only {result['allowed_rule'].upper()} Allowed")
    print(f"  📊 Total Detected: {result['total_vehicles']} vehicle(s)")
    print("-" * 64)

    # ── VERDICT BANNER ──
    if result["verdict"] == "VIOLATIONS_FOUND":
        print(f"  🚨 OVERALL VERDICT: ❌ {result['wrong_count']} VIOLATION(S) DETECTED")
    elif result["verdict"] == "ALL_COMPLIANT":
        print("  🌟 OVERALL VERDICT: ✅ ALL CLEAR (100% COMPLIANT)")
    else:
        print("  ℹ️  OVERALL VERDICT: NO VEHICLES DETECTED IN IMAGE")

    print("=" * 64)

    # ── WHAT IS WRONG ──
    print("\n  ❌ WHAT IS WRONG (Violations):")
    if not result["wrong_items"]:
        print("     ✓ None! No unauthorized vehicles or parking violations.")
    else:
        for idx, item in enumerate(result["wrong_items"], 1):
            print(f"     [{idx}] 🚫 UNAUTHORIZED: {item['label'].upper()} (Confidence: {item['confidence']:.1%})")
            print(f"         • Reason: {item['reason']}")
            print(f"         • Location BBox: {item['bbox']}")
            print(f"         • Recommended Action: {item['action']}")

    # ── WHAT IS RIGHT ──
    print("\n  ✅ WHAT IS RIGHT (Compliant / Permitted):")
    if not result["right_items"]:
        print("     • No compliant vehicles detected.")
    else:
        for idx, item in enumerate(result["right_items"], 1):
            print(f"     [{idx}] 🟢 ALLOWED: {item['label'].upper()} (Confidence: {item['confidence']:.1%})")
            print(f"         • Status: {item['reason']}")
            print(f"         • Location BBox: {item['bbox']}")

    print("\n" + "-" * 64)
    print(f"  💾 Annotated Result Image Saved → {result['output_path']}")
    print("=" * 64 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Vehicle Detection & Violation Inspector")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to image file (default: demo_input.jpg)")
    parser.add_argument("--allowed", type=str, default=None,
                        help="Allowed vehicle type (car/motorcycle/bus/truck/bicycle)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save annotated image")
    args = parser.parse_args()

    cfg = yaml.safe_load(open("config.yaml"))
    allowed_vehicle = (args.allowed or cfg["allowed_vehicle"]).lower().strip()
    model_path = cfg["model"]
    conf_thresh = float(cfg["confidence_threshold"])
    iou_thresh = float(cfg["iou_threshold"])

    img_path = args.image
    if not img_path:
        # Check if demo_input.jpg exists
        if os.path.exists("demo_input.jpg"):
            img_path = "demo_input.jpg"
        else:
            print("Please specify an image path: python detect_image.py --image path/to/image.jpg")
            sys.exit(1)

    try:
        report = analyze_image(
            image_path=img_path,
            allowed_vehicle=allowed_vehicle,
            model_path=model_path,
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
            output_path=args.output
        )
        print_report(report)
    except Exception as e:
        print(f"[Error] Failed to analyze image: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
