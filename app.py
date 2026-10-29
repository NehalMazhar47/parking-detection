"""
app.py
------
Flask web server for the Vehicle Detection & Parking Restriction System.

Endpoints:
  GET  /               → Dashboard HTML page
  GET  /video_feed     → MJPEG live stream with detection overlays
  GET  /api/stats      → JSON live stats (FPS, counts, alert status)
  POST /api/set_rule   → Change allowed vehicle type on the fly
  GET  /api/logs       → Last 50 detection events as JSON

Run:
    python app.py
Then open http://localhost:5000 in your browser.
"""

import threading
import time
import csv
import os

# Fix macOS webcam authorization in background threads
os.environ.setdefault("OPENCV_AVFOUNDATION_SKIP_AUTH", "1")
from collections import defaultdict
from pathlib import Path

import cv2
import yaml
from flask import Flask, Response, jsonify, render_template, request

from detector import VehicleDetector
from parking_rules import ParkingRuleEngine
from ui_overlay import draw_detections, draw_alert_banner, draw_dashboard, draw_timestamp
from logger import DetectionLogger

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

# ── Shared state (protected by lock) ─────────────────────────────────────────
lock          = threading.Lock()
latest_frame  = None          # JPEG bytes of the most-recent annotated frame
camera_enabled = True         # Live camera stream active flag
stats         = {
    "fps":             0.0,
    "allowed_vehicle": "car",
    "alert":           False,
    "camera_enabled":  True,
    "counts":          {},    # {label: {total, allowed, blocked}}
    "total_detections": 0,
}


# ── Config & components ───────────────────────────────────────────────────────
cfg             = yaml.safe_load(open("config.yaml"))
allowed_vehicle = cfg["allowed_vehicle"].lower().strip()
rules           = ParkingRuleEngine(allowed_vehicle)
logger          = DetectionLogger(cfg["log_file"]) if cfg["enable_logging"] else None
_detector       = None

def get_detector():
    global _detector
    if _detector is None:
        _detector = VehicleDetector(cfg["model"],
                                    float(cfg["confidence_threshold"]),
                                    float(cfg["iou_threshold"]))
    return _detector


# ── Background capture thread ─────────────────────────────────────────────────
capture_thread_started = False

def capture_loop():
    global latest_frame, stats, rules, allowed_vehicle, camera_enabled

    source = cfg.get("camera_source", 0)
    if str(source).isdigit():
        source = int(source)

    cap = None
    use_fallback_image = False

    try:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[WebApp] Camera source {source} unavailable. Using demo stream mode.")
            use_fallback_image = True
    except Exception as e:
        print(f"[WebApp] Camera error: {e}. Using demo stream mode.")
        use_fallback_image = True

    frame_id       = 0
    alert_counter  = 0
    alert_duration = int(cfg["alert_duration_frames"])
    fps_timer      = time.time()
    fps_count      = 0
    fps            = 0.0

    while True:
        with lock:
            is_active = camera_enabled

        if not is_active:
            time.sleep(0.2)
            continue

        if use_fallback_image or cap is None or not cap.isOpened():
            if os.path.exists("demo_input.jpg"):
                frame = cv2.imread("demo_input.jpg")
            else:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
            time.sleep(0.1)
        else:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                time.sleep(0.05)
                continue

        frame_id += 1

        # Resize for web display
        h, w = frame.shape[:2]
        if w > 960:
            scale = 960 / w
            frame = cv2.resize(frame, (960, int(h * scale)))

        # Detection (use current rules snapshot)
        with lock:
            current_rules = rules

        raw  = get_detector().detect(frame)
        evald = current_rules.evaluate_frame(raw)

        # Alert state
        if any(not d["allowed"] for d in evald):
            alert_counter = alert_duration
            unauth_labels = [d["label"] for d in evald if not d["allowed"]]
        else:
            alert_counter = max(0, alert_counter - 1)
            unauth_labels = []

        # Logging
        if logger and evald:
            logger.log_frame(frame_id, evald)

        # FPS
        fps_count += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            fps = fps_count / elapsed
            fps_count = 0
            fps_timer = time.time()

        # Draw overlays
        draw_alert_banner(frame, alert_counter, alert_duration, unauth_labels)
        draw_detections(frame, evald, show_confidence=True)
        draw_dashboard(frame, fps, allowed_vehicle, evald, show_dashboard=True)
        draw_timestamp(frame)

        # Encode to JPEG
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        jpg_bytes = buf.tobytes()

        # Build counts dict
        counts = defaultdict(lambda: {"total": 0, "allowed": 0, "blocked": 0})
        for d in evald:
            lbl = d["label"]
            counts[lbl]["total"] += 1
            if d["allowed"]:
                counts[lbl]["allowed"] += 1
            else:
                counts[lbl]["blocked"] += 1

        with lock:
            latest_frame = jpg_bytes
            stats["fps"]             = round(fps, 1)
            stats["alert"]           = alert_counter > 0
            stats["camera_enabled"]  = True
            stats["counts"]          = dict(counts)
            stats["allowed_vehicle"] = allowed_vehicle
            stats["total_detections"] += len(evald)


def ensure_capture_started():
    global capture_thread_started
    if not capture_thread_started:
        capture_thread_started = True
        t = threading.Thread(target=capture_loop, daemon=True)
        t.start()


# ── MJPEG generator ───────────────────────────────────────────────────────────
def generate_frames():
    ensure_capture_started()
    while True:
        with lock:
            is_active = camera_enabled
            frame = latest_frame

        if not is_active or frame is None:
            time.sleep(0.08)
            continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(1 / 30)   # cap at 30 FPS to browser


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html",
                           allowed_vehicle=stats["allowed_vehicle"])


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/stats")
def api_stats():
    with lock:
        return jsonify(stats)


@app.route("/api/camera/toggle", methods=["POST"])
def api_camera_toggle():
    global camera_enabled
    data = request.get_json(force=True) if request.is_json else {}
    with lock:
        if "enabled" in data:
            camera_enabled = bool(data["enabled"])
        else:
            camera_enabled = not camera_enabled
        stats["camera_enabled"] = camera_enabled
        status_str = "ON" if camera_enabled else "OFF"
    print(f"[WebApp] Camera toggled → {status_str}")
    return jsonify({"ok": True, "camera_enabled": camera_enabled})


@app.route("/api/set_rule", methods=["POST"])
def api_set_rule():
    global rules, allowed_vehicle
    data = request.get_json(force=True)
    vehicle = data.get("allowed_vehicle", "car").lower().strip()
    valid = {"car", "motorcycle", "bus", "truck", "bicycle"}
    if vehicle not in valid:
        return jsonify({"error": f"Invalid vehicle type. Choose from: {valid}"}), 400
    with lock:
        allowed_vehicle = vehicle
        rules = ParkingRuleEngine(vehicle)
    print(f"[WebApp] Rule changed → allowed: {vehicle}")
    return jsonify({"ok": True, "allowed_vehicle": vehicle})


@app.route("/api/logs")
def api_logs():
    log_path = Path(cfg["log_file"])
    if not log_path.exists():
        return jsonify([])
    rows = []
    with open(log_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return jsonify(rows[-50:][::-1])   # last 50, newest first


import base64
import numpy as np


def get_violation_reason(label: str, allowed: str) -> str:
    reasons = {
        "motorcycle": f"Motorcycles are prohibited in {allowed.upper()}-only parking bays.",
        "truck": f"Commercial trucks/heavy vehicles are not permitted in {allowed.upper()}-only areas.",
        "bus": f"Buses are not permitted in {allowed.upper()}-only parking bays.",
        "car": f"Standard passenger cars are not permitted in {allowed.upper()}-only spots.",
        "bicycle": f"Bicycles cannot occupy {allowed.upper()}-only parking spots.",
    }
    return reasons.get(label.lower(), f"Vehicle type '{label.capitalize()}' violates the '{allowed.upper()}' rule.")


@app.route("/api/detect_image", methods=["POST"])
def api_detect_image():
    """
    Analyzes an uploaded image, runs YOLOv8 detection, checks parking rules,
    and returns a full breakdown of WHAT IS WRONG vs WHAT IS RIGHT.
    """
    rule_vehicle = allowed_vehicle
    img_array = None
    conf_val = float(cfg.get("confidence_threshold", 0.25))

    # Check if multipart form upload
    if "image" in request.files:
        file = request.files["image"]
        rule_vehicle = request.form.get("allowed_vehicle", allowed_vehicle).lower().strip()
        if "confidence" in request.form:
            try:
                conf_val = float(request.form["confidence"])
            except Exception:
                pass
        file_bytes = file.read()
        np_arr = np.frombuffer(file_bytes, np.uint8)
        img_array = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    elif request.is_json:
        data = request.get_json(force=True)
        rule_vehicle = data.get("allowed_vehicle", allowed_vehicle).lower().strip()
        if "confidence" in data:
            try:
                conf_val = float(data["confidence"])
            except Exception:
                pass

        if "sample" in data:
            sample_path = data["sample"]
            if os.path.exists(sample_path):
                img_array = cv2.imread(sample_path)
            else:
                return jsonify({"error": f"Sample not found: {sample_path}"}), 404

        elif "image_base64" in data:
            b64_str = data["image_base64"]
            if "," in b64_str:
                b64_str = b64_str.split(",", 1)[1]
            file_bytes = base64.b64decode(b64_str)
            np_arr = np.frombuffer(file_bytes, np.uint8)
            img_array = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img_array is None:
        return jsonify({"error": "No valid image provided. Upload a file or send image_base64/sample."}), 400

    # Resize if image is huge for fast display
    h, w = img_array.shape[:2]
    max_w = 1280
    if w > max_w:
        scale = max_w / w
        img_array = cv2.resize(img_array, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)

    # Run detection with high-accuracy threshold
    rule_engine = ParkingRuleEngine(rule_vehicle)
    raw = get_detector().detect(img_array, conf=conf_val)
    evaluated = rule_engine.evaluate_frame(raw)

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
                "confidence": round(conf * 100, 1),
                "bbox": bbox,
                "reason": get_violation_reason(label, rule_vehicle),
                "action": "Violation: Move vehicle or issue parking citation"
            })
        else:
            right_items.append({
                "id": i,
                "label": label,
                "confidence": round(conf * 100, 1),
                "bbox": bbox,
                "reason": f"Authorized: Vehicle ({label.capitalize()}) is permitted in {rule_vehicle.upper()} parking zone.",
                "action": "Compliant: Authorized parking"
            })

    # Render annotated image
    annotated = img_array.copy()
    alert_counter = 30 if len(wrong_items) > 0 else 0
    draw_alert_banner(annotated, alert_counter, 30, [item["label"] for item in wrong_items])
    draw_detections(annotated, evaluated, show_confidence=True)
    draw_dashboard(annotated, fps=0.0, allowed_vehicle=rule_vehicle,
                   evaluated=evaluated, show_dashboard=True)
    draw_timestamp(annotated)

    # Encode annotated image to base64
    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    b64_annotated = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("utf-8")

    # Encode original image to base64 for comparison
    _, orig_buf = cv2.imencode(".jpg", img_array, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64_original = "data:image/jpeg;base64," + base64.b64encode(orig_buf.tobytes()).decode("utf-8")

    verdict = "VIOLATIONS_FOUND" if len(wrong_items) > 0 else ("ALL_COMPLIANT" if len(right_items) > 0 else "NO_VEHICLES")

    return jsonify({
        "verdict": verdict,
        "allowed_rule": rule_vehicle,
        "total_vehicles": len(evaluated),
        "wrong_count": len(wrong_items),
        "right_count": len(right_items),
        "wrong_items": wrong_items,
        "right_items": right_items,
        "annotated_image": b64_annotated,
        "original_image": b64_original,
        "detections": evaluated
    })


@app.route("/api/samples")
def api_samples():
    """Return available sample images."""
    samples = []
    preset_files = [
        ("sample_truck_motorcycle_combo.jpg", "🚛 🏍️ 🚗 Combined Trucks + Bikes + Cars"),
        ("demo_input.jpg", "🅿️ Aerial Drone Parking Lot (19 Cars)"),
        ("sample_bikes_cars.jpg", "🏍️ Parked Motorcycle Spot"),
        ("sample_truck_delivery.jpg", "🚛 Heavy Delivery Truck"),
        ("sample_truck_bike.jpg", "🚗 Passenger Car")
    ]
    for fname, label in preset_files:
        if os.path.exists(fname):
            samples.append({
                "id": fname,
                "name": label,
                "path": fname
            })
    return jsonify(samples)


if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  Vehicle Detection Web Dashboard")
    print("  Open → http://localhost:8080")
    print("=" * 55 + "\n")
    app.run(host="0.0.0.0", port=8080, debug=False, threaded=True)
