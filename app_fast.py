"""
app_fast.py — Fast-launch version of the parking detection web server.

Heavy ML imports (cv2, torch, ultralytics) are deferred to a background
thread so the Flask web UI is available within seconds.

Open http://localhost:8080 immediately after starting.
"""

import threading
import time
import csv
import os
import base64

os.environ.setdefault("OPENCV_AVFOUNDATION_SKIP_AUTH", "1")

from collections import defaultdict
from pathlib import Path

import yaml
from flask import Flask, Response, jsonify, render_template, request

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
cfg = yaml.safe_load(open("config.yaml"))
allowed_vehicle = cfg["allowed_vehicle"].lower().strip()

# ── Shared state ──────────────────────────────────────────────────────────────
lock = threading.Lock()
latest_frame = None
camera_enabled = True
ml_ready = False  # True once heavy libs are loaded

stats = {
    "fps":              0.0,
    "allowed_vehicle":  allowed_vehicle,
    "alert":            False,
    "camera_enabled":   True,
    "counts":           {},
    "total_detections": 0,
    "ml_ready":         False,
}

# Lazy-loaded globals (populated by background loader)
_cv2 = None
_np = None
_detector = None
_rules = None
_logger = None
_draw_detections = None
_draw_alert_banner = None
_draw_dashboard = None
_draw_timestamp = None
_ParkingRuleEngine = None


def _load_ml_libs():
    """Background thread: import heavy libraries and initialise components."""
    global _cv2, _np, _detector, _rules, _logger, ml_ready
    global _draw_detections, _draw_alert_banner, _draw_dashboard, _draw_timestamp
    global _ParkingRuleEngine

    print("[ML-Loader] Importing cv2 …")
    import cv2
    _cv2 = cv2

    print("[ML-Loader] Importing numpy …")
    import numpy as np
    _np = np

    print("[ML-Loader] Importing detector …")
    from detector import VehicleDetector
    from parking_rules import ParkingRuleEngine
    _ParkingRuleEngine = ParkingRuleEngine

    print("[ML-Loader] Importing UI overlay …")
    from ui_overlay import draw_detections, draw_alert_banner, draw_dashboard, draw_timestamp
    _draw_detections = draw_detections
    _draw_alert_banner = draw_alert_banner
    _draw_dashboard = draw_dashboard
    _draw_timestamp = draw_timestamp

    print("[ML-Loader] Importing logger …")
    from logger import DetectionLogger
    _logger_inst = DetectionLogger(cfg["log_file"]) if cfg["enable_logging"] else None

    print("[ML-Loader] Loading YOLO model …")
    det = VehicleDetector(cfg["model"],
                          float(cfg["confidence_threshold"]),
                          float(cfg["iou_threshold"]))

    with lock:
        _detector = det
        _rules = ParkingRuleEngine(allowed_vehicle)
        _logger = _logger_inst
        ml_ready = True
        stats["ml_ready"] = True

    print("[ML-Loader] ✅ All ML components ready!")
    _start_capture()


# ── Capture loop (only starts after ML is loaded) ────────────────────────────
def _start_capture():
    t = threading.Thread(target=_capture_loop, daemon=True)
    t.start()


def _capture_loop():
    global latest_frame, stats, _rules, allowed_vehicle, camera_enabled

    source = cfg.get("camera_source", 0)
    if str(source).isdigit():
        source = int(source)

    cap = None
    use_fallback_image = False

    try:
        cap = _cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[WebApp] Camera source {source} unavailable. Using demo mode.")
            use_fallback_image = True
    except Exception as e:
        print(f"[WebApp] Camera error: {e}. Using demo mode.")
        use_fallback_image = True

    frame_id = 0
    alert_counter = 0
    alert_duration = int(cfg["alert_duration_frames"])
    fps_timer = time.time()
    fps_count = 0
    fps = 0.0

    while True:
        with lock:
            is_active = camera_enabled

        if not is_active:
            time.sleep(0.2)
            continue

        if use_fallback_image or cap is None or not cap.isOpened():
            if os.path.exists("demo_input.jpg"):
                frame = _cv2.imread("demo_input.jpg")
            else:
                frame = _np.zeros((480, 640, 3), dtype=_np.uint8)
            time.sleep(0.1)
        else:
            ret, frame = cap.read()
            if not ret:
                cap.set(_cv2.CAP_PROP_POS_FRAMES, 0)
                time.sleep(0.05)
                continue

        frame_id += 1

        h, w = frame.shape[:2]
        if w > 960:
            scale = 960 / w
            frame = _cv2.resize(frame, (960, int(h * scale)))

        with lock:
            current_rules = _rules

        raw = _detector.detect(frame)
        evald = current_rules.evaluate_frame(raw)

        if any(not d["allowed"] for d in evald):
            alert_counter = alert_duration
            unauth_labels = [d["label"] for d in evald if not d["allowed"]]
        else:
            alert_counter = max(0, alert_counter - 1)
            unauth_labels = []

        if _logger and evald:
            _logger.log_frame(frame_id, evald)

        fps_count += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            fps = fps_count / elapsed
            fps_count = 0
            fps_timer = time.time()

        _draw_alert_banner(frame, alert_counter, alert_duration, unauth_labels)
        _draw_detections(frame, evald, show_confidence=True)
        _draw_dashboard(frame, fps, allowed_vehicle, evald, show_dashboard=True)
        _draw_timestamp(frame)

        _, buf = _cv2.imencode(".jpg", frame, [_cv2.IMWRITE_JPEG_QUALITY, 82])
        jpg_bytes = buf.tobytes()

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
            stats["fps"] = round(fps, 1)
            stats["alert"] = alert_counter > 0
            stats["camera_enabled"] = True
            stats["counts"] = dict(counts)
            stats["allowed_vehicle"] = allowed_vehicle
            stats["total_detections"] += len(evald)


# ── MJPEG generator ──────────────────────────────────────────────────────────
def generate_frames():
    while True:
        with lock:
            is_active = camera_enabled
            frame = latest_frame
        if not is_active or frame is None:
            time.sleep(0.08)
            continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(1 / 30)


# ── Helper ────────────────────────────────────────────────────────────────────
def get_violation_reason(label: str, allowed: str) -> str:
    reasons = {
        "motorcycle": f"Motorcycles are prohibited in {allowed.upper()}-only parking bays.",
        "truck": f"Commercial trucks/heavy vehicles are not permitted in {allowed.upper()}-only areas.",
        "bus": f"Buses are not permitted in {allowed.upper()}-only parking bays.",
        "car": f"Standard passenger cars are not permitted in {allowed.upper()}-only spots.",
        "bicycle": f"Bicycles cannot occupy {allowed.upper()}-only parking spots.",
    }
    return reasons.get(label.lower(), f"Vehicle type '{label.capitalize()}' violates the '{allowed.upper()}' rule.")


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
    global _rules, allowed_vehicle
    data = request.get_json(force=True)
    vehicle = data.get("allowed_vehicle", "car").lower().strip()
    valid = {"car", "motorcycle", "bus", "truck", "bicycle"}
    if vehicle not in valid:
        return jsonify({"error": f"Invalid vehicle type. Choose from: {valid}"}), 400
    with lock:
        allowed_vehicle = vehicle
        if _ParkingRuleEngine:
            _rules = _ParkingRuleEngine(vehicle)
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
    return jsonify(rows[-50:][::-1])


@app.route("/api/detect_image", methods=["POST"])
def api_detect_image():
    if not ml_ready:
        return jsonify({"error": "ML engine is still loading. Please wait a moment and retry."}), 503

    import numpy as np

    rule_vehicle = allowed_vehicle
    img_array = None
    conf_val = float(cfg.get("confidence_threshold", 0.25))

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
        img_array = _cv2.imdecode(np_arr, _cv2.IMREAD_COLOR)

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
                img_array = _cv2.imread(sample_path)
            else:
                return jsonify({"error": f"Sample not found: {sample_path}"}), 404

        elif "image_base64" in data:
            b64_str = data["image_base64"]
            if "," in b64_str:
                b64_str = b64_str.split(",", 1)[1]
            file_bytes = base64.b64decode(b64_str)
            np_arr = np.frombuffer(file_bytes, np.uint8)
            img_array = _cv2.imdecode(np_arr, _cv2.IMREAD_COLOR)

    if img_array is None:
        return jsonify({"error": "No valid image provided."}), 400

    h, w = img_array.shape[:2]
    max_w = 1280
    if w > max_w:
        scale = max_w / w
        img_array = _cv2.resize(img_array, (max_w, int(h * scale)), interpolation=_cv2.INTER_AREA)

    rule_engine = _ParkingRuleEngine(rule_vehicle)
    raw = _detector.detect(img_array, conf=conf_val)
    evaluated = rule_engine.evaluate_frame(raw)

    wrong_items = []
    right_items = []

    for i, det in enumerate(evaluated, start=1):
        label = det["label"]
        conf = det["confidence"]
        bbox = det["bbox"]

        if not det["allowed"]:
            wrong_items.append({
                "id": i, "label": label,
                "confidence": round(conf * 100, 1), "bbox": bbox,
                "reason": get_violation_reason(label, rule_vehicle),
                "action": "Violation: Move vehicle or issue parking citation"
            })
        else:
            right_items.append({
                "id": i, "label": label,
                "confidence": round(conf * 100, 1), "bbox": bbox,
                "reason": f"Authorized: Vehicle ({label.capitalize()}) is permitted in {rule_vehicle.upper()} parking zone.",
                "action": "Compliant: Authorized parking"
            })

    annotated = img_array.copy()
    alert_counter = 30 if len(wrong_items) > 0 else 0
    _draw_alert_banner(annotated, alert_counter, 30, [item["label"] for item in wrong_items])
    _draw_detections(annotated, evaluated, show_confidence=True)
    _draw_dashboard(annotated, fps=0.0, allowed_vehicle=rule_vehicle,
                    evaluated=evaluated, show_dashboard=True)
    _draw_timestamp(annotated)

    _, buf = _cv2.imencode(".jpg", annotated, [_cv2.IMWRITE_JPEG_QUALITY, 90])
    b64_annotated = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("utf-8")

    _, orig_buf = _cv2.imencode(".jpg", img_array, [_cv2.IMWRITE_JPEG_QUALITY, 85])
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
            samples.append({"id": fname, "name": label, "path": fname})
    return jsonify(samples)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Start ML loading in background thread
    loader = threading.Thread(target=_load_ml_libs, daemon=True)
    loader.start()

    print("\n" + "=" * 55)
    print("  Vehicle Detection Web Dashboard")
    print("  Open → http://localhost:8080")
    print("  (ML model loading in background...)")
    print("=" * 55 + "\n")
    app.run(host="0.0.0.0", port=8080, debug=False, threaded=True)
