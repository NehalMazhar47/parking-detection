# 🚗 Vehicle Detection & Parking Violation System

A complete **AI-powered vehicle classification, parking enforcement, and violation detection system** built with **YOLOv8 + OpenCV + Python + Flask**.

Put any image in (via Drag & Drop Web UI or CLI) to detect all vehicles and get an instant report on **WHAT IS WRONG (Violations)** and **WHAT IS RIGHT (Compliant)**. Also supports live webcam and RTSP video stream monitoring.

---

## 🎯 Key Features

- 📸 **Image Inspector (What is Wrong vs Right)**:
  - Drag & drop or browse any image (JPG, PNG, WEBP).
  - Instantly detects all vehicles (Cars, Motorcycles, Buses, Trucks, Bicycles).
  - Categorizes into **❌ What is Wrong** (unauthorized vehicles with explanation & citation recommendation) and **✅ What is Right** (permitted vehicles).
  - Color-coded bounding boxes + downloadable annotated results.
- 🌐 **Web Dashboard UI**:
  - Interactive web interface with tabbed switching (Image Inspector & Live Camera Feed).
  - Change allowed vehicle rules on the fly with instant re-evaluation.
- 📹 **Live Video & Webcam Detection**:
  - Real-time video stream processing with flashing alerts and live stats dashboard.
- 💻 **Dedicated CLI Tool (`detect_image.py`)**:
  - Run terminal audits with ASCII breakdown and saved annotated images.

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Web App (Recommended)

```bash
python app.py
```
Open **http://localhost:8080** in your browser:
- Use the **Image Inspector** tab to drop any image and see immediate violations vs compliant vehicles.
- Switch to the **Live Video** tab for real-time camera monitoring.

### 3. Run Single Image Detection via CLI

```bash
# Analyze image with default rule (Cars allowed)
python detect_image.py --image demo_input.jpg

# Check against motorcycle-only or bus-only rule
python detect_image.py --image demo_input.jpg --allowed motorcycle
python detect_image.py --image demo_input.jpg --allowed bus
```

### 4. Run Live Camera Stream

```bash
# Default webcam
python main.py

# Video file or RTSP stream
python main.py --source parking_lot.mp4 --allowed car
```
| `D` | Toggle dashboard panel on/off |

---

## 🛠 Configuration Reference (`config.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `camera_source` | `0` | Webcam index, video file path, or RTSP URL |
| `allowed_vehicle` | `car` | The only permitted vehicle type |
| `model` | `yolov8n.pt` | YOLO model (n=fast, s/m/l/x=accurate) |
| `confidence_threshold` | `0.45` | Min detection confidence (0–1) |
| `iou_threshold` | `0.45` | NMS IoU threshold (0–1) |
| `show_dashboard` | `true` | Show live stats panel |
| `show_confidence` | `true` | Show confidence % on bounding boxes |
| `display_scale` | `1.0` | Resize display (e.g. `0.75` = 75%) |
| `enable_logging` | `true` | Save detections to CSV |
| `log_file` | `logs/detection_log.csv` | Log file path |
| `alert_duration_frames` | `30` | Frames to show alert after last detection |

---

## 🚙 Supported Vehicle Types

| Label | Description |
|-------|-------------|
| `car` | Passenger cars, sedans, SUVs |
| `motorcycle` | Motorcycles, scooters |
| `bicycle` | Bicycles |
| `bus` | Buses, minibuses |
| `truck` | Trucks, lorries |

> These are all detected by the **YOLOv8 pre-trained COCO model** — no custom training needed!

---

## 📁 Project Structure

```
parking detection/
├── main.py              # Entry point
├── detector.py          # YOLOv8 detection logic
├── parking_rules.py     # Parking restriction engine
├── ui_overlay.py        # OpenCV drawing utilities
├── logger.py            # CSV event logger
├── config.yaml          # Configuration file
├── requirements.txt     # Python dependencies
├── logs/
│   └── detection_log.csv   # Auto-generated
└── screenshots/            # Auto-generated on 'S' press
```

---

## 📈 Upgrading the Model

For better accuracy (at the cost of speed), change the `model` in `config.yaml`:

| Model | Speed | Accuracy | Size |
|-------|-------|----------|------|
| `yolov8n.pt` | ⚡ Fastest | Good | ~6 MB |
| `yolov8s.pt` | Fast | Better | ~22 MB |
| `yolov8m.pt` | Moderate | Great | ~52 MB |
| `yolov8l.pt` | Slow | Excellent | ~87 MB |

---

## 📝 Detection Log Format

The CSV log at `logs/detection_log.csv`:

```
timestamp,frame_id,vehicle_type,status,confidence
2026-09-17 09:30:00,1,car,allowed,0.9231
2026-09-17 09:30:00,1,motorcycle,unauthorized,0.8745
```
