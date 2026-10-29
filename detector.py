import os
# Disable ultralytics auto-update checks and telemetry to run offline and fast
os.environ["YOLO_VERBOSE"] = "False"
os.environ["ULTRALYTICS_AUTOINSTALL"] = "0"

import torch
import numpy as np
from ultralytics import YOLO
from ultralytics.utils import SETTINGS

try:
    SETTINGS["sync"] = False
    SETTINGS["auto_update"] = False
except Exception:
    pass

# COCO class IDs for vehicles
VEHICLE_CLASSES = {
    1:  "bicycle",
    2:  "car",
    3:  "motorcycle",
    5:  "bus",
    7:  "truck",
}

VEHICLE_CLASS_IDS = list(VEHICLE_CLASSES.keys())


def _nms_boxes(boxes: list, scores: list, iou_thresh: float = 0.40) -> list[int]:
    """Applies Non-Maximum Suppression to deduplicate overlapping bounding boxes."""
    if not boxes:
        return []
    try:
        boxes_t = torch.tensor(boxes, dtype=torch.float32)
        scores_t = torch.tensor(scores, dtype=torch.float32)
        from torchvision.ops import nms
        return nms(boxes_t, scores_t, iou_thresh).tolist()
    except Exception:
        indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        keep = []
        while indices:
            current = indices.pop(0)
            keep.append(current)
            b1 = boxes[current]
            indices = [
                i for i in indices
                if _iou(b1, boxes[i]) < iou_thresh
            ]
        return keep


def _iou(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


class VehicleDetector:
    """
    High-Accuracy Vehicle Detector with multi-perspective support
    for both CCTV ground level and aerial / drone parking lots.
    """

    def __init__(self, model_path: str = "yolov8x.pt",
                 conf_threshold: float = 0.15,
                 iou_threshold: float = 0.40):
        try:
            threads = max(1, min(os.cpu_count() or 4, 8))
            torch.set_num_threads(threads)
        except Exception:
            pass

        self.device = "cpu"
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # Choose best available model: prefer yolov8x.pt if present, else fallback
        chosen_model = model_path
        if not os.path.exists(chosen_model):
            for candidate in ["yolov8x.pt", "yolov8m.pt", "yolov8s.pt", "yolov8n.pt"]:
                if os.path.exists(candidate):
                    chosen_model = candidate
                    break

        print(f"[Detector] Loading High-Accuracy Vision Model: {chosen_model} (Device: {self.device})")
        self.model = YOLO(chosen_model)
        self.model_path = chosen_model
        print("[Detector] High-Accuracy Model loaded successfully.")

    def detect(self, frame: np.ndarray, conf: float = None) -> list[dict]:
        """
        Run inference on a single BGR frame with multi-perspective support.

        Returns:
            List of detection dicts:
                - bbox       (x1, y1, x2, y2)
                - class_id   int
                - label      str
                - confidence float
        """
        conf_to_use = conf if conf is not None else self.conf_threshold
        h, w = frame.shape[:2]
        
        # Optimal resolution for top-view & street CCTV vehicle detection
        target_imgsz = 800 if max(h, w) <= 1280 else 1024

        results = self.model(
            frame,
            imgsz=target_imgsz,
            conf=conf_to_use,
            iou=self.iou_threshold,
            classes=VEHICLE_CLASS_IDS,
            device=self.device,
            verbose=False,
        )

        raw_boxes   = []
        raw_scores  = []
        raw_labels  = []
        raw_classes = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                cls_id = int(box.cls[0])
                score  = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                box_w = max(1, x2 - x1)
                box_h = max(1, y2 - y1)
                area  = box_w * box_h

                # Filter out microscopic noise and whole-frame boxes
                if area < 100 or area > (w * h * 0.95):
                    continue

                label = VEHICLE_CLASSES.get(cls_id, self.model.names.get(cls_id, "car"))

                raw_boxes.append([x1, y1, x2, y2])
                raw_scores.append(score)
                raw_labels.append(label)
                raw_classes.append(cls_id)

        # Apply clean NMS deduplication
        keep_indices = _nms_boxes(raw_boxes, raw_scores, iou_thresh=0.35)

        detections = []
        for idx in keep_indices:
            detections.append({
                "bbox":       tuple(raw_boxes[idx]),
                "class_id":   raw_classes[idx],
                "label":      raw_labels[idx],
                "confidence": raw_scores[idx],
            })

        return detections
