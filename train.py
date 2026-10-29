"""
train.py — Fine-tune YOLOv8 on the Top-View Vehicle Detection dataset
=====================================================================
This script transfers the weights from the pretrained yolov8m.pt (COCO)
and fine-tunes on aerial/top-view parking images so the model learns
overhead vehicle shapes — significantly boosting detection accuracy.

Usage:
    python train.py                   # default: 30 epochs
    python train.py --epochs 15       # faster run
    python train.py --model yolov8s.pt  # lighter model
"""

import os
import sys
import argparse
import shutil
from pathlib import Path

# ── Silence ultralytics noise ───────────────────────────────────────────────
os.environ["YOLO_VERBOSE"] = "False"
os.environ["ULTRALYTICS_AUTOINSTALL"] = "0"

from ultralytics import YOLO
from ultralytics.utils import SETTINGS

try:
    SETTINGS["sync"] = False
    SETTINGS["auto_update"] = False
except Exception:
    pass

# ── Paths ───────────────────────────────────────────────────────────────────
DATASET_BASE = Path(
    "/Users/ryan/.cache/kagglehub/datasets/"
    "farzadnekouei/top-view-vehicle-detection-image-dataset/versions/2/"
    "Vehicle_Detection_Image_Dataset"
)
DATA_YAML    = DATASET_BASE / "data.yaml"

PROJECT_DIR  = Path(__file__).parent
RUNS_DIR     = PROJECT_DIR / "runs" / "train"
OUTPUT_NAME  = "parking_finetune"
BEST_WEIGHTS = RUNS_DIR / OUTPUT_NAME / "weights" / "best.pt"
FINAL_PATH   = PROJECT_DIR / "parking_finetune.pt"


def patch_data_yaml():
    """
    Roboflow data.yaml uses relative paths (../train/images).
    YOLOv8 needs absolute paths when the cwd differs from the dataset dir.
    We write a patched copy next to the original.
    """
    patched = DATASET_BASE / "data_abs.yaml"

    train_images = str(DATASET_BASE / "train" / "images")
    valid_images = str(DATASET_BASE / "valid" / "images")

    content = (
        f"train: {train_images}\n"
        f"val:   {valid_images}\n"
        f"\nnc: 1\n"
        f"names: ['Vehicle']\n"
    )
    patched.write_text(content)
    print(f"[train.py] Patched data.yaml → {patched}")
    return str(patched)


def train(epochs: int = 30, model_path: str = "yolov8m.pt", imgsz: int = 640):
    print("=" * 60)
    print("  Parking Detection — YOLOv8 Fine-Tuning")
    print("=" * 60)
    print(f"  Base model   : {model_path}")
    print(f"  Dataset      : {DATASET_BASE}")
    print(f"  Epochs       : {epochs}")
    print(f"  Image size   : {imgsz}")
    print(f"  Output       : {FINAL_PATH}")
    print("=" * 60)

    if not DATASET_BASE.exists():
        print(f"\n[ERROR] Dataset not found at:\n  {DATASET_BASE}")
        print("Run the kagglehub download snippet first.")
        sys.exit(1)

    data_yaml = patch_data_yaml()

    # ── Load base model ──────────────────────────────────────────────────────
    print(f"\n[train.py] Loading base model: {model_path}")
    model = YOLO(model_path)

    # ── Train ────────────────────────────────────────────────────────────────
    print(f"[train.py] Starting fine-tuning for {epochs} epochs …\n")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=8,                   # safe for CPU RAM
        workers=2,
        project=str(RUNS_DIR),
        name=OUTPUT_NAME,
        exist_ok=True,             # overwrite previous run
        patience=10,               # early stopping
        lr0=0.001,                 # lower LR for fine-tuning
        lrf=0.01,
        warmup_epochs=2,
        hsv_h=0.015,               # mild augmentation for top-view
        hsv_s=0.5,
        hsv_v=0.3,
        flipud=0.5,                # top-view images benefit from vertical flip
        fliplr=0.5,
        mosaic=0.8,
        degrees=15.0,              # slight rotation (parking cameras tilt)
        translate=0.1,
        scale=0.4,
        device="cpu",
        verbose=True,
        plots=True,
    )

    # ── Copy best weights to project root ───────────────────────────────────
    if BEST_WEIGHTS.exists():
        shutil.copy2(BEST_WEIGHTS, FINAL_PATH)
        print(f"\n✅ Fine-tuned model saved → {FINAL_PATH}")
    else:
        print(f"\n[WARNING] best.pt not found at {BEST_WEIGHTS}")
        print("Check the runs/ directory manually.")
        return

    # ── Validate ─────────────────────────────────────────────────────────────
    print("\n[train.py] Running validation on held-out set …")
    fine_model = YOLO(str(FINAL_PATH))
    val_results = fine_model.val(
        data=data_yaml,
        imgsz=imgsz,
        device="cpu",
        verbose=True,
    )

    try:
        map50    = val_results.box.map50
        map5095  = val_results.box.map
        print("\n" + "=" * 60)
        print("  Validation Results")
        print("=" * 60)
        print(f"  mAP@50     : {map50:.4f}  ({map50 * 100:.1f}%)")
        print(f"  mAP@50-95  : {map5095:.4f}  ({map5095 * 100:.1f}%)")
        print("=" * 60)
        print(f"\n🚗 Fine-tuned model ready: {FINAL_PATH}")
        print("   Update config.yaml → model: parking_finetune.pt")
    except Exception:
        print("Validation complete. Check runs/ folder for metrics plots.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 for parking detection")
    parser.add_argument("--epochs", type=int,  default=30,         help="Training epochs (default: 30)")
    parser.add_argument("--model",  type=str,  default="yolov8m.pt", help="Base YOLO model (default: yolov8m.pt)")
    parser.add_argument("--imgsz",  type=int,  default=640,         help="Image size (default: 640)")
    args = parser.parse_args()

    train(epochs=args.epochs, model_path=args.model, imgsz=args.imgsz)
