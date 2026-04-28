"""
Fine-tuning script for YOLOv11 on the local traffic dataset.

This script is project-portable:
- no hardcoded absolute paths
- best model is copied from the actual run directory
"""

import os
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
RUNS_DIR = ROOT / "runs" / "train"
DATASET_YAML = ROOT / "data" / "data.yaml"

BASE_MODEL = MODELS_DIR / "yolo11n.pt"
FINAL_MODEL = MODELS_DIR / "best.pt"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device  : {DEVICE}")
print(f"Dataset : {DATASET_YAML}")
print(f"Model   : {BASE_MODEL}")

if not DATASET_YAML.exists():
    raise FileNotFoundError(f"Dataset YAML not found: {DATASET_YAML}")
if not BASE_MODEL.exists():
    raise FileNotFoundError(
        f"Base model not found: {BASE_MODEL}. Download/copy yolo11n.pt into models/."
    )

model = YOLO(str(BASE_MODEL))

results = model.train(
    data=str(DATASET_YAML),
    epochs=20,
    imgsz=320,
    batch=16,
    lr0=0.001,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    device=DEVICE,
    project=str(RUNS_DIR),
    name="yolov11_finetune",
    save=True,
    plots=True,
    patience=10,
    val=True,
)

run_dir = Path(results.save_dir)
best_model_path = run_dir / "weights" / "best.pt"
if not best_model_path.exists():
    raise FileNotFoundError(f"best.pt not found in expected run directory: {best_model_path}")

shutil.copy(best_model_path, FINAL_MODEL)
print(f"\nBest model saved -> {FINAL_MODEL}")

model_eval = YOLO(str(FINAL_MODEL))
metrics = model_eval.val(data=str(DATASET_YAML), device=DEVICE)

print("\n=== Evaluation ===")
print(f"mAP50     : {metrics.box.map50:.3f}")
print(f"mAP50-95  : {metrics.box.map:.3f}")
print(f"Precision : {metrics.box.mp:.3f}")
print(f"Recall    : {metrics.box.mr:.3f}")
