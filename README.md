# 🚦 Traffic Object Detection & Tracking

## Architecture

```
project/
│
├── main.py                  ← CLI entry point (model selector + GPU)
├── app.py                   ← Streamlit web interface
├── dashboard.py             ← Multi-scene comparison dashboard
│
├── utils/
│   ├── yolo_video.py        ← YOLOv11 simple detection (no tracking)
│   ├── yolo_tracker.py      ← YOLOv11 + ByteTrack + CSV logs
│   └── ssd_detector.py      ← SSD MobileNetV3 via torchvision
│
├── models/
│   └── best.pt              ← YOLOv11 fine-tuned weights (primary model)
│
├── logs/                    ← CSV logs (auto-generated)
├── results/                 ← Annotated output videos
└── data/                    ← Test videos
```

## Models

| Model | Weights | Classes | Speed | GPU |
|---|---|---|---|---|
| YOLOv11 + ByteTrack | best.pt (fine-tuned) | Traffic + stop sign | ⚡⚡⚡ | ✅ Auto |
| YOLOv11 simple | best.pt (fine-tuned) | Traffic + stop sign | ⚡⚡⚡ | ✅ Auto |
| SSD MobileNetV3 | auto-download (COCO) | 91 COCO classes | ⚡⚡ | ✅ Auto |

> `best.pt` is automatically used if found in `models/`. Falls back to `yolo11n.pt` otherwise.

## Install

```bash
conda activate aims_cv
pip install ultralytics streamlit plotly pandas opencv-python torch torchvision
```

## Usage (CLI)

```bash
# YOLOv11 + ByteTrack (default, recommended)
python main.py --model best --filepath data/traffic.mp4

# YOLOv11 + ByteTrack with specific classes
python main.py --model yolov11_track --filepath data/traffic.mp4 --classes car bus truck "stop sign"

# YOLOv11 simple detection (no tracking)
python main.py --model yolov11 --filepath data/traffic.mp4

# SSD MobileNetV3
python main.py --model ssd --filepath data/traffic.mp4 --classes car bus truck "stop sign"

# Force CPU
python main.py --model yolov11_track --filepath data/traffic.mp4 --device cpu

# Custom confidence threshold
python main.py --model yolov11_track --filepath data/traffic.mp4 --conf 0.5
```

## Web Interface

```bash
# Main detection interface
streamlit run app.py

# Multi-scene dashboard
streamlit run dashboard.py
```

## GPU

GPU is automatically detected. Check status:

```python
import torch
print(torch.cuda.is_available())     # True = GPU active
print(torch.cuda.get_device_name(0)) # GPU name
```

## Outputs

| File | Description |
|---|---|
| `results/yolo_tracked.avi` | Annotated video with bounding boxes and track IDs |
| `results/yolo_detection.avi` | Annotated video (no tracking) |
| `results/ssd_detection.avi` | SSD annotated video |
| `logs/log_YYYYMMDD_HHMMSS.csv` | Detection log with timestamps, classes, bbox, track IDs |

## CSV Log Format

```
timestamp_s, frame, track_id, class, x1, y1, x2, y2, confidence
0.04, 1, 3, car, 142, 88, 310, 201, 0.91
0.08, 2, 3, car, 145, 90, 312, 203, 0.93
```

## Detectable Classes

```
Traffic : car, truck, bus, motorbike, bicycle, person
Signs   : stop sign, traffic light
```

## License

MIT License — AIMS Senegal, Computer Vision Project 2, April 2026
