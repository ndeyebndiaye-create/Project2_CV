"""
SSD MobileNetV3 – Object detection via torchvision.
Detects traffic objects AND stop signs.
GPU forced when available.
No external model files needed – weights auto-downloaded.
"""

import cv2 as cv
import torch
import torchvision
from torchvision.transforms import functional as F
import numpy as np
import os
import csv
from datetime import datetime

COCO_CLASSES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "N/A", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A",
    "backpack", "umbrella", "N/A", "N/A", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "N/A", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "N/A",
    "dining table", "N/A", "N/A", "toilet", "N/A", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
    "sink", "refrigerator", "N/A", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]

# ✅ "motorbike" aligné avec yolo_tracker.py (cohérence du projet)
TRAFFIC_CLASSES = {
    "person", "bicycle", "car", "motorbike", "motorcycle",
    "bus", "truck", "stop sign", "traffic light"
}

np.random.seed(42)
COLORS     = np.random.randint(0, 255, size=(len(COCO_CLASSES), 3), dtype="uint8")
STOP_COLOR = (0, 0, 255)


class SSDDetector:
    """
    SSDLite MobileNetV3 pre-trained on COCO.
    - No training needed
    - Detects 91 COCO classes including stop sign
    - GPU accelerated
    """
    def __init__(self, filepath,
                 classes=None,
                 confidence_threshold=0.4,
                 device="cuda",
                 output_dir="logs"):

        self.filepath   = filepath
        self.classes    = classes if classes else TRAFFIC_CLASSES
        self.conf_thr   = confidence_threshold
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs("results",  exist_ok=True)

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Loading SSD MobileNetV3 on {self.device}...")

        weights = torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
        self.model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=weights)
        self.model.to(self.device)
        self.model.eval()
        print(f"Model ready on : {self.device}")

    def forward(self, show=True, save_video=True):
        cap = cv.VideoCapture(self.filepath)
        assert cap.isOpened(), "Cannot open video"

        w   = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv.CAP_PROP_FPS) or 25

        video_writer = None
        if save_video:
            video_writer = cv.VideoWriter(
                "results/ssd_detection.avi",
                cv.VideoWriter_fourcc(*"mp4v"),
                fps, (w, h)
            )

        log_path = os.path.join(
            self.output_dir,
            f"ssd_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )

        stats     = {}
        frame_idx = 0

        print(f"Running SSD detection...")
        print(f"Video : {w}x{h} @ {fps:.0f}fps")

        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            # ✅ Ajout timestamp_s pour cohérence avec yolo_tracker logs
            writer.writerow([
                "timestamp_s", "frame", "class",
                "x1", "y1", "x2", "y2", "confidence"
            ])

            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                timestamp = round(frame_idx / fps, 3)
                img_rgb   = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                tensor    = F.to_tensor(img_rgb).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    outputs = self.model(tensor)[0]

                boxes  = outputs["boxes"].cpu().numpy()
                labels = outputs["labels"].cpu().numpy()
                scores = outputs["scores"].cpu().numpy()

                no_object = True

                for box, label, score in zip(boxes, labels, scores):
                    if score < self.conf_thr:
                        continue

                    cls_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else "unknown"
                    # ✅ Normaliser "motorcycle" → "motorbike"
                    if cls_name == "motorcycle":
                        cls_name = "motorbike"
                    if cls_name not in self.classes:
                        continue

                    no_object = False
                    x1, y1, x2, y2 = map(int, box)

                    if cls_name == "stop sign":
                        color     = STOP_COLOR
                        thickness = 3
                        cv.putText(frame, "STOP SIGN",
                                   (x1, max(y1 - 20, 0)),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    else:
                        color     = [int(c) for c in COLORS[label]]
                        thickness = 2

                    cv.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                    cv.putText(frame, f"{cls_name}: {score:.2f}",
                               (x1, max(y1 - 8, 0)),
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    stats[cls_name] = stats.get(cls_name, 0) + 1
                    writer.writerow([
                        timestamp, frame_idx, cls_name,
                        x1, y1, x2, y2, round(float(score), 3)
                    ])

                if no_object:
                    cv.putText(frame, "No selected object detected",
                               (20, 50), cv.FONT_HERSHEY_SIMPLEX,
                               1.0, (0, 0, 255), 2)

                if save_video and video_writer:
                    video_writer.write(frame)
                if show:
                    cv.imshow("SSD Detection", frame)
                    if cv.waitKey(1) & 0xFF == ord("q"):
                        break

                frame_idx += 1

        cap.release()
        if video_writer:
            video_writer.release()
        if show:
            try:
                cv.destroyAllWindows()
            except cv.error:
                pass

        print(f"\n=== SSD Results ===")
        print(f"Log CSV : {log_path}")
        for cls, count in sorted(stats.items()):
            print(f"  {cls} : {count} detections")

        return log_path, stats
