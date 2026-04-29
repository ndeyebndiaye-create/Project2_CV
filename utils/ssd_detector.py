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
    - CSV logs follow the shared schema
    """
    def __init__(self, filepath,
                 classes=None,
                 confidence_threshold=0.4,
                 device="cuda",
                 output_dir="logs",
                 scene_name="scene_01",
                 group_id="group_01"):

        self.filepath   = filepath
        self.classes    = classes if classes else TRAFFIC_CLASSES
        self.conf_thr   = confidence_threshold
        self.output_dir = output_dir
        self.scene_name = scene_name
        self.group_id   = group_id
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

        video_name = os.path.basename(self.filepath)
        line_y     = h // 2  # ligne de comptage au centre

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
        prev_pos  = {}  # pour calcul vitesse {label_idx: (cx, cy)}

        print(f"Running SSD detection...")
        print(f"Video : {w}x{h} @ {fps:.0f}fps")

        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)

            # ── Header — schéma partagé ────────────────────────────────────────
            writer.writerow([
                'frame', 'timestamp_sec', 'scene_name', 'group_id', 'video_name',
                'track_id', 'class_name', 'confidence',
                'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2',
                'cx', 'cy', 'frame_width', 'frame_height',
                'crossed_line', 'direction', 'speed_px_s'
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

                    cls_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) \
                               else "unknown"
                    if cls_name == "motorcycle":
                        cls_name = "motorbike"
                    if cls_name not in self.classes:
                        continue

                    no_object = False
                    x1, y1, x2, y2 = map(int, box)

                    # ── Calculs schéma partagé ─────────────────────────────────
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    # Vitesse en px/s
                    key = int(label)
                    if key in prev_pos:
                        px, py = prev_pos[key]
                        dist   = ((cx-px)**2 + (cy-py)**2) ** 0.5
                        speed  = round(dist * fps, 2)
                    else:
                        speed = 0.0
                    prev_pos[key] = (cx, cy)

                    # Direction
                    if key in prev_pos and speed > 0:
                        px, py = prev_pos[key]
                        if abs(cx-px) > abs(cy-py):
                            direction = "right" if cx > px else "left"
                        else:
                            direction = "down" if cy > py else "up"
                    else:
                        direction = ""

                    # Traversée ligne de comptage
                    crossed = "true" if abs(cy - line_y) < 10 else "false"

                    # SSD n'a pas de track_id — on utilise -1
                    track_id = -1

                    # ── Dessin ────────────────────────────────────────────────
                    if cls_name == "stop sign":
                        color     = STOP_COLOR
                        thickness = 3
                        cv.putText(frame, "STOP SIGN",
                                   (x1, max(y1-20, 0)),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    else:
                        color     = [int(c) for c in COLORS[label]]
                        thickness = 2

                    cv.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                    cv.putText(frame, f"{cls_name}: {score:.2f}",
                               (x1, max(y1-8, 0)),
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    stats[cls_name] = stats.get(cls_name, 0) + 1

                    # ── Écriture CSV ───────────────────────────────────────────
                    writer.writerow([
                        frame_idx, timestamp,
                        self.scene_name, self.group_id, video_name,
                        track_id, cls_name, round(float(score), 3),
                        x1, y1, x2, y2,
                        cx, cy, w, h,
                        crossed, direction, speed
                    ])

                # Ligne de comptage visuelle
                cv.line(frame, (0, line_y), (w, line_y), (0, 255, 0), 2)
                cv.putText(frame, "Counting line",
                           (10, line_y - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

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
