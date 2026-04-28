"""
YOLOv11 + ByteTrack – Detection, tracking, and CSV logging.
Detects traffic objects AND stop signs.
GPU forced when available.
"""

import cv2 as cv
from ultralytics import YOLO
import csv
import os
from datetime import datetime

TRAFFIC_CLASSES = [
    'car', 'truck', 'bus', 'motorbike', 'bicycle',
    'person', 'traffic sign', 'traffic light'
]

CLASS_ALIASES = {
    "motorcycle": "motorbike",
    "trafficLight": "traffic light",
    "traffic_light": "traffic light",
    "pedestrian": "person",
}


def normalize_class_name(name: str) -> str:
    return CLASS_ALIASES.get(name, name)


class Tracker:
    """
    YOLOv11 + ByteTrack tracker.
    - Assigns unique IDs to each object
    - Logs every detection to CSV
    - Highlights stop signs with a special color (red)
    - GPU accelerated
    """
    def __init__(self, filepath, classes=None, device="cuda",
                 output_dir="logs", conf=0.4, min_box_area=900, min_track_hits=3):

        self.filepath   = filepath
        self.classes    = classes if classes else TRAFFIC_CLASSES
        self.device     = device
        self.output_dir = output_dir
        self.conf       = conf
        self.min_box_area = min_box_area
        self.min_track_hits = min_track_hits
        os.makedirs(output_dir, exist_ok=True)

        # ✅ Fallback : utilise best.pt si disponible, sinon yolo11n.pt
        finetuned = "models/yolo11n_finetuned.pt"
        base      = "models/yolo11n.pt"
        self.model_path = finetuned if os.path.exists(finetuned) else base
        print(f"[Tracker] Model : {self.model_path}")

    def forward(self, show=True, save_video=True):
        model = YOLO(self.model_path)
        cap   = cv.VideoCapture(self.filepath)
        assert cap.isOpened(), "Cannot open video"

        w     = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        h     = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        fps   = cap.get(cv.CAP_PROP_FPS) or 25
        total = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

        video_writer = None
        if save_video:
            os.makedirs("results", exist_ok=True)
            video_writer = cv.VideoWriter(
                "results/yolo_tracked.avi",
                cv.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
            )

        log_path = os.path.join(
            self.output_dir,
            f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )

        unique_ids = {}
        track_hits = {}
        frame_idx  = 0

        print(f"Running YOLOv11 + ByteTrack on {self.device.upper()}...")
        print(f"Video : {w}x{h} @ {fps:.0f}fps ({total} frames)")

        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp_s', 'frame', 'track_id',
                'class', 'x1', 'y1', 'x2', 'y2', 'confidence'
            ])

            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                timestamp = round(frame_idx / fps, 3)

                results = model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    conf=self.conf,       # ✅ seuil de confiance
                    device=self.device,
                    verbose=False
                )

                no_object = True

                if results[0].boxes is not None and results[0].boxes.id is not None:
                    for box in results[0].boxes:
                        track_id = int(box.id[0])
                        cls_name = normalize_class_name(model.names[int(box.cls[0])])
                        conf_val = round(float(box.conf[0]), 3)
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        box_area = max(0, (x2 - x1)) * max(0, (y2 - y1))

                        if cls_name not in self.classes:
                            continue
                        if conf_val < self.conf:
                            continue
                        if box_area < self.min_box_area:
                            continue

                        no_object = False
                        track_hits[track_id] = track_hits.get(track_id, 0) + 1
                        if track_hits[track_id] >= self.min_track_hits:
                            unique_ids.setdefault(track_id, cls_name)

                        writer.writerow([
                            timestamp, frame_idx, track_id,
                            cls_name, x1, y1, x2, y2, conf_val
                        ])

                annotated = results[0].plot()

                # Boîte rouge spéciale pour stop signs
                if results[0].boxes is not None:
                    for box in results[0].boxes:
                        cls_name = normalize_class_name(model.names[int(box.cls[0])])
                        if cls_name == 'stop sign':
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cv.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
                            cv.putText(annotated, "STOP SIGN",
                                       (x1, y1 - 10),
                                       cv.FONT_HERSHEY_SIMPLEX, 0.8,
                                       (0, 0, 255), 2)

                if no_object:
                    cv.putText(annotated, "No selected object detected",
                               (20, 50), cv.FONT_HERSHEY_SIMPLEX,
                               1.0, (0, 0, 255), 2)

                if save_video and video_writer:
                    video_writer.write(annotated)
                if show:
                    cv.imshow("YOLOv11 Tracking", annotated)
                    if cv.waitKey(1) & 0xFF == ord('q'):
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

        stats = {}
        for cls in unique_ids.values():
            stats[cls] = stats.get(cls, 0) + 1

        print(f"\n=== Results ===")
        print(f"Log CSV : {log_path}")
        for cls, count in sorted(stats.items()):
            print(f"  {cls} : {count} unique objects")

        return log_path, stats
