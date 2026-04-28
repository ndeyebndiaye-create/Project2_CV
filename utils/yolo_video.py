"""
YOLOv11 – Simple detection (no tracking).
Counts objects per frame inside a region of interest.
GPU forced when available.
"""

import cv2 as cv
from ultralytics import solutions
import os

DETECTION_CLASSES = [
    'car', 'truck', 'bus', 'motorbike', 'bicycle',
    'person', 'traffic sign', 'traffic light'
]


class Detector:
    def __init__(self, filepath, device="cuda"):
        self.filepath = filepath
        self.device   = device
        os.makedirs("results", exist_ok=True)

        
        finetuned = "models/yolo11n_finetuned.pt"
        base      = "models/yolo11n.pt"
        self.model_path = finetuned if os.path.exists(finetuned) else base
        print(f"[Detector] Model : {self.model_path}")

    def forward(self, show=True):
        cap = cv.VideoCapture(self.filepath)
        assert cap.isOpened(), "Cannot open video/image"

        w   = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv.CAP_PROP_FPS)) or 25

        #  region_points dynamiques selon la résolution de la vidéo
        margin_x = int(w * 0.02)
        margin_y = int(h * 0.05)
        top_y    = int(h * 0.35)
        bot_y    = int(h * 0.80)
        region_points = [
            (margin_x, bot_y),
            (w - margin_x, bot_y),
            (w - margin_x, top_y),
            (margin_x, top_y)
        ]

        video_writer = cv.VideoWriter(
            "results/yolo_detection.avi",
            cv.VideoWriter_fourcc(*"mp4v"),
            fps, (w, h)
        )

        counter = solutions.ObjectCounter(
            show=show,
            region=region_points,
            model=self.model_path,
            device=self.device,
        )

        print(f"Running YOLOv11 detection on {self.device.upper()}...")
        print(f"Video : {w}x{h} @ {fps}fps | Region : {region_points}")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            results = counter(frame)
            video_writer.write(results.plot_im)

        cap.release()
        video_writer.release()
        if show:
            try:
                cv.destroyAllWindows()
            except cv.error:
                pass
        print("Done → results/yolo_detection.avi")
