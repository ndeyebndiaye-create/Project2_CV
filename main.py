"""
Traffic Object Detection & Tracking
Supports: YOLOv11 (with/without tracking), SSD MobileNetV3
Uses best.pt (fine-tuned) if available, falls back to yolo11n.pt
GPU forced by default.
"""

import os
import argparse
import torch

os.makedirs("results", exist_ok=True)
os.makedirs("logs",    exist_ok=True)

from utils.yolo_video   import Detector
from utils.yolo_tracker import Tracker, TRAFFIC_CLASSES
from utils.ssd_detector import SSDDetector


def parse_args():
    parser = argparse.ArgumentParser(description="Traffic Object Detection & Tracking")
    parser.add_argument(
        "--model",
        type=str,
        choices=["yolov11", "yolov11_track", "best", "ssd"],
        default="yolov11_track",
        help="Model to use"
    )
    parser.add_argument("--filepath", type=str, required=True,
                        help="Path to video")
    parser.add_argument("--classes", nargs="+", default=TRAFFIC_CLASSES,
                        help="Classes to detect (e.g. car bus truck stop_sign)")
    parser.add_argument("--conf", type=float, default=0.4,
                        help="Confidence threshold")
    parser.add_argument("--min-area", type=int, default=900,
                        help="Minimum bounding-box area in pixels to keep detection")
    parser.add_argument("--min-track-hits", type=int, default=3,
                        help="Minimum frames a track must persist before counting")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cuda | cpu (default: auto-detect GPU)")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display live OpenCV window during processing (requires GUI-enabled OpenCV)."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"\n🖥️  Device : {device.upper()}")
    if device == "cuda":
        print(f"   GPU    : {torch.cuda.get_device_name(0)}")
    print(f"   Model  : {args.model}")
    print(f"   File   : {args.filepath}")
    print(f"   Classes: {args.classes}\n")

    if args.model == "yolov11":
        det = Detector(args.filepath, device=device)
        det.forward(show=args.show)

    elif args.model in ("yolov11_track", "best"):
        tracker = Tracker(
            args.filepath,
            classes=list(args.classes),
            device=device,
            conf=args.conf,
            min_box_area=args.min_area,
            min_track_hits=args.min_track_hits
        )
        log_path, stats = tracker.forward(show=args.show)
        print(f"\nLog CSV : {log_path}")
        print(f"Stats   : {stats}")

    elif args.model == "ssd":
        det = SSDDetector(
            args.filepath,
            classes=set(args.classes),
            confidence_threshold=args.conf,
            device=device
        )
        log_path, stats = det.forward(show=args.show)
        print(f"\nLog CSV : {log_path}")
        print(f"Stats   : {stats}")


if __name__ == "__main__":
    main()
