"""
Streamlit Web Interface – Traffic Detection
Supports: YOLOv11 (tracking) | SSD MobileNetV3
GPU forced automatically.
"""

import streamlit as st
import cv2 as cv
import tempfile
import os
import csv
import torch
import pandas as pd
import plotly.express as px
from ultralytics import YOLO
from datetime import datetime
from utils.yolo_tracker import normalize_class_name
#from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
#import av

if 'global_unique_ids' not in st.session_state:
    st.session_state.global_unique_ids = {}

def refine_class_by_shape(cls_name: str, x1: int, y1: int, x2: int, y2: int) -> str:
    """Reduce common false positives where vertical human boxes become 'car'."""
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    if cls_name == "car" and (h / w) > 1.2:
        return "person"
    return cls_name


# ── GPU Detection ──────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_NAME = torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU only"

TRAFFIC_CLASSES = ['car', 'truck', 'bus', 'motorbike', 'bicycle', 'person', 'traffic sign', 'traffic light']
YOLO_BASE_PATH = "models/yolo11n.pt"
YOLO_FINETUNED_PATH = "models/best.pt"
os.makedirs("logs", exist_ok=True)

class TrafficVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model       = YOLO(
            "models/best.pt" if os.path.exists("models/best.pt")
            else "models/yolo11n.pt"
        )
        self.unique_ids  = {}
        self.track_hits  = {}
        self.frame_idx   = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        results = self.model.track(
            img,
            persist  = True,
            tracker  = "bytetrack.yaml",
            conf     = 0.4,
            device   = DEVICE,
            verbose  = False
        )

        annotated = img.copy()
        no_object = True

        if results[0].boxes is not None and results[0].boxes.id is not None:
            for box in results[0].boxes:
                track_id = int(box.id[0])
                cls_name = self.model.names[int(box.cls[0])]
                conf_val = round(float(box.conf[0]), 3)
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                no_object = False
                self.track_hits[track_id] = self.track_hits.get(track_id, 0) + 1

                if self.track_hits[track_id] >= 3:
                    self.unique_ids.setdefault(track_id, cls_name)

                color = (0, 0, 255) if cls_name == "stop sign" else (0, 255, 0)
                label = f"{cls_name} ID:{track_id} {conf_val:.2f}"
                cv.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv.putText(annotated, label, (x1, max(y1 - 8, 0)),
                           cv.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        if no_object:
            cv.putText(annotated, "No selected object detected",
                       (20, 50), cv.FONT_HERSHEY_SIMPLEX,
                       1.0, (0, 0, 255), 2)

        # Afficher les compteurs sur la frame
        y_off = 30
        stats = {}
        for cls in self.unique_ids.values():
            stats[cls] = stats.get(cls, 0) + 1
        for cls_name, count in sorted(stats.items()):
            label = f"{cls_name}: {count} unique"
            cv.putText(annotated, label, (10, y_off),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y_off += 30

        self.frame_idx += 1
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")
    

st.set_page_config(page_title="Traffic Monitor", layout="wide")
st.title("🚦 Traffic Object Detection & Tracking")

# GPU status badge
if DEVICE == "cuda":
    st.success(f"⚡ GPU Active : {GPU_NAME}")
else:
    st.warning("⚠️ GPU not available – running on CPU")

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Configuration")

model_choice = st.sidebar.radio(
    "Model",
    ["YOLOv11 + ByteTrack (Base)", "YOLOv11 + ByteTrack (Fine-tuned)", "SSD MobileNetV3"],
    help="Choose base YOLO, fine-tuned YOLO, or SSD."
)

if model_choice == "YOLOv11 + ByteTrack (Base)":
    selected_yolo_path = YOLO_BASE_PATH
    st.sidebar.info("YOLO weights: base `models/yolo11n.pt`")
elif model_choice == "YOLOv11 + ByteTrack (Fine-tuned)":
    selected_yolo_path = YOLO_FINETUNED_PATH
    if os.path.exists(selected_yolo_path):
        st.sidebar.success("YOLO weights: fine-tuned `models/best.pt`")
    else:
        st.sidebar.error("`models/best.pt` not found. Train/copy your fine-tuned model first.")
else:
    selected_yolo_path = None

selected_classes = st.sidebar.multiselect(
    "Classes to detect",
    TRAFFIC_CLASSES,
    default=['car', 'truck', 'bus', 'traffic sign','person', 'traffic light']
)

confidence_threshold = st.sidebar.slider("Confidence threshold", 0.1, 1.0, 0.4)
min_box_area = st.sidebar.number_input("Min box area (px²)", min_value=0, value=1600, step=100)
min_track_hits = st.sidebar.number_input("Min track hits (YOLO only)", min_value=1, value=5, step=1)
shape_refine = st.sidebar.checkbox("Reclassify tall 'car' boxes as person", value=True)
person_car_ratio = st.sidebar.slider(
    "Person/Car shape ratio threshold (H/W)",
    min_value=0.8,
    max_value=2.0,
    value=1.2,
    step=0.05
)

uploaded_file = st.file_uploader(
    "📤 Upload a video (mp4, avi, mov)",
    type=["mp4", "avi", "mov"]
)
# ── Main ───────────────────────────────────────────────────────────────────────
if not selected_classes:
    st.warning("Select at least one class in the menu.")

elif uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    tfile.flush()

    col_video, col_stats = st.columns([2, 1])
    with col_video:
        st.subheader("📹 Annotated video")
        frame_display  = st.empty()
        status_display = st.empty()
    with col_stats:
        st.subheader("🔢 Unique object counters")
        counter_display = st.empty()
        progress_bar    = st.progress(0)

    if st.button("▶️ Start Detection", type="primary"):
        st.session_state.global_unique_ids = {}

        # ── YOLOv11 ─────────────────────────────────────────────────────────
        if model_choice in ("YOLOv11 + ByteTrack (Base)", "YOLOv11 + ByteTrack (Fine-tuned)"):
            if not os.path.exists(selected_yolo_path):
                st.error(f"Model file not found: `{selected_yolo_path}`")
                st.stop()

            model = YOLO(selected_yolo_path)
            cap   = cv.VideoCapture(tfile.name)
            fps   = cap.get(cv.CAP_PROP_FPS) or 25
            total = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

            #unique_ids = {}
            track_hits = {}
            logs       = []
            frame_idx  = 0
            log_path   = f"logs/log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            name_to_id = {v: k for k, v in model.names.items()}
            selected_ids = [name_to_id[c] for c in selected_classes if c in name_to_id]

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp = round(frame_idx / fps, 3)

                results = model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    conf=confidence_threshold,
                    classes=selected_ids if selected_ids else None,
                    device=DEVICE,          # ← GPU
                    verbose=False
                )

                no_object = True
                accepted = []

                if results[0].boxes is not None and results[0].boxes.id is not None:
                    for box in results[0].boxes:
                        track_id = int(box.id[0])
                        cls_name = normalize_class_name(model.names[int(box.cls[0])])
                        conf     = round(float(box.conf[0]), 3)
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        if shape_refine:
                            w = max(1, x2 - x1)
                            h = max(1, y2 - y1)
                            if cls_name == "car" and (h / w) > person_car_ratio:
                                cls_name = "person"
                        area = max(0, x2 - x1) * max(0, y2 - y1)

                        if cls_name not in selected_classes:
                            continue
                        if conf < confidence_threshold:
                            continue
                        if area < min_box_area:
                            continue

                        no_object = False
                        track_hits[track_id] = track_hits.get(track_id, 0) + 1
                        if track_hits[track_id] >= min_track_hits:
                            #unique_ids.setdefault(track_id, cls_name)
                            st.session_state.global_unique_ids.setdefault(track_id, cls_name)
                        accepted.append((track_id, cls_name, conf, x1, y1, x2, y2))
                        logs.append([timestamp, frame_idx, track_id,
                                     cls_name, x1, y1, x2, y2, conf])

                annotated = frame.copy()
                for track_id, cls_name, conf, x1, y1, x2, y2 in accepted:
                    color = (0, 0, 255) if cls_name == "stop sign" else (0, 255, 0)
                    thickness = 3 if cls_name == "stop sign" else 2
                    label = f"{cls_name} ID:{track_id} {conf:.2f}"
                    cv.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
                    cv.putText(annotated, label, (x1, max(y1 - 8, 0)),
                               cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if no_object:
                    cv.putText(annotated, "No selected object detected",
                               (20,50), cv.FONT_HERSHEY_SIMPLEX,
                               1.0, (0,0,255), 2)
                    status_display.warning("⚠️ No selected object in this frame")
                else:
                    status_display.empty()

                frame_display.image(
                    cv.cvtColor(annotated, cv.COLOR_BGR2RGB),
                    channels="RGB", use_column_width=True
                )

                stats = {}
                for cls in st.session_state.global_unique_ids.values():
                    stats[cls] = stats.get(cls, 0) + 1
                counter_display.markdown(
                    "\n".join([f"**{c}** : {n} unique" for c, n in sorted(stats.items())])
                    or "_Waiting for objects..._"
                )

                if total > 0:
                    progress_bar.progress(min(frame_idx / total, 1.0))
                frame_idx += 1

            cap.release()
            progress_bar.progress(1.0)

            with open(log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp_s','frame','track_id',
                                 'class','x1','y1','x2','y2','confidence'])
                writer.writerows(logs)

            st.success(f"✅ Done — {frame_idx} frames analyzed")

        # ── SSD ─────────────────────────────────────────────────────────────
        else:
            import torchvision
            from torchvision.transforms import functional as TF
            import numpy as np

            COCO_CLASSES = [
                "__background__","person","bicycle","car","motorcycle","airplane",
                "bus","train","truck","boat","traffic light","fire hydrant","N/A",
                "stop sign","parking meter","bench","bird","cat","dog","horse",
                "sheep","cow","elephant","bear","zebra","giraffe","N/A","backpack",
                "umbrella","N/A","N/A","handbag","tie","suitcase","frisbee","skis",
                "snowboard","sports ball","kite","baseball bat","baseball glove",
                "skateboard","surfboard","tennis racket","bottle","N/A","wine glass",
                "cup","fork","knife","spoon","bowl","banana","apple","sandwich",
                "orange","broccoli","carrot","hot dog","pizza","donut","cake",
                "chair","couch","potted plant","bed","N/A","dining table","N/A",
                "N/A","toilet","N/A","tv","laptop","mouse","remote","keyboard",
                "cell phone","microwave","oven","toaster","sink","refrigerator",
                "N/A","book","clock","vase","scissors","teddy bear","hair drier",
                "toothbrush"
            ]

            device = torch.device(DEVICE)
            weights = torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
            ssd_model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=weights)
            ssd_model.to(device)
            ssd_model.eval()

            cap       = cv.VideoCapture(tfile.name)
            fps       = cap.get(cv.CAP_PROP_FPS) or 25
            total     = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
            logs      = []
            frame_idx = 0
            log_path  = f"logs/ssd_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            np.random.seed(42)
            COLORS = np.random.randint(0, 255, size=(len(COCO_CLASSES), 3), dtype="uint8")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp = round(frame_idx / fps, 3)
                img_rgb   = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                tensor    = TF.to_tensor(img_rgb).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = ssd_model(tensor)[0]

                boxes_arr  = outputs["boxes"].cpu().numpy()
                labels_arr = outputs["labels"].cpu().numpy()
                scores_arr = outputs["scores"].cpu().numpy()

                no_object = True

                for box, label, score in zip(boxes_arr, labels_arr, scores_arr):
                    if score < confidence_threshold:
                        continue
                    cls_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else "unknown"
                    cls_name = normalize_class_name(cls_name)

                    x1, y1, x2, y2 = map(int, box)
                    if shape_refine:
                        w = max(1, x2 - x1)
                        h = max(1, y2 - y1)
                        if cls_name == "car" and (h / w) > person_car_ratio:
                            cls_name = "person"
                    if cls_name not in selected_classes:
                        continue
                    area = max(0, x2 - x1) * max(0, y2 - y1)
                    if area < min_box_area:
                        continue

                    no_object = False
                    color = (0,0,255) if cls_name == "stop sign" else [int(c) for c in COLORS[label]]

                    cv.rectangle(frame, (x1,y1),(x2,y2), color, 3 if cls_name=="stop sign" else 2)
                    label_text = "STOP SIGN" if cls_name=="stop sign" else f"{cls_name}: {score:.2f}"
                    cv.putText(frame, label_text, (x1, max(y1-8,0)),
                               cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    logs.append([timestamp, frame_idx, cls_name,
                                 x1, y1, x2, y2, round(float(score),3)])

                if no_object:
                    cv.putText(frame, "No selected object detected",
                               (20,50), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
                    status_display.warning("⚠️ No selected object in this frame")
                else:
                    status_display.empty()

                frame_display.image(
                    cv.cvtColor(frame, cv.COLOR_BGR2RGB),
                    channels="RGB", use_column_width=True
                )

                stats = {}
                for row in logs:
                    cls = row[2]
                    if cls not in stats :
                        stats[cls] = 1
                counter_display.markdown(
                    "\n".join([f"**{c}** : {n} unique" for c, n in sorted(stats.items())])
                    or "_Waiting..._"
                )

                if total > 0:
                    progress_bar.progress(min(frame_idx / total, 1.0))
                frame_idx += 1

            cap.release()
            progress_bar.progress(1.0)

            with open(log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp_s','frame','class',
                                 'x1','y1','x2','y2','confidence'])
                writer.writerows(logs)

            st.success(f"✅ Done — {frame_idx} frames analyzed")

        # ── Final Stats (common) ─────────────────────────────────────────────
        if logs:
            cols = (['timestamp_s','frame','track_id','class','x1','y1','x2','y2','confidence']
                    if model_choice in ("YOLOv11 + ByteTrack (Base)", "YOLOv11 + ByteTrack (Fine-tuned)")
                    else ['timestamp_s','frame','class','x1','y1','x2','y2','confidence'])
            df = pd.DataFrame(logs, columns=cols)

            st.subheader("📊 Final Statistics")
            ca, cb = st.columns(2)

            with ca:
                counts = df['class'].value_counts().reset_index()
                counts.columns = ['Class','Count']
                st.plotly_chart(
                    px.bar(counts, x='Class', y='Count',
                           title="Detections per class", color='Class'),
                    use_container_width=True
                )
            with cb:
                df['time_bin'] = (df['timestamp_s'] // 5) * 5
                intensity = df.groupby('time_bin')['class'].count().reset_index()
                intensity.columns = ['Time (s)','Detections']
                st.plotly_chart(
                    px.line(intensity, x='Time (s)', y='Detections',
                            title="Detection intensity over time"),
                    use_container_width=True
                )

            st.download_button(
                "📥 Download CSV logs",
                data=open(log_path).read(),
                file_name=os.path.basename(log_path),
                mime="text/csv"
            )

    os.unlink(tfile.name)

else:
    st.info("👆 Upload a video to start.")
