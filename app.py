"""
Streamlit Web Interface – Traffic Detection
Supports: YOLOv11 (tracking) | SSD MobileNetV3
GPU forced automatically.
Upload and processing are separated to avoid 403 errors.
CSV logs follow the shared schema.
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

# ── Session state init ─────────────────────────────────────────────────────────
if 'global_unique_ids' not in st.session_state:
    st.session_state.global_unique_ids = {}
if 'video_ready' not in st.session_state:
    st.session_state.video_ready = False
if 'tmp_path' not in st.session_state:
    st.session_state.tmp_path = None
if 'processing_done' not in st.session_state:
    st.session_state.processing_done = False

def refine_class_by_shape(cls_name, x1, y1, x2, y2, ratio=1.2):
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    if cls_name == "car" and (h / w) > ratio:
        return "person"
    return cls_name

# ── GPU Detection ──────────────────────────────────────────────────────────────
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
GPU_NAME = torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU only"

TRAFFIC_CLASSES     = ['car', 'truck', 'bus', 'motorbike', 'bicycle',
                       'person', 'traffic sign', 'traffic light']
YOLO_BASE_PATH      = "models/yolo11n.pt"
YOLO_FINETUNED_PATH = "models/best.pt"
os.makedirs("logs", exist_ok=True)

st.set_page_config(page_title="Traffic Monitor", layout="wide")
st.title("🚦 Traffic Object Detection & Tracking")

if DEVICE == "cuda":
    st.success(f"⚡ GPU Active : {GPU_NAME}")
else:
    st.warning("⚠️ GPU not available – running on CPU")

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Configuration")

model_choice = st.sidebar.radio(
    "Model",
    ["YOLOv11 + ByteTrack (Base)",
     "YOLOv11 + ByteTrack (Fine-tuned)",
     "SSD MobileNetV3"],
)

if model_choice == "YOLOv11 + ByteTrack (Base)":
    selected_yolo_path = YOLO_BASE_PATH
    st.sidebar.info("Weights: `models/yolo11n.pt`")
elif model_choice == "YOLOv11 + ByteTrack (Fine-tuned)":
    selected_yolo_path = YOLO_FINETUNED_PATH
    if os.path.exists(selected_yolo_path):
        st.sidebar.success("Weights: `models/best.pt` ✅")
    else:
        st.sidebar.error("`models/best.pt` not found.")
else:
    selected_yolo_path = None

selected_classes     = st.sidebar.multiselect(
    "Classes to detect", TRAFFIC_CLASSES,
    default=['car', 'truck', 'bus', 'person', 'traffic light']
)
confidence_threshold = st.sidebar.slider("Confidence threshold", 0.1, 1.0, 0.4)
min_box_area         = st.sidebar.number_input("Min box area (px²)", min_value=0, value=1600, step=100)
min_track_hits       = st.sidebar.number_input("Min track hits (YOLO only)", min_value=1, value=5, step=1)
shape_refine         = st.sidebar.checkbox("Reclassify tall 'car' as person", value=True)
frame_skip           = st.sidebar.slider("Process every N frames", 1, 10, 2)
person_car_ratio     = st.sidebar.slider("Person/Car ratio (H/W)", 0.8, 2.0, 1.2, 0.05)

# ── Métadonnées pour le schéma partagé ────────────────────────────────────────
st.sidebar.divider()
st.sidebar.subheader("📋 Log Metadata")
scene_name = st.sidebar.text_input("Scene name", value="scene_01",
                                    placeholder="intersection_A")
group_id   = st.sidebar.text_input("Group ID",   value="group_01",
                                    placeholder="group_01")

if not selected_classes:
    st.warning("Select at least one class in the sidebar.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 1 — UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("📤 Step 1 — Upload your video")

uploaded_file = st.file_uploader(
    "Choose a video file (mp4, avi, mov)",
    type=["mp4", "avi", "mov"]
)

if uploaded_file is not None:
    save_path = "temp_video.mp4"

    if not st.session_state.video_ready or st.session_state.tmp_path != save_path:
        with st.spinner("Saving video to server..."):
            with open(save_path, "wb") as f:
                uploaded_file.seek(0)
                while True:
                    chunk = uploaded_file.read(1024 * 1024)  # 1MB
                    if not chunk:
                        break
                    f.write(chunk)

            st.session_state.tmp_path          = save_path
            st.session_state.video_ready       = True
            st.session_state.global_unique_ids = {}
            st.session_state.processing_done   = False

    st.success(f"✅ Video ready: **{uploaded_file.name}**")

    cap_info = cv.VideoCapture(st.session_state.tmp_path)
    if not cap_info.isOpened():
        st.error("Error: Could not open video file.")
    else:
        fps_info   = cap_info.get(cv.CAP_PROP_FPS)
        w_info     = int(cap_info.get(cv.CAP_PROP_FRAME_WIDTH))
        h_info     = int(cap_info.get(cv.CAP_PROP_FRAME_HEIGHT))
        total_info = int(cap_info.get(cv.CAP_PROP_FRAME_COUNT))
        cap_info.release()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Resolution", f"{w_info}x{h_info}")
        c2.metric("FPS", f"{fps_info:.1f}")
        c3.metric("Frames", total_info)
        c4.metric("Duration", f"{total_info/fps_info:.1f}s" if fps_info > 0 else "N/A")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # ÉTAPE 2 — TRAITEMENT
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("▶️ Step 2 — Run Detection")

    start = st.button("▶️ Start Detection", type="primary",
                      disabled=st.session_state.processing_done)

    if start:
        st.session_state.processing_done   = False
        st.session_state.global_unique_ids = {}

        col_video, col_stats = st.columns([2, 1])
        with col_video:
            st.subheader("📹 Annotated video")
            frame_display  = st.empty()
            status_display = st.empty()
        with col_stats:
            st.subheader("🔢 Unique counters")
            counter_display = st.empty()
            progress_bar    = st.progress(0)

        tmp_path   = st.session_state.tmp_path
        video_name = uploaded_file.name
        logs       = []

        # ── YOLOv11 ───────────────────────────────────────────────────────────
        if model_choice in ("YOLOv11 + ByteTrack (Base)",
                            "YOLOv11 + ByteTrack (Fine-tuned)"):
            if not os.path.exists(selected_yolo_path):
                st.error(f"Model not found: `{selected_yolo_path}`")
                st.stop()

            model        = YOLO(selected_yolo_path)
            cap          = cv.VideoCapture(tmp_path)
            fps          = cap.get(cv.CAP_PROP_FPS) or 25
            total        = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
            w_vid        = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
            h_vid        = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            line_y       = h_vid // 2
            track_hits   = {}
            prev_pos     = {}
            frame_idx    = 0
            log_path     = f"logs/log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            name_to_id   = {v: k for k, v in model.names.items()}
            selected_ids = [name_to_id[c] for c in selected_classes if c in name_to_id]

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_skip != 0:
                    frame_idx += 1
                    continue

                timestamp = round(frame_idx / fps, 3)
                results   = model.track(
                    frame, persist=True, tracker="bytetrack.yaml",
                    conf=confidence_threshold,
                    classes=selected_ids if selected_ids else None,
                    device=DEVICE, verbose=False
                )

                no_object = True
                accepted  = []

                if results[0].boxes is not None and results[0].boxes.id is not None:
                    for box in results[0].boxes:
                        track_id = int(box.id[0])
                        cls_name = normalize_class_name(model.names[int(box.cls[0])])
                        conf     = round(float(box.conf[0]), 3)
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        if shape_refine:
                            cls_name = refine_class_by_shape(
                                cls_name, x1, y1, x2, y2, person_car_ratio)

                        area = max(0, x2-x1) * max(0, y2-y1)
                        if cls_name not in selected_classes: continue
                        if conf < confidence_threshold:       continue
                        if area < min_box_area:               continue

                        no_object = False
                        track_hits[track_id] = track_hits.get(track_id, 0) + 1
                        if track_hits[track_id] >= min_track_hits:
                            st.session_state.global_unique_ids.setdefault(
                                track_id, cls_name)

                        # ── Schéma partagé ─────────────────────────────────
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2

                        if track_id in prev_pos:
                            px_, py_ = prev_pos[track_id]
                            dist  = ((cx-px_)**2 + (cy-py_)**2) ** 0.5
                            speed = round(dist * fps, 2)
                            if abs(cx-px_) > abs(cy-py_):
                                direction = "right" if cx > px_ else "left"
                            else:
                                direction = "down" if cy > py_ else "up"
                        else:
                            speed, direction = 0.0, ""
                        prev_pos[track_id] = (cx, cy)

                        crossed = "true" if abs(cy - line_y) < 10 else "false"

                        accepted.append((track_id, cls_name, conf, x1, y1, x2, y2))
                        logs.append([
                            frame_idx, timestamp,
                            scene_name, group_id, video_name,
                            track_id, cls_name, conf,
                            x1, y1, x2, y2,
                            cx, cy, w_vid, h_vid,
                            crossed, direction, speed
                        ])

                annotated = frame.copy()

                # Ligne de comptage
                cv.line(annotated, (0, line_y), (w_vid, line_y), (0, 255, 0), 2)
                cv.putText(annotated, "Counting line", (10, line_y - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                for track_id, cls_name, conf, x1, y1, x2, y2 in accepted:
                    color     = (0, 0, 255) if cls_name == "stop sign" else (0, 255, 0)
                    thickness = 3 if cls_name == "stop sign" else 2
                    label     = f"{cls_name} ID:{track_id} {conf:.2f}"
                    cv.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
                    cv.putText(annotated, label, (x1, max(y1-8, 0)),
                               cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if no_object:
                    cv.putText(annotated, "No selected object detected",
                               (20, 50), cv.FONT_HERSHEY_SIMPLEX,
                               1.0, (0, 0, 255), 2)
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
                    "\n".join([f"**{c}** : {n} unique"
                               for c, n in sorted(stats.items())])
                    or "_Waiting for objects..._"
                )

                if total > 0:
                    progress_bar.progress(min(frame_idx / total, 1.0))
                frame_idx += 1

            cap.release()
            progress_bar.progress(1.0)

            with open(log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'frame', 'timestamp_sec', 'scene_name', 'group_id', 'video_name',
                    'track_id', 'class_name', 'confidence',
                    'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2',
                    'cx', 'cy', 'frame_width', 'frame_height',
                    'crossed_line', 'direction', 'speed_px_s'
                ])
                writer.writerows(logs)

            st.success(f"✅ Done — {frame_idx} frames analyzed")

        # ── SSD ───────────────────────────────────────────────────────────────
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

            device    = torch.device(DEVICE)
            weights   = torchvision.models.detection\
                        .SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
            ssd_model = torchvision.models.detection\
                        .ssdlite320_mobilenet_v3_large(weights=weights)
            ssd_model.to(device)
            ssd_model.eval()

            cap       = cv.VideoCapture(tmp_path)
            fps       = cap.get(cv.CAP_PROP_FPS) or 25
            total     = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
            w_vid     = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
            h_vid     = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            line_y    = h_vid // 2
            frame_idx = 0
            log_path  = f"logs/ssd_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            np.random.seed(42)
            COLORS       = np.random.randint(0, 255, size=(len(COCO_CLASSES), 3),
                                             dtype="uint8")
            seen_classes = set()
            prev_pos     = {}

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_skip != 0:
                    frame_idx += 1
                    continue

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
                    cls_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) \
                               else "unknown"
                    cls_name = normalize_class_name(cls_name)

                    x1, y1, x2, y2 = map(int, box)
                    if shape_refine:
                        cls_name = refine_class_by_shape(
                            cls_name, x1, y1, x2, y2, person_car_ratio)

                    area = max(0, x2-x1) * max(0, y2-y1)
                    if cls_name not in selected_classes: continue
                    if area < min_box_area:               continue

                    no_object = False
                    seen_classes.add(cls_name)

                    # ── Schéma partagé ─────────────────────────────────────
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    key = int(label)

                    if key in prev_pos:
                        px_, py_ = prev_pos[key]
                        dist  = ((cx-px_)**2 + (cy-py_)**2) ** 0.5
                        speed = round(dist * fps, 2)
                        if abs(cx-px_) > abs(cy-py_):
                            direction = "right" if cx > px_ else "left"
                        else:
                            direction = "down" if cy > py_ else "up"
                    else:
                        speed, direction = 0.0, ""
                    prev_pos[key] = (cx, cy)

                    crossed = "true" if abs(cy - line_y) < 10 else "false"

                    color = (0,0,255) if cls_name == "stop sign" \
                            else [int(c) for c in COLORS[label]]
                    cv.rectangle(frame, (x1,y1),(x2,y2), color,
                                 3 if cls_name=="stop sign" else 2)
                    label_text = "STOP SIGN" if cls_name=="stop sign" \
                                 else f"{cls_name}: {score:.2f}"
                    cv.putText(frame, label_text, (x1, max(y1-8,0)),
                               cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    logs.append([
                        frame_idx, timestamp,
                        scene_name, group_id, video_name,
                        -1, cls_name, round(float(score), 3),
                        x1, y1, x2, y2,
                        cx, cy, w_vid, h_vid,
                        crossed, direction, speed
                    ])

                # Ligne de comptage
                cv.line(frame, (0, line_y), (w_vid, line_y), (0, 255, 0), 2)
                cv.putText(frame, "Counting line", (10, line_y - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if no_object:
                    cv.putText(frame, "No selected object detected",
                               (20,50), cv.FONT_HERSHEY_SIMPLEX,
                               1.0, (0,0,255), 2)
                    status_display.warning("⚠️ No selected object in this frame")
                else:
                    status_display.empty()

                frame_display.image(
                    cv.cvtColor(frame, cv.COLOR_BGR2RGB),
                    channels="RGB", use_column_width=True
                )

                counter_display.markdown(
                    "\n".join([f"**{c}** : 1 unique" for c in sorted(seen_classes)])
                    or "_Waiting..._"
                )

                if total > 0:
                    progress_bar.progress(min(frame_idx / total, 1.0))
                frame_idx += 1

            cap.release()
            progress_bar.progress(1.0)

            with open(log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'frame', 'timestamp_sec', 'scene_name', 'group_id', 'video_name',
                    'track_id', 'class_name', 'confidence',
                    'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2',
                    'cx', 'cy', 'frame_width', 'frame_height',
                    'crossed_line', 'direction', 'speed_px_s'
                ])
                writer.writerows(logs)

            st.success(f"✅ Done — {frame_idx} frames analyzed")

        st.session_state.processing_done = True

        # ── Final Stats ────────────────────────────────────────────────────────
        if logs:
            cols = [
                'frame', 'timestamp_sec', 'scene_name', 'group_id', 'video_name',
                'track_id', 'class_name', 'confidence',
                'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2',
                'cx', 'cy', 'frame_width', 'frame_height',
                'crossed_line', 'direction', 'speed_px_s'
            ]
            df = pd.DataFrame(logs, columns=cols)

            st.subheader("📊 Final Statistics")
            ca, cb = st.columns(2)
            with ca:
                counts = df['class_name'].value_counts().reset_index()
                counts.columns = ['Class', 'Count']
                st.plotly_chart(
                    px.bar(counts, x='Class', y='Count',
                           title="Detections per class", color='Class'),
                    use_container_width=True
                )
            with cb:
                df['time_bin'] = (df['timestamp_sec'] // 5) * 5
                intensity = df.groupby('time_bin')['class_name'].count().reset_index()
                intensity.columns = ['Time (s)', 'Detections']
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

            if st.button("🔄 Process another video"):
                st.session_state.video_ready       = False
                st.session_state.tmp_path          = None
                st.session_state.processing_done   = False
                st.session_state.global_unique_ids = {}
                st.rerun()

else:
    st.session_state.video_ready     = False
    st.session_state.tmp_path        = None
    st.session_state.processing_done = False
    st.info("👆 Upload a video to start.")
