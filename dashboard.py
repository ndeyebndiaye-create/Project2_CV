import streamlit as st
import pandas as pd
import plotly.express as px
import glob
import os

st.set_page_config(page_title="Traffic Dashboard", layout="wide")
st.title("Traffic Dashboard — Multi-scene comparison")

# ── Load logs ──────────────────────────────────────────────────────────────────
log_files = sorted(glob.glob("logs/*.csv"))

if not log_files:
    st.warning("No logs found. Run a detection first via app.py or main.py.")
    st.stop()

dfs = []
for f in log_files:
    df = pd.read_csv(f)
    df['scene'] = os.path.basename(f).replace(".csv", "")
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)

# Handle both YOLO (has track_id) and SSD (no track_id) log formats
has_tracking = 'track_id' in data.columns
if has_tracking:
    unique = data.drop_duplicates(subset=['scene', 'track_id'])
else:
    unique = data.copy()

# ── Global metrics ─────────────────────────────────────────────────────────────
st.subheader("Global overview")
c1, c2, c3 = st.columns(3)
c1.metric("Total unique objects", len(unique))
c2.metric("Scenes analyzed",      len(log_files))
c3.metric("Classes detected",     data['class'].nunique())

st.divider()

# ── Scene selector ─────────────────────────────────────────────────────────────
scenes = ["All"] + sorted(data['scene'].unique().tolist())
selected_scene = st.selectbox("Filter by scene", scenes)

filtered = data if selected_scene == "All" else data[data['scene'] == selected_scene]
filtered_unique = unique if selected_scene == "All" else unique[unique['scene'] == selected_scene]

# ── Charts ─────────────────────────────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Detections by class")
    pie_data = filtered_unique['class'].value_counts().reset_index()
    pie_data.columns = ['Class', 'Count']
    st.plotly_chart(
        px.pie(pie_data, values='Count', names='Class', hole=0.3),
        use_container_width=True
    )

with col_b:
    st.subheader("Unique objects per scene and class")
    bar_data = filtered_unique.groupby(['scene', 'class']).size().reset_index(name='count')
    st.plotly_chart(
        px.bar(bar_data, x='scene', y='count', color='class', barmode='group'),
        use_container_width=True
    )

# ── Traffic intensity over time ────────────────────────────────────────────────
st.subheader("Traffic intensity over time")
filtered['time_bin'] = (filtered['timestamp_s'] // 10) * 10

if has_tracking:
    intensity = (filtered.groupby(['time_bin', 'scene'])['track_id']
                          .nunique().reset_index(name='active_objects'))
else:
    intensity = (filtered.groupby(['time_bin', 'scene'])['class']
                          .count().reset_index(name='active_objects'))

st.plotly_chart(
    px.line(intensity, x='time_bin', y='active_objects', color='scene',
            labels={'time_bin': 'Time (s)', 'active_objects': 'Active objects'}),
    use_container_width=True
)

# ── Confidence distribution ────────────────────────────────────────────────────
if 'confidence' in filtered.columns:
    st.subheader("Confidence score distribution")
    st.plotly_chart(
        px.histogram(filtered, x='confidence', color='class',
                     nbins=30, barmode='overlay',
                     labels={'confidence': 'Confidence score'}),
        use_container_width=True
    )

# ── Raw data ───────────────────────────────────────────────────────────────────
with st.expander("Raw data"):
    st.dataframe(filtered, use_container_width=True)

# ── Export ─────────────────────────────────────────────────────────────────────
st.download_button(
    "Download global CSV",
    data=filtered.to_csv(index=False),
    file_name="global_traffic_dataset.csv",
    mime="text/csv"
)
