# =============================================
#   SMART OBJECT DETECTION DASHBOARD (IMPROVED)
# =============================================

import os
import time
import math
from io import BytesIO
from datetime import datetime
from collections import Counter, defaultdict

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import altair as alt
from PIL import Image
from ultralytics import YOLO

# ----------------- CONFIG & CONSTANTS -----------------
DB_DIR = "db"
DB_NAME = os.path.join(DB_DIR, "sql_lite_db.db")
TEMP_MODEL_PATH = "models/uploaded_model.pt"
LOGGING_INTERVAL_FRAMES = 5  # only write to DB every N frames to avoid I/O bottleneck

# Ensure folders exist
os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(os.path.dirname(TEMP_MODEL_PATH), exist_ok=True)

# ----------------- DATABASE FUNCTIONS -----------------
import sqlite3


def get_conn():
    # check_same_thread=False avoids some issues when Streamlit reruns in different threads
    return sqlite3.connect(DB_NAME, check_same_thread=False)


def init_db():
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            Time_Stamp TEXT NOT NULL,
            Object_Name TEXT,
            Count INT,
            Confidence REAL
        );
    """)
    conn.commit()
    conn.close()


def insert_log(object_name, count_value, confidence):
    # lightweight insert
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
        INSERT INTO detections (Time_Stamp, Object_Name, Count, Confidence)
        VALUES (?, ?, ?, ?)
    """, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), object_name, count_value, confidence))
    conn.commit()
    conn.close()


def read_logs():
    conn = get_conn()
    df = pd.read_sql("SELECT * FROM detections ORDER BY ID DESC", conn)
    conn.close()
    return df


def export_excel(df):
    buffer = BytesIO()
    # pandas uses openpyxl by default for xlsx; ensure openpyxl is in requirements
    df.to_excel(buffer, index=False)
    buffer.seek(0)
    return buffer


# ----------------- MODEL LOADING -----------------
@st.cache_resource
def load_yolo_model_from_path(path: str):
    # caching prevents repeated expensive loads on reruns
    return YOLO(path)


def load_yolo_model_ui():
    uploaded_model = st.sidebar.file_uploader("Upload YOLO model (.pt)", type=["pt"])
    if uploaded_model is not None:
        with open(TEMP_MODEL_PATH, "wb") as f:
            f.write(uploaded_model.read())
        model = load_yolo_model_from_path(TEMP_MODEL_PATH)
        st.sidebar.success("Custom model loaded successfully!")
    else:
        # default to yolov8n if available in the environment
        model = load_yolo_model_from_path("yolov8n.pt")
        st.sidebar.info("Using default YOLO model (yolov8n.pt)")
    return model


# ----------------- GLOW EFFECT (DARK MODE) -----------------
def update_glow_phase():
    st.session_state.glow_phase += 0.15


def get_glow_color():
    phase = st.session_state.glow_phase
    r = 255
    g = int(140 + 115 * abs(math.sin(phase)))
    b = 0
    update_glow_phase()
    return (b, g, r)


# ----------------- OBJECT DETECTION CORE -----------------
def detect_objects(frame, model, dark_mode=False):
    # frame expected in RGB
    alerts = []
    annotated_frame = frame.copy()

    try:
        results = model(frame)
    except Exception as e:
        st.error(f"YOLO inference failed: {e}")
        return annotated_frame, alerts

    # update counters only for this frame
    frame_counter = Counter()
    frame_conf = defaultdict(list)

    for r in results:
        for box in r.boxes:
            try:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
            except Exception:
                continue

            object_name = model.names.get(cls, str(cls))

            color = (0, 255, 0) if not dark_mode else get_glow_color()
            label = f"{object_name} {conf:.2f}"

            # draw on RGB frame (we convert earlier)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255) if dark_mode else (0, 0, 0), 2)

            frame_counter[object_name] += 1
            frame_conf[object_name].append(conf)

    # merge per-frame counters into session counters
    for obj, cnt in frame_counter.items():
        st.session_state.counter[obj] = st.session_state.counter.get(obj, 0) + cnt
        st.session_state.confidence[obj].extend(frame_conf[obj])

        # log to DB only every LOGGING_INTERVAL_FRAMES frames to reduce I/O
        if st.session_state.frames % LOGGING_INTERVAL_FRAMES == 0:
            # log average confidence for the object in this batch
            avg_conf = float(np.mean(frame_conf[obj])) if frame_conf[obj] else 0.0
            insert_log(obj, st.session_state.counter[obj], avg_conf)

        if st.session_state.counter[obj] > 3:
            alerts.append(f"High number of {obj} detected!")

    st.session_state.frames += 1
    return annotated_frame, alerts


# ----------------- LIVE STATS UI -----------------
def display_live_stats(alerts):
    col1, col2 = st.columns([3, 1])
    frame_ph = col1.empty()
    stats_exp = col2.expander("Live Stats", expanded=True)
    chart_ph = stats_exp.empty()
    table_ph = stats_exp.empty()
    alert_ph = col2.empty()

    if alerts:
        for alert in alerts:
            alert_ph.warning(alert)

    return frame_ph, chart_ph, table_ph


# ----------------- UPDATE CHART & TABLE -----------------
def update_stats_display(chart_ph, table_ph):
    if st.session_state.counter:
        data = []
        for obj, count in st.session_state.counter.items():
            avg_conf = round(np.mean(st.session_state.confidence[obj]) if st.session_state.confidence[obj] else 0.0, 2)
            data.append({"Object": obj, "Count": count, "Avg Confidence": avg_conf})

        df = pd.DataFrame(data)
        df["Frames Processed"] = st.session_state.frames

        table_ph.table(df)

        chart_color = "#FFA500" if st.session_state.get("dark_mode", False) else "#32CD32"
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X("Object", sort=None),
            y="Count",
            tooltip=["Object", "Count", "Avg Confidence"]
        ).properties(title="Detected Objects Count")
        chart_ph.altair_chart(chart, width="stretch")


# ----------------- CAMERA STREAM HANDLING -----------------
def start_camera_capture(source):
    # create and store VideoCapture so we can reuse between reruns
    if "cap" not in st.session_state or st.session_state.cap_source != str(source):
        try:
            cap = cv2.VideoCapture(source)
            st.session_state.cap = cap
            st.session_state.cap_source = str(source)
        except Exception as e:
            st.error(f"Failed to open camera: {e}")
            st.session_state.cap = None


def stop_camera_capture():
    if "cap" in st.session_state and st.session_state.cap is not None:
        try:
            st.session_state.cap.release()
        except Exception:
            pass
        del st.session_state.cap
        st.session_state.cap = None
        st.session_state.cap_source = None


def read_one_frame_from_capture():
    cap = st.session_state.get("cap", None)
    if cap is None:
        return None
    ret, frame = cap.read()
    if not ret:
        return None
    return frame


# ----------------- IMAGE UPLOAD PROCESSING -----------------
def process_uploaded_image(image_file, model, dark_mode):
    image = Image.open(image_file).convert("RGB")
    frame = np.array(image)
    annotated_frame, alerts = detect_objects(frame, model, dark_mode)

    st.image(annotated_frame, caption="Detection Result", use_column_width=True)
    if alerts:
        for a in alerts:
            st.warning(a)

    update_stats_display(st.session_state.get('chart_ph', st.empty()), st.session_state.get('table_ph', st.empty()))


# ----------------- APPLY THEME CSS -----------------
def apply_theme(theme):
    if theme == "Dark Mode":
        st.markdown("""
        <style>
        .stApp {background: linear-gradient(135deg, #0f2027, #203a43, #2c5364); color:white;}
        h1,h2,h3,h4 {color:#FFD700; text-shadow:2px 2px 8px #000000;}
        div.stButton>button {font-size:20px; font-weight:bold; color:white; background: linear-gradient(to right, #FF512F, #DD2476); border-radius:10px; padding:12px; transition: transform 0.2s;} div.stButton>button:hover {transform: scale(1.05);}
        div[data-baseweb="tab-list"] button {color:white; font-size:18px; font-weight:bold; background:#203a43; border-radius:8px; margin:0 2px;}
        div[data-baseweb="tab-list"] button[aria-selected="true"] {color:#FFD700 !important; background:#2c5364 !important; box-shadow:0 0 10px #FFD700;}
        .stMarkdown p {font-size:18px; line-height:1.6; color:white;}
        div[role="radiogroup"] label { color: white !important; font-weight: 600 !important; }
        div.stDownloadButton>button { color: white !important; background: linear-gradient(to right, #FF512F, #DD2476); }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        .stApp {background-color:#f0f2f6; color:black;}
        div.stButton>button {font-size:20px; font-weight:bold; color:white; background: linear-gradient(to right, #36D1DC, #5B86E5); border-radius:10px; padding:12px; transition: transform 0.2s;} div.stButton>button:hover {transform: scale(1.05);}
        .stMarkdown p {font-size:18px; line-height:1.6; color:black;}
        </style>
        """, unsafe_allow_html=True)


# ----------------- LOGS TAB UI -----------------
def show_logs_tab():
    st.markdown("## Detection Logs")
    df_logs = read_logs()

    filter_option = st.radio("Filter logs by:", ["All Data", "Object Name", "Date Range"]) 
    df_filtered = df_logs.copy()

    if filter_option == "Object Name":
        object_list = df_logs['Object_Name'].unique().tolist() if not df_logs.empty else []
        selected_object = st.selectbox("Select Object", object_list)
        if selected_object:
            df_filtered = df_logs[df_logs['Object_Name'] == selected_object]

    elif filter_option == "Date Range":
        if df_logs.empty:
            st.info("No logs available yet.")
        else:
            min_date = pd.to_datetime(df_logs['Time_Stamp'].min())
            max_date = pd.to_datetime(df_logs['Time_Stamp'].max())
            start_date, end_date = st.date_input("Select Date Range", [min_date, max_date])
            df_filtered['Time_Stamp'] = pd.to_datetime(df_filtered['Time_Stamp'])
            df_filtered = df_filtered[(df_filtered['Time_Stamp'] >= pd.to_datetime(start_date)) &
                                      (df_filtered['Time_Stamp'] <= pd.to_datetime(end_date))]

    st.dataframe(df_filtered, width="stretch")
    excel_file = export_excel(df_filtered)
    st.download_button("Download Logs as Excel", excel_file, file_name="detection_logs.xlsx")


# ----------------- MAIN APP -----------------
def main():
    st.set_page_config(page_title="Smart Object Detection", layout="wide", page_icon="📷")

    # Initialize
    init_db()
    if "run" not in st.session_state: st.session_state.run = False
    if "counter" not in st.session_state: st.session_state.counter = {}
    if "confidence" not in st.session_state: st.session_state.confidence = defaultdict(list)
    if "frames" not in st.session_state: st.session_state.frames = 0
    if "glow_phase" not in st.session_state: st.session_state.glow_phase = 0
    if "last_time" not in st.session_state: st.session_state.last_time = time.time()

    # Sidebar
    st.sidebar.title("Control Panel")
    theme = st.sidebar.radio("Choose Theme", ["Light Mode", "Dark Mode"] )
    st.session_state.dark_mode = (theme == "Dark Mode")
    apply_theme(theme)

    st.session_state.model = load_yolo_model_ui()

    source_option = st.sidebar.radio("Input Source", ["USB Camera", "IP Camera", "Upload Image"]) 
    start_btn = st.sidebar.button("Start Detection")
    stop_btn = st.sidebar.button("Stop Detection")

    st.title("Smart Object Detection Dashboard")

    # Toolbar FPS
    fps_display = st.sidebar.empty()

    # Input Handling
    if source_option == "USB Camera":
        source = 0
    elif source_option == "IP Camera":
        source = st.sidebar.text_input("Enter IP camera URL")
    else:
        source = None

    if start_btn:
        st.session_state.run = True
        if source is not None:
            start_camera_capture(source)

    if stop_btn:
        st.session_state.run = False
        stop_camera_capture()

    # Camera / Image handling
    if source_option == "Upload Image":
        uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded:
            # reset counters for image inference session
            st.session_state.counter = {}
            st.session_state.confidence = defaultdict(list)
            st.session_state.frames = 0
            process_uploaded_image(uploaded, st.session_state.model, st.session_state.dark_mode)
    else:
        # live stream area placeholders
        frame_ph = st.empty()
        st.session_state['chart_ph'] = st.empty()
        st.session_state['table_ph'] = st.empty()

        if st.session_state.run:
            # Non-blocking single-frame loop: read a frame, display it, then rerun the app so Streamlit yields control.
            if "cap" not in st.session_state or st.session_state.cap is None:
                start_camera_capture(source)

            frame = read_one_frame_from_capture()
            if frame is None:
                st.warning("No frame available from the camera. Please check the source.")
            else:
                # convert BGR -> RGB once
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                annotated_frame, alerts = detect_objects(rgb_frame, st.session_state.model, st.session_state.dark_mode)

                # show frame
                frame_ph.image(annotated_frame, channels="RGB")

                # show alerts
                for a in alerts:
                    st.warning(a)

                # update charts/tables
                update_stats_display(st.session_state['chart_ph'], st.session_state['table_ph'])

                # fps calc
                now = time.time()
                fps = 1.0 / (now - st.session_state.last_time) if now != st.session_state.last_time else 0.0
                st.session_state.last_time = now
                fps_display.write(f"FPS: {fps:.2f}")

            # small sleep to avoid 100% CPU — this yields to Streamlit between reruns
            time.sleep(0.03)
            # trigger a lightweight rerun to fetch the next frame
            st.experimental_rerun()

    # Tabs
    about_tabs = st.tabs(["Overview", "How It Works", "Model Info", "Tips & Tricks", "Future Improvements", "References", "Logs"])

    with about_tabs[0]:
        st.markdown("**Project Overview:** YOLOv8 detection, glowing bounding boxes in Dark Mode, live stats, alerts, modern cinematic dashboard.")

    with about_tabs[1]:
        st.markdown("- Capture frame from camera or image.\n- YOLOv8 detects objects.\n- Display frame with glowing bounding boxes.\n- Update live stats and charts.")

    with about_tabs[2]:
        st.markdown("- YOLOv8 nano model (`yolov8n.pt`) - lightweight & fast.")

    with about_tabs[3]:
        st.markdown("- Good lighting improves detection accuracy.\n- Keep camera steady.\n- High-resolution images improve results.")

    with about_tabs[4]:
        st.markdown("- Support multiple cameras simultaneously.\n- Add sound alerts.\n- Recording feature.\n- Cloud storage & analysis.")

    with about_tabs[5]:
        st.markdown("- Ultralytics YOLO\n- Computer Vision & Deep Learning references.")

    with about_tabs[6]:
        show_logs_tab()


if __name__ == "__main__":
    main()
