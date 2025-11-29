# =============================================
#   SMART OBJECT DETECTION DASHBOARD 
#   FIXED FOR DEPLOYMENT WITH STREAMLIT-WEBRTC
# =============================================

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from collections import Counter, defaultdict
import pandas as pd
import altair as alt
from ultralytics import YOLO
import time
import math
from io import BytesIO
from datetime import datetime
import sqlite3
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# ----------------- CONFIG & CONSTANTS -----------------
DB_NAME = "db/sql_lite_db.db"
TEMP_MODEL_PATH = "models/uploaded_model.pt"

# RTC Configuration for deployment
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ----------------- DATABASE FUNCTIONS -----------------
def init_db():
    conn = sqlite3.connect(DB_NAME)
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
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        INSERT INTO detections (Time_Stamp, Object_Name, Count, Confidence)
        VALUES (?, ?, ?, ?)
    """, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), object_name, count_value, confidence))
    conn.commit()
    conn.close()

def read_logs():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql("SELECT * FROM detections ORDER BY ID DESC", conn)
    conn.close()
    return df

def export_excel(df):
    buffer = BytesIO()
    df.to_excel(buffer, index=False)
    buffer.seek(0)
    return buffer

# ----------------- MODEL LOADING -----------------
def load_yolo_model():
    uploaded_model = st.sidebar.file_uploader("Upload YOLO model (.pt)", type=["pt"])
    if uploaded_model is not None:
        with open(TEMP_MODEL_PATH, "wb") as f:
            f.write(uploaded_model.read())
        model = YOLO(TEMP_MODEL_PATH)
        st.sidebar.success("Custom model loaded successfully!")
    else:
        model = YOLO("yolov8n.pt")
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

# ----------------- VIDEO TRANSFORMER CLASS -----------------
class YOLOVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = st.session_state.model
        self.dark_mode = st.session_state.get("dark_mode", False)
        self.counter = Counter()
        self.confidence = defaultdict(list)
        self.frame_count = 0
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Run YOLO detection
        results = self.model(img)
        annotated_frame = img.copy()
        
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                object_name = self.model.names[cls]
                
                color = (0, 255, 0) if not self.dark_mode else get_glow_color()
                label = f"{object_name} {conf:.2f}"
                
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                           (255, 255, 255) if self.dark_mode else (0, 0, 0), 2)
                
                current_count = self.counter[object_name] + 1
                self.counter[object_name] = current_count
                self.confidence[object_name].append(conf)
                
                # Log detection
                try:
                    insert_log(object_name, current_count, conf)
                except:
                    pass  # Skip if database is locked
        
        self.frame_count += 1
        
        # Update session state
        st.session_state.counter = self.counter
        st.session_state.confidence = self.confidence
        st.session_state.frames = self.frame_count
        
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# ----------------- OBJECT DETECTION FOR IMAGES -----------------
def detect_objects_image(frame, model, dark_mode=False):
    results = model(frame)
    counter = Counter()
    confidence = defaultdict(list)
    alerts = []
    annotated_frame = frame.copy()

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            object_name = model.names[cls]

            color = (0, 255, 0) if not dark_mode else get_glow_color()
            label = f"{object_name} {conf:.2f}"

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255) if dark_mode else (0, 0, 0), 2)

            current_count = counter[object_name] + 1
            counter[object_name] = current_count
            confidence[object_name].append(conf)

            insert_log(object_name, current_count, conf)

            if current_count > 3:
                alerts.append(f"High number of {object_name} detected!")

    return annotated_frame, counter, confidence, alerts

# ----------------- UPDATE CHART & TABLE -----------------
def display_stats(counter, confidence, frame_count=0):
    if counter:
        data = []
        for obj, count in counter.items():
            avg_conf = round(np.mean(confidence[obj]), 2)
            data.append({"Object": obj, "Count": count, "Avg Confidence": avg_conf})

        df = pd.DataFrame(data)
        df["Frames Processed"] = frame_count

        st.table(df)

        chart_color = "#FFA500" if st.session_state.get("dark_mode", False) else "#32CD32"
        chart = alt.Chart(df).mark_bar(color=chart_color).encode(
            x=alt.X("Object", sort=None),
            y="Count",
            tooltip=["Object", "Count", "Avg Confidence"]
        ).properties(title="Detected Objects Count")
        st.altair_chart(chart, use_container_width=True)

# ----------------- IMAGE UPLOAD PROCESSING -----------------
def process_uploaded_image(image_file, model, dark_mode):
    image = Image.open(image_file)
    frame = np.array(image)
    annotated_frame, counter, confidence, alerts = detect_objects_image(frame, model, dark_mode)

    st.image(annotated_frame, channels="RGB", caption="Detection Result", use_container_width=True)
    
    if alerts:
        for a in alerts:
            st.warning(a)
    
    with st.expander("Detection Statistics", expanded=True):
        display_stats(counter, confidence)

# ----------------- APPLY THEME CSS -----------------
def apply_theme(theme):
    if theme == "Dark Mode":
        st.markdown("""
        <style>
        .stApp {background: linear-gradient(135deg, #0f2027, #203a43, #2c5364); color:white;}
        h1,h2,h3,h4 {color:#FFD700; text-shadow:2px 2px 8px #000000;}

        div.stButton>button {font-size:20px; font-weight:bold; color:white; 
            background: linear-gradient(to right, #FF512F, #DD2476); 
            border-radius:10px; padding:12px; transition: transform 0.2s;}
        div.stButton>button:hover {transform: scale(1.05); background: linear-gradient(to right,#FF8C00,#FF4500);}

        div[data-baseweb="tab-list"] button {color:white; font-size:18px; font-weight:bold; background:#203a43; border-radius:8px; margin:0 2px;}
        div[data-baseweb="tab-list"] button[aria-selected="true"] {color:#FFD700 !important; background:#2c5364 !important; box-shadow:0 0 10px #FFD700;}

        .stMarkdown p {font-size:18px; line-height:1.6; color:white;}

        div[role="radiogroup"] label, div[role="radiogroup"] span, div[role="group"] label, div[role="group"] span { color: white !important; }
        div.stDownloadButton>button { color: white !important; background: linear-gradient(to right, #FF512F, #DD2476); }

        div[role="radiogroup"] label {
            color: white !important;
            font-weight: 600 !important;
        }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        .stApp {background-color:#f0f2f6; color:black;}
        div.stButton>button {font-size:20px; font-weight:bold; color:white; 
            background: linear-gradient(to right, #36D1DC, #5B86E5); 
            border-radius:10px; padding:12px; transition: transform 0.2s;}
        div.stButton>button:hover {transform: scale(1.05); background: linear-gradient(to right,#1E90FF,#00BFFF);}
        .stMarkdown p {font-size:18px; line-height:1.6; color:black;}
        </style>
        """, unsafe_allow_html=True)

# ----------------- LOGS TAB UI -----------------
def show_logs_tab():
    st.markdown("## Detection Logs")
    df_logs = read_logs()

    filter_option = st.radio("Filter logs by:", ["All Data","Object Name","Date Range"])
    df_filtered = df_logs.copy()

    if filter_option=="Object Name":
        object_list = df_logs['Object_Name'].unique().tolist()
        selected_object = st.selectbox("Select Object", object_list)
        df_filtered = df_logs[df_logs['Object_Name']==selected_object]

    elif filter_option=="Date Range":
        min_date = pd.to_datetime(df_logs['Time_Stamp'].min())
        max_date = pd.to_datetime(df_logs['Time_Stamp'].max())
        start_date, end_date = st.date_input("Select Date Range", [min_date, max_date])
        df_filtered['Time_Stamp'] = pd.to_datetime(df_filtered['Time_Stamp'])
        df_filtered = df_filtered[(df_filtered['Time_Stamp']>=pd.to_datetime(start_date)) & 
                                  (df_filtered['Time_Stamp']<=pd.to_datetime(end_date))] 

    st.dataframe(df_filtered, use_container_width=True)
    excel_file = export_excel(df_filtered)
    st.download_button("Download Logs as Excel", excel_file, file_name="detection_logs.xlsx")

# ----------------- MAIN APP -----------------
def main():
    st.set_page_config(page_title="Smart Object Detection", layout="wide", page_icon="📷")

    # Initialize
    init_db()
    if "counter" not in st.session_state: st.session_state.counter = Counter()
    if "confidence" not in st.session_state: st.session_state.confidence = defaultdict(list)
    if "frames" not in st.session_state: st.session_state.frames = 0
    if "glow_phase" not in st.session_state: st.session_state.glow_phase = 0

    # Sidebar
    st.sidebar.title("🎛️ Control Panel")
    theme = st.sidebar.radio("Choose Theme", ["Light Mode", "Dark Mode"])
    st.session_state.dark_mode = (theme == "Dark Mode")
    apply_theme(theme)

    st.session_state.model = load_yolo_model()

    source_option = st.sidebar.radio("Input Source", ["Webcam (Deployed)","Upload Image"])

    st.title("📹 Smart Object Detection Dashboard")

    # Input Handling
    if source_option == "Webcam (Deployed)":
        st.markdown("### 🎥 Live Webcam Detection")
        st.info("Click 'START' to begin webcam detection. Works on deployed apps!")
        
        # Create columns for layout
        col1, col2 = st.columns([3, 1])
        
        with col1:
            webrtc_ctx = webrtc_streamer(
                key="object-detection",
                video_transformer_factory=YOLOVideoTransformer,
                rtc_configuration=RTC_CONFIGURATION,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
        
        with col2:
            st.markdown("### 📊 Live Stats")
            if webrtc_ctx.state.playing:
                stats_placeholder = st.empty()
                
                # Display stats periodically
                while webrtc_ctx.state.playing:
                    with stats_placeholder.container():
                        display_stats(
                            st.session_state.counter,
                            st.session_state.confidence,
                            st.session_state.frames
                        )
                    time.sleep(1)
                    
    elif source_option == "Upload Image":
        st.markdown("### 🖼️ Image Upload Detection")
        uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
        if uploaded:
            process_uploaded_image(uploaded, st.session_state.model, st.session_state.dark_mode)

    # Tabs
    st.markdown("---")
    about_tabs = st.tabs(["Overview","How It Works","Model Info","Tips & Tricks","Future Improvements","References","Logs"])

    with about_tabs[0]:
        st.markdown("""
        **Project Overview:** 
        - YOLOv8 real-time object detection
        - Works on deployed Streamlit apps using WebRTC
        - Glowing bounding boxes in Dark Mode
        - Live statistics and alerts
        - Modern cinematic dashboard design
        """)

    with about_tabs[1]:
        st.markdown("""
        **How It Works:**
        1. Webcam stream captured via WebRTC (works remotely!)
        2. Each frame processed by YOLOv8 model
        3. Objects detected with bounding boxes
        4. Live statistics updated in real-time
        5. All detections logged to database
        """)

    with about_tabs[2]:
        st.markdown("""
        **Model Information:**
        - Default: YOLOv8 nano (`yolov8n.pt`) - lightweight & fast
        - Upload custom models via sidebar
        - Supports all YOLOv8 model variants
        """)

    with about_tabs[3]:
        st.markdown("""
        **Tips & Tricks:**
        - ✅ Good lighting improves detection accuracy
        - ✅ Keep camera steady for best results
        - ✅ Allow camera permissions when prompted
        - ✅ Use high-resolution cameras for better detection
        - ✅ Try Dark Mode for a cinematic experience!
        """)

    with about_tabs[4]:
        st.markdown("""
        **Future Improvements:**
        - 🔄 Support multiple simultaneous cameras
        - 🔊 Sound alerts for specific objects
        - 📹 Recording feature with video download
        - ☁️ Cloud storage integration
        - 📈 Advanced analytics dashboard
        - 🤖 Custom model training interface
        """)

    with about_tabs[5]:
        st.markdown("""
        **References:**
        - [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
        - [Streamlit WebRTC](https://github.com/whitphx/streamlit-webrtc)
        - [OpenCV Documentation](https://opencv.org/)
        - Computer Vision & Deep Learning references
        """)

    with about_tabs[6]:
        show_logs_tab()

if __name__ == "__main__":
    main()
