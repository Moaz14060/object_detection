# ----------------- IMPORTS -----------------
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
from datetime import datetime
import database_connection as db

# ----------------- PAGE CONFIG -----------------
st.set_page_config(page_title="Smart Object Detection", layout="wide", page_icon="🎯")

# ----------------- THEME SELECTION -----------------
theme = st.sidebar.radio("Choose Theme", ["Light Mode", "Dark Mode"])

# ----------------- LOAD YOLO MODEL -----------------
model = YOLO(r"F:\object_detection\yolov8s.pt")

# ----------------- SESSION STATE -----------------
if "run" not in st.session_state: st.session_state.run = False
if "counter" not in st.session_state: st.session_state.counter = Counter()
if "confidence" not in st.session_state: st.session_state.confidence = defaultdict(list)
if "frames" not in st.session_state: st.session_state.frames = 0
if "glow_phase" not in st.session_state: st.session_state.glow_phase = 0

# ----------------- GLOW COLOR FUNCTION -----------------
def get_glow_color():
    phase = st.session_state.glow_phase
    r = 255
    g = int(140 + 115 * abs(math.sin(phase)))
    b = 0
    st.session_state.glow_phase += 0.15
    return (b, g, r)

# ----------------- DETECTION FUNCTION -----------------
def detect_object(frame, dark_mode=False):
    conn = db.get_db_connection()
    cursor = conn.cursor()
    results = model(frame)
    st.session_state.counter = Counter()
    st.session_state.confidence = defaultdict(list)
    alerts = []
    
    blurred_frame = frame.copy()
    if dark_mode:
        blurred_frame = cv2.GaussianBlur(frame, (15,15), 0)
    
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = f"{model.names[cls]} {conf:.2f}"

            color = (0,255,0)
            if dark_mode: color = get_glow_color()

            roi = frame[y1:y2, x1:x2]
            if dark_mode:
                blurred_frame[y1:y2, x1:x2] = roi
            cv2.rectangle(blurred_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(blurred_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255 if dark_mode else 0), 2)

            st.session_state.counter[model.names[cls]] +=1
            st.session_state.confidence[model.names[cls]].append(conf)
            if st.session_state.counter[model.names[cls]]>3:
                alerts.append(f"⚠️ High number of {model.names[cls]} detected!")
            
            if cursor:
                try:
                    cursor.execute("""
                        INSERT INTO Detections (Time_Stamp, Object_Name, Count, Confidence)
                        VALUES (?, ?, ?, ?)
                    """, (
                        datetime.now(),
                        model.names[cls],
                        st.session_state.counter[model.names[cls]],
                        round(conf, 2)
                    ))
                except Exception as e:
                    st.error(f"Database insert failed: {e}")

    # Commit once per frame
    if conn:
        conn.commit()
        cursor.close()
        conn.close()

    st.session_state.frames +=1
    return blurred_frame, alerts

# ----------------- CAMERA STREAM -----------------
def start_camera(source=0, dark_mode=False):
    col1, col2 = st.columns([3,1])
    frame_placeholder = col1.empty()
    with col2:
        stats_expander = st.expander("📊 Live Stats", expanded=True)
        chart_placeholder = stats_expander.empty()
        table_placeholder = stats_expander.empty()
        alert_placeholder = st.empty()

    cap = cv2.VideoCapture(source)
    while st.session_state.run:
        ret, frame = cap.read()
        if not ret: break
        detected, alerts = detect_object(frame,dark_mode)
        rgb_frame = cv2.cvtColor(detected, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

    if alerts:
        for a in alerts:
            if isinstance(a, str) and a.strip():
                alert_placeholder.warning(a)
            else:
                print(f"⚠️ Ignored alert (not string): {a}")

        if st.session_state.counter:
            data=[]
            for obj,count in st.session_state.counter.items():
                avg_conf=np.mean(st.session_state.confidence[obj])
                data.append({"Object":obj,"Count":count,"Avg Confidence":round(avg_conf,2)})
            df=pd.DataFrame(data)
            df['Frames Processed']=st.session_state.frames
            table_placeholder.table(df)
            chart_color = "#FFA500" if dark_mode else "#32CD32"
            chart=alt.Chart(df).mark_bar(color=chart_color).encode(
                x=alt.X('Object', sort=None),
                y='Count',
                tooltip=['Object','Count','Avg Confidence']
            ).properties(title="Detected Objects Count")
            chart_placeholder.altair_chart(chart,use_container_width=True)

        time.sleep(0.02)
    cap.release()
    frame_placeholder.empty()
    table_placeholder.empty()
    chart_placeholder.empty()
    alert_placeholder.empty()
    st.session_state.frames=0

# ----------------- CSS -----------------
if theme=="Dark Mode":
    st.markdown("""
    <style>
    .stApp {background: linear-gradient(135deg, #0f2027, #203a43, #2c5364); color:white;}
    h1,h2,h3,h4 {color:#FFD700; text-shadow:2px 2px 8px #000000;}
    div.stButton>button {font-size:20px; font-weight:bold; color:white; 
        background: linear-gradient(to right, #FF512F, #DD2476); 
        border-radius:10px; padding:12px; transition: transform 0.2s;}
    div.stButton>button:hover {transform: scale(1.05); background: linear-gradient(to right,#FF8C00,#FF4500);}
    div[data-baseweb="tab-list"] button {color:white; font-size:18px; font-weight:bold; transition: color 0.2s;}
    div[data-baseweb="tab-list"] button[aria-selected="true"] {color:red !important;}
    .stMarkdown p {font-size:18px; line-height:1.6; color:white;}
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

# ----------------- SIDEBAR -----------------
st.sidebar.title("🎯 Control Panel")
source_option = st.sidebar.radio("Input Source", ["USB Camera","IP Camera","Upload Image"])
start_btn = st.sidebar.button("▶️ Start Detection")
stop_btn = st.sidebar.button("⏹️ Stop Detection")

st.title("🎯 Smart Object Detection Dashboard")
dark_mode = True if theme=="Dark Mode" else False

# ----------------- INPUT HANDLING -----------------
if source_option=="USB Camera":
    if start_btn: st.session_state.run=True; start_camera(0,dark_mode)
    if stop_btn: st.session_state.run=False
elif source_option=="IP Camera":
    ip_url = st.sidebar.text_input("Enter IP camera URL")
    if start_btn:
        if ip_url: st.session_state.run=True; start_camera(ip_url,dark_mode)
        else: st.warning("Please enter a valid IP camera URL.")
    if stop_btn: st.session_state.run=False
elif source_option=="Upload Image":
    uploaded = st.file_uploader("📁 Upload an image", type=["jpg","jpeg","png"])
    if uploaded:
        image = Image.open(uploaded)
        frame = np.array(image)
        detected, alerts = detect_object(frame,dark_mode)
        st.image(detected, channels="RGB", caption="Detection Result", use_container_width=True)
        if alerts: [st.warning(a) for a in alerts]

        if st.session_state.counter:
            data=[]
            for obj,count in st.session_state.counter.items():
                avg_conf=np.mean(st.session_state.confidence[obj])
                data.append({"Object":obj,"Count":count,"Avg Confidence":round(avg_conf,2)})
            df=pd.DataFrame(data)
            df['Frames Processed']=st.session_state.frames
            st.table(df)
            chart_color = "#FFA500" if dark_mode else "#32CD32"
            chart=alt.Chart(df).mark_bar(color=chart_color).encode(
                x=alt.X('Object', sort=None),
                y='Count',
                tooltip=['Object','Count','Avg Confidence']
            ).properties(title="Detected Objects Count")
            st.altair_chart(chart,use_container_width=True)

# ----------------- ABOUT PROJECT TABS -----------------
st.markdown("## ✨ About Project")
about_tabs = st.tabs(["Overview","How It Works","Model Info","Tips & Tricks","Future Improvements","References"])

with about_tabs[0]:
    st.markdown("""
**Project Overview:** Animated Glow + Blur Background in Dark Mode, YOLOv8 detection, live stats, alerts, modern cinematic dashboard.
""")

with about_tabs[1]:
    st.markdown("""
- Capture frame from camera or image.  
- YOLOv8 detects objects.  
- Display frame with glowing bounding boxes and blurred background.  
- Update live stats and charts.
""")

with about_tabs[2]:
    st.markdown("""
- YOLOv8 nano model (`yolov8n.pt`)  
- Detects common objects.  
- Lightweight & fast.
""")

with about_tabs[3]:
    st.markdown("""
- Good lighting improves detection accuracy.  
- Keep camera steady.  
- High-resolution images improve results.
""")

with about_tabs[4]:
    st.markdown("""
- Support multiple cameras simultaneously.  
- Add sound alerts.  
- Recording feature.  
- Cloud storage & analysis.
""")

with about_tabs[5]:
    st.markdown("""
- [Ultralytics YOLO](https://ultralytics.com/)  
- Computer Vision & Deep Learning references.
""")
