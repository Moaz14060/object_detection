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
from io import BytesIO
from datetime import datetime
import sqlite3

# ==========================
#  DATABASE (SQLite)
# ==========================
DB_NAME = "object_detection.db"

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
    return buffer

# ----------------- PAGE CONFIG -----------------
st.set_page_config(page_title="Smart Object Detection", layout="wide", page_icon="🎯")

# ----------------- THEME SELECTION -----------------
theme = st.sidebar.radio("Choose Theme", ["Light Mode", "Dark Mode"])

# ----------------- LOAD YOLO MODEL (CUSTOM UPLOAD) -----------------
st.sidebar.subheader("📦 Load Custom YOLO Model")

uploaded_model = st.sidebar.file_uploader("Upload YOLO model (.pt)", type=["pt"])

if uploaded_model is not None:
    temp_model_path = "uploaded_model.pt"
    with open(temp_model_path, "wb") as f:
        f.write(uploaded_model.read())
    model = YOLO(temp_model_path)
    st.sidebar.success("✅ Custom model loaded successfully!")
else:
    model = YOLO("yolov8n.pt")
    st.sidebar.info("Using default YOLO model (yolov8n.pt)")

# ----------------- SESSION STATE -----------------
if "run" not in st.session_state: st.session_state.run = False
if "counter" not in st.session_state: st.session_state.counter = Counter()
if "confidence" not in st.session_state: st.session_state.confidence = defaultdict(list)
if "frames" not in st.session_state: st.session_state.frames = 0
if "glow_phase" not in st.session_state: st.session_state.glow_phase = 0

init_db()

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
    results = model(frame)
    st.session_state.counter = Counter()
    st.session_state.confidence = defaultdict(list)
    alerts = []
    
    blurred_frame = frame.copy()
    
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
            
            # insert into SQLite
            insert_log_both(model.names[cls], st.session_state.counter[model.names[cls]], conf)

            if st.session_state.counter[model.names[cls]]>3:
                alerts.append(f"⚠️ High number of {model.names[cls]} detected!")

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

        if alerts: [alert_placeholder.warning(a) for a in alerts]
        else: alert_placeholder.empty()

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

    div[data-baseweb="tab-list"] button {color:white; font-size:18px; font-weight:bold; background:#203a43; border-radius:8px; margin:0 2px;}
    div[data-baseweb="tab-list"] button[aria-selected="true"] {color:#FFD700 !important; background:#2c5364 !important; box-shadow:0 0 10px #FFD700;}

    .stMarkdown p {font-size:18px; line-height:1.6; color:white;}

    div[role="radiogroup"] label, div[role="radiogroup"] span, div[role="group"] label, div[role="group"] span { color: white !important; }
    div.stDownloadButton>button { color: white !important; background: linear-gradient(to right, #FF512F, #DD2476); }

    .css-1d391kg {background: linear-gradient(135deg, #0f2027, #203a43, #2c5364); color:white;}
    .css-1v3fvcr {color:white;}
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
                data.append({"Object":obj,"Count":"{0}".format(count),"Avg Confidence":round(avg_conf,2)})
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

# ----------------- ABOUT PROJECT + LOGS TABS -----------------
about_tabs = st.tabs(["Overview","How It Works","Model Info","Tips & Tricks","Future Improvements","References","Logs"])

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

# ----------------- LOGS TAB -----------------
with about_tabs[6]:
    st.markdown("## 📂 Detection Logs")
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
    st.download_button("⬇ Download Logs as Excel", excel_file, file_name="detection_logs.xlsx")
    
    
    
# ==========================
#  SQL SERVER CONNECTION
# ==========================
#import pyodbc

#def get_sql_connection():
#    try:
#        conn = pyodbc.connect(
#            "DRIVER={ODBC Driver 17 for SQL Server};"
#            "SERVER=192.168.1.8,1433;"
#            "DATABASE=ObjectDetection;"
#            "UID=sa;"
#            "PWD=test1234;"
#        )
#        return conn
#    except Exception as e:
#        st.sidebar.error(f"SQL Connection Error: {e}")
#       return None


# ==========================
# CREATE TABLE IN SQL SERVER
# ==========================
#def init_sql_table():
#   conn = get_sql_connection()
#   if conn is None:
#       return
#   cursor = conn.cursor()
#   cursor.execute("""
#       IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='Detections' AND xtype='U')
#       CREATE TABLE Detections (
#           ID INT IDENTITY(1,1) PRIMARY KEY,
#           Time_Stamp DATETIME NOT NULL,
#           Object_Name NVARCHAR(255),
#           Count_Value INT,
#           Confidence FLOAT
#       );
#   """)
#   conn.commit()
#   conn.close()


# ==========================
# INSERT INTO SQL SERVER
# ==========================
#def insert_sql_log(object_name, count_value, confidence):
#    conn = get_sql_connection()
#   if conn is None:
#        return
#   cursor = conn.cursor()
#   try:
#       cursor.execute("""
#           INSERT INTO Detections (Time_Stamp, Object_Name, Count_Value, Confidence)
#           VALUES (?, ?, ?, ?)
#       """, (datetime.now(), object_name, count_value, confidence))
#       conn.commit()
#   except Exception as e:
#       st.error(f"Error inserting into SQL Server: {e}")
#   finally:
#       conn.close()


# ==========================
# ENABLE SQL SERVER INSERT
# ==========================
#init_sql_table()

# تعديل بسيط: إضافة إدخال SQL Server بجانب SQLite
#def insert_log_both(object_name, count_value, confidence):
#   insert_log(object_name, count_value, confidence)   # SQLite
#   insert_sql_log(object_name, count_value, confidence)  # SQL Server


#def get_sql_connection():
#   try:
#       conn = pyodbc.connect(
#           "DRIVER={ODBC Driver 17 for SQL Server};"
#           "SERVER=192.168.1.8,1433;"
#           "DATABASE=object_detection;"
#           "UID=sa;"
#           "PWD=test1234;"
#       )
#      return conn
#    except Exception as e:
#       st.error(f"❌ SQL Connection Failed: {e}")
#       return None
#if st.button("📤 Send Logs to SQL Server"):
#   conn = get_sql_connection()
#   if conn is not None:
#       cursor = conn.cursor()
#       for _, row in df_filtered.iterrows():
#           cursor.execute("""
#               INSERT INTO detections (Time_Stamp, Object_Name, Count, Confidence)
#               VALUES (?, ?, ?, ?)
#           """, row["Time_Stamp"], row["Object_Name"], row["Count"], row["Confidence"])
#       conn.commit()
#       conn.close()
#       st.success("✅ Logs sent to SQL Server successfully!")






"""
#Explanation of code

#Smart Object Detection Dashboard - Detailed Explanation

1. Imports:
- streamlit (st): Web framework for creating interactive dashboards and UI components.
- cv2: OpenCV library for image/video capture, frame processing, and drawing bounding boxes.
- numpy (np): Efficient numerical operations on arrays, used for frame manipulation.
- PIL.Image: Handles image loading and conversion for processing in OpenCV or displaying in Streamlit.
- collections.Counter, defaultdict: Track object counts and store multiple confidence scores per object.
- pandas (pd): Data manipulation, filtering, and exporting logs as DataFrames.
- altair: Create interactive bar charts for live object count visualization.
- ultralytics.YOLO: Provides YOLOv8 model for object detection.
- time: Used for timing, frame delays, and animation effects.
- math: Calculate sinusoidal glow effects for bounding boxes in Dark Mode.
- io.BytesIO: In-memory file handling for exporting Excel logs.
- datetime: Timestamp generation for logs.
- sqlite3: SQLite database management for storing and retrieving detection records.

2. Database Functions:
- init_db():
    - Connects to a local SQLite database (object_detection.db).
    - Creates a 'detections' table if it does not already exist.
    - Columns include ID (auto-increment primary key), Time_Stamp, Object_Name, Count, and Confidence.
    - Ensures persistent storage of detected objects.
- insert_log(object_name, count_value, confidence):
    - Inserts a new row into the SQLite table for every detection event.
    - Records timestamp, object name, number of occurrences, and confidence score.
    - Allows real-time tracking and historical analysis.
- read_logs():
    - Reads all records from SQLite into a pandas DataFrame.
    - Orders entries by ID descending, so the most recent detections appear first.
    - Provides data for display in the logs tab or charts.
- export_excel(df):
    - Converts a pandas DataFrame to an in-memory Excel file using BytesIO.
    - Enables downloading logs directly from the dashboard without saving files on disk.

3. Streamlit Configuration:
- st.set_page_config(): Sets the web page title, layout (wide), and icon.
- Sidebar radio button for theme selection (Light Mode / Dark Mode).
- File uploader allows users to provide a custom YOLO model (.pt file).
- If no model is uploaded, defaults to YOLOv8 nano (`yolov8n.pt`) for lightweight fast detection.

4. Session State:
- Maintains persistent variables across reruns:
    - "run": Boolean for whether detection is active.
    - "counter": Counts of each detected object in the current frame/session.
    - "confidence": List of confidence scores for each detected object.
    - "frames": Number of processed frames since detection started.
    - "glow_phase": Tracks animation phase for pulsing glow effect in Dark Mode.

5. Glow Color Function:
- get_glow_color():
    - Computes a pulsing orange-yellow color for bounding boxes in Dark Mode.
    - Uses `sin()` of a phase value to oscillate green channel intensity smoothly.
    - Returns a BGR tuple for OpenCV rectangle drawing.
    - Updates glow_phase to animate over consecutive frames.

6. Object Detection Function:
- detect_object(frame, dark_mode=False):
    - Runs YOLO model inference on the provided frame.
    - Initializes counters and confidence dictionaries for fresh frame stats.
    - For each detection:
        - Extracts bounding box coordinates (x1, y1, x2, y2).
        - Retrieves class index and confidence score.
        - Forms label with object name and confidence.
        - Chooses color: green for Light Mode, glowing for Dark Mode.
        - Draws rectangle and label on the frame.
        - Updates counters and stores confidence values.
        - Inserts detection log into database(s) using insert_log_both().
        - Triggers alerts if object count exceeds a threshold (e.g., >3).
    - Returns processed frame with annotations and list of alerts.

7. Camera Stream Function:
- start_camera(source=0, dark_mode=False):
    - Initializes video capture for USB or IP camera.
    - Sets up Streamlit placeholders for:
        - Video frames.
        - Live statistics table.
        - Altair chart for object counts.
        - Alerts for high object count events.
    - Continuously:
        - Reads frame from camera.
        - Runs detect_object() to annotate frame and get alerts.
        - Converts BGR to RGB and displays frame in Streamlit.
        - Updates table and bar chart for object counts and average confidence.
        - Shows alerts as Streamlit warnings.
        - Sleeps briefly (20 ms) to allow smooth frame updates.
    - Cleans up resources when stopped (release capture, empty placeholders).

8. CSS/Theming:
- Applies custom CSS for Light and Dark Modes.
- Dark Mode:
    - Gradient background, golden headers with shadow.
    - Buttons with gradient, hover effect, and scaling.
    - Tab styling with selected tab highlighting.
    - Download buttons styled to match theme.
- Light Mode:
    - Light background with blue gradient buttons.
    - Hover effect scaling and color changes.
- Ensures visually consistent and modern UI.

9. Input Handling:
- USB Camera:
    - Starts cv2.VideoCapture(0) when Start Detection is pressed.
- IP Camera:
    - User provides URL; validates URL before starting detection.
- Upload Image:
    - Allows one-time detection on static images.
    - Updates table and chart like live camera feed.
- Stop button halts detection loop by updating session_state["run"].

10. Tabs and Logs:
- Tabs include Overview, How It Works, Model Info, Tips, Future Improvements, References, Logs.
- Logs tab:
    - Displays DataFrame of detection logs.
    - Filters available by object name or date range.
    - Converts filtered DataFrame to Excel for download.
    - Provides visual representation with Altair bar chart.

11. SQL Server Integration (commented):
- Optional code to send detection logs to a remote SQL Server database.
- Functions include:
    - get_sql_connection(): Connects to server with provided credentials.
    - init_sql_table(): Creates Detections table if it doesn't exist.
    - insert_sql_log(): Inserts individual detection into SQL Server.
    - insert_log_both(): Sends logs to both SQLite and SQL Server.
- Button in UI triggers bulk upload of filtered logs to SQL Server.

Overall:
- Modular design separates detection, UI, database, and styling.
- Dark/Light Mode, live charts, alerts, and log export provide full dashboard functionality.
- Can easily be extended to multiple cameras, sound alerts, or cloud storage.


"""
