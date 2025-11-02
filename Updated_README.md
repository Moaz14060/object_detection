# Real-Time Object Detection for Autonomous Vehicles

## Project Description

This project implements a deep learning-based, real-time object detection system designed to enhance road safety and situational awareness for autonomous vehicles. The system processes live video feeds (from USB cameras, IP cameras, or uploaded images) to identify and classify objects such as pedestrians, vehicles, and traffic signs with low-latency inference.

The application is built as a modern, interactive dashboard using **Streamlit**, featuring a cinematic dark mode with glowing bounding boxes and a blurred background effect for detected objects.

## Features

*   **Real-Time Detection:** Utilizes the lightweight **YOLOv8n** model for fast, real-time object detection.
*   **Interactive Dashboard:** A user-friendly interface built with Streamlit for easy control and visualization.
*   **Dynamic Visualization:** Features glowing bounding boxes, a blurred background in dark mode, and live statistical charts (object count, average confidence).
*   **Multiple Input Sources:** Supports detection from USB cameras, IP camera URLs, and image uploads.
*   **Alert System:** Triggers warnings for high concentrations of detected objects.
*   **MLOps Ready:** Integrated with **MLflow** for tracking experiments, managing models, and logging performance metrics.

## Tech Stack / Tools

The core technologies and tools used in this project are:

| Category | Tool/Library | Purpose |
| :--- | :--- | :--- |
| **Framework** | Streamlit | Web application framework for the interactive dashboard. |
| **Model** | YOLOv8n (Ultralytics) | Deep learning model for real-time object detection. |
| **MLOps** | MLflow | Experiment tracking, model management, and performance logging. |
| **Data Analysis** | NumPy, Pandas, Altair | Data manipulation, statistical analysis, and chart generation. |
| **Computer Vision** | OpenCV (`cv2`) | Handling video streams and image processing. |
| **Database** | pyodbc, SQL Server | Optional integration for MLOps data logging and monitoring. |
| **Language** | Python 3.8+ | Primary programming language. |

## Project Structure

The project is structured as follows:

| Path | Description |
| :--- | :--- |
| `src/app.py` | The main Streamlit application file. Handles the UI, camera stream processing, YOLO inference, and visualization logic. |
| `src/database_connection.py` | Contains the function to establish a connection to a SQL Server database using `pyodbc` and Streamlit secrets for configuration. |
| `models/` | Directory for storing pre-trained model weights. The application uses `yolov8n.pt`. |
| `db/database_creation.sql` | SQL script for creating the necessary database tables (e.g., for logging detection data). |
| `requirements.txt` | List of all Python dependencies for easy installation. |
| `README.md` | This project documentation file. |
| `LICENSE` | Project license details. |

## 🚀 Getting Started

This guide provides detailed steps to set up and run the object detection application on your local machine.

### Prerequisites

1.  **Python 3.8+**
2.  **Git** (for cloning the repository)
3.  **A virtual environment** (highly recommended to manage dependencies)

### Installation

Follow these steps to get the project running:

1.  **Clone the repository:**
    Open your terminal or command prompt and clone the project:
    ```bash
    git clone https://github.com/Moaz14060/object_detection
    cd object_detection
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # Create environment (e.g., named 'venv')
    python3 -m venv venv
    
    # Activate the environment (Linux/macOS)
    source venv/bin/activate
    
    # Activate the environment (Windows)
    .\venv\Scripts\activate
    ```

3.  **Install dependencies:**
    Since the provided `requirements.txt` is empty, we will create one and use it for installation.

    **A. Create `requirements.txt`:**
    ```bash
    # Create the requirements.txt file
    echo "streamlit" > requirements.txt
    echo "opencv-python" >> requirements.txt
    echo "numpy" >> requirements.txt
    echo "Pillow" >> requirements.txt
    echo "pandas" >> requirements.txt
    echo "altair" >> requirements.txt
    echo "ultralytics" >> requirements.txt
    echo "pyodbc" >> requirements.txt
    echo "mlflow" >> requirements.txt
    ```

    **B. Install the packages:**
    ```bash
    pip install -r requirements.txt
    ```
    ***Note on `pyodbc`:** If you encounter issues installing `pyodbc`, you may need to install the appropriate ODBC driver for your operating system. For a basic run without database logging, you can skip `pyodbc` and comment out the database-related imports in `src/app.py` and `src/database_connection.py`.*

4.  **Model Weights:**
    The application uses the `yolov8n.pt` model. The `ultralytics` library will automatically download this model the first time it is run if it's not present. No manual download is required.

## Usage

The application is a Streamlit dashboard and is run using the `streamlit run` command.

1.  **Run the application:**
    Ensure your virtual environment is active, and run the main file:
    ```bash
    streamlit run src/app.py
    ```

2.  **Access the Dashboard:**
    The command will open the application in your default web browser (usually at `http://localhost:8501`).

### In-App Usage

1.  **Control Panel:** Use the sidebar on the left to control the application.
2.  **Theme Selection:** Choose between "Light Mode" and the cinematic "Dark Mode."
3.  **Input Source:** Select your input source (USB Camera, IP Camera, or Upload Image).
4.  **Start/Stop:** Click the **▶️ Start Detection** button to begin the live feed analysis and **⏹️ Stop Detection** to end it.
5.  **Live Stats:** The right column displays a live table and chart of detected objects and their average confidence scores.

## Model Details

### Object Detection Model

*   **Model:** YOLOv8n (You Only Look Once, version 8, nano variant)
*   **Framework:** Ultralytics
*   **Purpose:** Chosen for its balance of speed and accuracy, making it suitable for real-time, low-latency inference required in autonomous vehicle applications.

### MLOps and Experiment Tracking (MLflow)

The project incorporates **MLflow** to manage the machine learning lifecycle, specifically for:

*   **Experiment Tracking:** Logging parameters, metrics (like mAP, IoU, FPS), and artifacts (model checkpoints) during training runs.
*   **Model Registry:** Storing and versioning the trained YOLO models, including the `best.pt` file found in the `models/` directory.
*   **Reproducibility:** Ensuring that the environment and code used to produce a model can be easily reproduced.

## Results

The project aims to achieve the following objectives, which define the expected results:

1.  **Real-Time Inference:** Successfully deploy a model that provides low-latency object detection from live video feeds.
2.  **Robust Detection:** Optimize detection for varying conditions (lighting, weather, and traffic).
3.  **Scalable Deployment:** Demonstrate the model's operation in a scalable environment (the Streamlit dashboard).
4.  **MLOps Readiness:** Establish the framework for MLOps pipelines for monitoring, retraining, and maintenance through the database and MLflow integration.

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.
