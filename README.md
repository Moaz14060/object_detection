# Real-Time Object Detection for Autonomous Vehicles

## Project Description

This project implements a deep learning-based, real-time object detection system designed to enhance road safety and situational awareness for autonomous vehicles. The system processes live video feeds (from USB cameras, IP cameras, or uploaded images) to identify and classify objects such as pedestrians, vehicles, and traffic signs with low-latency inference.

The application is built as a modern, interactive dashboard using **Streamlit**, featuring a cinematic dark mode with glowing bounding boxes and a blurred background effect for detected objects.

## Features

*   **Real-Time Detection:** Utilizes the lightweight **YOLOv8n** model for fast, real-time object detection.
*   **Interactive Dashboard:** A user-friendly interface built with Streamlit for easy control and visualization.
*   **Dynamic Visualization:** Features glowing bounding boxes, a blurred background in dark mode, and live statistical charts (object count, average confidence).
*   **Multiple Input Sources:** Supports detection from USB cameras, IP camera URLs, and image uploads.
*   **MLOps Ready:** Integrated with **MLflow** for tracking experiments, managing models, and logging performance metrics.
*   **Performance Tracking:** Includes a dedicated script to log model performance metrics (`mAP`, `IoU`, `Precision`, `Recall`, `F1_score`) to MLflow.

## Tech Stack / Tools

The core technologies and tools used in this project are:

| Category | Tool/Library | Purpose |
| :--- | :--- | :--- |
| **Framework** | Streamlit | Web application framework for the interactive detection dashboard. |
| **Model** | YOLOv8n (Ultralytics) | Deep learning model for real-time object detection. |
| **MLOps** | **MLflow** | **Experiment tracking, model management, and performance logging.** |
| **Data Analysis** | NumPy, Pandas, Altair | Data manipulation, statistical analysis, and chart generation. |
| **Computer Vision** | OpenCV (`cv2`) | Handling video streams and image processing. |
| **Database** | pyodbc, SQL Server | Optional integration for MLOps data logging and monitoring. |
| **Visualization** | Power BI | External dashboard for visualizing detection data and model metrics. |
| **Language** | Python 3.8+ | Primary programming language. |

## Project Structure

The project is structured as follows:

| Path | Description |
| :--- | :--- |
| `src/app.py` | The main Streamlit application file for real-time detection. |
| `src/database_connection.py` | Contains the function to establish a connection to a SQL Server database. |
| `log_models.py` | **MLflow script to log model performance metrics and artifacts.** |
| `models/` | Directory containing different model weights (`1.pt`, `2.pt`, `best.pt`). |
| `mlruns/` | **Local MLflow tracking server directory for experiment data and runs.** |
| `dashboard/` | Contains the Power BI file (`Detections Dashboard.pbix`) for data visualization. |
| `db/database_creation.sql` | SQL script for creating the necessary database tables. |
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

## Usage

The application is a Streamlit dashboard and is run using the `streamlit run` command.

1.  **Run the application:**
    Ensure your virtual environment is active, and run the main file:
    ```bash
    streamlit run src/app.py
    ```

2.  **Access the Dashboard:**
    The command will open the application in your default web browser (usually at `http://localhost:8501`).

### MLOps Usage (MLflow)

To log the model performance metrics and artifacts to the local MLflow tracking server:

1.  **Run the logging script:**
    ```bash
    python log_models.py
    ```
2.  **Start the MLflow UI:**
    ```bash
    mlflow ui
    ```
3.  **Access MLflow:**
    Open your browser to `http://localhost:5000` (or the port specified by MLflow) to view the "AutonomousVehicle_ObjectDetection" experiment, model metrics, and logged artifacts.

## Model Details

### Object Detection Model

*   **Model:** YOLOv8n (You Only Look Once, version 8, nano variant)
*   **Framework:** Ultralytics
*   **Purpose:** Chosen for its balance of speed and accuracy, making it suitable for real-time, low-latency inference required in autonomous vehicle applications.

### MLOps and Experiment Tracking (MLflow)

The project uses **MLflow** to manage the machine learning lifecycle, specifically for:

*   **Experiment Tracking:** The `log_models.py` script logs key performance indicators (KPIs) for three model variants.
*   **Metrics Logged:** The script logs `mAP`, `IoU`, `Precision`, `Recall`, and `F1_score` for each model variant, providing a clear comparison of model performance.
*   **Model Registry:** The MLflow structure is set up to register and version the trained models, facilitating easy deployment and rollback.

## Results

The project aims to achieve the following objectives, which define the expected results:

1.  **Real-Time Inference:** Successfully deploy a model that provides low-latency object detection from live video feeds.
2.  **Robust Detection:** Optimize detection for varying conditions (lighting, weather, and traffic).
3.  **Scalable Deployment:** Demonstrate the model's operation in a scalable environment (the Streamlit dashboard).
4.  **MLOps Readiness:** Establish the framework for MLOps pipelines for monitoring, retraining, and maintenance through the database and MLflow integration.

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.
