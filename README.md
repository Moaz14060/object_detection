# Real-Time Object Detection for Autonomous Vehicles

## Project Overview

This project delivers a deep learning-based, real-time object detection system designed to enhance road safety and situational awareness, primarily for autonomous vehicle applications. The system processes live video feeds (from USB cameras, IP cameras, or uploaded images) to identify and classify objects such as pedestrians, vehicles, and traffic signs with low-latency inference.

The application is presented as a modern, interactive dashboard built with **Streamlit**, featuring a cinematic dark mode with glowing bounding boxes and a blurred background effect for detected objects.

## Key Features

*   **Real-Time Detection:** Utilizes the lightweight **YOLOv8n** model for fast, real-time object detection.
*   **Interactive Dashboard:** A user-friendly interface built with Streamlit for easy control and visualization.
*   **Dynamic Visualization:** Features glowing bounding boxes, a blurred background in dark mode, and live statistical charts (object count, average confidence).
*   **Multiple Input Sources:** Supports detection from USB cameras, IP camera URLs, and image uploads.
*   **MLOps Ready:** Integrated with **MLflow** for tracking experiments, managing models, and logging performance metrics.
*   **Performance Tracking:** Includes a dedicated script to log model performance metrics (`mAP`, `IoU`, `Precision`, `Recall`, `F1_score`) to MLflow.

## Quick Access and Live Demos

You can explore the project's capabilities through the live Streamlit application and the associated Power BI dashboard without needing to download or run the code locally.

### 1. Live Streamlit Application

Experience the real-time object detection dashboard directly:

[**Streamlit App: Object Detection Demo**](https://objectdetection-erv7rq3koga6qoq6qhhnsd.streamlit.app/)

> **Note on Usage:** This live application is hosted on a cloud platform (Streamlit Cloud) and serves as a fully functional alternative to running the repository locally. Due to the cloud hosting environment, the option to connect via a local IP address is not available. Please use the provided URL for access.

### 2. Power BI Dashboard

View the project's data and model performance metrics through the interactive Power BI dashboard:

[**Power BI Dashboard: Detection Metrics**](https://app.powerbi.com/view?r=eyJrIjoiZGRkYWUyNjEtMGZmOC00YWY4LTk4NjAtNWQyNGU3MTJmZDYzIiwidCI6ImVhZjYyNGM4LWEwYzQtNDE5NS04N2QyLTQ0M2U1ZDc1MTZjZCIsImMiOjh9&embedImagePlaceholder=true)

## Technology Stack

The core technologies and tools used in this project are:

| Category | Tool/Library | Purpose |
| :--- | :--- | :--- |
| **Framework** | Streamlit | Web application framework for the interactive detection dashboard. |
| **Model** | YOLOv8n (Ultralytics) | Deep learning model for real-time object detection. |
| **MLOps** | **MLflow** | **Experiment tracking, model management, and performance logging.** |
| **Data Analysis** | NumPy, Pandas, Altair | Data manipulation, statistical analysis, and chart generation. |
| **Computer Vision** | OpenCV (`cv2`) | Handling video streams and image processing. |
| **Databases** | **SQL Server, SQLite** | **SQL Server for production MLOps data logging; SQLite for local MLflow tracking.** |
| **Visualization** | Power BI | External dashboard for visualizing detection data and model metrics. |
| **Language** | Python 3.8+ | Primary programming language. |

## Database Integration

The project utilizes a dual-database approach to manage both production-level MLOps data and local experiment tracking:

| Database | Primary Use Case | Integration Details |
| :--- | :--- | :--- |
| **SQL Server** | **Production MLOps Data Logging** | Used for persistent storage of detection results, performance metrics, and other critical data in a production or enterprise environment. The `src/database_connection.py` file is designed to connect to this server using `pyodbc`. |
| **SQLite** | **Local MLflow Tracking** | Used by default by MLflow to store local experiment metadata, parameters, and metrics within the `mlruns/` directory. This provides a lightweight, file-based database for development and local testing. |

## Project Structure

The repository is organized into the following directories and key files:

| Path | Description |
| :--- | :--- |
| `src/` | Contains the core Python source code for the application. |
| `src/app.py` | The main Streamlit application file for the real-time object detection dashboard. |
| `src/database_connection.py` | Contains the function to establish a connection to the SQL Server database. |
| `log_models.py` | A standalone script utilizing MLflow to log model performance metrics and artifacts. |
| `models/` | Directory containing different model weights (e.g., `1.pt`, `2.pt`, `best.pt`). |
| `mlruns/` | **Local MLflow tracking server directory for experiment data and runs (uses SQLite internally).** |
| `dashboard/` | Contains the Power BI file (`Detections Dashboard.pbix`) used for external data visualization. |
| `db/` | Contains database-related scripts and files. |
| `db/database_creation.sql` | SQL script for creating the necessary tables in the SQL Server database. |
| `.gitignore` | Specifies files and directories to be ignored by Git. |
| `requirements.txt` | List of all Python dependencies for easy installation. |
| `LICENSE` | Project license details (MIT License). |

## 🚀 Getting Started

This guide provides detailed steps to set up and run the object detection application on your local machine.

### Prerequisites

1.  **Python 3.8+**
2.  **Git** (for cloning the repository)
3.  **A virtual environment** (highly recommended to manage dependencies)
4.  **ODBC Driver** (Required for SQL Server connection via `pyodbc`. See `pyodbc` documentation for installation.)

### Installation

Follow these steps to get the project running:

1.  **Clone the repository:**
    
    ```shell
    git clone https://github.com/Moaz14060/object_detection
    cd object_detection
    ```
    
2.  **Create and activate a virtual environment:**
    
    ```shell
    # Create environment (e.g., named 'venv')
    python3 -m venv venv
    
    # Activate the environment (Linux/macOS)
    source venv/bin/activate
    
    # Activate the environment (Windows)
    .\venv\Scripts\activate
    ```
    
3.  **Install dependencies:**
    
    The project dependencies are listed in `requirements.txt`. Install them using pip:
    
    ```shell
    pip install -r requirements.txt
    ```
    
    > **Note on `pyodbc`:** If you encounter issues installing `pyodbc`, you may need to install the appropriate ODBC driver for your operating system. For a basic run without SQL Server data logging, you can temporarily skip `pyodbc` and comment out the database-related imports in `src/app.py` and `src/database_connection.py`.

## Usage

The application is a Streamlit dashboard and is run using the `streamlit run` command.

1.  **Run the application:** Ensure your virtual environment is active, and run the main file:
    
    ```shell
    streamlit run src/app.py
    ```
    
2.  **Access the Dashboard:** The command will open the application in your default web browser (usually at `http://localhost:8501`).

### MLOps Usage (MLflow)

To log the model performance metrics and artifacts to the local MLflow tracking server:

1.  **Run the logging script:**
    
    ```shell
    python log_models.py
    ```
    
2.  **Start the MLflow UI:**
    
    ```shell
    mlflow ui
    ```
    
3.  **Access MLflow:** Open your browser to `http://localhost:5000` (or the port specified by MLflow) to view the "AutonomousVehicle\_ObjectDetection" experiment, model metrics, and logged artifacts.

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

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

## Authors

*   Moaz Ahmed Abdel Ghaffar
*   Salma Ayman
*   Mohamed Ashraf Abdel-Aziz
*   Shahd Medhat
*   Mohamed Ibrahim
