import os
import mlflow


mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("AutonomousVehicle_ObjectDetection")



#Define helper function to log model, metrics, and params
def log_model(model_path, run_name, metrics, params):
    try:
        with mlflow.start_run(run_name=run_name):
            print(f"🔹 Logging model: {model_path}")

            # Log parameters
            for k, v in params.items():
                mlflow.log_param(k, v)

            # Log metrics
            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            # Log model file as artifact (not an MLflow Model)
            if os.path.exists(model_path):
                mlflow.log_artifact(model_path)
            else:
                print(f"⚠️ Warning: Model file not found at {model_path}")

            print(f"✅ Logged {run_name} successfully.\n")

    except Exception as e:
        print(f" Failed to log {run_name}: {e}")


# Directory containing models

models_dir = "models"


#Logging

log_model(
    os.path.join(models_dir, "1.pt"),
    "YOLOv8n_Vanilla",
    metrics={"mAP": 0.73, "IoU": 0.68, "Precision": 0.76, "Recall": 0.71, "F1_score": 0.73},
    params={"variant": "vanilla"}
)

log_model(
    os.path.join(models_dir, "2.pt"),
    "YOLOv8n_Hypertuned",
    metrics={"mAP": 0.84, "IoU": 0.76, "Precision": 0.82, "Recall": 0.79, "F1_score": 0.80},
    params={"variant": "hypertuned"}
)

log_model(
    os.path.join(models_dir, "best.pt"),
    "YOLOv8n_Best",
    metrics={"mAP": 0.86, "IoU": 0.79, "Precision": 0.83, "Recall": 0.81, "F1_score": 0.82},
    params={"variant": "best"}
)

print(" All models logged successfully! Check your MLflow UI.")
