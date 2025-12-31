import json
import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DRIFT_FLAG_PATH = os.path.join(BASE_DIR, "drift_flag.json")
TRAINING_DATA_PATH = os.path.join(BASE_DIR, "training_data.csv")
LIVE_DATA_PATH = os.path.join(BASE_DIR, "live_predictions.csv")

MODEL_NAME = "demo_classifier"

# ==========================
# CHECK DRIFT
# ==========================
with open(DRIFT_FLAG_PATH) as f:
    drift_info = json.load(f)

if not drift_info["drift_detected"]:
    print("No drift detected. Skipping retraining.")
    exit()

print("Drift detected → Retraining model")

# ==========================
# LOAD & MERGE DATA
# ==========================
train_df = pd.read_csv(TRAINING_DATA_PATH)
live_df = pd.read_csv(LIVE_DATA_PATH)

# Combine old + new data
full_df = pd.concat([train_df, live_df], ignore_index=True)

X = full_df.filter(like="feature_")
y = full_df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================
# TRAIN NEW MODEL
# ==========================
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# ==========================
# LOG TO MLFLOW
# ==========================
mlflow.set_experiment("Auto-Retraining")

with mlflow.start_run():
    mlflow.log_param("n_neighbors", 5)
    mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name=MODEL_NAME)

print("New model trained & logged to MLflow")
