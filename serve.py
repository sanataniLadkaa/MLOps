import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
import os

# ==========================
# CONFIG
# ==========================
MODEL_NAME = "demo_classifier"
MODEL_STAGE = "Staging"
PREDICTION_LOG_PATH = "live_predictions.csv"

# ==========================
# LOAD MODEL
# ==========================
model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{MODEL_NAME}/{MODEL_STAGE}"
)

app = FastAPI(title="MLflow FastAPI Inference")

# ==========================
# REQUEST SCHEMA
# ==========================
class PredictRequest(BaseModel):
    features: list

# ==========================
# UTIL: LOG LIVE PREDICTION
# ==========================
def log_prediction(features, prediction):
    record = {
        "timestamp": datetime.utcnow(),
        "prediction": int(prediction),
    }

    for i, val in enumerate(features):
        record[f"feature_{i}"] = val

    df = pd.DataFrame([record])

    # Append to CSV
    if os.path.exists(PREDICTION_LOG_PATH):
        df.to_csv(PREDICTION_LOG_PATH, mode="a", header=False, index=False)
    else:
        df.to_csv(PREDICTION_LOG_PATH, index=False)

    # Log to MLflow
    with mlflow.start_run(run_name="live_prediction", nested=True):
        mlflow.log_metric("prediction", int(prediction))
        mlflow.log_artifact(PREDICTION_LOG_PATH)

# ==========================
# ROUTES
# ==========================
@app.get("/")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "stage": MODEL_STAGE
    }

@app.post("/predict")
def predict(req: PredictRequest):
    X = np.array(req.features).reshape(1, -1)
    pred = model.predict(X)[0]

    log_prediction(req.features, pred)

    return {"prediction": int(pred)}
