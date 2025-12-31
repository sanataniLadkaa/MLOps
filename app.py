from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import mlflow.pyfunc
import pandas as pd
import os

# =========================
# CONFIG
# =========================
MODEL_NAME = "demo_classifier"
MODEL_STAGE = os.getenv("MODEL_STAGE", "Staging")  # default safe stage

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LIVE_DATA_PATH = os.path.join(BASE_DIR, "live_predictions.csv")

FEATURE_NAMES = [f"feature_{i}" for i in range(15)]

# =========================
# FASTAPI APP
# =========================
app = FastAPI(
    title="ML Inference Service",
    description="Stable MLflow inference service",
    version="1.0.0",
)

# =========================
# LOAD MODEL SAFELY
# =========================
model = None

def load_model():
    global model
    try:
        # model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        model = mlflow.pyfunc.load_model("model")
        # print(f"✅ Loaded model: {model_uri}")
    except Exception as e:
        print(f"❌ Model load failed: {e}")
        model = None

@app.on_event("startup")
def startup_event():
    load_model()

# =========================
# INPUT SCHEMA
# =========================
class PredictionInput(BaseModel):
    features: List[float]

# =========================
# PREDICT ENDPOINT
# =========================
@app.post("/predict")
def predict(data: PredictionInput):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Check MLflow registry."
        )

    if len(data.features) != len(FEATURE_NAMES):
        raise HTTPException(
            status_code=400,
            detail=f"Expected {len(FEATURE_NAMES)} features, got {len(data.features)}"
        )

    input_df = pd.DataFrame(
        [data.features],
        columns=FEATURE_NAMES
    )

    prediction = int(model.predict(input_df)[0])

    log_df = input_df.copy()
    log_df["prediction"] = prediction

    if os.path.exists(LIVE_DATA_PATH):
        log_df.to_csv(LIVE_DATA_PATH, mode="a", header=False, index=False)
    else:
        log_df.to_csv(LIVE_DATA_PATH, index=False)

    return {"prediction": prediction}
