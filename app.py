from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import mlflow.pyfunc
import pandas as pd
import os

# =========================
# CONFIG
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model")
LIVE_DATA_PATH = os.path.join(BASE_DIR, "live_predictions.csv")

FEATURE_NAMES = [f"feature_{i}" for i in range(15)]

# =========================
# FASTAPI APP
# =========================
app = FastAPI(
    title="ML Inference Service",
    description="Bundled MLflow model inference",
    version="1.0.0",
)

# =========================
# LOAD MODEL (ONCE)
# =========================
try:
    model = mlflow.pyfunc.load_model(MODEL_PATH)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None

# =========================
# INPUT SCHEMA
# =========================
class PredictionInput(BaseModel):
    features: List[float]

# =========================
# HEALTH CHECK (IMPORTANT)
# =========================
@app.get("/")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None
    }

# =========================
# PREDICT ENDPOINT
# =========================
@app.post("/predict")
def predict(data: PredictionInput):

    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Model not loaded"
        )

    if len(data.features) != len(FEATURE_NAMES):
        raise HTTPException(
            status_code=400,
            detail=f"Expected {len(FEATURE_NAMES)} features, got {len(data.features)}"
        )

    input_df = pd.DataFrame([data.features], columns=FEATURE_NAMES)
    prediction = int(model.predict(input_df)[0])

    log_df = input_df.copy()
    log_df["prediction"] = prediction

    if os.path.exists(LIVE_DATA_PATH):
        log_df.to_csv(LIVE_DATA_PATH, mode="a", header=False, index=False)
    else:
        log_df.to_csv(LIVE_DATA_PATH, index=False)

    return {"prediction": prediction}
