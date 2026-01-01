from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List
import mlflow.pyfunc
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model")
LIVE_DATA_PATH = os.path.join(BASE_DIR, "live_predictions.csv")

FEATURE_NAMES = [f"feature_{i}" for i in range(15)]
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

app = FastAPI(
    title="ML Inference Service",
    description="Stable MLflow inference service",
    version="1.0.0",
)

# Load model
try:
    model = mlflow.pyfunc.load_model(MODEL_PATH)
    print("✅ Model loaded")
except Exception as e:
    print("❌ Model load failed:", e)
    model = None

class PredictionInput(BaseModel):
    features: List[float]

@app.get("/", response_class=HTMLResponse)
def ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
def predict(data: PredictionInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if len(data.features) != len(FEATURE_NAMES):
        raise HTTPException(
            status_code=400,
            detail=f"Expected {len(FEATURE_NAMES)} features"
        )

    df = pd.DataFrame([data.features], columns=FEATURE_NAMES)
    pred = int(model.predict(df)[0])

    df["prediction"] = pred
    df.to_csv(
        LIVE_DATA_PATH,
        mode="a",
        header=not os.path.exists(LIVE_DATA_PATH),
        index=False
    )

    return {"prediction": pred}
