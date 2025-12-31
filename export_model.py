# export_model.py
import mlflow
import os

MODEL_NAME = "demo_classifier"
MODEL_ALIAS = "champion"

OUTPUT_DIR = "model_artifacts"

os.makedirs(OUTPUT_DIR, exist_ok=True)

model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

mlflow.pyfunc.save_model(
    dst_path=OUTPUT_DIR,
    python_model=mlflow.pyfunc.load_model(model_uri)
)

print("✅ Model exported to model_artifacts/")
