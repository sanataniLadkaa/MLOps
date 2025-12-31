import mlflow.pyfunc
import pandas as pd

model = mlflow.pyfunc.load_model("models:/demo_classifier/Staging")

# Underlying sklearn model
sk_model = model._model_impl.sklearn_model

print("Expected features:", sk_model.n_features_in_)
