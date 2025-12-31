import os
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from mlflow.models.signature import infer_signature

# =========================
# CONFIG
# =========================
MODEL_NAME = "demo_classifier"
N_FEATURES = 15
FEATURE_NAMES = [f"feature_{i}" for i in range(N_FEATURES)]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAINING_DATA_PATH = os.path.join(BASE_DIR, "training_data.csv")
MODEL_EXPORT_PATH = os.path.join(BASE_DIR, "model")  # 🔥 bundled model

# MLflow only for tracking (local)
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("demo_classifier_experiment")

# =========================
# GENERATE DATA
# =========================
X, y = make_classification(
    n_samples=2000,
    n_features=N_FEATURES,
    n_informative=8,
    n_redundant=2,
    n_classes=2,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train_df = pd.DataFrame(X_train, columns=FEATURE_NAMES)
X_test_df = pd.DataFrame(X_test, columns=FEATURE_NAMES)

# Save training data (for drift later)
train_df = X_train_df.copy()
train_df["target"] = y_train
train_df.to_csv(TRAINING_DATA_PATH, index=False)

# =========================
# TRAIN MODEL
# =========================
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_df, y_train)

# =========================
# EVALUATE
# =========================
y_pred = model.predict(X_test_df)
accuracy = accuracy_score(y_test, y_pred)

# =========================
# LOG + EXPORT MODEL
# =========================
with mlflow.start_run() as run:
    mlflow.log_param("model_type", "KNN")
    mlflow.log_param("n_neighbors", 5)
    mlflow.log_param("n_features", N_FEATURES)
    mlflow.log_metric("accuracy", accuracy)

    signature = infer_signature(X_train_df, model.predict(X_train_df))
    input_example = X_train_df.iloc[:5]

    # Log to MLflow (tracking only)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        input_example=input_example,
    )

    # 🔥 EXPORT MODEL LOCALLY (KEY FIX)
    mlflow.sklearn.save_model(
        sk_model=model,
        path=MODEL_EXPORT_PATH,
        signature=signature,
        input_example=input_example,
    )

    print(f"✅ Model trained | accuracy={accuracy:.4f}")
    print(f"📦 Model exported to: {MODEL_EXPORT_PATH}")
