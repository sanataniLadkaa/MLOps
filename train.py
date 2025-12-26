import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score
from data import generate_data
from models import get_models

# ==========================
# CONFIG
# ==========================
EXPERIMENT_NAME = "Multi_Model_Training"
REGISTERED_MODEL_NAME = "demo_classifier"
ACCURACY_GATE = 0.85

mlflow.set_experiment(EXPERIMENT_NAME)

# ==========================
# DATA
# ==========================
X_train, X_test, y_train, y_test = generate_data()

# ==========================
# SAVE TRAINING BASELINE (FOR DRIFT)
# ==========================
train_df = pd.DataFrame(
    X_train,
    columns=[f"feature_{i}" for i in range(X_train.shape[1])]
)
train_df.to_csv("training_data.csv", index=False)

# ==========================
# MODELS
# ==========================
models = get_models()
results = {}

best_run = None
best_acc = 0.0

# ==========================
# TRAINING LOOP
# ==========================
for name, model in models.items():

    with mlflow.start_run(run_name=name) as run:

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        # ---- Log params ----
        mlflow.log_param("model_name", name)
        for p, v in model.get_params().items():
            mlflow.log_param(p, v)

        # ---- Log metrics ----
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        # ---- Drift baseline stats ----
        mlflow.log_metric("feature_mean", float(np.mean(X_train)))
        mlflow.log_metric("feature_std", float(np.std(X_train)))

        # ---- Log model ----
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"{name} | accuracy={acc:.4f}")

        # ==========================
        # METRIC GATE
        # ==========================
        if acc >= ACCURACY_GATE:
            print(f"✅ {name} passed metric gate")

            model_uri = f"runs:/{run.info.run_id}/model"
            mv = mlflow.register_model(
                model_uri=model_uri,
                name=REGISTERED_MODEL_NAME
            )

            if acc > best_acc:
                best_acc = acc
                best_run = mv
        else:
            print(f"❌ {name} failed metric gate")

# ==========================
# PROMOTE BEST MODEL
# ==========================
if best_run:
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=best_run.version,
        stage="Staging",
        archive_existing_versions=True
    )

    print(
        f"\n🏆 Model v{best_run.version} promoted to STAGING "
        f"(accuracy={best_acc:.4f})"
    )
else:
    print("\n❌ No model passed metric gate")
