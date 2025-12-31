import os
import json
import pandas as pd

from evidently.report import Report
from evidently.metrics import DataDriftTable

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAINING_DATA_PATH = os.path.join(BASE_DIR, "training_data.csv")
LIVE_DATA_PATH = os.path.join(BASE_DIR, "live_predictions.csv")
DRIFT_REPORT_PATH = os.path.join(BASE_DIR, "drift_report.html")
DRIFT_FLAG_PATH = os.path.join(BASE_DIR, "drift_flag.json")

DRIFT_THRESHOLD = 0.30

# ==========================
# LOAD DATA
# ==========================
train_df = pd.read_csv(TRAINING_DATA_PATH)
live_df = pd.read_csv(LIVE_DATA_PATH)

train_features = train_df.filter(like="feature_")
live_features = live_df.filter(like="feature_")

# Align columns
live_features = live_features[train_features.columns]

# ==========================
# DRIFT REPORT
# ==========================
report = Report(metrics=[
    DataDriftTable()
])

report.run(
    reference_data=train_features,
    current_data=live_features
)

# Save HTML
report.save_html(DRIFT_REPORT_PATH)

# ==========================
# DRIFT LOGIC
# ==========================
report_dict = report.as_dict()

drifted = sum(
    1
    for col in report_dict["metrics"][0]["result"]["drift_by_columns"].values()
    if col["drift_detected"]
)

total = train_features.shape[1]
drift_ratio = drifted / total
drift_detected = drift_ratio >= DRIFT_THRESHOLD

with open(DRIFT_FLAG_PATH, "w") as f:
    json.dump(
        {
            "drift_detected": drift_detected,
            "drift_ratio": drift_ratio,
            "threshold": DRIFT_THRESHOLD,
        },
        f,
        indent=2,
    )

print("Drift ratio:", drift_ratio)
print("Drift detected:", drift_detected)
