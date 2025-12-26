import pandas as pd
import json
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# ==========================
# CONFIG
# ==========================
TRAINING_DATA_PATH = "training_data.csv"
LIVE_DATA_PATH = "live_predictions.csv"
DRIFT_REPORT_PATH = "drift_report.html"
DRIFT_FLAG_PATH = "drift_flag.json"

DRIFT_THRESHOLD = 0.30   # 30% features drifting

# ==========================
# LOAD DATA
# ==========================
train_df = pd.read_csv(TRAINING_DATA_PATH)
live_df = pd.read_csv(LIVE_DATA_PATH)

train_features = train_df.filter(like="feature_")
live_features = live_df.filter(like="feature_")

# ==========================
# RUN DRIFT REPORT
# ==========================
report = Report(metrics=[DataDriftPreset()])
report.run(
    reference_data=train_features,
    current_data=live_features
)

report.save_html(DRIFT_REPORT_PATH)

# ==========================
# EXTRACT DRIFT SCORE
# ==========================
report_dict = report.as_dict()

drifted_features = sum(
    1 for f in report_dict["metrics"][0]["result"]["drift_by_columns"].values()
    if f["drift_detected"]
)

total_features = train_features.shape[1]
drift_ratio = drifted_features / total_features

drift_detected = drift_ratio >= DRIFT_THRESHOLD

# ==========================
# SAVE DRIFT FLAG
# ==========================
with open(DRIFT_FLAG_PATH, "w") as f:
    json.dump({
        "drift_detected": drift_detected,
        "drift_ratio": drift_ratio,
        "threshold": DRIFT_THRESHOLD
    }, f, indent=2)

print("Drift ratio:", drift_ratio)
print("Drift detected:", drift_detected)
