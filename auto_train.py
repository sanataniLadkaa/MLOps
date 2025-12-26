import json
import subprocess
import sys

DRIFT_FLAG_PATH = "drift_flag.json"

with open(DRIFT_FLAG_PATH, "r") as f:
    drift_info = json.load(f)

if drift_info["drift_detected"]:
    print("🚨 Drift detected — retraining started")
    subprocess.run([sys.executable, "train.py"], check=True)
else:
    print("✅ No significant drift — retraining skipped")
