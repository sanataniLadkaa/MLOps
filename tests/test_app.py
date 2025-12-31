from fastapi.testclient import TestClient
from unittest.mock import MagicMock
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import app

# 🔥 MOCK MODEL
app.model = MagicMock()
app.model.predict.return_value = [1]

client = TestClient(app.app)

def test_predict():
    payload = {"features": [0.1] * 15}
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json()["prediction"] == 1
