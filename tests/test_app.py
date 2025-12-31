from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_predict():
    payload = {
        "features": [0.1] * 15
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()
