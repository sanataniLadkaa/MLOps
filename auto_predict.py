import requests
import random
import time

URL = "http://127.0.0.1:8000/predict"

HEADERS = {
    "Content-Type": "application/json"
}

for i in range(35):  # send 20 live requests
    features = [random.randint(0, 5) for _ in range(15)]

    payload = {
        "features": features
    }

    response = requests.post(URL, json=payload, headers=HEADERS)

    print(f"Request {i+1} | Features: {features} | Response: {response.json()}")

    time.sleep(1)  # 1 second gap
