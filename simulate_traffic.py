import random
import requests
import time

URL = "http://127.0.0.1:8000/predict"

def random_features():
    return [random.uniform(-5, 5) for _ in range(15)]

for _ in range(50):
    payload = {"features": random_features()}
    r = requests.post(URL, json=payload)
    print(r.json())
    time.sleep(0.5)
