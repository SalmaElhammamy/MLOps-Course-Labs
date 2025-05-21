from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert "Welcome" in response.json()["message"]

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "OK"

def test_predict():
    response = client.post("/predict", json={"feature1": 1.5, "feature2": 2.3})
    assert response.status_code == 200
    assert "prediction" in response.json()
