from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 422
    assert response.json() == {"message": "Welcome to the Churn Prediction API"}

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 422
    assert response.json() == {"status": "OK"}