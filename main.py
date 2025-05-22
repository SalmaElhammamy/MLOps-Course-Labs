from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
import time
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response

app = FastAPI(title="Bank Customer Churn Prediction API")

# Load your trained model
model = joblib.load("model.pkl")

# Define the input schema
class CustomerData(BaseModel):
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

# Prometheus metrics
REQUEST_COUNT = Counter(
    'http_requests_total', 'Total HTTP requests',
    ['method', 'endpoint', 'http_status']
)

REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds', 'Duration of HTTP requests in seconds',
    ['method', 'endpoint']
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Churn Prediction API"}

@app.get("/health")
def health_check():
    return {"status": "OK"}

@app.post("/predict")
async def predict(data: CustomerData, request: Request):
    start_time = time.time()

    input_df = data.dict()
    input_df = [input_df]
    prediction = model.predict(input_df)

    duration = time.time() - start_time

    # Update Prometheus metrics
    REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path, http_status=200).inc()
    REQUEST_LATENCY.labels(method=request.method, endpoint=request.url.path).observe(duration)

    return {"prediction": int(prediction[0])}

@app.get("/metrics")
def metrics():
    # Expose Prometheus metrics
    return Response(generate_latest(), media_type="text/plain")
