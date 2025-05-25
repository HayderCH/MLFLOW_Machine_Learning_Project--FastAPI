from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from model import load_model

app = FastAPI(
    title="Parkinson's Disease Prediction API",
    description="API for predicting Parkinson's disease from features using a trained ML model.",
    version="1.0.0",
)


# Define the input data schema
class SampleInput(BaseModel):
    features: list[float]


@app.on_event("startup")
def load_artifacts():
    global model, scaler
    model, scaler = load_model()


@app.post("/predict")
def predict(input_data: SampleInput):
    sample = np.array(input_data.features).reshape(1, -1)
    sample_scaled = scaler.transform(sample)
    prediction = model.predict(sample_scaled)[0]
    return {"prediction": int(prediction)}


# Optional: root endpoint
@app.get("/")
def read_root():
    return {"message": "API is up. Use /docs for interactive Swagger UI."}
