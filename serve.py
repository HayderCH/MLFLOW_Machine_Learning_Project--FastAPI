from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from model import load_model
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import json
from elasticsearch import Elasticsearch
from time import time
import os

app = FastAPI(
    title="Parkinson's Disease Prediction API",
    description="API for predicting Parkinson's disease from features using a trained ML model.",
    version="1.0.0",
)

# SQLAlchemy setup (SQLite for simplicity)
engine = create_engine("sqlite:///./predictions.db")
Base = declarative_base()
SessionLocal = sessionmaker(bind=engine)


class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    features = Column(String)
    prediction = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(bind=engine)


# Define the input data schema
class SampleInput(BaseModel):
    features: list[float]


@app.on_event("startup")
def load_artifacts():
    global model, scaler, es
    model, scaler = load_model()
    es_host = os.getenv("ELASTIC_HOST", "localhost")
    es_port = int(os.getenv("ELASTIC_PORT", 9200))
    es = Elasticsearch([{"host": es_host, "port": es_port, "scheme": "http"}])


@app.post("/predict")
def predict(input_data: SampleInput):
    start = time()
    sample = np.array(input_data.features).reshape(1, -1)
    sample_scaled = scaler.transform(sample)
    prediction = model.predict(sample_scaled)[0]
    elapsed = time() - start
    # Log prediction to DB
    db = SessionLocal()
    db.add(
        Prediction(
            features=json.dumps(input_data.features),
            prediction=int(prediction),
            timestamp=datetime.utcnow(),
        )
    )
    db.commit()
    db.close()
    # Log to Elasticsearch
    try:
        es.index(
            index="parkinsons_predictions",
            document={
                "features": input_data.features,
                "prediction": int(prediction),
                "timestamp": datetime.utcnow().isoformat(),
                "latency": elapsed,
            },
        )
    except Exception as e:
        print(f"Failed to log to Elasticsearch: {e}")
    return {"prediction": int(prediction)}


@app.get("/history")
def get_history():
    db = SessionLocal()
    results = db.query(Prediction).order_by(Prediction.timestamp.desc()).limit(50).all()
    db.close()
    return [
        {
            "id": p.id,
            "features": json.loads(p.features),
            "prediction": p.prediction,
            "timestamp": p.timestamp.isoformat(),
        }
        for p in results
    ]


@app.get("/")
def read_root():
    return {"message": "API is up. Use /docs for interactive Swagger UI."}
