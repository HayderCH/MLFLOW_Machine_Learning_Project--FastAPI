# Parkinson's Disease Prediction — MLOps Project

**Academic year:** 2024-2025  
**Student:** HayderCH  
**Project:** MLOps Excellence — End-to-End ML System with Monitoring

---

## 🚀 Project Overview

This repository demonstrates a full MLOps workflow for Parkinson's Disease prediction, starting from modularized ML code to automated CI/CD, experiment tracking, REST API deployment, containerization, and advanced monitoring with Elastic Stack (Elasticsearch & Kibana).

- **Model:** XGBoost-based classifier predicting Parkinson's Disease from biomedical features.
- **API:** FastAPI serving predictions and storing history.
- **Experiment Tracking:** MLflow with model registry and custom backend.
- **CI/CD:** Makefile, GitHub Actions, linters, auto-formatting, and test automation.
- **Containerization:** Docker & Docker Compose.
- **Monitoring:** Real-time prediction logs and metrics in Kibana.

---

## 📁 Project Structure

```
.
├── data.py              # Data loading and preprocessing module
├── model.py             # Model training, evaluation, and saving
├── train.py             # MLflow experiment tracking & model registration
├── serve.py             # FastAPI app with prediction, logging, and monitoring
├── requirements.txt     # Python dependencies
├── Dockerfile           # Build FastAPI app container
├── docker-compose.yml   # Multi-service orchestration (API, Elasticsearch, Kibana)
├── Makefile             # CI/CD automation commands
├── .github/workflows/   # GitHub Actions for CI/CD
└── ...                  # Notebooks, configs, and other files
```

---

## 💡 Features for Excellence

### 1. **Modularized & Clean Code**
- Clear separation between data, modeling, API, and utility logic.
- No circular dependencies; each file serves a single responsibility.

### 2. **Automated CI/CD**
- **Makefile** for local automation (linting, formatting, testing).
- **GitHub Actions:** Automated pipeline for code quality, formatting, and unit testing.
- Linters: Pylint, Flake8, MyPy.
- Code formatter: Black.
- Easily extensible to Jenkins, SonarQube, or email notifications.

### 3. **MLflow Integration**
- Tracks experiments, hyperparameters, and metrics.
- Custom SQLite backend for portability.
- **Model Registry:** Automatic registration and promotion to production stage.
- Model versioning and artifacts tracked.

### 4. **REST API with FastAPI**
- `/predict` endpoint: Get predictions from the latest model.
- `/history` endpoint: View recent prediction history (persisted in SQLite).
- **Well-documented:** OpenAPI/Swagger UI at `/docs`.
- Ready for extension: add `/retrain` endpoint or alternative frameworks (Flask, Django).

### 5. **Docker & Docker Compose**
- One container for FastAPI, another for Elasticsearch, another for Kibana.
- SQLite used for fast local prototyping; easily swapped for PostgreSQL.
- Multi-container orchestration for production-like workflow.

### 6. **Real-Time Monitoring & Logging**
- Every prediction is logged to **Elasticsearch** (features, result, latency, timestamp).
- **Kibana dashboard:** Visualize predictions, latency, and API usage in real time.
- Ready for further monitoring (system metrics, error rates, etc.).

---

## 🏗️ How to Run

### 1. **Clone the repo & set up environment**
```sh
git clone https://github.com/HayderCH/MLFLOW_Machine_Learning_Project--FastAPI.git
cd MLFLOW_Machine_Learning_Project--FastAPI
```

### 2. **Build and Start All Services**
```sh
docker-compose up --build
```
- This will launch:
  - FastAPI app at [http://localhost:8000](http://localhost:8000)
  - Elasticsearch at [http://localhost:9200](http://localhost:9200)
  - Kibana dashboard at [http://localhost:5601](http://localhost:5601)

### 3. **Test the API**
- Visit [http://localhost:8000/docs](http://localhost:8000/docs) for interactive documentation.
- Example prediction:
  ```json
  POST /predict
  {
    "features": [197.9, 0.00755, 0.00007, 0.00370, 0.02087, 0.00017, 0.00577, 0.00981, 0.01650, 0.01747, 0.02448, 0.00220, 0.00322, 0.00566, 0.00681, 0.00867, 0.00908, 0.01660, 0.00023, 0.00088, 0.00000, 0.00000]
  }
  ```

### 4. **Monitor Predictions in Kibana**
- Go to [http://localhost:5601](http://localhost:5601)
- Create an index pattern with `parkinsons_predictions*`
- Explore predictions, latency, and history.

---

## 🧑‍🔬 What to Demonstrate

- **Modular Code:** Show structure, no code duplication, each file’s responsibility.
- **CI/CD:** Show Makefile, GitHub Actions, sample lint/test output.
- **MLflow Registry:** Show UI or code for model registration and promotion.
- **API:** Live prediction demo via Swagger UI.
- **Monitoring:** Make a prediction, show it appears in Kibana instantly.
- **Docker Compose:** All services up with a single command.

---

## 🌟 Pistes d’Excellence Realized

- Modular, independent code structure.
- Multi-stage CI/CD with linters, formatters, and tests.
- MLflow: tracking, registry, and custom backend.
- Docker Compose with multi-container orchestration.
- Real-time monitoring with Elastic/Kibana.
- Easily extensible to system monitoring, retraining endpoint, or alternative APIs.

---

## 📚 Further Improvements (Ideas for Extra Excellence)

- Add `/retrain` endpoint to trigger model retraining via API.
- Log system/container metrics to Elasticsearch or Prometheus/Grafana.
- Integrate with Slack/email for CI notifications.
- Add more tests (API, integration, edge cases).
- Use Postgres for persistent prediction history.
- Deploy MLflow server as a service for remote experiment tracking.

---

## 📄 License

MIT — For educational purposes.

---

