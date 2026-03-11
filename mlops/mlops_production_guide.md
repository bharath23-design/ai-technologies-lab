# MLOps Production Guide — Start to End

A complete reference for taking ML models from development to production.

---

## Table of Contents

1. [MLOps Overview](#1-mlops-overview)
2. [Data Versioning — DVC](#2-data-versioning--dvc)
3. [Experiment Tracking — MLflow](#3-experiment-tracking--mlflow)
4. [Model Registry](#4-model-registry)
5. [Feature Store](#5-feature-store)
6. [Pipeline Orchestration — Airflow](#6-pipeline-orchestration--airflow)
7. [Model Serving — FastAPI + Docker](#7-model-serving--fastapi--docker)
8. [CI/CD — GitHub Actions](#8-cicd--github-actions)
9. [Monitoring — Evidently + Prometheus + Grafana](#9-monitoring--evidently--prometheus--grafana)
10. [Infrastructure — Kubernetes](#10-infrastructure--kubernetes)
11. [MLOps Maturity Levels](#11-mlops-maturity-levels)
12. [Full Stack Summary](#12-full-stack-summary)

---

## 1. MLOps Overview

MLOps = DevOps principles applied to the ML lifecycle. It covers everything from data ingestion to model retirement.

```
Data → Feature Engineering → Training → Evaluation → Registry → Serving → Monitoring → Retraining
```

### Core Pillars

| Pillar | What it solves | Tools |
|---|---|---|
| Data versioning | Reproducibility of datasets | DVC, LakeFS |
| Experiment tracking | Compare runs, metrics, params | MLflow, W&B |
| Model registry | Versioned model promotion | MLflow Registry |
| Feature store | Prevent train/serve skew | Feast, Tecton |
| Orchestration | Automate pipelines end-to-end | Airflow, Prefect |
| Serving | Low-latency inference APIs | FastAPI, TorchServe, Triton |
| CI/CD | Automated test → train → deploy | GitHub Actions, Jenkins |
| Monitoring | Detect drift, degradation | Evidently, Prometheus, Grafana |
| Infrastructure | Scalable container orchestration | Kubernetes, Helm |

---

## 2. Data Versioning — DVC

**Problem**: Git can't track large datasets or model artifacts.
**Solution**: DVC tracks data with Git-like semantics, stores actual files in remote storage (S3, GCS, Azure).

### Setup

```bash
pip install dvc dvc-s3        # install DVC with S3 remote
git init
dvc init                      # creates .dvc/ config dir
git add .dvc .dvcignore
git commit -m "init dvc"
```

### Add Remote Storage

```bash
# S3 remote
dvc remote add -d myremote s3://my-bucket/dvc-store
dvc remote modify myremote region us-east-1

# OR local remote (for testing)
dvc remote add -d localremote /tmp/dvc-storage
```

### Track Data

```bash
dvc add data/raw/train.csv         # creates data/raw/train.csv.dvc
git add data/raw/train.csv.dvc data/raw/.gitignore
git commit -m "track raw training data v1"
dvc push                           # push actual data to remote
```

### Reproduce & Pull

```bash
dvc pull                           # download data from remote
dvc repro                          # reproduce pipeline stages
```

### DVC Pipeline (dvc.yaml)

```yaml
stages:
  preprocess:
    cmd: python src/preprocess.py
    deps:
      - data/raw/train.csv
      - src/preprocess.py
    outs:
      - data/processed/train_clean.csv

  train:
    cmd: python src/train.py
    deps:
      - data/processed/train_clean.csv
      - src/train.py
    params:
      - params.yaml:
          - model.n_estimators
          - model.max_depth
    outs:
      - models/model.pkl
    metrics:
      - metrics/scores.json:
          cache: false
```

```bash
dvc repro          # runs only changed stages
dvc dag            # visualize pipeline DAG
dvc params diff    # see param changes
dvc metrics show   # show metrics
```

---

## 3. Experiment Tracking — MLflow

**Problem**: No way to compare training runs, store artifacts, or reproduce results.
**Solution**: MLflow logs params, metrics, artifacts per run with a UI.

### Setup

```bash
pip install mlflow scikit-learn
mlflow ui                    # starts UI at http://localhost:5000
```

### Basic Tracking

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

mlflow.set_tracking_uri("http://localhost:5000")  # remote or local
mlflow.set_experiment("iris-classifier")

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run(run_name="rf-baseline"):
    # Log hyperparameters
    n_estimators = 100
    max_depth = 5
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # Train
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy_score(y_test, preds))
    mlflow.log_metric("f1_score", f1_score(y_test, preds, average="weighted"))

    # Log model
    mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name="IrisClassifier")

    # Log artifacts (e.g., confusion matrix plot)
    # mlflow.log_artifact("plots/confusion_matrix.png")

print(f"Run ID: {mlflow.active_run().info.run_id}")
```

### Autologging (zero-code tracking)

```python
mlflow.sklearn.autolog()   # logs all params, metrics, model automatically
# same for: mlflow.xgboost.autolog(), mlflow.pytorch.autolog(), mlflow.tensorflow.autolog()
```

### Remote Tracking Server (Production)

```bash
# Start MLflow server with PostgreSQL backend + S3 artifact store
mlflow server \
  --backend-store-uri postgresql://user:pass@localhost/mlflow \
  --default-artifact-root s3://my-bucket/mlflow-artifacts \
  --host 0.0.0.0 \
  --port 5000
```

---

## 4. Model Registry

**Problem**: No controlled promotion of models between dev → staging → production.
**Solution**: MLflow Model Registry manages model versions with lifecycle stages.

### Register & Promote a Model

```python
from mlflow.tracking import MlflowClient

client = MlflowClient(tracking_uri="http://localhost:5000")

# Register a model from a run
run_id = "your-run-id-here"
model_uri = f"runs:/{run_id}/model"
mv = client.create_model_version(name="IrisClassifier", source=model_uri, run_id=run_id)
print(f"Model version: {mv.version}")

# Promote to Staging
client.transition_model_version_stage(
    name="IrisClassifier",
    version=mv.version,
    stage="Staging",
    archive_existing_versions=False
)

# Promote to Production after validation
client.transition_model_version_stage(
    name="IrisClassifier",
    version=mv.version,
    stage="Production",
    archive_existing_versions=True   # archives old production version
)
```

### Load a Production Model

```python
import mlflow.pyfunc

model = mlflow.pyfunc.load_model("models:/IrisClassifier/Production")
predictions = model.predict(X_test)
```

### Model Aliases (MLflow 2.x+)

```python
client.set_registered_model_alias("IrisClassifier", "champion", version=3)
model = mlflow.pyfunc.load_model("models:/IrisClassifier@champion")
```

---

## 5. Feature Store

**Problem**: Training and serving compute features differently → silent model degradation (train/serve skew).
**Solution**: A feature store computes features once, serves them consistently for both training and inference.

### Feast Setup

```bash
pip install feast[redis]
feast init feature_repo
cd feature_repo
```

### feature_store.yaml

```yaml
project: my_ml_project
registry: data/registry.db
provider: local
online_store:
  type: redis
  connection_string: "localhost:6379"
offline_store:
  type: file
```

### Define Features (features.py)

```python
from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64

customer = Entity(name="customer_id", join_keys=["customer_id"])

customer_stats_source = FileSource(
    path="data/customer_stats.parquet",
    timestamp_field="event_timestamp",
)

customer_stats_fv = FeatureView(
    name="customer_stats",
    entities=[customer],
    ttl=timedelta(days=1),
    schema=[
        Field(name="purchase_count", dtype=Int64),
        Field(name="avg_order_value", dtype=Float32),
        Field(name="days_since_last_purchase", dtype=Int64),
    ],
    online=True,
    source=customer_stats_source,
)
```

### Materialize & Retrieve

```bash
feast apply                          # register features
feast materialize-incremental $(date -u +"%Y-%m-%dT%H:%M:%S")  # sync to online store
```

```python
from feast import FeatureStore

store = FeatureStore(repo_path="feature_repo/")

# Training: offline retrieval
entity_df = pd.DataFrame({"customer_id": [1, 2, 3], "event_timestamp": [...]})
training_df = store.get_historical_features(
    entity_df=entity_df,
    features=["customer_stats:purchase_count", "customer_stats:avg_order_value"]
).to_df()

# Serving: online retrieval (low latency)
features = store.get_online_features(
    features=["customer_stats:purchase_count", "customer_stats:avg_order_value"],
    entity_rows=[{"customer_id": 1}]
).to_dict()
```

---

## 6. Pipeline Orchestration — Airflow

**Problem**: Training, evaluation, and deployment steps run manually → not reproducible, not scheduled.
**Solution**: Airflow DAGs define the full pipeline as code with scheduling and dependency management.

### Install

```bash
pip install apache-airflow
airflow db init
airflow webserver --port 8080    # UI
airflow scheduler                # in another terminal
```

### ML Training DAG

```python
# dags/ml_training_pipeline.py
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "mlops-team",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": True,
    "email": ["mlops@company.com"],
}

def ingest_data(**context):
    """Pull latest data from source."""
    import pandas as pd
    df = pd.read_parquet("s3://bucket/raw/data.parquet")
    df.to_parquet("/tmp/raw_data.parquet")
    context["ti"].xcom_push(key="row_count", value=len(df))

def preprocess(**context):
    """Clean and feature engineer."""
    import pandas as pd
    df = pd.read_parquet("/tmp/raw_data.parquet")
    # ... cleaning logic
    df.to_parquet("/tmp/processed_data.parquet")

def train_model(**context):
    """Train and log to MLflow."""
    import mlflow
    import pandas as pd
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    df = pd.read_parquet("/tmp/processed_data.parquet")
    X, y = df.drop("target", axis=1), df["target"]

    with mlflow.start_run():
        model = GradientBoostingClassifier(n_estimators=200)
        model.fit(X, y)
        auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
        mlflow.log_metric("auc", auc)
        mlflow.sklearn.log_model(model, "model")
        context["ti"].xcom_push(key="auc", value=auc)

def check_model_quality(**context):
    """Branch: promote if AUC > threshold, else alert."""
    auc = context["ti"].xcom_pull(task_ids="train", key="auc")
    return "promote_model" if auc >= 0.85 else "alert_low_performance"

def promote_model(**context):
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    # promote latest version to production
    print("Model promoted to production")

with DAG(
    dag_id="ml_training_pipeline",
    default_args=default_args,
    schedule_interval="0 2 * * 1",   # every Monday at 2am
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["ml", "production"],
) as dag:

    ingest   = PythonOperator(task_id="ingest",   python_callable=ingest_data)
    preproc  = PythonOperator(task_id="preprocess", python_callable=preprocess)
    train    = PythonOperator(task_id="train",    python_callable=train_model)
    evaluate = BranchPythonOperator(task_id="evaluate", python_callable=check_model_quality)
    promote  = PythonOperator(task_id="promote_model", python_callable=promote_model)
    alert    = BashOperator(task_id="alert_low_performance", bash_command="echo 'Low AUC — skipping promotion'")

    ingest >> preproc >> train >> evaluate >> [promote, alert]
```

---

## 7. Model Serving — FastAPI + Docker

**Problem**: Models need to be accessible as REST APIs with low latency and health checks.
**Solution**: FastAPI wraps the model, Docker containerizes it, Gunicorn handles concurrency.

### FastAPI App (app/main.py)

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import numpy as np
import time
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(title="ML Model API", version="1.0.0")

# Load model from registry at startup
model = None

@app.on_event("startup")
def load_model():
    global model
    model = mlflow.pyfunc.load_model("models:/IrisClassifier/Production")

# Expose Prometheus metrics at /metrics
Instrumentator().instrument(app).expose(app)

class PredictRequest(BaseModel):
    features: list[float]

class PredictResponse(BaseModel):
    prediction: int
    probability: float
    latency_ms: float

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.get("/ready")
def ready():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    start = time.time()
    input_array = np.array(request.features).reshape(1, -1)
    pred = model.predict(input_array)
    # For classifiers with predict_proba
    latency = (time.time() - start) * 1000
    return PredictResponse(prediction=int(pred[0]), probability=0.95, latency_ms=latency)
```

### requirements.txt

```
fastapi==0.115.0
uvicorn[standard]==0.30.0
mlflow==2.17.0
scikit-learn==1.5.0
numpy==1.26.0
prometheus-fastapi-instrumentator==7.0.0
pydantic==2.8.0
gunicorn==22.0.0
```

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app/ ./app/

# Non-root user for security
RUN adduser --disabled-password --gecos "" appuser
USER appuser

EXPOSE 8000

# Gunicorn with Uvicorn workers for production
CMD ["gunicorn", "app.main:app", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120", \
     "--access-logfile", "-"]
```

### docker-compose.yml

```yaml
version: "3.9"

services:
  ml-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
      - MODEL_NAME=IrisClassifier
      - MODEL_STAGE=Production
    depends_on:
      - mlflow
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.17.0
    ports:
      - "5000:5000"
    command: >
      mlflow server
      --backend-store-uri sqlite:///mlflow.db
      --default-artifact-root /mlflow/artifacts
      --host 0.0.0.0
    volumes:
      - mlflow_data:/mlflow

  prometheus:
    image: prom/prometheus:v2.53.0
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:11.1.0
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
    depends_on:
      - prometheus

volumes:
  mlflow_data:
  grafana_data:
```

---

## 8. CI/CD — GitHub Actions

**Problem**: Manual testing and deployment is error-prone and slow.
**Solution**: Automated pipeline that runs tests → trains → evaluates → deploys on every merge.

### .github/workflows/ml_pipeline.yml

```yaml
name: ML CI/CD Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
  AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
  AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}

jobs:
  # ── 1. Code Quality ──────────────────────────────────────────────────
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"
      - run: pip install -r requirements.txt -r requirements-dev.txt
      - run: ruff check .                    # linting
      - run: mypy src/                       # type checking
      - run: pytest tests/ -v --cov=src --cov-report=xml
      - uses: codecov/codecov-action@v4

  # ── 2. Data Validation ───────────────────────────────────────────────
  data-validation:
    runs-on: ubuntu-latest
    needs: lint-and-test
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install great-expectations dvc
      - run: dvc pull data/processed/        # pull processed data
      - run: great_expectations checkpoint run data_validation_checkpoint

  # ── 3. Train & Evaluate ──────────────────────────────────────────────
  train-and-evaluate:
    runs-on: ubuntu-latest
    needs: data-validation
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -r requirements.txt
      - run: dvc pull
      - name: Train model
        run: python src/train.py --config configs/train_config.yaml
      - name: Evaluate against champion
        run: python src/evaluate.py --challenger-run-id ${{ steps.train.outputs.run_id }}
      - name: Gate on AUC threshold
        run: |
          AUC=$(python src/get_metric.py --metric auc)
          if (( $(echo "$AUC < 0.85" | bc -l) )); then
            echo "AUC $AUC below threshold 0.85 — failing"
            exit 1
          fi

  # ── 4. Build & Push Docker Image ─────────────────────────────────────
  build-and-push:
    runs-on: ubuntu-latest
    needs: train-and-evaluate
    steps:
      - uses: actions/checkout@v4
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - uses: docker/build-push-action@v5
        with:
          push: true
          tags: ghcr.io/${{ github.repository }}/ml-api:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  # ── 5. Deploy to Kubernetes ───────────────────────────────────────────
  deploy:
    runs-on: ubuntu-latest
    needs: build-and-push
    environment: production
    steps:
      - uses: actions/checkout@v4
      - uses: azure/k8s-set-context@v3
        with:
          kubeconfig: ${{ secrets.KUBECONFIG }}
      - name: Rolling deploy
        run: |
          kubectl set image deployment/ml-api \
            ml-api=ghcr.io/${{ github.repository }}/ml-api:${{ github.sha }} \
            -n ml-production
          kubectl rollout status deployment/ml-api -n ml-production --timeout=5m
```

### Model Evaluation Gate Script (src/evaluate.py)

```python
"""Compare challenger vs champion. Fail if challenger doesn't improve."""
import argparse
import mlflow
from mlflow.tracking import MlflowClient
import sys

def evaluate(challenger_run_id: str, metric: str = "auc", threshold: float = 0.85):
    client = MlflowClient()

    # Get challenger metric
    challenger_run = client.get_run(challenger_run_id)
    challenger_score = challenger_run.data.metrics[metric]

    # Get champion (current production) metric
    try:
        prod_versions = client.get_latest_versions("IrisClassifier", stages=["Production"])
        champion_run_id = prod_versions[0].run_id
        champion_run = client.get_run(champion_run_id)
        champion_score = champion_run.data.metrics[metric]
    except IndexError:
        champion_score = threshold  # no champion yet, use threshold

    print(f"Champion {metric}: {champion_score:.4f}")
    print(f"Challenger {metric}: {challenger_score:.4f}")

    if challenger_score < threshold:
        print(f"FAIL: challenger below absolute threshold {threshold}")
        sys.exit(1)
    if challenger_score < champion_score:
        print(f"FAIL: challenger does not beat champion")
        sys.exit(1)

    print("PASS: challenger beats champion — promoting")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenger-run-id", required=True)
    args = parser.parse_args()
    evaluate(args.challenger_run_id)
```

---

## 9. Monitoring — Evidently + Prometheus + Grafana

**Problem**: Models degrade silently in production due to data drift or concept drift.
**Solution**: Track data distributions and model metrics continuously; alert on anomalies.

### Types of Drift to Monitor

| Type | What drifts | Detection method |
|---|---|---|
| Data drift | Input feature distribution | PSI, KS test, Jensen-Shannon |
| Concept drift | Relationship between X and y | Monitor prediction errors over time |
| Label drift | Target distribution | Compare training vs production label dist |
| Model quality | Accuracy, F1, AUC | Compare to baseline thresholds |

### Evidently Drift Report (Batch Monitoring)

```python
# monitoring/drift_report.py
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, ClassificationPreset

# Load reference (training) and current (production) data
reference_df = pd.read_parquet("data/reference/training_data.parquet")
current_df   = pd.read_parquet("data/production/last_7_days.parquet")

# Data drift + quality report
report = Report(metrics=[
    DataDriftPreset(),
    DataQualityPreset(),
    ClassificationPreset(),   # requires "target" and "prediction" columns
])

report.run(reference_data=reference_df, current_data=current_df)
report.save_html("reports/drift_report.html")

# Parse results programmatically
result = report.as_dict()
drift_detected = result["metrics"][0]["result"]["dataset_drift"]
if drift_detected:
    print("DATA DRIFT DETECTED — alerting team")
    # send_slack_alert("Drift detected in production features!")
```

### Evidently Test Suite (Pass/Fail Gates)

```python
from evidently.test_suite import TestSuite
from evidently.tests import (
    TestNumberOfMissingValues,
    TestShareOfDriftedColumns,
    TestColumnDrift,
    TestAccuracyScore,
)

suite = TestSuite(tests=[
    TestNumberOfMissingValues(lt=0.05),           # < 5% missing
    TestShareOfDriftedColumns(lt=0.3),            # < 30% features drifted
    TestColumnDrift(column_name="feature_1"),
    TestAccuracyScore(gte=0.80),                  # accuracy >= 80%
])

suite.run(reference_data=reference_df, current_data=current_df)
suite.save_html("reports/test_suite.html")

if not suite.as_dict()["summary"]["all_passed"]:
    raise Exception("Monitoring tests failed — check drift report")
```

### Real-time Monitoring with Prometheus

```python
# app/metrics.py — custom Prometheus metrics
from prometheus_client import Counter, Histogram, Gauge

PREDICTION_COUNTER = Counter(
    "ml_predictions_total",
    "Total number of predictions",
    ["model_version", "prediction_class"]
)

PREDICTION_LATENCY = Histogram(
    "ml_prediction_latency_seconds",
    "Prediction latency in seconds",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

FEATURE_VALUE = Histogram(
    "ml_feature_value",
    "Distribution of feature values",
    ["feature_name"]
)

MODEL_ACCURACY = Gauge(
    "ml_model_accuracy",
    "Current model accuracy on recent predictions",
    ["model_name"]
)
```

### Prometheus Config (monitoring/prometheus.yml)

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

alerting:
  alertmanagers:
    - static_configs:
        - targets: ["alertmanager:9093"]

rule_files:
  - "alert_rules.yml"

scrape_configs:
  - job_name: "ml-api"
    static_configs:
      - targets: ["ml-api:8000"]
    metrics_path: /metrics

  - job_name: "node-exporter"
    static_configs:
      - targets: ["node-exporter:9100"]
```

### Alert Rules (monitoring/alert_rules.yml)

```yaml
groups:
  - name: ml-model-alerts
    rules:
      - alert: HighPredictionLatency
        expr: histogram_quantile(0.95, ml_prediction_latency_seconds_bucket) > 0.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "ML API p95 latency > 500ms"

      - alert: ModelAccuracyDrop
        expr: ml_model_accuracy < 0.80
        for: 10m
        labels:
          severity: critical
        annotations:
          summary: "Model accuracy dropped below 80%"

      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "API error rate > 5%"
```

---

## 10. Infrastructure — Kubernetes

**Problem**: Single-server deployment doesn't scale and has no self-healing.
**Solution**: Kubernetes manages container orchestration, auto-scaling, and rolling deployments.

### Namespace & ConfigMap

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: ml-production
  labels:
    environment: production
```

### Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-api
  namespace: ml-production
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-api
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0         # zero-downtime deployment
  template:
    metadata:
      labels:
        app: ml-api
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      containers:
        - name: ml-api
          image: ghcr.io/myorg/ml-api:latest
          ports:
            - containerPort: 8000
          env:
            - name: MLFLOW_TRACKING_URI
              valueFrom:
                secretKeyRef:
                  name: mlflow-secret
                  key: tracking-uri
          resources:
            requests:
              cpu: "500m"
              memory: "512Mi"
            limits:
              cpu: "2"
              memory: "2Gi"
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /ready
              port: 8000
            initialDelaySeconds: 15
            periodSeconds: 5
```

### Service & Ingress

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: ml-api-service
  namespace: ml-production
spec:
  selector:
    app: ml-api
  ports:
    - port: 80
      targetPort: 8000
  type: ClusterIP

---
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ml-api-ingress
  namespace: ml-production
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/rate-limit: "100"
spec:
  tls:
    - hosts:
        - api.mycompany.com
      secretName: ml-api-tls
  rules:
    - host: api.mycompany.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: ml-api-service
                port:
                  number: 80
```

### Horizontal Pod Autoscaler

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-api-hpa
  namespace: ml-production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-api
  minReplicas: 2
  maxReplicas: 20
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Pods
      pods:
        metric:
          name: ml_prediction_latency_seconds_p95
        target:
          type: AverageValue
          averageValue: "200m"   # 200ms
```

---

## 11. MLOps Maturity Levels

| Level | Name | Characteristics |
|---|---|---|
| **0** | Manual ML | Notebooks only, no automation, no versioning |
| **1** | ML pipelines | DVC + MLflow, manual deploys, some monitoring |
| **2** | Automated retraining | Airflow DAGs, CI/CD, basic drift detection |
| **3** | Full MLOps | Auto-retraining on drift, shadow deployments, full observability |

### Progression Path

```
Level 0 → Add DVC + MLflow           (reproducibility)
Level 1 → Add CI/CD + Docker         (reliable deployment)
Level 2 → Add Airflow + monitoring   (automation + observability)
Level 3 → Add auto-retraining + A/B  (closed-loop system)
```

---

## 12. Full Stack Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                        MLOps Production Stack                    │
├─────────────────┬───────────────────────────────────────────────┤
│ Data            │ DVC → S3/GCS (versioned datasets)             │
│ Features        │ Feast (offline + online store)                │
│ Experiments     │ MLflow Tracking Server + PostgreSQL backend   │
│ Model Registry  │ MLflow Registry (Staging → Production stages) │
│ Orchestration   │ Apache Airflow (scheduled retraining DAGs)    │
│ Serving         │ FastAPI + Gunicorn + Docker                   │
│ CI/CD           │ GitHub Actions (lint→test→train→build→deploy) │
│ Infrastructure  │ Kubernetes + HPA + Rolling deployments        │
│ Monitoring      │ Evidently (drift) + Prometheus + Grafana      │
│ Alerting        │ Alertmanager → Slack / PagerDuty              │
└─────────────────┴───────────────────────────────────────────────┘
```

### Key Production Principles

1. **Never deploy without a validation gate** — compare challenger vs champion metric before promoting
2. **Version everything** — code (Git), data (DVC), models (MLflow), configs (YAML in Git)
3. **Monitor from day one** — don't wait for the model to fail visibly
4. **Automate retraining triggers** — drift detected → trigger DAG → retrain → evaluate → promote
5. **Zero-downtime deployments** — use rolling updates, canary, or blue/green strategies
6. **Separate concerns** — feature pipelines, training, serving, and monitoring are independent services
7. **Fail fast in CI** — catch data quality issues and low metrics before they reach production

---

## References

- [MLOps Principles](https://ml-ops.org/content/mlops-principles)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [DVC Documentation](https://dvc.org/doc)
- [Feast Feature Store](https://docs.feast.dev/)
- [Evidently AI](https://docs.evidentlyai.com/)
- [Airflow MLflow Integration](https://medium.com/@syeda.areeba.nadeem/from-data-to-deployment-implementing-mlops-with-airflow-dvc-mlflow-and-kubernetes-6b2df084afba)
- [FastAPI + Prometheus Monitoring](https://github.com/jeremyjordan/ml-monitoring)
- [MLOps Best Practices 2025](https://www.thirstysprout.com/post/mlops-best-practices)
