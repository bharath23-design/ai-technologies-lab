# Data Drift Detection — Complete Implementation & Production Guide

End-to-end notes covering what was built locally, how the Evidently UI dashboard was set up, and how to take everything to production.

---

## Table of Contents

1. [What is Data Drift?](#1-what-is-data-drift)
2. [Local Implementation](#2-local-implementation)
3. [Code Walkthrough](#3-code-walkthrough)
4. [Native Evidently Alerts](#4-native-evidently-alerts)
5. [Evidently UI Dashboard Setup](#5-evidently-ui-dashboard-setup)
6. [Moving to Production](#6-moving-to-production)
7. [Production Steps](#7-production-steps)
8. [Containerization](#8-containerization)
9. [Environment Variables](#9-environment-variables)
10. [Minimal First Production Setup](#10-minimal-first-production-setup)
11. [Production Checklist](#11-production-checklist)
12. [Tech Stack Summary](#12-tech-stack-summary)
13. [API Evolution — Old vs New](#13-api-evolution--old-vs-new)

---

## 1. What is Data Drift?

Data drift occurs when the statistical properties of model input data change over time compared to the training data (reference). This causes model performance to degrade **silently** in production without any code changes.

**Types of drift:**

| Type | Description |
|---|---|
| Feature drift | Distribution of input features shifts |
| Label drift | Distribution of the target variable shifts |
| Concept drift | Relationship between features and target changes |

---

## 2. Local Implementation

### Folder Structure

```
mlops/
├── data_drift_detection.py        # Main modular implementation
├── data_drift_production_guide.md # This file
├── mlops_production_guide.md      # General MLOps production reference
├── evidently_ai_mlops.ipynb       # Original exploratory notebook
├── drift_report.html              # Generated HTML drift report
├── drift_report.json              # Generated JSON drift report
└── evidently_workspace/           # Evidently UI workspace (snapshots DB)
```

### Quick Start

```bash
# Install dependencies
pip install evidently scikit-learn pandas

# Run drift detection
python mlops/data_drift_detection.py

# Launch the live dashboard
evidently ui \
  --workspace mlops/evidently_workspace \
  --port 8085

# Open in browser
# http://localhost:8085
```

### Pipeline Flow

```
load_iris_data()
      │
split_reference_current()
      │
      ├── reference_df  (rows 0–70)    ← training baseline
      └── current_df    (rows 70–150)  ← production / current data
                │
        DataDriftDetector.run()
                │
      ┌─────────┼──────────────────────┬──────────────────────┐
      │         │                      │                      │
validate_    run_drift_          fire_test_         stamp_snapshot()
schema()   detection()           alerts()         (tags + metadata
               │              (PASS/FAIL per       on SnapshotModel)
     ┌─────────┴──────┐        column + dataset)         │
     │                │          WARNING logs      log_to_evidently_ui()
save_html_      save_json_        to console       (writes SnapshotModel
report()        report()                           with tags/metadata
     │                │                            to workspace —
drift_report   drift_report                        visible in Reports
   .html           .json                           list Tags column)
               │
       parse_drift_summary()  ← includes alerts[]
               │
          console output
```

### Sample Output

```
2026-03-18 14:44:52 — INFO    — Iris dataset loaded: 150 rows, 5 columns.
2026-03-18 14:44:52 — INFO    — Split — reference: 70 rows | current: 80 rows.
2026-03-18 14:44:52 — INFO    — Report built with drift method: 'psi'.
2026-03-18 14:44:52 — INFO    — Schema validation passed: True
2026-03-18 14:44:52 — INFO    — Running drift detection…
2026-03-18 14:44:52 — INFO    — Drift detection complete.
2026-03-18 14:44:52 — INFO    — HTML report saved → drift_report.html
2026-03-18 14:44:52 — INFO    — JSON report saved → drift_report.json
2026-03-18 14:44:52 — WARNING — [DRIFT ALERT] Share of Drifted Columns: Less 0.500 — Actual value 1.000 >= 0.500 (status=FAIL)
2026-03-18 14:44:52 — WARNING — [DRIFT ALERT] Value Drift for column sepal length (cm) — Drift score is 3.66. PSI threshold is 0.10. (status=FAIL)
2026-03-18 14:44:52 — WARNING — [DRIFT ALERT] Value Drift for column sepal width (cm) — Drift score is 1.56. PSI threshold is 0.10. (status=FAIL)
2026-03-18 14:44:52 — WARNING — [DRIFT ALERT] Value Drift for column petal length (cm) — Drift score is 10.55. PSI threshold is 0.10. (status=FAIL)
2026-03-18 14:44:52 — WARNING — [DRIFT ALERT] Value Drift for column petal width (cm) — Drift score is 9.90. PSI threshold is 0.10. (status=FAIL)
2026-03-18 14:44:52 — WARNING — [DRIFT ALERT] Value Drift for column target — Drift score is 11.82. PSI threshold is 0.10. (status=FAIL)
2026-03-18 14:44:52 — WARNING — [DRIFT ALERT] 6 test(s) failed. Investigate the drifted columns above.
2026-03-18 14:44:53 — INFO    — Snapshot 'drift_run_2026-03-18_14-44-52' logged

=== Drift Detection Summary ===
  Dataset drift detected : True
  Features evaluated     : 5
  Features drifted       : 5
  Drifted columns        : ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)', 'target']
  Alerts fired           : 6
================================

=== Evidently Test Alerts (FAIL) ===
  [FAIL] Share of Drifted Columns: Less 0.500
         Share of Drifted Columns: Actual value 1.000 >= 0.500
  [FAIL] Value Drift for column sepal length (cm)
         Drift score is 3.66. The drift detection method is PSI. The drift threshold is 0.10.
  [FAIL] Value Drift for column sepal width (cm)
         Drift score is 1.56. The drift detection method is PSI. The drift threshold is 0.10.
  [FAIL] Value Drift for column petal length (cm)
         Drift score is 10.55. The drift detection method is PSI. The drift threshold is 0.10.
  [FAIL] Value Drift for column petal width (cm)
         Drift score is 9.90. The drift detection method is PSI. The drift threshold is 0.10.
  [FAIL] Value Drift for column target
         Drift score is 11.82. The drift detection method is PSI. The drift threshold is 0.10.
====================================
```

---

## 3. Code Walkthrough

### Configuration — `DataDriftConfig`

All settings are in one dataclass so nothing is hardcoded across functions:

```python
@dataclass
class DataDriftConfig:
    html_report_path: str = "mlops/drift_report.html"
    json_report_path: str = "mlops/drift_report.json"
    drift_method: str = "psi"           # psi | ks | chisquare | wasserstein | jensenshannon
    drift_share_threshold: float = 0.5  # fraction of drifted columns to flag dataset drift
    workspace_path: str = "mlops/evidently_workspace"
    project_name: str = "Iris Data Drift Monitoring"
```

### Drift Methods

| Method | Full Name | Best For |
|---|---|---|
| `psi` | Population Stability Index | Numerical, industry standard |
| `ks` | Kolmogorov-Smirnov | Numerical, small samples |
| `chisquare` | Chi-Square test | Categorical features |
| `wasserstein` | Wasserstein Distance | Continuous distributions |
| `jensenshannon` | Jensen-Shannon Divergence | Numerical and categorical |

### Key Functions

| Name | Purpose |
|---|---|
| `load_iris_data()` | Load dataset — swap for DB connector in production |
| `load_csv_data()` | Load from CSV file |
| `split_reference_current()` | Split DataFrame into reference and current sets |
| `validate_schema()` | Ensure both datasets share the same columns |
| `build_report()` | Create Evidently `Report` with `DataDriftPreset(include_tests=True)` |
| `run_drift_detection()` | Execute drift detection, returns a `Snapshot` object |
| `save_html_report()` | Save interactive HTML report to disk |
| `save_json_report()` | Save machine-readable JSON report, returns dict |
| `fire_test_alerts()` | Read `snapshot.tests_results`, log every FAIL/ERROR as `WARNING` |
| `stamp_snapshot()` | Write `tags` + `metadata` into `SnapshotModel` before logging to workspace |
| `log_to_evidently_ui()` | Serialize `SnapshotModel` with tags/metadata via `write_snapshot()` — visible in Reports list |
| `parse_drift_summary()` | Parse result dict into structured summary, includes `alerts[]` list |
| `DataDriftDetector.run()` | Orchestrates all of the above end-to-end |

---

## 4. Native Evidently Alerts

Alerts are powered entirely by **Evidently's built-in test layer** — no external library or service needed.

### How It Works

Setting `include_tests=True` on both `DataDriftPreset` and `Report` activates Evidently's test engine. It auto-generates one `MetricTestResult` per check:

| Test | Trigger | Alert condition |
|---|---|---|
| `LessThanMetricTest` | Dataset level | FAIL if share of drifted columns ≥ `drift_share` (default 0.5) |
| `ValueDriftTest` | Per column | FAIL if PSI score > threshold (default 0.1) |

```python
# build_report() — activates the test layer
report = Report(
    metrics=[DataDriftPreset(method="psi", include_tests=True)],
    include_tests=True,
)
```

### `fire_test_alerts(snapshot)`

Reads `snapshot.tests_results` and emits a `WARNING` log for every `FAIL` or `ERROR`:

```python
from evidently.tests.numerical_tests import TestStatus

def fire_test_alerts(snapshot) -> list[dict]:
    alerts = []
    for test_result in snapshot.tests_results:
        if test_result.status in (TestStatus.FAIL, TestStatus.ERROR):
            alerts.append({
                "test_name":   test_result.name,
                "description": test_result.description,
                "status":      test_result.status.value,
            })
            logger.warning(
                "[DRIFT ALERT] %s — %s (status=%s)",
                test_result.name,
                test_result.description,
                test_result.status.value,
            )
    return alerts
```

### Where to See Alerts

**1. Console** — WARNING lines appear immediately when the script runs.

**2. Evidently UI — Reports list** — each row shows tags and metadata at a glance:

```
Reports tab
  ┌───────────────────────────────┬──────────────────────┬──────────────────────────────────┐
  │ Report ID                     │ Tags                 │ Metadata                         │
  ├───────────────────────────────┼──────────────────────┼──────────────────────────────────┤
  │ drift_run_2026-03-18_14-59-40 │ drift_detected, psi  │ dataset_drift: True              │
  │                               │                      │ n_features: 5                    │
  │                               │                      │ n_drifted: 5                     │
  │                               │                      │ drifted_columns: sepal length... │
  └───────────────────────────────┴──────────────────────┴──────────────────────────────────┘
```

- **Filter by Tags** dropdown → select `drift_detected` → shows only runs with drift
- **Search in Metadata** box → type a column name or `n_drifted` to filter runs

**3. Evidently UI — individual run → Tests tab** (top-right corner of the report):

```
Reports tab → click a run → Tests tab
  ┌──────────┬───────────┬──────┬───────┐
  │ SUCCESS  │  WARNING  │ FAIL │ ERROR │
  │    0     │     0     │  6   │   0   │
  └──────────┴───────────┴──────┴───────┘

  ┌─────────────────────────────────────────────────────────┐
  │ ⊗  Share of Drifted Columns: Less 0.500                 │  ← red border = FAIL alert
  │    Actual value 1.000 >= 0.500                          │
  ├─────────────────────────────────────────────────────────┤
  │ ⊗  Value Drift for column sepal length (cm)             │
  │    Drift score is 3.66. Threshold is 0.10.              │
  └─────────────────────────────────────────────────────────┘
```

**4. Dashboard Alerts tab** — counter panel (latest drift share) + bar panel (per-column PSI across runs).

### `stamp_snapshot(snapshot, summary)` — Reports List Labels

Evidently serialises snapshots via `to_snapshot_model()` before writing to disk. Changes made to `snapshot.report.tags` do **not** propagate through that path. `stamp_snapshot` therefore directly mutates the `SnapshotModel`:

```python
def stamp_snapshot(snapshot, summary: dict) -> None:
    drift_tag = "drift_detected" if summary["dataset_drift"] else "no_drift"
    snapshot.report.tags    = [drift_tag, "psi"]
    snapshot.report.metadata = {
        "dataset_drift":   str(summary["dataset_drift"]),
        "n_features":      str(summary["n_features"]),
        "n_drifted":       str(summary["n_drifted"]),
        "drifted_columns": ", ".join(summary["drifted_columns"]) or "none",
    }
```

`log_to_evidently_ui` then calls `to_snapshot_model()`, copies these values onto the model, and writes it via `workspace.state.write_snapshot()` — bypassing `workspace.add_run()` which would discard them:

```python
snapshot_model = snapshot.to_snapshot_model()
snapshot_model.name     = run_name
snapshot_model.tags     = snapshot.report.tags
snapshot_model.metadata = snapshot.report.metadata

from evidently.sdk.models import new_id
snapshot_id = new_id()
workspace.state.write_snapshot(project.id, snapshot_id, snapshot_model)
```

| Field | Values | UI location |
|---|---|---|
| `tags` | `drift_detected` or `no_drift`, `psi` | "Filter by Tags" dropdown |
| `metadata.dataset_drift` | `True` / `False` | Metadata column |
| `metadata.n_drifted` | e.g. `5` | Metadata column |
| `metadata.drifted_columns` | comma-separated column names | Searchable via "Search in Metadata" |

---

### `alerts[]` in the Summary Dict

`DataDriftDetector.run()` returns a summary dict that now includes:

```python
{
    "dataset_drift":   True,
    "n_features":      5,
    "n_drifted":       5,
    "drifted_columns": ["sepal length (cm)", ...],
    "alerts": [
        {"test_name": "Share of Drifted Columns: Less 0.500",
         "description": "Actual value 1.000 >= 0.500",
         "status": "FAIL"},
        {"test_name": "Value Drift for column sepal length (cm)",
         "description": "Drift score is 3.66. Threshold is 0.10.",
         "status": "FAIL"},
        ...
    ]
}
```

Use `summary["alerts"]` to route to external systems (Airflow, PagerDuty, etc.) if needed.

---

### How `parse_drift_summary` Works

The 2026 Evidently API returns a flat list of metrics in the result dict:

```json
{
  "metrics": [
    { "metric_name": "DriftedColumnsCount(drift_share=0.5,method=psi)", "value": {"count": 5.0, "share": 1.0} },
    { "metric_name": "ValueDrift(column=sepal length (cm),method=psi,threshold=0.1)", "value": 3.65 },
    ...
  ]
}
```

- `DriftedColumnsCount` → total drifted columns + share (used for dataset-level drift flag)
- `ValueDrift` → per-column PSI score (drifted if score > threshold, default 0.1)
- `dataset_drift = True` when `share >= drift_share_threshold` (default 0.5)

---

## 5. Evidently UI Dashboard Setup

This is a **self-hosted** local dashboard — no cloud account or sign-up needed.

### Step 1 — Workspace

A `Workspace` is a local folder that acts as a lightweight database. It stores all snapshots as JSON files on disk using SQLite.

```python
from evidently.ui.workspace import Workspace

workspace = Workspace.create("mlops/evidently_workspace")
```

- `Workspace.create()` creates the folder if it doesn't exist, or opens it if it already does.
- No external database required — SQLite + file system under the hood.

### Step 2 — Project

A `Project` groups related runs together under one name in the UI.

```python
project = workspace.create_project("Iris Data Drift Monitoring")
project.description = "Automated data drift monitoring using PSI method."
project.save()
```

- On the **first run**, a new project is created.
- On **subsequent runs**, the existing project is reused (`workspace.list_projects()` is checked first to avoid duplicates).

### Step 3 — Dashboard Panels

Four panels are configured across two dashboard tabs — **Drift Overview** for metrics trends and **Alerts** for test-status visualisation.

```python
from evidently.sdk.models import DashboardPanelPlot, PanelMetric

# Panel 1 — bar chart: drifted column count per run
dashboard.add_panel(
    DashboardPanelPlot(
        title="Drifted Columns Count Over Time",
        subtitle="Number of features with drift detected per run",
        values=[
            PanelMetric(
                legend="Drifted Columns",
                metric="DriftedColumnsCount",   # matches metric_name in result dict
                metric_labels={},
                tags=[], metadata={}, view_params={},
            )
        ],
        plot_params={"plot_type": "bar"},
    ),
    tab="Drift Overview",
)

# Panel 2 — line chart: PSI score per feature per run
dashboard.add_panel(
    DashboardPanelPlot(
        title="Per-Column PSI Drift Score Over Time",
        subtitle="PSI value per feature — higher = more drift",
        values=[
            PanelMetric(
                legend=col,
                metric="ValueDrift",            # matches metric_name prefix
                metric_labels={"column": col},  # filters by column name
                tags=[], metadata={}, view_params={},
            )
            for col in ["sepal length (cm)", "sepal width (cm)",
                        "petal length (cm)", "petal width (cm)", "target"]
        ],
        plot_params={"plot_type": "line"},
    ),
    tab="Drift Overview",
)
```

- Panels are created **only once** on first project creation.
- `metric` matches the prefix of `metric_name` in the Evidently result dict.
- `metric_labels` filters per-column metrics by their column name label.

### Step 4 — Logging Snapshots

Each time the script runs, the result is saved as a **timestamped snapshot**:

```python
run_name = f"drift_run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
workspace.add_run(project.id, snapshot, name=run_name)
```

- `snapshot` is the object returned by `report.run()` — it is already a `Snapshot` type (no conversion needed).
- The timestamp in the name makes each run distinguishable in the UI.
- Each new run adds a new data point to the trend charts automatically.

### Step 5 — Starting the Server

Port 8000 was occupied by an SSH tunnel on this machine, so port **8085** is used.

```bash
evidently ui \
  --workspace /Users/bharathkumar/Desktop/Main_code/ai-technologies-lab/mlops/evidently_workspace \
  --port 8085
```

Open **http://localhost:8085** in your browser.

### Where to Navigate in the UI

```
http://localhost:8085
  └── Click "Iris Data Drift Monitoring"   ← your project
        │
        ├── "Drift Overview" tab
        │     ├── Bar chart  — Drifted Columns Count Over Time
        │     └── Line chart — Per-Column PSI Drift Score Over Time
        │
        └── "Reports" / "Runs" (left sidebar)
              └── Each row = one timestamped snapshot (drift_run_YYYY-MM-DD_HH-MM-SS)
                    └── Click any row → full interactive drift report for that run
```

### Dashboard Panels Summary

**Drift Overview tab**

| Panel | Chart Type | What It Shows |
|---|---|---|
| Drifted Columns Count Over Time | Bar | Number of drifted features per run |
| Per-Column PSI Drift Score Over Time | Line | PSI value per feature across all runs |

**Alerts tab**

| Panel | Chart Type | What It Shows |
|---|---|---|
| Dataset Drift Test — Share of Drifted Columns | Counter | Latest drift share value (≥0.5 = alert) |
| Per-Column Drift Test — PSI Score Over Time | Bar | PSI per column — bars above 0.1 = FAIL alert |

**Reports tab → Tests tab** (per individual run)

| Counter | Meaning |
|---|---|
| SUCCESS | Tests passed — no drift |
| FAIL | Tests failed — drift alerts (red-bordered cards) |
| WARNING / ERROR | Edge cases |

---

## 6. Moving to Production

### What Changes

| Layer | Local | Production |
|---|---|---|
| Data source | `sklearn.datasets` (static) | Live DB / feature store |
| File storage | Local paths | S3 / GCS / Azure Blob |
| Scheduling | Manual script run | Airflow DAG / cron |
| Alerting | Console print | Slack / PagerDuty / email |
| On drift | Nothing | Trigger retraining pipeline |
| Dashboard | Local Evidently UI | Evidently UI on server / Grafana / MLflow |

### Production Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Data Sources                            │
│  Training DB / Feature Store  +  Production Prediction Logs │
└───────────────────┬─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│              Scheduler  (Airflow DAG — daily)               │
│                                                             │
│   load_reference_data()   load_current_data()               │
│           │                       │                         │
│           └──────────┬────────────┘                         │
│                      ▼                                      │
│           DataDriftDetector.run()                           │
└──────────────────────┬──────────────────────────────────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
    drift_report   drift_report   parse_drift_summary()
       .html          .json              │
          │            │         ┌──────┴──────┐
          └────────────┘         │             │
               S3 / GCS       No drift      Drift!
                               │             │
                          Log metrics    Slack alert
                          to MLflow    + Retrain DAG
```

---

## 7. Production Steps

### Step 1 — Replace Data Loading with DB Connectors

```python
import sqlalchemy
import pandas as pd

def load_reference_data(conn_str: str, table: str) -> pd.DataFrame:
    """Load training baseline from data warehouse."""
    engine = sqlalchemy.create_engine(conn_str)
    return pd.read_sql(f"SELECT * FROM {table} WHERE split = 'train'", engine)

def load_current_data(conn_str: str, table: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Load recent production predictions."""
    engine = sqlalchemy.create_engine(conn_str)
    query = f"""
        SELECT * FROM {table}
        WHERE prediction_date BETWEEN '{start_date}' AND '{end_date}'
    """
    return pd.read_sql(query, engine)
```

**Data warehouse options:**

| Tool | Best For |
|---|---|
| Snowflake | Enterprise, large scale |
| BigQuery | GCP ecosystem |
| Redshift | AWS ecosystem |
| PostgreSQL | Small / mid-scale |

**What to log from production model:**
- All input features (exactly as fed to the model)
- Model prediction output
- Prediction timestamp
- Model version

---

### Step 2 — Cloud Storage for Reports

#### AWS S3

```python
import boto3

def save_html_to_s3(result, bucket: str, key: str) -> None:
    result.save_html("/tmp/drift_report.html")
    boto3.client("s3").upload_file("/tmp/drift_report.html", bucket, key)
    print(f"HTML report saved → s3://{bucket}/{key}")

def save_json_to_s3(result_dict: dict, bucket: str, key: str) -> None:
    import json
    boto3.client("s3").put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(result_dict, default=str),
        ContentType="application/json",
    )
    print(f"JSON report saved → s3://{bucket}/{key}")
```

#### Google Cloud Storage

```python
from google.cloud import storage

def save_html_to_gcs(result, bucket_name: str, blob_name: str) -> None:
    result.save_html("/tmp/drift_report.html")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    bucket.blob(blob_name).upload_from_filename("/tmp/drift_report.html")
```

**Recommended S3/GCS folder structure:**
```
s3://your-bucket/
└── drift-reports/
    └── 2026/
        └── 03/
            └── 17/
                ├── drift_report.html
                └── drift_report.json
```

---

### Step 3 — Scheduling with Apache Airflow

```python
# dags/drift_detection_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from mlops.data_drift_detection import DataDriftDetector, DataDriftConfig

default_args = {
    "owner": "mlops-team",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

def run_drift_check(**context):
    execution_date = context["ds"]  # YYYY-MM-DD

    reference_df = load_reference_data(conn_str=CONN_STR, table="training_features")
    current_df = load_current_data(
        conn_str=CONN_STR,
        table="prediction_logs",
        start_date=execution_date,
        end_date=execution_date,
    )

    config = DataDriftConfig(
        html_report_path=f"s3://your-bucket/drift-reports/{execution_date}/drift_report.html",
        json_report_path=f"s3://your-bucket/drift-reports/{execution_date}/drift_report.json",
        drift_method="psi",
    )

    detector = DataDriftDetector(config=config)
    summary = detector.run(reference_df=reference_df, current_df=current_df)

    if summary["dataset_drift"]:
        send_slack_alert(summary)
        trigger_retraining_dag()

with DAG(
    dag_id="data_drift_detection",
    default_args=default_args,
    schedule_interval="0 0 * * *",   # daily at midnight
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["mlops", "drift"],
) as dag:

    drift_check = PythonOperator(
        task_id="run_drift_check",
        python_callable=run_drift_check,
    )
```

**Scheduling options:**

| Tool | When to Use |
|---|---|
| Apache Airflow | Complex pipelines, task dependencies |
| Prefect | Lighter than Airflow, easier local testing |
| GitHub Actions cron | Simple projects, no infra needed |
| AWS EventBridge + Lambda | Serverless, event-driven |

---

### Step 4 — Alerting

#### Native Evidently Alerts (built-in — no extra library)

This is what is already implemented. `fire_test_alerts()` reads `snapshot.tests_results` and logs every `FAIL` as a `WARNING`:

```python
from evidently.tests.numerical_tests import TestStatus

alerts = fire_test_alerts(result)
# Returns list[dict] — test_name, description, status
# Each FAIL is immediately logged: WARNING — [DRIFT ALERT] ...
```

View alerts in the UI: `Reports` tab → click a run → **Tests tab** → red cards = FAIL alerts.

#### Using `alerts[]` from the Summary to Route to External Systems

`DataDriftDetector.run()` returns `summary["alerts"]` — use it to fan out to any channel:

```python
summary = detector.run(reference_df=reference_df, current_df=current_df)

if summary["alerts"]:
    # Route to external system — Airflow, PagerDuty, Slack, etc.
    for alert in summary["alerts"]:
        print(f"[{alert['status']}] {alert['test_name']}: {alert['description']}")
```

**Alerting options (production routing):**

| Tool | Use Case |
|---|---|
| Evidently Tests tab (built-in) | Visual FAIL/PASS per run in the UI — already implemented |
| `summary["alerts"]` list | Programmatic routing — trigger any downstream action |
| Airflow `on_failure_callback` | Fail the DAG task if drift is detected |
| PagerDuty API | On-call / critical production alerts |
| Slack webhook (via `requests`) | Team notifications |
| AWS SES | Email alerts |

---

### Step 5 — Metrics & Monitoring Dashboard

#### Log to MLflow

```python
import mlflow

def log_drift_to_mlflow(summary: dict, report_path: str) -> None:
    with mlflow.start_run(run_name="drift_detection"):
        mlflow.log_metric("n_features", summary["n_features"])
        mlflow.log_metric("n_drifted_features", summary["n_drifted"])
        mlflow.log_metric(
            "drift_share", summary["n_drifted"] / max(summary["n_features"], 1)
        )
        mlflow.log_param("dataset_drift", summary["dataset_drift"])
        mlflow.log_artifact(report_path)
```

**Dashboard options:**

| Tool | Purpose |
|---|---|
| Grafana + Prometheus | Plot drift score as time-series, threshold alerts |
| MLflow UI | View metrics per run, compare across dates |
| Evidently Cloud | Managed drift dashboard (paid) |
| Evidently UI (self-hosted) | What we built locally — free, runs anywhere |
| Metabase / Superset | BI dashboard over drift results table |

---

### Step 6 — Trigger Retraining on Drift

```python
def trigger_retraining(method: str = "airflow") -> None:
    if method == "airflow":
        import requests
        requests.post(
            "http://airflow-webserver:8080/api/v1/dags/model_retraining/dagRuns",
            json={"conf": {"reason": "drift_detected"}},
            auth=("admin", "password"),
        )

    elif method == "github_actions":
        import requests
        requests.post(
            "https://api.github.com/repos/org/repo/actions/workflows/retrain.yml/dispatches",
            headers={"Authorization": f"Bearer {GITHUB_TOKEN}"},
            json={"ref": "main"},
        )

    elif method == "sagemaker":
        import boto3
        boto3.client("sagemaker").start_pipeline_execution(
            PipelineName="model-retraining-pipeline"
        )
```

---

## 8. Containerization

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY mlops/data_drift_detection.py .

CMD ["python", "data_drift_detection.py"]
```

```bash
# Build and run
docker build -t drift-detector .
docker run --env-file .env drift-detector
```

---

## 9. Environment Variables

```bash
# Database
DB_CONN_STR=postgresql://user:password@host:5432/db
REFERENCE_TABLE=training_features
CURRENT_TABLE=prediction_logs

# Cloud storage
S3_BUCKET=your-mlops-bucket
GCS_BUCKET=your-mlops-bucket

# Alerting
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/xxx/yyy/zzz
ALERT_EMAIL=mlops-team@yourcompany.com

# MLflow
MLFLOW_TRACKING_URI=http://mlflow-server:5000
```

---

## 10. Minimal First Production Setup

If you want to go live quickly without full infrastructure:

```
Step 1 — Containerize    Docker image of data_drift_detection.py
Step 2 — Schedule        GitHub Actions cron (free, no infra needed)
Step 3 — Store reports   Push HTML + JSON to S3
Step 4 — Alert           One Slack webhook on drift detected
```

This covers 80% of production needs. Add Airflow, MLflow, and retraining triggers as the system matures.

---

## 11. Production Checklist

- [x] Native Evidently alerts via `include_tests=True` + `fire_test_alerts()` — already implemented
- [x] FAIL alerts visible in UI: `Reports` tab → run → `Tests` tab (red-bordered cards)
- [x] `stamp_snapshot()` writes `tags` + `metadata` to `SnapshotModel` — visible in Reports list
- [x] Filter by Tags (`drift_detected` / `no_drift`) and Search in Metadata in the UI
- [x] `summary["alerts"]` returned by `DataDriftDetector.run()` for downstream routing
- [ ] Model inputs logged to database with timestamps
- [ ] Reference (training) data accessible from a stable table
- [ ] `DataDriftConfig` paths point to cloud storage
- [ ] Drift detection scheduled (Airflow / cron)
- [ ] Route `summary["alerts"]` to external channel (Slack / PagerDuty) if needed
- [ ] Drift metrics logged to MLflow or Grafana
- [ ] Retraining pipeline triggered on drift
- [ ] Docker image built and tested
- [ ] Evidently UI deployed on a server (or replaced with Grafana)
- [ ] Environment variables stored in secrets manager (AWS Secrets Manager / Vault)

---

## 12. Tech Stack Summary

| Layer | Local | Production |
|---|---|---|
| Data | `sklearn.datasets` | Snowflake / BigQuery / PostgreSQL |
| Scheduling | Manual script run | Airflow / Prefect / GitHub Actions |
| Storage | Local file system | AWS S3 / GCS / Azure Blob |
| Alerting | Console print | Slack / PagerDuty / SES |
| Monitoring | Evidently UI (local) | Grafana / MLflow / Evidently Cloud |
| Retraining | Manual | Airflow DAG / SageMaker Pipeline |
| Deployment | Python script | Docker + Kubernetes / ECS |

---

## 13. API Evolution — Old vs New

This implementation uses the **current 2026 Evidently API**. The old API (`Profile` + `DataDriftProfileSection`) is deprecated and removed in recent versions.

| | Old API (pre-2023) | New API (2026) |
|---|---|---|
| Import | `from evidently.model_profile import Profile` | `from evidently import Report` |
| Preset | `DataDriftProfileSection()` | `DataDriftPreset(method="psi", include_tests=True)` |
| Run | `profile.calculate(ref, cur)` | `report.run(reference_data=ref, current_data=cur)` |
| Output dict | `json.loads(profile.json())` | `result.dict()` |
| Save HTML | Not straightforward | `result.save_html("report.html")` |
| Alerts | Not available | `snapshot.tests_results` → `MetricTestResult` (PASS/FAIL) |
| Test statuses | Not available | `TestStatus.FAIL / SUCCESS / WARNING / ERROR` |
| Dashboard | Not available | `Workspace` + `add_run()` |
| Result type | Plain dict from JSON string | `Snapshot` object |
| Dashboard panels | Not available | `DashboardPanelPlot` + `PanelMetric` |

---

## References

- [Evidently AI Documentation](https://docs.evidentlyai.com)
- [DataDriftPreset API](https://docs.evidentlyai.com/metrics/preset_data_drift)
- [Evidently UI Self-hosting](https://docs.evidentlyai.com/docs/setup/self-hosting)
