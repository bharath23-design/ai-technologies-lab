# MLOps — Data Drift Detection

End-to-end data drift detection implementation using **Evidently AI (2026 API)**, with a live monitoring dashboard, HTML/JSON report exports, **native built-in test alerts (PASS/FAIL)**, and a clear path to production deployment.

---

## Folder Structure

```
mlops/
├── data_drift_detection.py        # Main modular implementation
├── data_drift_production_guide.md # Step-by-step production setup guide
├── mlops_production_guide.md      # General MLOps production reference
├── evidently_ai_mlops.ipynb       # Original notebook (exploratory)
├── drift_report.html              # Generated HTML drift report
├── drift_report.json              # Generated JSON drift report
└── evidently_workspace/           # Evidently UI workspace (snapshots DB)
```

---

## What This Does

Detects **data drift** between a reference dataset (training data) and a current dataset (production data) using **Population Stability Index (PSI)**. Outputs:

1. An interactive **HTML report**
2. A machine-readable **JSON report**
3. **Native Evidently test alerts** — FAIL/PASS per column and dataset level, logged to console immediately
4. A **timestamped snapshot** logged to the Evidently UI live dashboard with an **Alerts tab**

---

## Quick Start

### 1. Install dependencies

```bash
pip install evidently scikit-learn pandas
```

### 2. Run drift detection

```bash
python mlops/data_drift_detection.py
```

### 3. Launch the live dashboard

```bash
evidently ui --workspace mlops/evidently_workspace --port 8000
```

Open **http://localhost:8000** in your browser.

---

## Implementation

### Architecture

```
load_iris_data()
      │
split_reference_current()
      │
      ├── reference_df  (rows 0–70)   ← training baseline
      └── current_df    (rows 70–150) ← production data
                │
        DataDriftDetector.run()
                │
      ┌─────────┼──────────────────┬───────────────────┐
      │         │                  │                   │
validate_     run_drift_      fire_test_          stamp_snapshot()
schema()    detection()       alerts()          (tags + metadata
                │              (PASS/FAIL        on SnapshotModel)
      ┌─────────┴──────┐     per column &              │
      │                │     dataset level)      log_to_evidently_ui()
save_html_       save_json_    │                (Drift Overview +
report()         report()   WARNING logs         Alerts tab +
      │                │     to console          Tags/Metadata in
drift_report    drift_report                     Reports list)
   .html            .json
                │
        parse_drift_summary()  ← includes alerts[]
                │
           console output
```

### Key Classes and Functions

| Name | Type | Purpose |
|---|---|---|
| `DataDriftConfig` | dataclass | All paths and settings in one place |
| `DataDriftDetector` | class | Orchestrates the full pipeline |
| `load_iris_data()` | function | Load dataset (swap for DB connector in production) |
| `load_csv_data()` | function | Load from CSV file |
| `split_reference_current()` | function | Split into reference and current sets |
| `validate_schema()` | function | Ensure both datasets have matching columns |
| `build_report()` | function | Create Evidently Report with `DataDriftPreset(include_tests=True)` |
| `run_drift_detection()` | function | Execute drift detection, returns Snapshot |
| `save_html_report()` | function | Save interactive HTML report |
| `save_json_report()` | function | Save machine-readable JSON report |
| `fire_test_alerts()` | function | Read `snapshot.tests_results`, log FAIL/ERROR as WARNING alerts |
| `stamp_snapshot()` | function | Write `tags` + `metadata` into `SnapshotModel` so Reports list shows drift status |
| `log_to_evidently_ui()` | function | Serialize `SnapshotModel` with tags/metadata and write to workspace |
| `parse_drift_summary()` | function | Extract structured summary from result dict (includes `alerts[]`) |

---

## Configuration

All settings are in `DataDriftConfig`:

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
| `psi` | Population Stability Index | Numerical features, industry standard |
| `ks` | Kolmogorov-Smirnov | Numerical features, small samples |
| `chisquare` | Chi-Square test | Categorical features |
| `wasserstein` | Wasserstein Distance | Continuous distributions |
| `jensenshannon` | Jensen-Shannon Divergence | Both numerical and categorical |

---

## Evidently UI Dashboard

### How the UI Was Set Up

The live dashboard uses the **Evidently UI self-hosted** option — no sign-up or cloud account needed. Everything runs locally.

#### Where to See Alerts in the UI

```
http://localhost:8000
  └── Iris Data Drift Monitoring → Reports tab
        │
        ├── Reports list — each row shows:
        │     Report ID          Tags                  Metadata
        │     drift_run_..._14-59   drift_detected, psi   dataset_drift: True
        │                                               n_drifted: 5
        │                                               drifted_columns: sepal...
        │
        ├── Filter by Tags dropdown → select "drift_detected"
        │     → shows only runs where drift was detected
        │
        ├── Search in Metadata box → type a column name or "n_drifted"
        │     → filters runs matching that metadata value
        │
        └── Click any run → Tests tab (top-right corner)
              ├── Counter: SUCCESS | WARNING | FAIL | ERROR
              ├── [FAIL] Share of Drifted Columns: Less 0.500
              │         Actual value 1.000 >= 0.500
              ├── [FAIL] Value Drift for column sepal length (cm)
              │         Drift score is 3.66. Threshold is 0.10.
              └── ... one red-bordered card per failing column
```

The **Tags** and **Metadata** on each row let you spot drifted runs at a glance without clicking into each report. The **Tests tab** inside a report gives the full per-column breakdown.

#### Step 1 — Workspace

A `Workspace` is a local folder (`evidently_workspace/`) that acts as a lightweight database. It stores all snapshots (drift run results) as JSON files on disk.

```python
from evidently.ui.workspace import Workspace

workspace = Workspace.create("mlops/evidently_workspace")
```

- `Workspace.create()` creates the folder if it doesn't exist, or opens it if it does.
- No external database required — SQLite + file system under the hood.

#### Step 2 — Project

A `Project` groups related runs together under one name in the UI.

```python
project = workspace.create_project("Iris Data Drift Monitoring")
project.description = "Automated data drift monitoring using PSI method."
project.save()
```

- On the first run, a new project is created.
- On subsequent runs, the existing project is reused (`workspace.list_projects()` is checked first).

#### Step 3 — Dashboard Panels

Two time-series panels are configured on the project dashboard so results are visualised as trends across runs, not just as static individual reports.

```python
from evidently.sdk.models import DashboardPanelPlot, PanelMetric

# Panel 1 — bar chart of drifted column count per run
dashboard.add_panel(
    DashboardPanelPlot(
        title="Drifted Columns Count Over Time",
        values=[PanelMetric(legend="Drifted Columns", metric="DriftedColumnsCount", ...)],
        plot_params={"plot_type": "bar"},
    ),
    tab="Drift Overview",
)

# Panel 2 — line chart of PSI score per feature per run
dashboard.add_panel(
    DashboardPanelPlot(
        title="Per-Column PSI Drift Score Over Time",
        values=[PanelMetric(legend=col, metric="ValueDrift", metric_labels={"column": col}, ...)
                for col in COLUMNS],
        plot_params={"plot_type": "line"},
    ),
    tab="Drift Overview",
)
```

- Panels are created **only once** (on first project creation).
- `metric` matches the metric name returned by Evidently (`DriftedColumnsCount`, `ValueDrift`).
- `metric_labels` filters per-column metrics by column name.

#### Step 4 — Logging Snapshots

Each time `data_drift_detection.py` runs, the result is saved as a **timestamped snapshot**:

```python
workspace.add_run(project.id, snapshot, name="drift_run_2026-03-17_18-04-56")
```

- `snapshot` is the object returned by `report.run()` — it is already a `Snapshot` type.
- The timestamp in the name makes each run distinguishable in the UI.
- Each new run adds a new data point to the trend charts automatically.

#### Step 5 — Starting the Server

Port 8000 was occupied by an SSH tunnel on this machine, so port **8085** is used instead.

```bash
evidently ui \
  --workspace /Users/bharathkumar/Desktop/Main_code/ai-technologies-lab/mlops/evidently_workspace \
  --port 8085
```

Open **http://localhost:8085** in your browser.

#### Where to Look in the UI

```
http://localhost:8085
  └── Click "Iris Data Drift Monitoring"
        └── Click "Drift Overview" tab
              ├── Bar chart  — Drifted Columns Count Over Time
              └── Line chart — Per-Column PSI Drift Score Over Time

        └── Click "Reports" / "Runs" (left sidebar)
              └── Each row = one timestamped drift run
                    └── Click any row → full interactive drift report
```

---

### Dashboard Summary

Each script run logs a new timestamped snapshot (`drift_run_YYYY-MM-DD_HH-MM-SS`) to the workspace. The dashboard is pre-configured with four panels across two tabs:

**Drift Overview tab**

| Panel | Type | Shows |
|---|---|---|
| Drifted Columns Count Over Time | Bar chart | Number of drifted features per run |
| Per-Column PSI Drift Score Over Time | Line chart | PSI value per feature across runs |

**Alerts tab**

| Panel | Type | Shows |
|---|---|---|
| Dataset Drift Test — Share of Drifted Columns | Counter | Latest drift share value |
| Per-Column Drift Test — PSI Score Over Time | Bar chart | PSI bars per column — above threshold = FAIL alert |

**Reports tab → individual run → Tests tab**

| Counter | Meaning |
|---|---|
| SUCCESS | Tests that passed (no drift) |
| FAIL | Tests that failed — drift detected — these are your alerts |
| WARNING / ERROR | Edge cases |

Each FAIL card shows the column name, drift score, and threshold that triggered the alert (red border).

```bash
# Start dashboard
evidently ui \
  --workspace /Users/bharathkumar/Desktop/Main_code/ai-technologies-lab/mlops/evidently_workspace \
  --port 8085
```

Open **http://localhost:8085**

---

## Sample Output

```
2026-03-18 14:44:52 — INFO  — Iris dataset loaded: 150 rows, 5 columns.
2026-03-18 14:44:52 — INFO  — Split — reference: 70 rows | current: 80 rows.
2026-03-18 14:44:52 — INFO  — Schema validation passed: True
2026-03-18 14:44:52 — INFO  — Running drift detection…
2026-03-18 14:44:52 — INFO  — Drift detection complete.
2026-03-18 14:44:52 — INFO  — HTML report saved → drift_report.html
2026-03-18 14:44:52 — INFO  — JSON report saved → drift_report.json
2026-03-18 14:44:52 — WARNING — [DRIFT ALERT] Share of Drifted Columns: Less 0.500 — Actual value 1.000 >= 0.500 (status=FAIL)
2026-03-18 14:44:52 — WARNING — [DRIFT ALERT] Value Drift for column sepal length (cm) — Drift score is 3.66. PSI threshold is 0.10. (status=FAIL)
2026-03-18 14:44:52 — WARNING — [DRIFT ALERT] Value Drift for column sepal width (cm) — Drift score is 1.56. PSI threshold is 0.10. (status=FAIL)
2026-03-18 14:44:52 — WARNING — [DRIFT ALERT] Value Drift for column petal length (cm) — Drift score is 10.55. PSI threshold is 0.10. (status=FAIL)
2026-03-18 14:44:52 — WARNING — [DRIFT ALERT] Value Drift for column petal width (cm) — Drift score is 9.90. PSI threshold is 0.10. (status=FAIL)
2026-03-18 14:44:52 — WARNING — [DRIFT ALERT] Value Drift for column target — Drift score is 11.82. PSI threshold is 0.10. (status=FAIL)
2026-03-18 14:44:52 — WARNING — [DRIFT ALERT] 6 test(s) failed.
2026-03-18 14:44:53 — INFO  — Snapshot 'drift_run_2026-03-18_14-44-52' logged

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
  ...
====================================
```

---

## API Evolution (Old vs New)

This implementation uses the **current 2026 Evidently API**. The old API used in earlier production projects is deprecated.

| | Old API (pre-2023) | New API (2026) |
|---|---|---|
| Import | `from evidently.model_profile import Profile` | `from evidently import Report` |
| Preset | `DataDriftProfileSection()` | `DataDriftPreset(method="psi", include_tests=True)` |
| Run | `profile.calculate(ref, cur)` | `report.run(reference_data=ref, current_data=cur)` |
| Output | `profile.json()` | `result.dict()` / `result.save_html()` |
| Alerts | Not available | `snapshot.tests_results` → `MetricTestResult` (PASS/FAIL) |
| Dashboard | Not available | `Workspace` + `add_run()` |
| Result type | `dict` from JSON string | `Snapshot` object |

---

## Moving to Production

See `data_drift_production_guide.md` for the full production setup, covering:

- Replacing `load_iris_data()` with database connectors (Snowflake, BigQuery, PostgreSQL)
- Storing reports in cloud storage (AWS S3, GCS)
- Scheduling with Apache Airflow (daily DAG example)
- Alerting via Slack webhook or email
- Logging metrics to MLflow
- Triggering model retraining on drift detection
- Docker containerization

### Minimal production checklist

- [ ] Model inputs logged to a database with timestamps
- [ ] `DataDriftConfig` paths updated to cloud storage
- [ ] Drift detection scheduled (Airflow / GitHub Actions cron)
- [ ] Slack or email alert on drift
- [ ] Retraining pipeline triggered on drift

---

## References

- [Evidently AI Documentation](https://docs.evidentlyai.com)
- [DataDriftPreset API](https://docs.evidentlyai.com/metrics/preset_data_drift)
- [Evidently UI Self-hosting](https://docs.evidentlyai.com/docs/setup/self-hosting)
