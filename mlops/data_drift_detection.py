"""
Data Drift Detection Module (Evidently AI - 2026 API)
=====================================================
Modular implementation using the current Report + DataDriftPreset API.
Uses Evidently's native test layer (include_tests=True) for built-in alerting
via MetricTestResult (PASS/FAIL) and workspace UI dashboard alert panels.
"""

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import pandas as pd
from pandas import DataFrame
from sklearn import datasets
from datetime import datetime
from evidently import Report
from evidently.presets import DataDriftPreset
from evidently.sdk.models import DashboardPanelPlot, PanelMetric
from evidently.tests.numerical_tests import TestStatus
from evidently.ui.workspace import Workspace

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DataDriftConfig:
    """Paths and settings for the drift detection run."""
    html_report_path: str = "/Users/bharathkumar/Desktop/Main_code/ai-technologies-lab/mlops/drift_report.html"
    json_report_path: str = "/Users/bharathkumar/Desktop/Main_code/ai-technologies-lab/mlops/drift_report.json"
    drift_method: str = "psi"          # psi | ks | chisquare | wasserstein | jensenshannon
    drift_share_threshold: float = 0.5  # fraction of drifted columns to flag dataset drift
    # Evidently UI — set workspace_path to enable live dashboard logging
    workspace_path: str = "/Users/bharathkumar/Desktop/Main_code/ai-technologies-lab/mlops/evidently_workspace"
    project_name: str = "Iris Data Drift Monitoring"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_iris_data() -> DataFrame:
    """Load the Iris dataset as a DataFrame."""
    iris = datasets.load_iris(as_frame=True)
    df = iris.frame
    logger.info("Iris dataset loaded: %d rows, %d columns.", *df.shape)
    return df


def load_csv_data(file_path: str) -> DataFrame:
    """Load data from a CSV file."""
    df = pd.read_csv(file_path)
    logger.info("CSV data loaded from '%s': %d rows, %d columns.", file_path, *df.shape)
    return df


def split_reference_current(
    df: DataFrame,
    reference_end_idx: int,
) -> tuple[DataFrame, DataFrame]:
    """Split a DataFrame into reference (train) and current (test) sets."""
    reference_df = df.iloc[:reference_end_idx]
    current_df = df.iloc[reference_end_idx:]
    logger.info(
        "Split — reference: %d rows | current: %d rows.",
        len(reference_df),
        len(current_df),
    )
    return reference_df, current_df


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def validate_schema(reference_df: DataFrame, current_df: DataFrame) -> bool:
    """Ensure reference and current datasets share the same columns."""
    ref_cols = set(reference_df.columns)
    cur_cols = set(current_df.columns)

    missing_in_current = ref_cols - cur_cols
    extra_in_current = cur_cols - ref_cols

    if missing_in_current:
        logger.warning("Columns missing in current data: %s", missing_in_current)
    if extra_in_current:
        logger.warning("Extra columns in current data (not in reference): %s", extra_in_current)

    is_valid = not missing_in_current and not extra_in_current
    logger.info("Schema validation passed: %s", is_valid)
    return is_valid


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------

def build_report(method: str = "psi") -> Report:
    """
    Create an Evidently Report configured for data drift.

    include_tests=True on both DataDriftPreset and Report activates Evidently's
    native test layer:
      - LessThanMetricTest  → FAIL if share of drifted columns >= drift_share threshold
      - ValueDriftTest      → FAIL per column if PSI score > per-column threshold
    Results are stored in snapshot.tests_results (list of MetricTestResult).
    """
    report = Report(
        metrics=[DataDriftPreset(method=method, include_tests=True)],
        include_tests=True,
    )
    logger.info("Report built with drift method: '%s'.", method)
    return report


def run_drift_detection(
    report: Report,
    reference_df: DataFrame,
    current_df: DataFrame,
):
    """Run drift detection and return the evaluation result."""
    logger.info("Running drift detection…")
    result = report.run(reference_data=reference_df, current_data=current_df)
    logger.info("Drift detection complete.")
    return result


def stamp_snapshot(snapshot, summary: dict) -> None:
    """
    Write drift results into the snapshot's tags and metadata so each report
    is clearly labelled in the Evidently UI Reports list.

    Tags  — visible in the "Filter by Tags" dropdown:
        "drift_detected" or "no_drift"
        "psi"  (drift method used)

    Metadata — visible in the "Metadata" column and searchable:
        dataset_drift, n_features, n_drifted, drifted_columns
    """
    drift_tag = "drift_detected" if summary["dataset_drift"] else "no_drift"
    snapshot.report.tags = [drift_tag, "psi"]

    snapshot.report.metadata.update({
        "dataset_drift": str(summary["dataset_drift"]),
        "n_features":    str(summary["n_features"]),
        "n_drifted":     str(summary["n_drifted"]),
        "drifted_columns": ", ".join(summary["drifted_columns"]) or "none",
    })
    logger.info(
        "Snapshot stamped — tags=%s | drifted=%d/%d columns.",
        snapshot.report.tags,
        summary["n_drifted"],
        summary["n_features"],
    )


# ---------------------------------------------------------------------------
# Reporting & persistence
# ---------------------------------------------------------------------------

def save_html_report(result, output_path: str) -> None:
    """Save the drift report as an HTML file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.save_html(output_path)
    logger.info("HTML report saved → '%s'.", output_path)


def save_json_report(result, output_path: str) -> dict:
    """Save the drift report as a JSON file and return the dict."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result_dict = result.dict()
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2, default=str)
    logger.info("JSON report saved → '%s'.", output_path)
    return result_dict


# ---------------------------------------------------------------------------
# Drift summary parsing
# ---------------------------------------------------------------------------

def parse_drift_summary(result_dict: dict, drift_share_threshold: float = 0.5) -> dict:
    """
    Extract a human-readable drift summary from the result dict.

    The 2026 Evidently API produces a flat list of metrics:
      - DriftedColumnsCount  → count / share of drifted columns
      - ValueDrift           → per-column PSI / stat value + threshold

    Returns a dict with keys:
        dataset_drift   — bool, share of drifted columns >= drift_share_threshold
        n_features      — total number of features evaluated
        n_drifted       — number of features where drift was detected
        drifted_columns — list of column names that drifted
    """
    summary = {
        "dataset_drift": False,
        "n_features": 0,
        "n_drifted": 0,
        "drifted_columns": [],
    }

    try:
        metrics = result_dict.get("metrics", [])
        value_drift_metrics = []

        for metric in metrics:
            metric_name: str = metric.get("metric_name", "")
            config: dict = metric.get("config", {})
            value = metric.get("value")

            # DriftedColumnsCount carries the dataset-level summary
            if "DriftedColumnsCount" in metric_name:
                n_drifted = int(value.get("count", 0)) if isinstance(value, dict) else 0
                share = float(value.get("share", 0.0)) if isinstance(value, dict) else 0.0
                summary["n_drifted"] = n_drifted
                summary["dataset_drift"] = share >= drift_share_threshold

            # ValueDrift carries per-column drift score
            if "ValueDrift" in metric_name:
                column = config.get("column", "")
                threshold = config.get("threshold", 0.1)
                drift_score = float(value) if value is not None else 0.0
                value_drift_metrics.append({
                    "column": column,
                    "drift_score": drift_score,
                    "threshold": threshold,
                    "drifted": drift_score > threshold,
                })

        summary["n_features"] = len(value_drift_metrics)
        summary["drifted_columns"] = [
            m["column"] for m in value_drift_metrics if m["drifted"]
        ]
        # Recalculate n_drifted from per-column data if DriftedColumnsCount was absent
        if summary["n_drifted"] == 0 and summary["drifted_columns"]:
            summary["n_drifted"] = len(summary["drifted_columns"])

    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not fully parse drift summary: %s", exc)

    logger.info(
        "Drift summary — dataset_drift=%s | %d/%d features drifted | drifted columns: %s",
        summary["dataset_drift"],
        summary["n_drifted"],
        summary["n_features"],
        summary["drifted_columns"] or "none",
    )
    return summary


# ---------------------------------------------------------------------------
# Evidently native alert layer — MetricTestResult (PASS / FAIL)
# ---------------------------------------------------------------------------

def fire_test_alerts(snapshot) -> list[dict]:
    """
    Read Evidently's built-in test results from the snapshot and raise
    structured alerts for every FAIL / ERROR.

    Evidently generates one MetricTestResult per check when
    ``include_tests=True`` is set on the Report and DataDriftPreset:

    - ``LessThanMetricTest``  → dataset-level alert: share of drifted columns
      >= drift_share threshold.
    - ``ValueDriftTest``      → per-column alert: PSI score exceeds threshold.

    Each FAIL is logged at WARNING level (visible in the console immediately)
    and returned as a dict so callers can route the alert further if needed.

    Returns
    -------
    alerts : list[dict]  — one entry per failed test, keys:
        test_name, description, status
    """
    alerts = []

    for test_result in snapshot.tests_results:
        status: TestStatus = test_result.status

        if status in (TestStatus.FAIL, TestStatus.ERROR):
            alert = {
                "test_name": test_result.name,
                "description": test_result.description,
                "status": status.value,
            }
            alerts.append(alert)
            logger.warning(
                "[DRIFT ALERT] %s — %s (status=%s)",
                test_result.name,
                test_result.description,
                status.value,
            )

    if not alerts:
        logger.info("[DRIFT ALERT] All tests passed — no drift alerts.")
    else:
        logger.warning(
            "[DRIFT ALERT] %d test(s) failed. Investigate the drifted columns above.",
            len(alerts),
        )

    return alerts


# ---------------------------------------------------------------------------
# Evidently UI — live dashboard logging
# ---------------------------------------------------------------------------

COLUMNS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
    "target",
]


def _setup_dashboard_panels(project) -> None:
    """
    Configure time-series panels on the project dashboard so each run's
    metrics are plotted as a trend over time.
    Called only once when a project is first created.
    """
    dashboard = project.dashboard

    # Panel 1 — drifted column count over time
    dashboard.add_panel(
        DashboardPanelPlot(
            title="Drifted Columns Count Over Time",
            subtitle="Number of features with drift detected per run",
            values=[
                PanelMetric(
                    legend="Drifted Columns",
                    metric="DriftedColumnsCount",
                    metric_labels={},
                    tags=[],
                    metadata={},
                    view_params={},
                )
            ],
            plot_params={"plot_type": "bar"},
        ),
        tab="Drift Overview",
    )

    # Panel 2 — per-column PSI drift score over time
    dashboard.add_panel(
        DashboardPanelPlot(
            title="Per-Column PSI Drift Score Over Time",
            subtitle="PSI value per feature — higher = more drift",
            values=[
                PanelMetric(
                    legend=col,
                    metric="ValueDrift",
                    metric_labels={"column": col},
                    tags=[],
                    metadata={},
                    view_params={},
                )
                for col in COLUMNS
            ],
            plot_params={"plot_type": "line"},
        ),
        tab="Drift Overview",
    )

    # -----------------------------------------------------------------------
    # Alert Panels — native Evidently test layer (include_tests=True)
    # These panels reflect MetricTestResult statuses stored in each snapshot.
    # A rising FAIL counter in the workspace UI is the visual drift alert.
    # -----------------------------------------------------------------------

    # Panel 3 — dataset-level drift share test (PASS/FAIL counter)
    dashboard.add_panel(
        DashboardPanelPlot(
            title="Dataset Drift Test — Share of Drifted Columns",
            subtitle="FAIL = share of drifted columns exceeded threshold. Counter shows latest value.",
            values=[
                PanelMetric(
                    legend="Drift Share (actual)",
                    metric="DriftedColumnsCount",
                    metric_labels={"value_type": "share"},
                    tags=[],
                    metadata={},
                    view_params={},
                )
            ],
            plot_params={"plot_type": "counter"},
        ),
        tab="Alerts",
    )

    # Panel 4 — per-column drift test status over time (bar per run)
    dashboard.add_panel(
        DashboardPanelPlot(
            title="Per-Column Drift Test — PSI Score Over Time",
            subtitle="Bars above threshold = FAIL alert for that column",
            values=[
                PanelMetric(
                    legend=col,
                    metric="ValueDrift",
                    metric_labels={"column": col},
                    tags=[],
                    metadata={},
                    view_params={},
                )
                for col in COLUMNS
            ],
            plot_params={"plot_type": "bar"},
        ),
        tab="Alerts",
    )

    logger.info("Dashboard panels configured on project '%s'.", project.name)


def log_to_evidently_ui(
    snapshot,
    workspace_path: str,
    project_name: str,
) -> None:
    """
    Log a completed run (Snapshot) into the Evidently UI workspace.

    Tags and metadata are stamped via stamp_snapshot() before this is called,
    but Evidently serialises via to_snapshot_model() which does NOT carry over
    changes made to snapshot.report.tags / .metadata directly.  We therefore:
      1. Convert to SnapshotModel manually.
      2. Inject the tags + metadata into the model.
      3. Write it directly through workspace.state.write_snapshot().

    This makes Tags and Metadata visible in the Reports list UI.
    """
    from evidently.sdk.models import new_id

    workspace = Workspace.create(workspace_path)

    existing = [p for p in workspace.list_projects() if p.name == project_name]
    is_new_project = len(existing) == 0

    project = workspace.create_project(project_name) if is_new_project else existing[0]
    project.description = "Automated data drift monitoring using PSI method."
    project.save()

    # Configure dashboard panels on first creation only
    if is_new_project:
        _setup_dashboard_panels(project)

    # Build timestamped name
    run_name = f"drift_run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    # Convert snapshot → SnapshotModel and inject tags + metadata before writing
    snapshot_model = snapshot.to_snapshot_model()
    snapshot_model.name = run_name
    snapshot_model.tags = snapshot.report.tags          # set by stamp_snapshot()
    snapshot_model.metadata = snapshot.report.metadata  # set by stamp_snapshot()

    snapshot_id = new_id()
    workspace.state.write_snapshot(project.id, snapshot_id, snapshot_model)

    logger.info(
        "Snapshot '%s' logged → workspace '%s' | project '%s'.",
        run_name,
        workspace_path,
        project_name,
    )
    logger.info(
        "Start dashboard:  evidently ui --workspace %s --port 8000",
        workspace_path,
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class DataDriftDetector:
    """End-to-end data drift detection pipeline."""

    def __init__(self, config: DataDriftConfig):
        self.config = config
        self._report = build_report(method=config.drift_method)

    def run(
        self,
        reference_df: DataFrame,
        current_df: DataFrame,
    ) -> dict:
        """
        Execute drift detection, persist reports, and return a drift summary.

        Parameters
        ----------
        reference_df : baseline / training data
        current_df   : new / production data

        Returns
        -------
        drift_summary : dict with keys dataset_drift, n_features, n_drifted,
                        drifted_columns
        """
        # 1. Validate schema
        if not validate_schema(reference_df, current_df):
            raise ValueError(
                "Reference and current datasets do not share the same schema."
            )

        # 2. Run detection
        result = run_drift_detection(self._report, reference_df, current_df)

        # 3. Persist reports
        save_html_report(result, self.config.html_report_path)
        result_dict = save_json_report(result, self.config.json_report_path)

        # 4. Fire Evidently native test alerts (PASS/FAIL from include_tests=True)
        #    Each FAIL is logged at WARNING level immediately in the console.
        alerts = fire_test_alerts(result)

        # 5. Parse summary early so we can stamp the snapshot before logging
        summary = parse_drift_summary(result_dict, drift_share_threshold=self.config.drift_share_threshold)
        summary["alerts"] = alerts

        # 6. Stamp tags + metadata onto snapshot so Reports list shows drift status
        stamp_snapshot(result, summary)

        # 7. Log snapshot to Evidently UI live dashboard
        log_to_evidently_ui(
            snapshot=result,
            workspace_path=self.config.workspace_path,
            project_name=self.config.project_name,
        )

        return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    config = DataDriftConfig(
        html_report_path="/Users/bharathkumar/Desktop/Main_code/ai-technologies-lab/mlops/drift_report.html",
        json_report_path="/Users/bharathkumar/Desktop/Main_code/ai-technologies-lab/mlops/drift_report.json",
        drift_method="psi",
    )

    # Load & split data
    df = load_iris_data()
    reference_df, current_df = split_reference_current(df, reference_end_idx=70)

    # Run detector
    detector = DataDriftDetector(config=config)
    summary = detector.run(reference_df=reference_df, current_df=current_df)

    # Final output
    print("\n=== Drift Detection Summary ===")
    print(f"  Dataset drift detected : {summary['dataset_drift']}")
    print(f"  Features evaluated     : {summary['n_features']}")
    print(f"  Features drifted       : {summary['n_drifted']}")
    print(f"  Drifted columns        : {summary['drifted_columns'] or 'none'}")
    print(f"  Alerts fired           : {len(summary['alerts'])}")
    print("================================")

    if summary["alerts"]:
        print("\n=== Evidently Test Alerts (FAIL) ===")
        for alert in summary["alerts"]:
            print(f"  [{alert['status']}] {alert['test_name']}")
            print(f"         {alert['description']}")
        print("====================================\n")
    else:
        print("  No drift alerts — all tests passed.\n")


if __name__ == "__main__":
    main()