# monitoring/drift_check.py
"""
Data drift monitoring using Evidently AI.

What this script does:
  1. Loads the training CSV as the reference dataset.
  2. Loads recent predictions from the SQLite predictions database
     as the current dataset.
  3. Runs the Evidently DataDriftPreset report.
  4. Prints PSI / drift scores per feature to stdout.
  5. Saves an HTML drift report to monitoring/reports/.
  6. If > 2 features have drifted, prints a retraining alert.

Usage:
    python monitoring/drift_check.py
    python monitoring/drift_check.py --days 7  # look back 7 days of predictions
    python monitoring/drift_check.py --output monitoring/reports/drift_report.html
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config import DB_PATH, LOG_FORMAT, LOG_LEVEL, RAW_DATA_PATH

logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature columns that exist in both the raw training CSV and the predictions
# table (after renaming).  We align on these for drift comparison.
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = ["tenure", "monthly_charges", "total_charges"]
CATEGORICAL_FEATURES = [
    "gender",
    "senior_citizen",
    "partner",
    "dependents",
    "phone_service",
    "multiple_lines",
    "internet_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "contract",
    "paperless_billing",
    "payment_method",
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Mapping from raw CSV column names → DB column names
CSV_TO_DB_RENAME: dict[str, str] = {
    "MonthlyCharges": "monthly_charges",
    "TotalCharges": "total_charges",
    "SeniorCitizen": "senior_citizen",
    "Partner": "partner",
    "Dependents": "dependents",
    "PhoneService": "phone_service",
    "MultipleLines": "multiple_lines",
    "InternetService": "internet_service",
    "OnlineSecurity": "online_security",
    "OnlineBackup": "online_backup",
    "DeviceProtection": "device_protection",
    "TechSupport": "tech_support",
    "StreamingTV": "streaming_tv",
    "StreamingMovies": "streaming_movies",
    "Contract": "contract",
    "PaperlessBilling": "paperless_billing",
    "PaymentMethod": "payment_method",
}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_reference_data(csv_path: Path) -> pd.DataFrame:
    """Load and minimally preprocess the training CSV as the reference set."""
    df = pd.read_csv(csv_path)
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    if "Churn" in df.columns:
        df = df.drop(columns=["Churn"])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    # Rename to match DB column names
    df = df.rename(columns=CSV_TO_DB_RENAME)

    # Lowercase remaining column names
    df.columns = [c.lower() if c not in df.columns else c for c in df.columns]

    # Ensure SeniorCitizen is present under new name
    available = [c for c in ALL_FEATURES if c in df.columns]
    return df[available]


def load_current_data(db_path: Path, days_back: int = 30) -> pd.DataFrame:
    """Load recent predictions from SQLite as the current dataset."""
    if not db_path.exists():
        logger.error("Database not found at %s. Run the API first.", db_path)
        sys.exit(1)

    cutoff = (
        datetime.now(timezone.utc) - timedelta(days=days_back)
    ).isoformat()

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    query = """
        SELECT
            gender, senior_citizen, partner, dependents, tenure,
            phone_service, multiple_lines, internet_service,
            online_security, online_backup, device_protection, tech_support,
            streaming_tv, streaming_movies, contract, paperless_billing,
            payment_method, monthly_charges, total_charges
        FROM predictions
        WHERE timestamp >= ?
        ORDER BY id DESC
    """
    rows = conn.execute(query, (cutoff,)).fetchall()
    conn.close()

    if not rows:
        logger.warning(
            "No predictions found in the last %d days. "
            "Cannot perform drift check.",
            days_back,
        )
        sys.exit(0)

    df = pd.DataFrame([dict(r) for r in rows])
    available = [c for c in ALL_FEATURES if c in df.columns]
    return df[available]


# ---------------------------------------------------------------------------
# Drift check
# ---------------------------------------------------------------------------

def run_drift_check(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_path: Path,
) -> int:
    """
    Run Evidently DataDriftPreset and return the count of drifted features.

    Saves an HTML report to *output_path* and prints a summary to stdout.
    """
    try:
        from evidently import ColumnMapping
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report
    except ImportError:
        logger.error(
            "Evidently is not installed. Run: pip install evidently"
        )
        sys.exit(1)

    logger.info(
        "Running drift check: reference=%d rows, current=%d rows",
        len(reference),
        len(current),
    )

    # Evidently ColumnMapping
    column_mapping = ColumnMapping(
        numerical_features=NUMERIC_FEATURES,
        categorical_features=[c for c in CATEGORICAL_FEATURES if c in reference.columns],
    )

    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=reference,
        current_data=current,
        column_mapping=column_mapping,
    )

    # Save HTML report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(output_path))
    logger.info("Drift report saved to %s", output_path)

    # Extract drift results
    report_dict = report.as_dict()
    drift_results = report_dict["metrics"][0]["result"]
    number_of_drifted = drift_results.get("number_of_drifted_columns", 0)
    share_drifted = drift_results.get("share_of_drifted_columns", 0.0)

    print("\n" + "=" * 60)
    print("  DATA DRIFT REPORT SUMMARY")
    print("=" * 60)
    print(f"  Reference rows  : {len(reference):,}")
    print(f"  Current rows    : {len(current):,}")
    print(f"  Features checked: {drift_results.get('number_of_columns', 0)}")
    print(f"  Drifted features: {number_of_drifted}  ({share_drifted:.1%})")
    print("-" * 60)

    # Per-feature drift breakdown
    per_feature = drift_results.get("drift_by_columns", {})
    if per_feature:
        print(f"  {'Feature':<35} {'Drifted':<10} {'Score':>10}")
        print(f"  {'-'*35} {'-'*10} {'-'*10}")
        for feat_name, feat_info in sorted(per_feature.items()):
            drifted = "✅ YES" if feat_info.get("drift_detected", False) else "  no"
            score = feat_info.get("stattest_threshold", None)
            score_str = f"{score:.4f}" if score is not None else "  N/A"
            print(f"  {feat_name:<35} {drifted:<10} {score_str:>10}")

    print("=" * 60)

    if number_of_drifted > 2:
        print(
            "\n🚨 RETRAINING ALERT: More than 2 features have drifted.\n"
            "   It is recommended to retrain the model with fresh data.\n"
            "   Run: python ml/train.py --data data/raw/telco_churn.csv"
        )
    else:
        print(
            f"\n✅ Drift within acceptable limits "
            f"({number_of_drifted}/2 threshold). No immediate action required."
        )

    return number_of_drifted


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run data drift check using Evidently.")
    parser.add_argument(
        "--data",
        type=Path,
        default=RAW_DATA_PATH,
        help="Path to the raw training CSV (reference dataset).",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="How many days of recent predictions to use as current dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("monitoring/reports/drift_report.html"),
        help="Output path for the HTML drift report.",
    )
    args = parser.parse_args()

    reference = load_reference_data(args.data)
    current = load_current_data(Path(DB_PATH), days_back=args.days)

    n_drifted = run_drift_check(reference, current, args.output)
    sys.exit(1 if n_drifted > 2 else 0)


if __name__ == "__main__":
    main()
