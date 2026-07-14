# ml/train.py
"""
Standalone training script for the Customer Churn Prediction model.

Usage:
    python ml/train.py --data data/raw/telco_churn.csv

What this script does:
  1. Loads and preprocesses the IBM Telco Churn CSV.
  2. Trains an sklearn Pipeline: StandardScaler → LogisticRegression.
  3. Evaluates on a stratified 80/20 split.
  4. Logs all params + metrics to MLflow under the 'customer-churn-prediction' experiment.
  5. Saves the pipeline, feature columns, and SHAP explainer to models/.
  6. Registers the model in the MLflow Model Registry as 'ChurnPredictor'.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Ensure project root is importable when run from repo root or ml/ dir
# ---------------------------------------------------------------------------
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config import (
    CHURN_THRESHOLD,
    LOG_FORMAT,
    LOG_LEVEL,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_MODEL_REGISTRY_NAME,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    RAW_DATA_PATH,
)

logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def load_and_preprocess(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load raw Telco CSV and return (X, y) ready for train/test split.

    Steps:
      - Drop customerID (no predictive value)
      - Fix TotalCharges: coerce to numeric, fill NaN with 0
      - Encode target: Yes → 1, No → 0
      - One-hot encode all remaining categoricals with drop_first=True
    """
    logger.info("Loading data from %s", csv_path)
    df = pd.read_csv(csv_path)
    logger.info("Raw shape: %s", df.shape)

    # Drop ID column
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Fix TotalCharges (whitespace strings in the dataset)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # Encode target
    y = (df["Churn"].str.strip().str.capitalize() == "Yes").astype(int)
    df = df.drop(columns=["Churn"])

    # One-hot encode
    df = pd.get_dummies(df, drop_first=True)

    logger.info("Processed shape: %s  |  Churn rate: %.2f%%", df.shape, y.mean() * 100)
    return df, y


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def compute_metrics(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = CHURN_THRESHOLD,
) -> dict[str, float]:
    """Compute AUC, F1, Precision, Recall, PR-AUC at the given threshold."""
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    return {
        "auc_roc": round(roc_auc_score(y_test, y_proba), 4),
        "pr_auc": round(average_precision_score(y_test, y_proba), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "threshold": threshold,
    }


# ---------------------------------------------------------------------------
# SHAP explainer
# ---------------------------------------------------------------------------

def build_shap_explainer(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
) -> shap.LinearExplainer:
    """Fit a LinearExplainer on the scaled training data."""
    scaler: StandardScaler = pipeline.named_steps["scaler"]
    X_train_scaled = scaler.transform(X_train)
    explainer = shap.LinearExplainer(
        pipeline.named_steps["clf"],
        X_train_scaled,
        feature_perturbation="interventional",
    )
    return explainer


# ---------------------------------------------------------------------------
# Artefact saving
# ---------------------------------------------------------------------------

def save_artifacts(
    pipeline: Pipeline,
    feature_columns: list[str],
    explainer: shap.LinearExplainer,
    models_dir: Path,
) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)

    pipeline_path = models_dir / "churn_model_pipeline.pkl"
    features_path = models_dir / "feature_columns.pkl"
    shap_path = models_dir / "shap_explainer.pkl"

    joblib.dump(pipeline, pipeline_path)
    joblib.dump(feature_columns, features_path)
    joblib.dump(explainer, shap_path)

    logger.info("Saved pipeline  → %s", pipeline_path)
    logger.info("Saved features  → %s", features_path)
    logger.info("Saved explainer → %s", shap_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(csv_path: Path) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # ---- Preprocessing ----
    X, y = load_and_preprocess(csv_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    logger.info(
        "Split: train=%d  test=%d", len(X_train), len(X_test)
    )

    # ---- MLflow ----
    # ---- MLflow ----
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_registry_uri("http://localhost:5000")
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="logistic_regression_baseline") as run:
        run_id = run.info.run_id
        logger.info("MLflow run ID: %s", run_id)

        # ---- Train ----
        pipeline = build_pipeline()
        pipeline.fit(X_train, y_train)
        logger.info("Training complete.")

        # ---- Evaluate ----
        metrics = compute_metrics(pipeline, X_test, y_test)
        logger.info("Metrics: %s", metrics)

        # ---- SHAP ----
        explainer = build_shap_explainer(pipeline, X_train)

        # ---- Log to MLflow ----
        lr_params = pipeline.named_steps["clf"].get_params()
        mlflow.log_params(
            {
                "model_type": "LogisticRegression",
                "class_weight": lr_params["class_weight"],
                "max_iter": lr_params["max_iter"],
                "solver": lr_params["solver"],
                "random_state": lr_params["random_state"],
                "n_features": X_train.shape[1],
                "train_size": len(X_train),
                "test_size": len(X_test),
                "churn_threshold": CHURN_THRESHOLD,
            }
        )
        mlflow.log_metrics(metrics)

        # Log sklearn model to MLflow
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name=MLFLOW_MODEL_REGISTRY_NAME,
        )

        # ---- Save artifacts locally ----
        feature_columns = list(X_train.columns)
        save_artifacts(pipeline, feature_columns, explainer, MODELS_DIR)

        # Log artifacts to MLflow too
        mlflow.log_artifact(str(MODELS_DIR / "churn_model_pipeline.pkl"), artifact_path="artifacts")
        mlflow.log_artifact(str(MODELS_DIR / "feature_columns.pkl"), artifact_path="artifacts")

        logger.info(
            "✅ Training complete!\n"
            "  AUC-ROC : %.4f\n"
            "  PR-AUC  : %.4f\n"
            "  F1      : %.4f\n"
            "  Recall  : %.4f\n"
            "  Precision: %.4f",
            metrics["auc_roc"],
            metrics["pr_auc"],
            metrics["f1"],
            metrics["recall"],
            metrics["precision"],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Customer Churn model.")
    parser.add_argument(
        "--data",
        type=Path,
        default=RAW_DATA_PATH,
        help="Path to the raw Telco Churn CSV.",
    )
    args = parser.parse_args()
    main(args.data)
