# ml/predictor.py
"""
ChurnPredictor: loads trained artifacts and exposes predict() for single
and batch inference. Handles full one-hot feature construction for ALL
~30 model features (not just the 7 previously exposed in the prototype).
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Bootstrap logging before config import so import errors are visible
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config import with graceful fallback so predictor works standalone too
# ---------------------------------------------------------------------------
try:
    from config import (
        CHURN_THRESHOLD,
        FEATURE_COLUMNS_PATH,
        LEGACY_FEATURE_COLUMNS_PATH,
        LEGACY_MODEL_PIPELINE_PATH,
        LEGACY_SHAP_EXPLAINER_PATH,
        MODEL_PIPELINE_PATH,
        SHAP_EXPLAINER_PATH,
    )
except ImportError:
    _base = Path(__file__).resolve().parent.parent
    MODEL_PIPELINE_PATH = _base / "models" / "churn_model_pipeline.pkl"
    FEATURE_COLUMNS_PATH = _base / "models" / "feature_columns.pkl"
    SHAP_EXPLAINER_PATH = _base / "models" / "shap_explainer.pkl"
    LEGACY_MODEL_PIPELINE_PATH = _base / "churn_model_pipeline.pkl"
    LEGACY_FEATURE_COLUMNS_PATH = _base / "feature_columns.pkl"
    LEGACY_SHAP_EXPLAINER_PATH = _base / "shap_explainer.pkl"
    CHURN_THRESHOLD = 0.35


def _resolve_path(primary: Path, fallback: Path) -> Path:
    """Return *primary* if it exists, else *fallback*."""
    if primary.exists():
        return primary
    if fallback.exists():
        logger.warning(
            "Primary artifact not found at %s; falling back to %s",
            primary,
            fallback,
        )
        return fallback
    raise FileNotFoundError(
        f"Artifact not found at either {primary} or {fallback}."
    )


# ---------------------------------------------------------------------------
# ChurnPredictor
# ---------------------------------------------------------------------------

class ChurnPredictor:
    """
    Wraps the trained sklearn Pipeline, feature columns list, and SHAP
    explainer into a single class that provides a clean predict() interface.

    Usage
    -----
    >>> predictor = ChurnPredictor()
    >>> result = predictor.predict(customer_data_dict)
    >>> # result → {"churn": "Yes"/"No", "probability": 0.72,
    >>> #            "risk_level": "High", "top_factors": [...]}
    """

    def __init__(self) -> None:
        self._load_artifacts()

    # ------------------------------------------------------------------
    # Artifact loading
    # ------------------------------------------------------------------

    def _load_artifacts(self) -> None:
        """Load pipeline, feature columns, and SHAP explainer from disk."""
        pipeline_path = _resolve_path(MODEL_PIPELINE_PATH, LEGACY_MODEL_PIPELINE_PATH)
        features_path = _resolve_path(FEATURE_COLUMNS_PATH, LEGACY_FEATURE_COLUMNS_PATH)
        shap_path = _resolve_path(SHAP_EXPLAINER_PATH, LEGACY_SHAP_EXPLAINER_PATH)

        logger.info("Loading pipeline from %s", pipeline_path)
        self.pipeline = joblib.load(pipeline_path)

        logger.info("Loading feature columns from %s", features_path)
        self.feature_columns: list[str] = list(joblib.load(features_path))

        logger.info("Loading SHAP explainer from %s", shap_path)
        self.explainer = joblib.load(shap_path)

        # Cache the scaler step so we can call transform() for SHAP
        self.scaler = self.pipeline.named_steps["scaler"]

        logger.info(
            "ChurnPredictor ready — %d feature columns loaded",
            len(self.feature_columns),
        )

    # ------------------------------------------------------------------
    # Feature construction  (THE KEY FIX: all ~30 features handled)
    # ------------------------------------------------------------------

    def build_input_df(self, raw: dict[str, Any]) -> pd.DataFrame:
        """
        Construct a single-row DataFrame that matches the model's expected
        one-hot-encoded feature space.

        Parameters
        ----------
        raw : dict
            Keys match the *human-readable* field names from CustomerInput:
            gender, SeniorCitizen, Partner, Dependents, tenure,
            PhoneService, MultipleLines, InternetService, OnlineSecurity,
            OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
            StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
            MonthlyCharges, TotalCharges

        Returns
        -------
        pd.DataFrame with exactly the columns in self.feature_columns,
        all zeros by default, with correct values filled in.
        """
        # Start from an all-zero row with every model feature
        row: dict[str, int | float] = {col: 0 for col in self.feature_columns}

        # ---- Numeric / binary features --------------------------------
        _safe_set(row, "tenure", raw.get("tenure", 0), self.feature_columns)
        _safe_set(row, "MonthlyCharges", raw.get("MonthlyCharges", 0.0), self.feature_columns)
        _safe_set(row, "TotalCharges", raw.get("TotalCharges", 0.0), self.feature_columns)
        _safe_set(row, "SeniorCitizen", int(raw.get("SeniorCitizen", 0)), self.feature_columns)

        # ---- Categorical features → one-hot --------------------------
        # get_dummies(drop_first=True) drops the *first* alphabetical level.
        # We replicate that logic: only set the indicator if the value is
        # NOT the dropped reference category.

        _set_dummies(row, "gender", raw.get("gender", ""), self.feature_columns)
        _set_dummies(row, "Partner", raw.get("Partner", ""), self.feature_columns)
        _set_dummies(row, "Dependents", raw.get("Dependents", ""), self.feature_columns)
        _set_dummies(row, "PhoneService", raw.get("PhoneService", ""), self.feature_columns)
        _set_dummies(row, "MultipleLines", raw.get("MultipleLines", ""), self.feature_columns)
        _set_dummies(row, "InternetService", raw.get("InternetService", ""), self.feature_columns)
        _set_dummies(row, "OnlineSecurity", raw.get("OnlineSecurity", ""), self.feature_columns)
        _set_dummies(row, "OnlineBackup", raw.get("OnlineBackup", ""), self.feature_columns)
        _set_dummies(row, "DeviceProtection", raw.get("DeviceProtection", ""), self.feature_columns)
        _set_dummies(row, "TechSupport", raw.get("TechSupport", ""), self.feature_columns)
        _set_dummies(row, "StreamingTV", raw.get("StreamingTV", ""), self.feature_columns)
        _set_dummies(row, "StreamingMovies", raw.get("StreamingMovies", ""), self.feature_columns)
        _set_dummies(row, "Contract", raw.get("Contract", ""), self.feature_columns)
        _set_dummies(row, "PaperlessBilling", raw.get("PaperlessBilling", ""), self.feature_columns)
        _set_dummies(row, "PaymentMethod", raw.get("PaymentMethod", ""), self.feature_columns)

        df = pd.DataFrame([row], columns=self.feature_columns)
        return df

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, raw: dict[str, Any]) -> dict[str, Any]:
        """
        Run inference for a single customer.

        Parameters
        ----------
        raw : dict  (see build_input_df for field names)

        Returns
        -------
        {
            "churn":        "Yes" | "No",
            "probability":  float  (0.0 – 1.0),
            "risk_level":   "Low" | "Medium" | "High",
            "top_factors":  [{"feature": str, "impact": float}, ...]  # 5 items
        }
        """
        input_df = self.build_input_df(raw)
        probability: float = float(self.pipeline.predict_proba(input_df)[0][1])
        churn_label = "Yes" if probability >= CHURN_THRESHOLD else "No"
        risk_level = _risk_level(probability)

        top_factors = self._shap_top_factors(input_df, n=5)

        return {
            "churn": churn_label,
            "probability": round(probability, 6),
            "risk_level": risk_level,
            "top_factors": top_factors,
        }

    def predict_batch(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Run predict() for each record; returns a list of result dicts."""
        return [self.predict(r) for r in records]

    # ------------------------------------------------------------------
    # SHAP
    # ------------------------------------------------------------------

    def _shap_top_factors(self, input_df: pd.DataFrame, n: int = 5) -> list[dict[str, Any]]:
        """
        Return the top-n SHAP features sorted by absolute impact.

        Returns
        -------
        [{"feature": str, "impact": float}, ...]
        """
        try:
            input_scaled = self.scaler.transform(input_df)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                shap_values = self.explainer(input_scaled)

            # shap_values.values shape: (1, n_features) or (1, n_features, 2)
            values = shap_values.values
            if values.ndim == 3:
                # multi-output: take the positive-class column
                values = values[:, :, 1]

            shap_row = values[0]  # shape (n_features,)
            feature_names = self.feature_columns

            pairs = sorted(
                zip(feature_names, shap_row),
                key=lambda x: abs(x[1]),
                reverse=True,
            )[:n]

            return [
                {"feature": feat, "impact": round(float(imp), 6)}
                for feat, imp in pairs
            ]
        except Exception as exc:  # noqa: BLE001
            logger.error("SHAP computation failed: %s", exc, exc_info=True)
            return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_set(
    row: dict[str, Any],
    col: str,
    value: Any,
    feature_columns: list[str],
) -> None:
    """Set *col* in *row* if it exists in *feature_columns*; warn otherwise."""
    if col in feature_columns:
        row[col] = value
    else:
        logger.warning(
            "Feature '%s' not found in model feature columns — skipping.", col
        )


def _set_dummies(
    row: dict[str, Any],
    prefix: str,
    value: str,
    feature_columns: list[str],
) -> None:
    """
    Replicate pd.get_dummies(drop_first=True) one-hot logic for *prefix*.

    Sets ``{prefix}_{value}`` to 1 if the column exists in *feature_columns*.
    If *value* is the reference (dropped) category, no column will exist and
    this is a no-op — which is exactly the correct behaviour.
    If *value* is non-empty but the column is not found, emit a warning.
    """
    if not value:
        return
    col_name = f"{prefix}_{value}"
    if col_name in feature_columns:
        row[col_name] = 1
    else:
        # This is normal for the reference (dropped) category; only warn
        # when the value is plausible but not in the schema.
        all_prefix_cols = [c for c in feature_columns if c.startswith(f"{prefix}_")]
        if all_prefix_cols:
            # Prefix exists in schema → value is the dropped reference → silently OK
            pass
        else:
            logger.warning(
                "No columns found for prefix '%s' in feature schema — "
                "feature '%s' may be entirely absent from this model.",
                prefix,
                prefix,
            )


def _risk_level(probability: float) -> str:
    if probability < CHURN_THRESHOLD:
        return "Low"
    elif probability < 0.65:
        return "Medium"
    else:
        return "High"
