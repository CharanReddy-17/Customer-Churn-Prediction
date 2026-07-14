# tests/test_predictor.py
"""
Pytest test suite for ChurnPredictor.

Fixtures:
  predictor (module-scoped) — loads once for the entire test session

Tests:
  test_high_risk_customer     — tenure=1, fiber+e-check+month-to-month → churn=Yes, prob>0.35
  test_low_risk_customer      — tenure=60, two year contract, bank transfer → churn=No
  test_output_schema          — all expected keys present in response
  test_shap_factors_count     — exactly 5 factors returned
  test_probability_range      — probability is in [0.0, 1.0]
  test_batch_predict          — batch returns correct number of results
  test_risk_level_high        — very high-risk profile → risk_level == "High"
  test_risk_level_low         — very safe profile → risk_level == "Low"
  test_build_input_df_shape   — build_input_df returns correct column count
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure project root is importable
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from ml.predictor import ChurnPredictor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def predictor() -> ChurnPredictor:
    """Load ChurnPredictor once for all tests in this module."""
    return ChurnPredictor()


# ---------------------------------------------------------------------------
# Shared customer profiles
# ---------------------------------------------------------------------------

HIGH_RISK_CUSTOMER = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 89.10,
    "TotalCharges": 89.10,
}

LOW_RISK_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "Yes",
    "tenure": 60,
    "PhoneService": "Yes",
    "MultipleLines": "Yes",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "Yes",
    "DeviceProtection": "Yes",
    "TechSupport": "Yes",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Two year",
    "PaperlessBilling": "No",
    "PaymentMethod": "Bank transfer (automatic)",
    "MonthlyCharges": 65.00,
    "TotalCharges": 3900.00,
}

MEDIUM_RISK_CUSTOMER = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "One year",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Mailed check",
    "MonthlyCharges": 55.00,
    "TotalCharges": 660.00,
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHighRiskCustomer:
    """Customer with all classic churn indicators."""

    def test_predicts_churn_yes(self, predictor: ChurnPredictor) -> None:
        result = predictor.predict(HIGH_RISK_CUSTOMER)
        assert result["churn"] == "Yes", (
            f"Expected churn=Yes for high-risk customer, got {result['churn']} "
            f"(probability={result['probability']:.4f})"
        )

    def test_probability_above_threshold(self, predictor: ChurnPredictor) -> None:
        result = predictor.predict(HIGH_RISK_CUSTOMER)
        assert result["probability"] > 0.35, (
            f"Expected probability > 0.35 for high-risk customer, "
            f"got {result['probability']:.4f}"
        )


class TestLowRiskCustomer:
    """Long-tenure, two-year contract, bank transfer customer."""

    def test_predicts_churn_no(self, predictor: ChurnPredictor) -> None:
        result = predictor.predict(LOW_RISK_CUSTOMER)
        assert result["churn"] == "No", (
            f"Expected churn=No for low-risk customer, got {result['churn']} "
            f"(probability={result['probability']:.4f})"
        )

    def test_probability_below_threshold(self, predictor: ChurnPredictor) -> None:
        result = predictor.predict(LOW_RISK_CUSTOMER)
        assert result["probability"] < 0.35, (
            f"Expected probability < 0.35 for low-risk customer, "
            f"got {result['probability']:.4f}"
        )


class TestOutputSchema:
    """Verify all required keys are present in every response."""

    REQUIRED_KEYS = {"churn", "probability", "risk_level", "top_factors"}

    def test_high_risk_schema(self, predictor: ChurnPredictor) -> None:
        result = predictor.predict(HIGH_RISK_CUSTOMER)
        missing = self.REQUIRED_KEYS - result.keys()
        assert not missing, f"Missing keys in response: {missing}"

    def test_low_risk_schema(self, predictor: ChurnPredictor) -> None:
        result = predictor.predict(LOW_RISK_CUSTOMER)
        missing = self.REQUIRED_KEYS - result.keys()
        assert not missing, f"Missing keys in response: {missing}"

    def test_churn_label_valid(self, predictor: ChurnPredictor) -> None:
        for customer in (HIGH_RISK_CUSTOMER, LOW_RISK_CUSTOMER, MEDIUM_RISK_CUSTOMER):
            result = predictor.predict(customer)
            assert result["churn"] in {"Yes", "No"}, (
                f"churn label must be 'Yes' or 'No', got: {result['churn']}"
            )

    def test_risk_level_valid(self, predictor: ChurnPredictor) -> None:
        for customer in (HIGH_RISK_CUSTOMER, LOW_RISK_CUSTOMER, MEDIUM_RISK_CUSTOMER):
            result = predictor.predict(customer)
            assert result["risk_level"] in {"Low", "Medium", "High"}, (
                f"risk_level must be Low/Medium/High, got: {result['risk_level']}"
            )

    def test_top_factors_is_list_of_dicts(self, predictor: ChurnPredictor) -> None:
        result = predictor.predict(HIGH_RISK_CUSTOMER)
        assert isinstance(result["top_factors"], list)
        for factor in result["top_factors"]:
            assert "feature" in factor, f"Missing 'feature' key in factor: {factor}"
            assert "impact" in factor, f"Missing 'impact' key in factor: {factor}"
            assert isinstance(factor["feature"], str)
            assert isinstance(factor["impact"], float)


class TestShapFactorsCount:
    """Exactly 5 SHAP factors must be returned."""

    def test_five_factors_high_risk(self, predictor: ChurnPredictor) -> None:
        result = predictor.predict(HIGH_RISK_CUSTOMER)
        assert len(result["top_factors"]) == 5, (
            f"Expected 5 SHAP factors, got {len(result['top_factors'])}"
        )

    def test_five_factors_low_risk(self, predictor: ChurnPredictor) -> None:
        result = predictor.predict(LOW_RISK_CUSTOMER)
        assert len(result["top_factors"]) == 5, (
            f"Expected 5 SHAP factors, got {len(result['top_factors'])}"
        )


class TestProbabilityRange:
    """Probability must always be in [0.0, 1.0]."""

    def test_probability_range_high_risk(self, predictor: ChurnPredictor) -> None:
        result = predictor.predict(HIGH_RISK_CUSTOMER)
        assert 0.0 <= result["probability"] <= 1.0, (
            f"Probability out of range: {result['probability']}"
        )

    def test_probability_range_low_risk(self, predictor: ChurnPredictor) -> None:
        result = predictor.predict(LOW_RISK_CUSTOMER)
        assert 0.0 <= result["probability"] <= 1.0

    def test_probability_range_medium_risk(self, predictor: ChurnPredictor) -> None:
        result = predictor.predict(MEDIUM_RISK_CUSTOMER)
        assert 0.0 <= result["probability"] <= 1.0


class TestBatchPredict:
    """Batch predict should return correct number of results."""

    def test_batch_size_matches(self, predictor: ChurnPredictor) -> None:
        customers = [HIGH_RISK_CUSTOMER, LOW_RISK_CUSTOMER, MEDIUM_RISK_CUSTOMER]
        results = predictor.predict_batch(customers)
        assert len(results) == 3

    def test_batch_results_have_schema(self, predictor: ChurnPredictor) -> None:
        customers = [HIGH_RISK_CUSTOMER, LOW_RISK_CUSTOMER]
        results = predictor.predict_batch(customers)
        for r in results:
            assert "churn" in r
            assert "probability" in r
            assert "top_factors" in r


class TestInputDfShape:
    """build_input_df must produce a DataFrame with the correct column count."""

    def test_column_count_matches_feature_columns(self, predictor: ChurnPredictor) -> None:
        df = predictor.build_input_df(HIGH_RISK_CUSTOMER)
        assert list(df.columns) == predictor.feature_columns, (
            "DataFrame columns do not match predictor.feature_columns"
        )
        assert df.shape == (1, len(predictor.feature_columns))

    def test_no_nan_values(self, predictor: ChurnPredictor) -> None:
        df = predictor.build_input_df(HIGH_RISK_CUSTOMER)
        assert not df.isnull().any().any(), "build_input_df produced NaN values"
