# api/schemas.py
"""
Pydantic v2 request / response schemas for the Churn Prediction API.
All allowed categorical values are imported from config so they stay in sync
with the predictor and the dashboard.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

try:
    from config import (
        CONTRACT_VALUES,
        DEPENDENTS_VALUES,
        DEVICE_PROTECTION_VALUES,
        GENDER_VALUES,
        INTERNET_ADDON_VALUES,
        INTERNET_SERVICE_VALUES,
        MULTIPLE_LINES_VALUES,
        ONLINE_BACKUP_VALUES,
        ONLINE_SECURITY_VALUES,
        PAPERLESS_BILLING_VALUES,
        PARTNER_VALUES,
        PAYMENT_METHOD_VALUES,
        PHONE_SERVICE_VALUES,
        STREAMING_MOVIES_VALUES,
        STREAMING_TV_VALUES,
        TECH_SUPPORT_VALUES,
    )
except ImportError:
    # Fallback definitions for standalone use
    GENDER_VALUES = ["Male", "Female"]
    PARTNER_VALUES = ["Yes", "No"]
    DEPENDENTS_VALUES = ["Yes", "No"]
    PHONE_SERVICE_VALUES = ["Yes", "No"]
    MULTIPLE_LINES_VALUES = ["No phone service", "No", "Yes"]
    INTERNET_SERVICE_VALUES = ["DSL", "Fiber optic", "No"]
    INTERNET_ADDON_VALUES = ["No internet service", "No", "Yes"]
    ONLINE_SECURITY_VALUES = INTERNET_ADDON_VALUES
    ONLINE_BACKUP_VALUES = INTERNET_ADDON_VALUES
    DEVICE_PROTECTION_VALUES = INTERNET_ADDON_VALUES
    TECH_SUPPORT_VALUES = INTERNET_ADDON_VALUES
    STREAMING_TV_VALUES = INTERNET_ADDON_VALUES
    STREAMING_MOVIES_VALUES = INTERNET_ADDON_VALUES
    CONTRACT_VALUES = ["Month-to-month", "One year", "Two year"]
    PAPERLESS_BILLING_VALUES = ["Yes", "No"]
    PAYMENT_METHOD_VALUES = [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]


# ---------------------------------------------------------------------------
# Helper: build a reusable validator factory
# ---------------------------------------------------------------------------

def _make_allowed_validator(allowed: list[str], field_label: str):
    """Return a field_validator function that checks against *allowed*."""

    def _validator(cls, v: str) -> str:  # noqa: N805
        if v not in allowed:
            raise ValueError(
                f"'{v}' is not a valid value for {field_label}. "
                f"Allowed: {allowed}"
            )
        return v

    return classmethod(_validator)


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------

class CustomerInput(BaseModel):
    """All 18 raw customer features required by the Churn model."""

    gender: str = Field(
        ...,
        description="Customer gender",
        examples=["Male"],
    )
    SeniorCitizen: int = Field(
        ...,
        ge=0,
        le=1,
        description="1 if senior citizen, 0 otherwise",
        examples=[0],
    )
    Partner: str = Field(..., examples=["Yes"])
    Dependents: str = Field(..., examples=["No"])
    tenure: int = Field(..., ge=0, le=120, description="Months with company", examples=[12])
    PhoneService: str = Field(..., examples=["Yes"])
    MultipleLines: str = Field(..., examples=["No"])
    InternetService: str = Field(..., examples=["Fiber optic"])
    OnlineSecurity: str = Field(..., examples=["No"])
    OnlineBackup: str = Field(..., examples=["No"])
    DeviceProtection: str = Field(..., examples=["No"])
    TechSupport: str = Field(..., examples=["No"])
    StreamingTV: str = Field(..., examples=["No"])
    StreamingMovies: str = Field(..., examples=["No"])
    Contract: str = Field(..., examples=["Month-to-month"])
    PaperlessBilling: str = Field(..., examples=["Yes"])
    PaymentMethod: str = Field(..., examples=["Electronic check"])
    MonthlyCharges: float = Field(..., ge=0.0, examples=[70.35])
    TotalCharges: float = Field(..., ge=0.0, examples=[843.0])

    # ---- Field validators ------------------------------------------------

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, v: str) -> str:
        if v not in GENDER_VALUES:
            raise ValueError(f"gender must be one of {GENDER_VALUES}")
        return v

    @field_validator("Partner")
    @classmethod
    def validate_partner(cls, v: str) -> str:
        if v not in PARTNER_VALUES:
            raise ValueError(f"Partner must be one of {PARTNER_VALUES}")
        return v

    @field_validator("Dependents")
    @classmethod
    def validate_dependents(cls, v: str) -> str:
        if v not in DEPENDENTS_VALUES:
            raise ValueError(f"Dependents must be one of {DEPENDENTS_VALUES}")
        return v

    @field_validator("PhoneService")
    @classmethod
    def validate_phone_service(cls, v: str) -> str:
        if v not in PHONE_SERVICE_VALUES:
            raise ValueError(f"PhoneService must be one of {PHONE_SERVICE_VALUES}")
        return v

    @field_validator("MultipleLines")
    @classmethod
    def validate_multiple_lines(cls, v: str) -> str:
        if v not in MULTIPLE_LINES_VALUES:
            raise ValueError(f"MultipleLines must be one of {MULTIPLE_LINES_VALUES}")
        return v

    @field_validator("InternetService")
    @classmethod
    def validate_internet_service(cls, v: str) -> str:
        if v not in INTERNET_SERVICE_VALUES:
            raise ValueError(f"InternetService must be one of {INTERNET_SERVICE_VALUES}")
        return v

    @field_validator("OnlineSecurity")
    @classmethod
    def validate_online_security(cls, v: str) -> str:
        if v not in ONLINE_SECURITY_VALUES:
            raise ValueError(f"OnlineSecurity must be one of {ONLINE_SECURITY_VALUES}")
        return v

    @field_validator("OnlineBackup")
    @classmethod
    def validate_online_backup(cls, v: str) -> str:
        if v not in ONLINE_BACKUP_VALUES:
            raise ValueError(f"OnlineBackup must be one of {ONLINE_BACKUP_VALUES}")
        return v

    @field_validator("DeviceProtection")
    @classmethod
    def validate_device_protection(cls, v: str) -> str:
        if v not in DEVICE_PROTECTION_VALUES:
            raise ValueError(f"DeviceProtection must be one of {DEVICE_PROTECTION_VALUES}")
        return v

    @field_validator("TechSupport")
    @classmethod
    def validate_tech_support(cls, v: str) -> str:
        if v not in TECH_SUPPORT_VALUES:
            raise ValueError(f"TechSupport must be one of {TECH_SUPPORT_VALUES}")
        return v

    @field_validator("StreamingTV")
    @classmethod
    def validate_streaming_tv(cls, v: str) -> str:
        if v not in STREAMING_TV_VALUES:
            raise ValueError(f"StreamingTV must be one of {STREAMING_TV_VALUES}")
        return v

    @field_validator("StreamingMovies")
    @classmethod
    def validate_streaming_movies(cls, v: str) -> str:
        if v not in STREAMING_MOVIES_VALUES:
            raise ValueError(f"StreamingMovies must be one of {STREAMING_MOVIES_VALUES}")
        return v

    @field_validator("Contract")
    @classmethod
    def validate_contract(cls, v: str) -> str:
        if v not in CONTRACT_VALUES:
            raise ValueError(f"Contract must be one of {CONTRACT_VALUES}")
        return v

    @field_validator("PaperlessBilling")
    @classmethod
    def validate_paperless_billing(cls, v: str) -> str:
        if v not in PAPERLESS_BILLING_VALUES:
            raise ValueError(f"PaperlessBilling must be one of {PAPERLESS_BILLING_VALUES}")
        return v

    @field_validator("PaymentMethod")
    @classmethod
    def validate_payment_method(cls, v: str) -> str:
        if v not in PAYMENT_METHOD_VALUES:
            raise ValueError(f"PaymentMethod must be one of {PAYMENT_METHOD_VALUES}")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": 29.85,
            }
        }
    }


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class ShapFactor(BaseModel):
    """A single SHAP feature-impact pair."""

    feature: str
    impact: float


class PredictionResponse(BaseModel):
    """Single-prediction API response."""

    request_id: str = Field(..., description="UUID for this request")
    churn: Literal["Yes", "No"]
    probability: float = Field(..., ge=0.0, le=1.0)
    risk_level: Literal["Low", "Medium", "High"]
    top_factors: list[ShapFactor]
    latency_ms: float = Field(..., description="End-to-end prediction latency in ms")

    model_config = {
        "json_schema_extra": {
            "example": {
                "request_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                "churn": "Yes",
                "probability": 0.712,
                "risk_level": "High",
                "top_factors": [
                    {"feature": "Contract_Month-to-month", "impact": 0.45},
                    {"feature": "tenure", "impact": -0.31},
                    {"feature": "InternetService_Fiber optic", "impact": 0.28},
                    {"feature": "MonthlyCharges", "impact": 0.22},
                    {"feature": "PaymentMethod_Electronic check", "impact": 0.19},
                ],
                "latency_ms": 12.4,
            }
        }
    }


class BatchPredictionItem(BaseModel):
    """A single item inside a batch response."""

    index: int
    request_id: str
    churn: Literal["Yes", "No"]
    probability: float
    risk_level: Literal["Low", "Medium", "High"]
    top_factors: list[ShapFactor]
    error: str | None = None


class BatchPredictionResponse(BaseModel):
    """Batch-prediction API response."""

    batch_request_id: str
    total: int
    successful: int
    failed: int
    latency_ms: float
    predictions: list[BatchPredictionItem]


# ---------------------------------------------------------------------------
# Feedback schema
# ---------------------------------------------------------------------------

class FeedbackPatch(BaseModel):
    """Payload for PATCH /predictions/{id}/feedback."""

    actual_churn: Literal["Yes", "No"] = Field(
        ..., description="The real outcome observed for this customer"
    )
    notes: str | None = Field(None, description="Optional free-text notes")


# ---------------------------------------------------------------------------
# History item schema
# ---------------------------------------------------------------------------

class PredictionHistoryItem(BaseModel):
    """One row from the predictions history table."""

    id: int
    request_id: str
    timestamp: str
    churn: str
    probability: float
    risk_level: str
    actual_churn: str | None = None
    notes: str | None = None
