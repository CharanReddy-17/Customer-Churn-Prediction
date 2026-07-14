# api/api.py
"""
FastAPI application for the Customer Churn Prediction service.

Endpoints:
  GET  /health                        → liveness check
  POST /predict                       → single prediction
  POST /predict/batch                 → batch predictions (up to 10,000)
  GET  /predictions/history           → paginated prediction log from SQLite
  PATCH /predictions/{id}/feedback    → record actual outcome for a prediction
"""

from __future__ import annotations

import logging
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

import sys
import os

# Ensure project root is on sys.path when run from api/ directory
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    BatchPredictionItem,
    BatchPredictionResponse,
    CustomerInput,
    FeedbackPatch,
    PredictionHistoryItem,
    PredictionResponse,
    ShapFactor,
)
from config import DB_PATH, LOG_FORMAT, LOG_LEVEL
from ml.predictor import ChurnPredictor

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id    TEXT    NOT NULL UNIQUE,
    timestamp     TEXT    NOT NULL,
    -- Raw inputs (stored as TEXT for portability)
    gender                TEXT,
    senior_citizen        INTEGER,
    partner               TEXT,
    dependents            TEXT,
    tenure                INTEGER,
    phone_service         TEXT,
    multiple_lines        TEXT,
    internet_service      TEXT,
    online_security       TEXT,
    online_backup         TEXT,
    device_protection     TEXT,
    tech_support          TEXT,
    streaming_tv          TEXT,
    streaming_movies      TEXT,
    contract              TEXT,
    paperless_billing     TEXT,
    payment_method        TEXT,
    monthly_charges       REAL,
    total_charges         REAL,
    -- Outputs
    churn                 TEXT    NOT NULL,
    probability           REAL    NOT NULL,
    risk_level            TEXT    NOT NULL,
    top_factors_json      TEXT,
    latency_ms            REAL,
    -- Feedback
    actual_churn          TEXT,
    feedback_notes        TEXT,
    feedback_at           TEXT
);
"""


def _get_db() -> sqlite3.Connection:
    """Create (or open) the SQLite database and ensure the table exists."""
    db_path = Path(DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute(_CREATE_TABLE_SQL)
    conn.commit()
    return conn


def _log_prediction(
    conn: sqlite3.Connection,
    request_id: str,
    timestamp: str,
    customer: CustomerInput,
    result: dict[str, Any],
    latency_ms: float,
) -> None:
    """Insert a prediction row into SQLite."""
    import json

    top_factors_json = json.dumps(result.get("top_factors", []))
    conn.execute(
        """
        INSERT OR IGNORE INTO predictions (
            request_id, timestamp,
            gender, senior_citizen, partner, dependents, tenure,
            phone_service, multiple_lines, internet_service,
            online_security, online_backup, device_protection, tech_support,
            streaming_tv, streaming_movies, contract, paperless_billing,
            payment_method, monthly_charges, total_charges,
            churn, probability, risk_level, top_factors_json, latency_ms
        ) VALUES (
            ?, ?,  ?, ?, ?, ?, ?,  ?, ?, ?,
            ?, ?, ?, ?,  ?, ?, ?, ?,  ?, ?, ?,
            ?, ?, ?, ?, ?
        )
        """,
        (
            request_id,
            timestamp,
            customer.gender,
            customer.SeniorCitizen,
            customer.Partner,
            customer.Dependents,
            customer.tenure,
            customer.PhoneService,
            customer.MultipleLines,
            customer.InternetService,
            customer.OnlineSecurity,
            customer.OnlineBackup,
            customer.DeviceProtection,
            customer.TechSupport,
            customer.StreamingTV,
            customer.StreamingMovies,
            customer.Contract,
            customer.PaperlessBilling,
            customer.PaymentMethod,
            customer.MonthlyCharges,
            customer.TotalCharges,
            result["churn"],
            result["probability"],
            result["risk_level"],
            top_factors_json,
            latency_ms,
        ),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

predictor: ChurnPredictor | None = None
db_conn: sqlite3.Connection | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global predictor, db_conn
    logger.info("Starting up — loading ChurnPredictor and SQLite …")
    predictor = ChurnPredictor()
    db_conn = _get_db()
    logger.info("Startup complete.")
    yield
    if db_conn:
        db_conn.close()
    logger.info("Shutdown complete.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "Production-grade REST API for predicting telecom customer churn "
        "using a trained sklearn Pipeline with SHAP explanations."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Dependency: validated predictor
# ---------------------------------------------------------------------------

def get_predictor() -> ChurnPredictor:
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    return predictor


def get_db() -> sqlite3.Connection:
    if db_conn is None:
        raise HTTPException(status_code=503, detail="Database not ready.")
    return db_conn


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", tags=["System"])
def health_check() -> dict[str, str]:
    """Liveness check. Returns 200 when the service is ready."""
    model_status = "loaded" if predictor is not None else "not_loaded"
    return {
        "status": "ok",
        "model": model_status,
        "version": app.version,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_single(
    customer: CustomerInput,
    pred: ChurnPredictor = Depends(get_predictor),
    conn: sqlite3.Connection = Depends(get_db),
) -> PredictionResponse:
    """
    Predict churn for a single customer.

    Returns churn label, probability, risk level, top-5 SHAP factors,
    and end-to-end latency. Every call is logged to SQLite.
    """
    request_id = str(uuid.uuid4())
    from datetime import datetime, timezone

    timestamp = datetime.now(timezone.utc).isoformat()

    t0 = time.perf_counter()
    try:
        result = pred.predict(customer.model_dump())
    except Exception as exc:
        logger.error("Prediction error for request %s: %s", request_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    logger.info(
        "request_id=%s churn=%s probability=%.4f latency_ms=%.1f",
        request_id,
        result["churn"],
        result["probability"],
        latency_ms,
    )

    _log_prediction(conn, request_id, timestamp, customer, result, latency_ms)

    return PredictionResponse(
        request_id=request_id,
        churn=result["churn"],  # type: ignore[arg-type]
        probability=result["probability"],
        risk_level=result["risk_level"],  # type: ignore[arg-type]
        top_factors=[ShapFactor(**f) for f in result["top_factors"]],
        latency_ms=latency_ms,
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch(
    customers: list[CustomerInput],
    pred: ChurnPredictor = Depends(get_predictor),
    conn: sqlite3.Connection = Depends(get_db),
) -> BatchPredictionResponse:
    """
    Predict churn for a batch of up to 10,000 customers.

    Each record is logged individually. Errors in individual records are
    captured per-item without failing the entire batch.
    """
    if len(customers) > 10_000:
        raise HTTPException(
            status_code=422,
            detail="Batch size exceeds maximum of 10,000 records.",
        )

    from datetime import datetime, timezone

    batch_request_id = str(uuid.uuid4())
    t_batch_start = time.perf_counter()

    items: list[BatchPredictionItem] = []
    successful = 0
    failed = 0

    for idx, customer in enumerate(customers):
        request_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        try:
            t0 = time.perf_counter()
            result = pred.predict(customer.model_dump())
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            _log_prediction(conn, request_id, timestamp, customer, result, latency_ms)
            items.append(
                BatchPredictionItem(
                    index=idx,
                    request_id=request_id,
                    churn=result["churn"],  # type: ignore[arg-type]
                    probability=result["probability"],
                    risk_level=result["risk_level"],  # type: ignore[arg-type]
                    top_factors=[ShapFactor(**f) for f in result["top_factors"]],
                )
            )
            successful += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("Batch item %d failed: %s", idx, exc)
            items.append(
                BatchPredictionItem(
                    index=idx,
                    request_id=request_id,
                    churn="No",
                    probability=0.0,
                    risk_level="Low",
                    top_factors=[],
                    error=str(exc),
                )
            )
            failed += 1

    total_latency_ms = round((time.perf_counter() - t_batch_start) * 1000, 2)

    return BatchPredictionResponse(
        batch_request_id=batch_request_id,
        total=len(customers),
        successful=successful,
        failed=failed,
        latency_ms=total_latency_ms,
        predictions=items,
    )


@app.get(
    "/predictions/history",
    response_model=list[PredictionHistoryItem],
    tags=["History"],
)
def get_predictions_history(
    limit: int = Query(default=100, ge=1, le=1000, description="Max rows to return"),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
    conn: sqlite3.Connection = Depends(get_db),
) -> list[PredictionHistoryItem]:
    """
    Return paginated prediction history from SQLite.

    Results are ordered newest-first.
    """
    rows = conn.execute(
        """
        SELECT id, request_id, timestamp, churn, probability, risk_level,
               actual_churn, feedback_notes AS notes
        FROM predictions
        ORDER BY id DESC
        LIMIT ? OFFSET ?
        """,
        (limit, offset),
    ).fetchall()

    return [
        PredictionHistoryItem(
            id=row["id"],
            request_id=row["request_id"],
            timestamp=row["timestamp"],
            churn=row["churn"],
            probability=row["probability"],
            risk_level=row["risk_level"],
            actual_churn=row["actual_churn"],
            notes=row["notes"],
        )
        for row in rows
    ]


@app.patch(
    "/predictions/{prediction_id}/feedback",
    tags=["History"],
)
def record_feedback(
    prediction_id: str,
    feedback: FeedbackPatch,
    conn: sqlite3.Connection = Depends(get_db),
) -> dict[str, str]:
    """
    Record the actual churn outcome for a previously made prediction.

    *prediction_id* can be the integer row ID **or** the UUID request_id.
    """
    from datetime import datetime, timezone

    feedback_at = datetime.now(timezone.utc).isoformat()

    # Try UUID first, then integer ID
    result = conn.execute(
        "SELECT id FROM predictions WHERE request_id = ?",
        (prediction_id,),
    ).fetchone()

    if result is None:
        try:
            int_id = int(prediction_id)
            result = conn.execute(
                "SELECT id FROM predictions WHERE id = ?", (int_id,)
            ).fetchone()
        except ValueError:
            pass

    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Prediction '{prediction_id}' not found.",
        )

    row_id = result["id"]
    conn.execute(
        """
        UPDATE predictions
        SET actual_churn = ?, feedback_notes = ?, feedback_at = ?
        WHERE id = ?
        """,
        (feedback.actual_churn, feedback.notes, feedback_at, row_id),
    )
    conn.commit()

    logger.info(
        "Feedback recorded: prediction_id=%s actual_churn=%s",
        prediction_id,
        feedback.actual_churn,
    )
    return {"status": "updated", "prediction_id": prediction_id}
