# README.md — Customer Churn Prediction

# 📡 Customer Churn Prediction System

A production-grade ML system for predicting telecom customer churn, built on the
IBM Telco dataset. Includes a FastAPI REST API, Streamlit dashboard, MLflow experiment
tracking, SHAP explanations, Evidently drift monitoring, and a full pytest test suite.

**Model performance:** AUC-ROC 0.845 · Recall (Churn class) 79% · Threshold 0.35

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        USER / CLIENT                                 │
└────────────────────┬──────────────────────┬──────────────────────────┘
                     │                      │
          ┌──────────▼──────────┐   ┌──────▼──────────────┐
          │  Streamlit Dashboard│   │  FastAPI REST API    │
          │  dashboard/app.py   │   │  api/api.py          │
          │  (port 8501)        │   │  (port 8000)         │
          └──────────┬──────────┘   └──────┬───────────────┘
                     │                     │
                     └──────────┬──────────┘
                                │
                    ┌───────────▼────────────┐
                    │   ChurnPredictor        │
                    │   ml/predictor.py       │
                    │   · Pipeline (.pkl)     │
                    │   · Feature cols (.pkl) │
                    │   · SHAP explainer (.pkl│
                    └───────────┬────────────┘
                                │
              ┌─────────────────┼──────────────────┐
              │                 │                  │
   ┌──────────▼──────┐ ┌───────▼──────┐  ┌───────▼──────────┐
   │   SQLite DB      │ │  MLflow       │  │  Evidently        │
   │   predictions.db │ │  mlruns/      │  │  drift_check.py   │
   │   (audit log)    │ │  (tracking)   │  │  monitoring/      │
   └─────────────────┘ └──────────────┘  └──────────────────┘
```

---

## 🚀 Quickstart (3 commands)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy existing pkl files to models/ (first-time setup)
mkdir -p models
cp churn_model_pipeline.pkl models/
cp feature_columns.pkl models/
cp shap_explainer.pkl models/

# 3a. Run the Streamlit dashboard (standalone — no API required)
streamlit run dashboard/app.py

# 3b. Run the FastAPI service
uvicorn api.api:app --reload --port 8000
```

Open the dashboard at **http://localhost:8501**
Open the API docs at **http://localhost:8000/docs**

---

## 📁 Project Structure

```
churn-prediction/
├── api/
│   ├── api.py              ← FastAPI app (5 endpoints)
│   └── schemas.py          ← Pydantic v2 request/response models
├── ml/
│   ├── predictor.py        ← ChurnPredictor class (full feature set)
│   └── train.py            ← Standalone training script + MLflow
├── models/                 ← Trained artifacts (.pkl files)
│   ├── churn_model_pipeline.pkl
│   ├── feature_columns.pkl
│   └── shap_explainer.pkl
├── data/
│   └── raw/
│       └── telco_churn.csv ← IBM Telco dataset
├── dashboard/
│   └── app.py              ← Streamlit UI (18-feature sidebar)
├── monitoring/
│   ├── drift_check.py      ← Evidently drift detection
│   └── reports/            ← Generated HTML drift reports
├── tests/
│   └── test_predictor.py   ← Pytest suite (15 tests)
├── config.py               ← Centralized config & constants
├── Dockerfile              ← Container image for FastAPI
├── .dockerignore
├── requirements.txt
└── README.md
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness check — returns model status |
| `POST` | `/predict` | Single customer churn prediction |
| `POST` | `/predict/batch` | Batch predictions (up to 10,000) |
| `GET` | `/predictions/history` | Paginated prediction log from SQLite |
| `PATCH` | `/predictions/{id}/feedback` | Record actual churn outcome |

### Example: Single Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "TotalCharges": 89.10
  }'
```

### Example Response

```json
{
  "request_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "churn": "Yes",
  "probability": 0.7124,
  "risk_level": "High",
  "top_factors": [
    {"feature": "Contract_Month-to-month", "impact": 0.452},
    {"feature": "tenure",                  "impact": -0.318},
    {"feature": "InternetService_Fiber optic", "impact": 0.281},
    {"feature": "MonthlyCharges",          "impact": 0.221},
    {"feature": "PaymentMethod_Electronic check", "impact": 0.193}
  ],
  "latency_ms": 11.4
}
```

---

## 🔁 How to Retrain

```bash
# Place updated CSV in data/raw/
python ml/train.py --data data/raw/telco_churn.csv

# View results in MLflow UI
mlflow ui --port 5000
# Open http://localhost:5000
```

The script will:
1. Preprocess the raw CSV (fix TotalCharges, one-hot encode, stratified split)
2. Train a `StandardScaler → LogisticRegression(class_weight='balanced')` pipeline
3. Log all params + metrics (AUC, F1, Recall, PR-AUC) to MLflow
4. Save the 3 pkl artifacts to `models/`
5. Register the model as `ChurnPredictor` in the MLflow Model Registry

---

## 📊 How to Run Drift Detection

```bash
# After running the API for at least a few predictions:
python monitoring/drift_check.py

# Look back 7 days instead of default 30:
python monitoring/drift_check.py --days 7

# Custom output path for the HTML report:
python monitoring/drift_check.py --output monitoring/reports/my_report.html
```

Exit code `1` and a **RETRAINING ALERT** is printed if more than 2 features drift.
The HTML report is saved to `monitoring/reports/drift_report.html`.

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=ml --cov-report=term-missing
```

Expected output: **15 tests passing** in < 5 seconds (model loads once via module fixture).

---

## 🐳 Docker

```bash
# Build the image
docker build -t churn-api:latest .

# Run the container
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/predictions.db:/app/predictions.db \
  churn-api:latest

# Health check
curl http://localhost:8000/health
```

---

## 🐛 Known Bugs Fixed

| Bug | Status |
|-----|--------|
| Original `app.py` silently zeroed ~23 features (only 7 exposed in UI) | ✅ Fixed in `ml/predictor.py` (`build_input_df`) — all 18 raw features map to all ~30 one-hot columns |
| Threshold hardcoded to 0.5 in old UI | ✅ Centralized in `config.py` as `CHURN_THRESHOLD = 0.35` |

---

## 📦 Tech Stack

| Layer | Technology |
|-------|-----------|
| Model | scikit-learn Pipeline (StandardScaler + LogisticRegression) |
| Explainability | SHAP LinearExplainer |
| API | FastAPI + Pydantic v2 |
| Dashboard | Streamlit |
| Tracking | MLflow |
| Monitoring | Evidently AI |
| Storage | SQLite (prediction log) |
| Container | Docker (python:3.11-slim) |
| Tests | Pytest |

---

*Built by **Charan Reddy** · Customer Churn ML Project*
