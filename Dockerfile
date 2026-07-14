FROM python:3.11-slim AS base

# ---- System dependencies ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- Install Python dependencies (separate layer for caching) ----
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ---- Copy application code ----
COPY config.py .
COPY ml/ ml/
COPY api/ api/
COPY monitoring/ monitoring/
COPY dashboard/ dashboard/

# ---- Copy model artifacts ----
# The models/ directory must exist and contain the .pkl files before building.
# In production, mount this as a volume or use a model registry pull step.
COPY models/ models/

# ---- Runtime environment ----
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# ---- Expose the API port ----
EXPOSE 8000

# ---- Healthcheck ----
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
    || exit 1

# ---- Default command: run FastAPI via uvicorn ----
CMD ["uvicorn", "api.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--log-level", "info"]
