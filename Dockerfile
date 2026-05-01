# ─────────────────────────────────────────────
# Loan Default Prediction — Dockerfile
# ─────────────────────────────────────────────
FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /app

# Install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY app.py model.py drift.py explain.py ./
COPY data/ ./data/

# Copy trained model if it already exists (optional – can be mounted at runtime)
COPY model.pkl ./model.pkl 2>/dev/null || true

# MLflow artifact directory
RUN mkdir -p mlruns

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=model.pkl
ENV MLFLOW_TRACKING_URI=./mlruns

# Expose FastAPI port and MLflow UI port
EXPOSE 8000
EXPOSE 5000

# Default: start the API server
# To run MLflow UI instead: docker run ... mlflow ui --host 0.0.0.0 --port 5000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
