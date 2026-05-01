"""
app.py - FastAPI REST API for Loan Default Prediction
Endpoints:
  POST /predict        → single prediction
  POST /predict_batch  → batch predictions
  GET  /health         → health check
  GET  /feature_importance → top SHAP features
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import numpy as np
import pandas as pd
import joblib
import uvicorn
import os
import json

app = FastAPI(
    title="Loan Default Prediction API",
    description="Predicts the likelihood of a borrower defaulting on a loan.",
    version="1.0.0",
)

# ─────────────────────────────────────────────
# Load model at startup
# ─────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")
pipeline = None


@app.on_event("startup")
def load_model():
    global pipeline
    if os.path.exists(MODEL_PATH):
        pipeline = joblib.load(MODEL_PATH)
        print(f"[app] Model loaded from {MODEL_PATH}")
    else:
        print(f"[app] WARNING: model not found at {MODEL_PATH}. Run model.py first.")


# ─────────────────────────────────────────────
# Request / Response schemas
# ─────────────────────────────────────────────
class LoanApplication(BaseModel):
    no_of_dependents: Optional[float] = Field(0, description="Number of dependents")
    education: Optional[str] = Field("Graduate", description="Graduate / Not Graduate")
    self_employed: Optional[str] = Field("No", description="Yes / No")
    income_annum: Optional[float] = Field(500000, description="Annual income (INR)")
    loan_amount: Optional[float] = Field(1000000, description="Loan amount (INR)")
    loan_term: Optional[float] = Field(12, description="Loan term in months")
    cibil_score: Optional[float] = Field(650, description="CIBIL credit score (300–900)")
    residential_assets_value: Optional[float] = Field(0, description="Residential assets")
    commercial_assets_value: Optional[float] = Field(0, description="Commercial assets")
    luxury_assets_value: Optional[float] = Field(0, description="Luxury assets")
    bank_asset_value: Optional[float] = Field(0, description="Bank assets")

    class Config:
        schema_extra = {
            "example": {
                "no_of_dependents": 2,
                "education": "Graduate",
                "self_employed": "No",
                "income_annum": 900000,
                "loan_amount": 2000000,
                "loan_term": 18,
                "cibil_score": 720,
                "residential_assets_value": 3000000,
                "commercial_assets_value": 500000,
                "luxury_assets_value": 200000,
                "bank_asset_value": 150000,
            }
        }


class PredictionResponse(BaseModel):
    default_probability: float
    prediction: str
    risk_level: str
    confidence: float


class BatchRequest(BaseModel):
    applications: List[LoanApplication]


class BatchResponse(BaseModel):
    results: List[PredictionResponse]
    total: int


# ─────────────────────────────────────────────
# Helper: preprocess a single dict to DataFrame
# ─────────────────────────────────────────────
def _prepare_input(app_data: dict) -> pd.DataFrame:
    df = pd.DataFrame([app_data])

    # Encode categoricals same way as model.py
    cat_map = {
        "education": {"Graduate": 1, "Not Graduate": 0},
        "self_employed": {"Yes": 1, "No": 0},
    }
    for col, mapping in cat_map.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0)

    # Feature engineering (mirrors model.py)
    if "income_annum" in df.columns and "loan_amount" in df.columns:
        df["loan_to_income"] = df["loan_amount"] / (df["income_annum"] + 1)
    if "loan_amount" in df.columns and "loan_term" in df.columns:
        df["monthly_payment_est"] = df["loan_amount"] / (df["loan_term"] + 1)
    if "cibil_score" in df.columns:
        df["good_credit"] = (df["cibil_score"] >= 700).astype(int)
    if "no_of_dependents" in df.columns and "income_annum" in df.columns:
        df["income_per_dependent"] = df["income_annum"] / (df["no_of_dependents"] + 1)

    return df


def _risk_level(prob: float) -> str:
    if prob < 0.3:
        return "LOW"
    elif prob < 0.6:
        return "MEDIUM"
    return "HIGH"


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": pipeline is not None,
        "model_path": MODEL_PATH,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(application: LoanApplication):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run model.py first.")

    try:
        X = _prepare_input(application.dict())
        prob = float(pipeline.predict_proba(X)[0][1])
        prediction = "Default" if prob >= 0.5 else "No Default"
        confidence = max(prob, 1 - prob)

        return PredictionResponse(
            default_probability=round(prob, 4),
            prediction=prediction,
            risk_level=_risk_level(prob),
            confidence=round(confidence, 4),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_batch", response_model=BatchResponse)
def predict_batch(batch: BatchRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    results = []
    for application in batch.applications:
        try:
            X = _prepare_input(application.dict())
            prob = float(pipeline.predict_proba(X)[0][1])
            prediction = "Default" if prob >= 0.5 else "No Default"
            confidence = max(prob, 1 - prob)
            results.append(PredictionResponse(
                default_probability=round(prob, 4),
                prediction=prediction,
                risk_level=_risk_level(prob),
                confidence=round(confidence, 4),
            ))
        except Exception as e:
            results.append(PredictionResponse(
                default_probability=-1,
                prediction="ERROR",
                risk_level="UNKNOWN",
                confidence=0.0,
            ))

    return BatchResponse(results=results, total=len(results))


@app.get("/feature_importance")
def feature_importance():
    """Return SHAP-based feature importances if available."""
    import os
    if os.path.exists("shap_importance.csv"):
        df = pd.read_csv("shap_importance.csv")
        return df.head(10).to_dict(orient="records")
    return {"message": "Run explain.py first to generate SHAP importances."}


# ─────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
