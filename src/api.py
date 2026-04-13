"""
Fraud Detection REST API — Kaggle CC Fraud Dataset Schema
==========================================================
FastAPI service exposing:
  - POST /predict      → Binary fraud decision + probability
  - POST /risk-score   → Detailed risk classification + explanation
  - GET  /health       → Service health check
  - GET  /model-info   → Model metadata and cost config

Input schema matches the Kaggle Credit Card Fraud dataset:
  V1–V28: PCA-transformed behavioral features (floats)
  Amount: Transaction amount in EUR (raw, will be log-transformed)
  Time:   Seconds since dataset start (will be cyclic-encoded)
"""

import os
import sys
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import engineer_features, FEATURE_COLS
from src.loss_functions import DEFAULT_C_FP, DEFAULT_C_FN
import pandas as pd

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
ASYM_MODEL_PATH = os.path.join(MODELS_DIR, "asymmetric_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")

# --- Global state ---
_model = None
_scaler = None
_feature_names = None


# ─── Pydantic Request/Response Models ─────────────────────────────────────────

class TransactionRequest(BaseModel):
    """
    Transaction features matching Kaggle Credit Card Fraud dataset schema.
    V1–V28 are PCA-transformed (already anonymized by the data provider).
    Amount and Time are the only raw features.
    """
    # PCA-transformed behavioral features
    V1: float;  V2: float;  V3: float;  V4: float;  V5: float
    V6: float;  V7: float;  V8: float;  V9: float;  V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float

    # Raw features (will be engineered in API)
    Amount: float = Field(..., ge=0.0, description="Transaction amount in EUR")
    Time: float = Field(..., ge=0.0,
                        description="Seconds elapsed since first transaction in dataset")

    model_config = {"json_schema_extra": {"example": {
        "V1": -1.3598, "V2": -0.0728, "V3": 2.5363, "V4": 1.3782,
        "V5": -0.3383, "V6": 0.4624, "V7": 0.2396, "V8": 0.0987,
        "V9": 0.3638, "V10": 0.0908, "V11": -0.5516, "V12": -0.6178,
        "V13": -0.9914, "V14": -0.3112, "V15": 1.4682, "V16": -0.4704,
        "V17": 0.2080, "V18": 0.0258, "V19": 0.4040, "V20": 0.2514,
        "V21": -0.0183, "V22": 0.2778, "V23": -0.1105, "V24": 0.0669,
        "V25": 0.1285, "V26": -0.1891, "V27": 0.1336, "V28": -0.0211,
        "Amount": 149.62, "Time": 406.0
    }}}


class PredictResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    decision: str
    threshold_used: float
    model_version: str


class RiskScoreResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    risk_tier: str
    risk_score: float
    decision: str
    explanation: list[str]
    top_risk_factors: list[dict]
    expected_loss_eur: float
    recommendation: str
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    scaler_loaded: bool


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Asymmetric Fraud Detection API",
    description=(
        "Production-grade fraud detection on the Kaggle Credit Card Fraud dataset. "
        "Uses asymmetric cost-sensitive XGBoost with Prospect Theory-inspired loss. "
        "FN penalty is 4× higher than FP (C_FN=$20, C_FP=$5)."
    ),
    version="1.0.0",
)


def load_artifacts():
    global _model, _scaler, _feature_names
    if not os.path.exists(ASYM_MODEL_PATH):
        raise RuntimeError(
            f"Model not found at {ASYM_MODEL_PATH}. Run: python -m src.train"
        )
    if not os.path.exists(SCALER_PATH):
        raise RuntimeError(
            f"Scaler not found at {SCALER_PATH}. Run: python -m src.train"
        )
    bundle = joblib.load(ASYM_MODEL_PATH)
    _model = bundle
    _scaler = joblib.load(SCALER_PATH)
    _feature_names = bundle.get("feature_names", FEATURE_COLS)
    print(f"[API] Asymmetric model loaded (threshold={bundle['threshold']:.4f})")


@app.on_event("startup")
async def startup_event():
    try:
        load_artifacts()
    except RuntimeError as e:
        print(f"[API WARNING] {e}")


# ─── Utilities ────────────────────────────────────────────────────────────────

def transaction_to_array(tx: TransactionRequest) -> np.ndarray:
    """
    Convert request → engineered features → scaled array.
    Applies the same transformations as training preprocessing.
    """
    if _scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")

    # Build raw row matching dataset schema
    row = {f"V{i}": getattr(tx, f"V{i}") for i in range(1, 29)}
    row["Amount"] = tx.Amount
    row["Time"] = tx.Time

    df = pd.DataFrame([row])
    df = engineer_features(df)  # Amount_log, Time_sin, Time_cos; drops Amount & Time

    X = df[FEATURE_COLS].values
    return _scaler.transform(X)


def get_fraud_proba(X_scaled: np.ndarray) -> float:
    """Run inference via XGBoost booster, return sigmoid probability."""
    import xgboost as xgb
    dmat = xgb.DMatrix(X_scaled, feature_names=_feature_names)
    raw = _model["model"].predict(dmat)
    return float(1.0 / (1.0 + np.exp(-raw[0])))


def classify_risk(proba: float) -> tuple[str, str]:
    """
    Map fraud probability to operational risk tier.

    Tiers calibrated for 0.17% base rate and C_FN/C_FP = 4:
      LOW      (< 0.05): Allow — very low risk
      MEDIUM   (0.05–0.30): Challenge with step-up auth
      HIGH     (0.30–0.70): Block and notify cardholder
      CRITICAL (> 0.70):  Hard block, alert fraud team
    """
    if proba < 0.05:
        return "LOW", "ALLOW"
    elif proba < 0.30:
        return "MEDIUM", "CHALLENGE"
    elif proba < 0.70:
        return "HIGH", "BLOCK"
    else:
        return "CRITICAL", "HARD_BLOCK"


def generate_explanation(tx: TransactionRequest, proba: float) -> list[str]:
    """Human-readable explanation based on transaction features."""
    reasons = []

    # Time-of-day from raw Time field
    hour = (tx.Time % 86400) / 3600
    if hour < 4 or hour > 22:
        reasons.append(f"🕐 Late-night transaction (hour ≈ {hour:.0f}:00)")

    if tx.Amount > 1000:
        reasons.append(f"💰 High transaction amount (€{tx.Amount:,.2f})")
    elif tx.Amount < 1:
        reasons.append(f"⚠️ Unusually small amount (€{tx.Amount:.2f}) — possible card test")

    # Key fraud-discriminative PCA features (well-established in literature)
    if tx.V14 < -5:
        reasons.append("🔍 Strong behavioral anomaly (V14 — highly fraud-discriminative)")
    if tx.V4 > 5:
        reasons.append("🔍 Unusual pattern in V4 (correlated with fraud in this dataset)")
    if tx.V12 < -5:
        reasons.append("🔍 Anomalous behavioral signal in V12")
    if tx.V10 < -5:
        reasons.append("🔍 Anomalous behavioral signal in V10")

    if not reasons:
        reasons.append("⚠️ Combined PCA feature pattern elevated risk above threshold")

    return reasons


def get_top_risk_factors(X_scaled: np.ndarray) -> list[dict]:
    """Return top 5 features by XGBoost gain importance."""
    if _model is None:
        return []
    scores = _model["model"].get_score(importance_type="gain")
    feature_names = _feature_names or FEATURE_COLS
    factors = [
        {
            "feature": f,
            "scaled_value": round(float(X_scaled[0, i]), 4),
            "importance_gain": round(scores.get(f, 0.0), 4),
        }
        for i, f in enumerate(feature_names)
    ]
    factors.sort(key=lambda x: x["importance_gain"], reverse=True)
    return factors[:5]


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    return {
        "status": "healthy" if _model else "degraded",
        "model_loaded": _model is not None,
        "scaler_loaded": _scaler is not None,
    }


@app.get("/model-info", tags=["System"])
def model_info():
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_type": "Asymmetric XGBoost",
        "dataset": "Kaggle Credit Card Fraud (MLG-ULB) — 284,807 transactions",
        "version": "1.0.0",
        "decision_threshold": _model["threshold"],
        "cost_config": {"C_FP_USD": DEFAULT_C_FP, "C_FN_USD": DEFAULT_C_FN},
        "asymmetry_ratio": f"{DEFAULT_C_FN / DEFAULT_C_FP:.1f}x",
        "feature_count": len(_feature_names or FEATURE_COLS),
        "features": "V1–V28 (PCA) + Amount_log + Time_sin + Time_cos",
        "theoretical_optimal_threshold": round(
            DEFAULT_C_FP / (DEFAULT_C_FP + DEFAULT_C_FN), 4
        ),
    }


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(tx: TransactionRequest):
    """Binary fraud prediction with probability."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")
    X = transaction_to_array(tx)
    proba = get_fraud_proba(X)
    threshold = _model["threshold"]
    return {
        "fraud_probability": round(proba, 6),
        "is_fraud": proba >= threshold,
        "decision": "BLOCK" if proba >= threshold else "ALLOW",
        "threshold_used": round(threshold, 4),
        "model_version": "asymmetric-kaggle-v1.0",
    }


@app.post("/risk-score", response_model=RiskScoreResponse, tags=["Prediction"])
def risk_score(tx: TransactionRequest):
    """Detailed risk assessment with tier, explanation, and top features."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")

    X = transaction_to_array(tx)
    proba = get_fraud_proba(X)
    threshold = _model["threshold"]
    risk_tier, decision = classify_risk(proba)
    explanation = generate_explanation(tx, proba)
    top_factors = get_top_risk_factors(X)
    expected_loss_eur = round(proba * tx.Amount, 2)

    recommendation_map = {
        "ALLOW":      "Process transaction normally.",
        "CHALLENGE":  "Proceed with step-up authentication (OTP / biometric).",
        "BLOCK":      "Block transaction. Notify cardholder for verification.",
        "HARD_BLOCK": "Block immediately. Alert fraud team. Consider card suspension.",
    }

    return {
        "fraud_probability": round(proba, 6),
        "is_fraud": proba >= threshold,
        "risk_tier": risk_tier,
        "risk_score": round(proba * 100, 1),
        "decision": decision,
        "explanation": explanation,
        "top_risk_factors": top_factors,
        "expected_loss_eur": expected_loss_eur,
        "recommendation": recommendation_map[decision],
        "model_version": "asymmetric-kaggle-v1.0",
    }


if __name__ == "__main__":
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)