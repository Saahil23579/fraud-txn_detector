"""
============================================================
  STEP 2 — Fraud Detection API  (FastAPI)
  
  Install dependencies:
      pip install fastapi uvicorn joblib scikit-learn numpy pandas

  Run the server:
      uvicorn api:app --reload --port 8000

  API Docs (auto-generated):
      http://127.0.0.1:8000/docs
============================================================
"""

import json
import numpy as np
import pandas as pd
import joblib

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal

# ── 1. LOAD MODEL & METADATA ─────────────────────────────
# These files were created by save_model.py
model = joblib.load("fraud_model.pkl")

with open("model_meta.json") as f:
    meta = json.load(f)

THRESHOLD = meta["threshold"]
FEATURES = meta["features"]

# ── 2. FEATURE ENCODING MAPS (same as fraud.py) ──────────
COUNTRY_MAP = {"Germany": 0, "India": 1, "Singapore": 2, "UAE": 3, "UK": 4, "USA": 5}
PM_MAP = {"Credit Card": 0, "Debit Card": 1, "NetBanking": 2, "UPI": 3, "Wallet": 4}
DEVICE_MAP = {"Laptop": 0, "Mobile": 1, "Tablet": 2}

# ── 3. FASTAPI APP ────────────────────────────────────────
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time transaction fraud scoring using a trained ML model.",
    version="1.0.0",
)


# ── 4. REQUEST & RESPONSE SCHEMAS ────────────────────────
class Transaction(BaseModel):
    """Input: one financial transaction."""
    country: Literal["Germany", "India", "Singapore", "UAE", "UK", "USA"]
    amount: float
    payment_method: Literal["Credit Card", "Debit Card", "NetBanking", "UPI", "Wallet"]
    device: Literal["Laptop", "Mobile", "Tablet"]
    hour: int



model_config = {
    "json_schema_extra": {
        "example": {
            "country": "USA",
            "amount": 150.0,
            "payment_method": "Credit Card",
            "device": "Laptop",
            "hour": 14,
        }
    }
}

class PredictionResponse(BaseModel):
    """Output: fraud score + decision."""
    fraud_probability: float
    decision: Literal["APPROVE", "STEP-UP AUTH", "BLOCK"]
    risk_tier: str
    threshold_used: float


# ── 5. FEATURE ENGINEERING (mirrors fraud.py) ────────────
def engineer_features(txn: dict) -> pd.DataFrame:
    row = txn.copy()

    # Temporal
    row["is_night"] = int(row["hour"] < 6)
    row["is_evening"] = int(18 <= row["hour"] < 24)
    row["hour_sin"] = np.sin(2 * np.pi * row["hour"] / 24)
    row["hour_cos"] = np.cos(2 * np.pi * row["hour"] / 24)

    # Amount
    row["amount_log"] = np.log1p(row["amount"])
    row["amount_bucket"] = min(int(row["amount"] // 1000), 4)

    # High-risk combos
    row["mobile_night"] = int(row["device"] == "Mobile" and row["is_night"])
    row["wallet_night"] = int(row["payment_method"] == "Wallet" and row["is_night"])
    row["high_amt_mobile"] = int(row["amount"] > 3000 and row["device"] == "Mobile")

    # Encode categoricals
    row["country_enc"] = COUNTRY_MAP.get(row["country"], 0)
    row["payment_method_enc"] = PM_MAP.get(row["payment_method"], 0)
    row["device_enc"] = DEVICE_MAP.get(row["device"], 0)

    return pd.DataFrame([row])[FEATURES]


# ── 6. ROUTES ─────────────────────────────────────────────

@app.get("/")
def root():
    """Health check."""
    return {"status": "ok", "model": meta["best_model_name"], "threshold": THRESHOLD}


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    """
    Score a single transaction and return a fraud decision.

    - **APPROVE**       → Low risk  (prob < threshold)
    - **STEP-UP AUTH**  → Medium risk (threshold ≤ prob < 0.70)
    - **BLOCK**         → High risk  (prob ≥ 0.70)
    """
    try:
        X_new = engineer_features(transaction.dict())
        prob = float(model.predict_proba(X_new)[0][1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    if prob >= 0.70:
        risk_tier = "🔴 HIGH RISK"
        decision = "BLOCK"
    elif prob >= THRESHOLD:
        risk_tier = "🟡 MEDIUM RISK"
        decision = "STEP-UP AUTH"
    else:
        risk_tier = "🟢 LOW RISK"
        decision = "APPROVE"

    return PredictionResponse(
        fraud_probability=round(prob, 4),
        decision=decision,
        risk_tier=risk_tier,
        threshold_used=round(THRESHOLD, 4),
    )


@app.post("/predict/batch", response_model=list[PredictionResponse])
def predict_batch(transactions: list[Transaction]):
    """
    Score multiple transactions in one request (max 100).
    """
    if len(transactions) > 100:
        raise HTTPException(status_code=400, detail="Batch size must be ≤ 100.")

    results = []
    for txn in transactions:
        results.append(predict(txn))  # reuse single-predict logic
    return results
