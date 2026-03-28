from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import mlflow
import mlflow.sklearn
import os

app = FastAPI(title="ML Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None

@app.on_event("startup")
async def load_model():
    global model
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        print("Warning: MLFLOW_TRACKING_URI not set — running without model")
        return

    mlflow.set_tracking_uri(tracking_uri)

    try:
        model = mlflow.sklearn.load_model(
            "models:/energy-demand-champion/Production"
        )
        print("Champion model loaded from MLflow.")
    except Exception as e:
        print(f"Warning: Could not load model: {e}")
        print("Train a model first with: modal run backend/ml/training/train.py")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
    }

@app.post("/api/forecast")
async def forecast(body: dict):
    if model is None:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=503,
            detail="Model not loaded yet — train one first"
        )

    import pandas as pd
    import numpy as np

    # Build feature vector from request
    features = {
        "load_lag_1h":          body.get("load_lag_1h", 0),
        "load_lag_24h":         body.get("load_lag_24h", 0),
        "load_lag_168h":        body.get("load_lag_168h", 0),
        "hour":                 body.get("hour", 0),
        "weekday":              body.get("weekday", 0),
        "is_holiday":           int(body.get("is_holiday", False)),
        "month":                body.get("month", 1),
        "is_weekend":           int(body.get("weekday", 0) >= 5),
        "temperature_c":        body.get("temperature_c", 12),
        "wind_speed_ms":        body.get("wind_speed_ms", 5),
        "oil_price_brent":      body.get("oil_price_brent", 100),
        "oil_delta_pct_30m":    body.get("oil_delta_pct_30m", 0),
        "grid_imbalance_price": body.get("grid_imbalance_price", 100),
        "hour_sin":             np.sin(2 * np.pi * body.get("hour", 0) / 24),
        "hour_cos":             np.cos(2 * np.pi * body.get("hour", 0) / 24),
        "weekday_sin":          np.sin(2 * np.pi * body.get("weekday", 0) / 7),
        "weekday_cos":          np.cos(2 * np.pi * body.get("weekday", 0) / 7),
        "load_rolling_mean_7d": body.get("load_lag_168h", 0),
        "load_rolling_std_7d":  0.0,
    }

    X = pd.DataFrame([features])
    pred_mw = float(model.predict(X)[0])

    # Compute shock-aware confidence intervals
    shock_score = body.get("shock_score", 0)
    base_margin = pred_mw * 0.03
    shock_margin = shock_score * pred_mw * 0.12
    total_margin = base_margin + shock_margin

    return {
        "forecast_mw":    round(pred_mw, 1),
        "lower_bound_mw": round(pred_mw - total_margin, 1),
        "upper_bound_mw": round(pred_mw + total_margin, 1),
        "shock_score":    shock_score,
        "shock_active":   shock_score >= 0.6,
        "confidence_pct": 90,
    }