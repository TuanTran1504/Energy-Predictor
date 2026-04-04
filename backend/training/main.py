"""
FastAPI ML inference service for BTC/ETH 24h direction prediction.

Endpoints:
  GET  /health              — liveness + which models are loaded
  GET  /models              — model metadata (n_estimators, n_features)
  POST /predict             — single inference from recent candles
  GET  /backtest            — historical predicted vs actual (for chart overlay)
  POST /reload              — hot-reload model from DagsHub without restart
  POST /cache/bust          — expire Redis cache entry

Start:
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Dependencies (pip install):
  fastapi uvicorn[standard] python-dotenv pandas xgboost joblib mlflow redis
"""

import sys
import os
import json
import asyncio
import logging
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

# ── BLOCK 2: fix path + load .env — before ANY other import ──────────────────
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

def _find_env_files() -> list[Path]:
    """Return [root/.env, backend/.env] — both loaded so all vars are available."""
    found = []
    p = Path(__file__).resolve().parent
    for _ in range(6):
        candidate = p / ".env"
        if candidate.exists():
            found.append(candidate)
        p = p.parent
    return list(reversed(found))   # root first, backend overrides

for _env_file in _find_env_files():
    load_dotenv(_env_file)

# ── BLOCK 3: third-party imports — after env is loaded ────────────────────────
import pandas as pd
import redis as redis_lib
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model_store import bust_cache, load_all_models, HORIZONS, SYMBOLS
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
log = logging.getLogger(__name__)

# ── Redis helper ───────────────────────────────────────────────────────────────
PREDICTION_TTL = 90_000   # 25 hours — ensures cache survives even if daily job is late
_redis: Optional[redis_lib.Redis] = None

def _get_redis() -> Optional[redis_lib.Redis]:
    global _redis
    if _redis is not None:
        return _redis
    url = os.getenv("REDIS_URL", "")
    if not url:
        return None
    if url.startswith("redis://"):
        url = "rediss://" + url[len("redis://"):]
    try:
        r = redis_lib.from_url(url, decode_responses=True, socket_timeout=3)
        r.ping()
        _redis = r
        return r
    except Exception as e:
        log.warning(f"Redis unavailable: {e}")
        return None

def _pred_cache_key(symbol: str, horizon: int) -> str:
    return f"prediction:{symbol.upper()}:{horizon}d"

# ── Feature column list ────────────────────────────────────────────────────────
# Defined here — NOT imported from feature_engineering — so this file has zero
# dependency on feature_engineering at prediction time. The list must exactly
# match get_feature_columns() in feature_engineering.py.
FEATURE_COLS: list[str] = [
    # Price momentum
    "ret_3d", "ret_7d", "ret_14d",
    # Volatility regime
    "volatility_7d",
    # Price position / range
    "price_position", "hl_range",
    # Volume
    "vol_trend",
    # Momentum indicators
    "rsi_14", "macd_hist",
    # Regime
    "bull_regime",
    # Market structure
    "btc_eth_ratio_7d_change",
    # Sentiment
    "fear_greed",
    # Derivatives positioning
    "funding_rate_avg", "funding_extreme_long",
    # Macro (as-of joined in training; neutral defaults at inference)
    "macro_fed_rate", "macro_cpi_surprise", "macro_nfp_surprise",
    # Calendar
    "day_of_week",
]

# ── Per-symbol UP thresholds ───────────────────────────────────────────────────
# BTC shows a persistent UP bias in validation (recall UP ~0.76, DOWN ~0.33)
# caused by the training period being predominantly bullish (2021-2025).
# Requiring higher confidence before calling UP reduces false UP signals.
# Configurable via env vars so you can tune without redeploying.
UP_THRESHOLDS: dict[str, float] = {
    "BTC": float(os.getenv("BTC_UP_THRESHOLD", "0.55")),
    "ETH": float(os.getenv("ETH_UP_THRESHOLD", "0.50")),
}

SYMBOLS: list[str] = ["BTC", "ETH"]

# ── Global model registry — populated at startup ───────────────────────────────
models: dict[str, object] = {}
_executor = ThreadPoolExecutor(max_workers=2)


def precompute_predictions() -> dict:
    """
    Builds features once per symbol, runs all horizon models, stores results
    in Redis with a 25h TTL. Called at startup and every 24h by the scheduler.
    Returns the computed predictions dict for convenience.
    """
    from feature_engineering import build_features

    results = {}
    for symbol in SYMBOLS:
        symbol_models = {k: m for k, m in models.items() if k.startswith(symbol + "_")}
        if not symbol_models:
            log.warning(f"[precompute] No models loaded for {symbol} — skipping")
            continue

        try:
            df = build_features(symbol, lookahead=1)   # features are horizon-independent
        except Exception as e:
            log.error(f"[precompute] build_features failed for {symbol}: {e}")
            continue

        if df.empty:
            log.warning(f"[precompute] Empty feature df for {symbol}")
            continue

        X = df[FEATURE_COLS].fillna(0).tail(1)
        r = _get_redis()

        for key, model in symbol_models.items():
            # key format: "BTC_1d" → horizon = 1
            horizon = int(key.split("_")[1].replace("d", ""))

            if X.shape[1] != model.n_features_in_:
                log.warning(f"[precompute] Feature mismatch for {key} — skipping")
                continue

            proba     = model.predict_proba(X)[0]
            up_prob   = float(proba[1])
            down_prob = float(proba[0])
            threshold = UP_THRESHOLDS.get(symbol, 0.5)
            direction = "UP" if up_prob >= threshold else "DOWN"
            confidence = up_prob if direction == "UP" else down_prob

            payload = {
                "symbol":       symbol,
                "direction":    direction,
                "confidence":   round(confidence, 4),
                "up_prob":      round(up_prob, 4),
                "down_prob":    round(down_prob, 4),
                "threshold":    threshold,
                "predicted_at": datetime.utcnow().isoformat(),
            }
            results[key] = payload

            if r:
                cache_key = _pred_cache_key(symbol, horizon)
                r.setex(cache_key, PREDICTION_TTL, json.dumps(payload))
                log.info(f"[precompute] Cached {cache_key} → {direction} ({confidence:.1%})")

    log.info(f"[precompute] Done — {len(results)} predictions cached")
    return results


async def _precompute_scheduler():
    """Runs precompute at startup (after a short delay) then every 24h.
    Uses a Redis lock so only one worker runs precompute when multiple workers are active."""
    await asyncio.sleep(5)   # let startup finish first
    loop = asyncio.get_event_loop()
    while True:
        r = _get_redis()
        acquired = False
        if r:
            # nx=True → only sets if key doesn't exist (atomic lock)
            # ex=3600 → lock expires after 1h so a crashed worker can't block forever
            acquired = r.set("precompute:lock", "1", nx=True, ex=3600)
        else:
            acquired = True  # no Redis — single worker, always run

        if acquired:
            log.info("[scheduler] Running daily prediction precompute...")
            await loop.run_in_executor(_executor, precompute_predictions)
            if r:
                r.delete("precompute:lock")  # release lock immediately after done
        else:
            log.info("[scheduler] Skipping precompute — another worker is running it")

        await asyncio.sleep(24 * 3600)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, start daily precompute scheduler, clear on shutdown."""
    global models
    log.info("Loading models via model_store...")
    models = load_all_models()
    if models:
        log.info(f"Ready — loaded: {list(models.keys())}")
    else:
        log.warning(
            "No models loaded. Run train.py first, then restart the service "
            "or call POST /reload."
        )

    # Start background scheduler (non-blocking)
    task = asyncio.create_task(_precompute_scheduler())

    yield

    task.cancel()
    models.clear()


app = FastAPI(title="ML Inference Service — BTC/ETH direction", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite dev server
        "http://localhost:8080",   # Go backend
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
# Schemas
# ══════════════════════════════════════════════════════════════════════════════

class CandleIn(BaseModel):
    """One daily OHLCV candle. Send the last 20 for a valid prediction."""
    date:       str    # ISO date string, e.g. "2025-03-29"
    open_usd:   float
    high_usd:   float
    low_usd:    float
    close_usd:  float
    volume_usd: float
    change_pct: float  # percent change from previous day close, e.g. 1.23 = +1.23%


class PredictRequest(BaseModel):
    symbol:  Literal["BTC", "ETH"]
    candles: list[CandleIn]   # minimum 15, recommended 20


class PredictionOut(BaseModel):
    symbol:       str
    direction:    Literal["UP", "DOWN"]
    confidence:   float   # probability of the predicted class, 0-1
    up_prob:      float   # raw P(UP)
    down_prob:    float   # raw P(DOWN)
    threshold:    float   # UP threshold applied (symbol-specific, e.g. 0.55 for BTC)
    predicted_at: str     # UTC ISO timestamp


class BacktestPoint(BaseModel):
    date:          str
    actual_price:  float
    predicted_dir: Literal["UP", "DOWN"]
    up_prob:       float
    correct:       bool


class BacktestOut(BaseModel):
    symbol:   str
    days:     int
    accuracy: float
    points:   list[BacktestPoint]


# ══════════════════════════════════════════════════════════════════════════════
# Feature engineering (inference-only, no DB needed)
# ══════════════════════════════════════════════════════════════════════════════

def _vol_trend(df: pd.DataFrame, i: int) -> float:
    """Volume today vs 7-day average."""
    vol_today = float(df.iloc[i]["volume_usd"]) or 0.0
    vol_avg   = df.iloc[i - 7:i]["volume_usd"].astype(float).mean()
    return (vol_today / vol_avg - 1.0) if vol_avg > 0 else 0.0


def _price_position(df: pd.DataFrame, i: int, close: float) -> float:
    """Where is today's close within the 14-day high-low range?"""
    hi = float(df.iloc[i - 14:i + 1]["high_usd"].max())
    lo = float(df.iloc[i - 14:i + 1]["low_usd"].min())
    return (close - lo) / (hi - lo) if hi > lo else 0.5


def _compute_rsi_series(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    return (100 - (100 / (1 + rs))).fillna(50.0)


def candles_to_features(candles: list[CandleIn]) -> pd.DataFrame:
    """
    Convert a list of daily candles into a single-row feature DataFrame
    ready for model.predict_proba().

    Requires at least 15 candles (14 for lookback + 1 current).
    Sending 20 is recommended so all windows are fully populated.

    Macro/shock features are set to neutral defaults because they are not
    available in real-time from Binance candles alone. This is intentional —
    the model degrades gracefully on those features rather than breaking.
    """
    if len(candles) < 15:
        raise ValueError(f"Need at least 15 candles, got {len(candles)}")

    df = pd.DataFrame([c.model_dump() for c in candles])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    close_series = df["close_usd"].astype(float)
    ret_series   = close_series.pct_change().fillna(0.0)

    vol_7d         = ret_series.rolling(7, min_periods=1).std().fillna(0.0)
    rsi_14         = _compute_rsi_series(close_series, 14)
    ema_12_series  = close_series.ewm(span=12, adjust=False, min_periods=1).mean()
    ema_26_series  = close_series.ewm(span=26, adjust=False, min_periods=1).mean()
    macd_line      = ema_12_series - ema_26_series
    macd_signal    = macd_line.ewm(span=9, adjust=False, min_periods=1).mean()
    macd_hist_series = macd_line - macd_signal
    ma_200_series  = close_series.rolling(200, min_periods=1).mean()

    i      = len(df) - 1    # index of the most recent candle
    close  = float(df.iloc[i]["close_usd"])
    date   = df.iloc[i]["date"]
    ma_200 = float(ma_200_series.iloc[i])

    row: dict = {
        # ── Price momentum ──────────────────────────────────────────────────
        "ret_3d":  close / float(df.iloc[i - 3]["close_usd"]) - 1.0,
        "ret_7d":  close / float(df.iloc[i - 7]["close_usd"]) - 1.0,
        "ret_14d": close / float(df.iloc[i - 14]["close_usd"]) - 1.0,

        # ── Volatility / range ──────────────────────────────────────────────
        "volatility_7d": float(vol_7d.iloc[i]),
        "hl_range": (
            float(df.iloc[i]["high_usd"]) - float(df.iloc[i]["low_usd"])
        ) / close,

        # ── Price position / volume ─────────────────────────────────────────
        "price_position": _price_position(df, i, close),
        "vol_trend":      _vol_trend(df, i),

        # ── Momentum indicators ─────────────────────────────────────────────
        "rsi_14":    float(rsi_14.iloc[i]),
        "macd_hist": float(macd_hist_series.iloc[i]),

        # ── Regime ──────────────────────────────────────────────────────────
        # bull_regime: 1 when price is above the 200-day MA.
        # With only 20 candles provided, ma_200 is just the mean of those
        # candles (not a real 200MA), so we default to 0 (unknown regime).
        # For accurate bull_regime at inference, send 200+ candles.
        "bull_regime": int(close > ma_200) if len(df) >= 200 else 0,

        # ── Market structure ────────────────────────────────────────────────
        # Ratio change not computable without the other symbol's candles.
        "btc_eth_ratio_7d_change": 0.0,

        # ── Sentiment — neutral default ─────────────────────────────────────
        # Fear & Greed not available from raw candles.
        "fear_greed": 0.5,

        # ── Funding rate — neutral defaults ─────────────────────────────────
        # Not available from spot candles.
        "funding_rate_avg":     0.0,
        "funding_extreme_long": 0,

        # ── Macro — neutral defaults ─────────────────────────────────────────
        # As-of macro values are not available at candle-only inference time.
        # macro_fed_rate defaults to 5.25 (approximate current US rate level).
        # Override via FED_RATE_DEFAULT env var if the rate changes significantly.
        "macro_fed_rate":     float(os.getenv("FED_RATE_DEFAULT", "5.25")),
        "macro_cpi_surprise": 0.0,
        "macro_nfp_surprise": 0.0,

        # ── Calendar ────────────────────────────────────────────────────────
        "day_of_week": int(date.dayofweek),
    }

    # Select columns in exact training order
    return pd.DataFrame([row])[FEATURE_COLS]


# ══════════════════════════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    """
    Liveness check. models_loaded tells you which symbols are ready.
    If the list is empty, training hasn't run yet or model_store failed.
    """
    return {
        "status":        "ok",
        "models_loaded": list(models.keys()),
        "feature_count": len(FEATURE_COLS),
        "timestamp":     datetime.utcnow().isoformat(),
    }


@app.get("/models")
def list_models():
    """Returns XGBoost metadata for each loaded model."""
    info: dict = {}
    for symbol, model in models.items():
        try:
            info[symbol] = {
                "n_estimators":   int(model.n_estimators),
                "n_features":     int(model.n_features_in_),
                "best_iteration": getattr(model, "best_iteration", None),
            }
        except Exception:
            info[symbol] = {"status": "loaded"}
    return {"models": info}


@app.post("/predict", response_model=PredictionOut)
def predict(req: PredictRequest):
    """
    Predict BTC or ETH direction for the next 24h.

    Send the last 20 daily candles (minimum 15). Each candle needs:
      date, open_usd, high_usd, low_usd, close_usd, volume_usd, change_pct

    Returns direction (UP/DOWN), confidence (0-1), and raw probabilities.
    """
    model = models.get(req.symbol)
    if model is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Model for {req.symbol} is not loaded. "
                f"Run train.py first, then POST /reload or restart the service. "
                f"Currently loaded: {list(models.keys()) or 'none'}"
            ),
        )

    try:
        X = candles_to_features(req.candles)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except KeyError as e:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Feature column mismatch: {e}. "
                f"Ensure FEATURE_COLS in main.py matches the training feature set."
            ),
        )

    # Validate feature count matches what the model expects
    if X.shape[1] != model.n_features_in_:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Model expects {model.n_features_in_} features, "
                f"but inference built {X.shape[1]}. "
                f"FEATURE_COLS in main.py must match the training feature set."
            ),
        )

    proba     = model.predict_proba(X)[0]   # shape: (2,) → [P(DOWN), P(UP)]
    up_prob   = float(proba[1])
    down_prob = float(proba[0])
    threshold = UP_THRESHOLDS.get(req.symbol, 0.5)
    direction = "UP" if up_prob >= threshold else "DOWN"
    confidence = up_prob if direction == "UP" else down_prob

    return PredictionOut(
        symbol       = req.symbol,
        direction    = direction,
        confidence   = round(confidence, 4),
        up_prob      = round(up_prob, 4),
        down_prob    = round(down_prob, 4),
        threshold    = threshold,
        predicted_at = datetime.utcnow().isoformat(),
    )


@app.get("/backtest", response_model=BacktestOut)
def backtest(
    symbol:   Literal["BTC", "ETH"] = Query(..., description="BTC or ETH"),
    days:     int                    = Query(default=30, ge=7, le=180),
    lookahead: int                   = Query(default=1, ge=1, le=30, description="Prediction horizon in days"),
):
    """
    Runs the model over the last N days of DB data and returns each day's
    actual price + predicted direction + whether the prediction was correct.

    Used by the React ML tab to draw the predicted-vs-actual chart overlay.
    Requires feature_engineering.build_features() which reads from Supabase.
    This call is slow (~2-5s) due to the DB read — it is only triggered when
    the ML tab mounts, not on every prediction.
    """
    model_key = f"{symbol}_{lookahead}d"
    model = models.get(model_key)
    if model is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_key}' not loaded. Run train.py with LOOKAHEAD_DAYS={lookahead} then POST /reload.",
        )

    try:
        from feature_engineering import build_features
        df = build_features(symbol, lookahead=lookahead)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Feature build failed (DB read error): {e}",
        )

    if df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No historical data found for {symbol} in Supabase.",
        )

    df = df.tail(days).reset_index(drop=True)
    X  = df[FEATURE_COLS].fillna(0)

    if X.shape[1] != model.n_features_in_:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Feature mismatch: model expects {model.n_features_in_} features, "
                f"got {X.shape[1]}. Use the fixed feature_engineering.py."
            ),
        )

    proba    = model.predict_proba(X)   # shape: (N, 2)
    up_probs = proba[:, 1]

    points: list[BacktestPoint] = []
    correct_count = 0

    for idx, row in df.iterrows():
        up_prob    = float(up_probs[idx])
        pred_dir   = "UP" if up_prob >= 0.5 else "DOWN"
        actual_dir = "UP" if int(row["target"]) == 1 else "DOWN"
        is_correct = pred_dir == actual_dir
        if is_correct:
            correct_count += 1

        points.append(BacktestPoint(
            date          = row["date"].strftime("%b %d"),
            actual_price  = round(float(row["close_usd"]), 2),
            predicted_dir = pred_dir,
            up_prob       = round(up_prob, 4),
            correct       = is_correct,
        ))

    accuracy = correct_count / len(points) if points else 0.0

    return BacktestOut(
        symbol   = symbol,
        days     = days,
        accuracy = round(accuracy, 4),
        points   = points,
    )


@app.get("/predict/live/all")
def predict_live_all(
    symbol: Literal["BTC", "ETH"] = Query(..., description="BTC or ETH"),
):
    """
    Returns predictions for all horizons for one symbol.
    Serves from Redis cache (precomputed daily) — falls back to live computation on cache miss.
    """
    r = _get_redis()
    log.info(f"[predict/live/all] redis={r is not None}, HORIZONS={HORIZONS}")
    results = {}
    missing_horizons = []

    # Try Redis cache first
    for horizon in HORIZONS:
        key = f"{symbol}_{horizon}d"
        cache_key = _pred_cache_key(symbol, horizon)
        cached = r.get(cache_key) if r else None
        log.info(f"[predict/live/all] {cache_key} → {'HIT' if cached else 'MISS'}")
        if cached:
            results[key] = json.loads(cached)
        else:
            missing_horizons.append(horizon)

    # Cache miss — compute live for missing horizons
    if missing_horizons:
        log.info(f"[predict/live/all] Cache miss for {symbol} horizons={missing_horizons} — computing live")
        from feature_engineering import build_features
        try:
            df = build_features(symbol, lookahead=1)
        except Exception as e:
            if results:
                return results   # return whatever we got from cache
            raise HTTPException(status_code=500, detail=f"Feature build failed: {e}")

        if not df.empty:
            X = df[FEATURE_COLS].fillna(0).tail(1)
            for horizon in missing_horizons:
                key = f"{symbol}_{horizon}d"
                model = models.get(key)
                if model is None or X.shape[1] != model.n_features_in_:
                    continue
                proba     = model.predict_proba(X)[0]
                up_prob   = float(proba[1])
                down_prob = float(proba[0])
                direction = "UP" if up_prob >= 0.5 else "DOWN"
                confidence = up_prob if direction == "UP" else down_prob
                payload = {
                    "symbol":       symbol,
                    "direction":    direction,
                    "confidence":   round(confidence, 4),
                    "up_prob":      round(up_prob, 4),
                    "down_prob":    round(down_prob, 4),
                    "predicted_at": datetime.utcnow().isoformat(),
                }
                results[key] = payload
                if r:
                    r.setex(_pred_cache_key(symbol, horizon), PREDICTION_TTL, json.dumps(payload))

    return results


@app.get("/predict/live", response_model=PredictionOut)
def predict_live(
    symbol:    Literal["BTC", "ETH"] = Query(..., description="BTC or ETH"),
    lookahead: int                    = Query(default=1, ge=1, le=30, description="Prediction horizon in days"),
):
    """
    Predict direction using the most recent row from the database.
    No candles need to be sent — uses the same feature pipeline as training.
    Called by the Go service to populate live prediction cards in the dashboard.
    """
    model_key = f"{symbol}_{lookahead}d"
    model = models.get(model_key)
    if model is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_key}' not loaded. Run train.py with LOOKAHEAD_DAYS={lookahead} then POST /reload.",
        )

    try:
        from feature_engineering import build_features
        df = build_features(symbol, lookahead=lookahead)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Feature build failed (DB read error): {e}",
        )

    if df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No historical data found for {symbol}.",
        )

    X = df[FEATURE_COLS].fillna(0).tail(1)

    if X.shape[1] != model.n_features_in_:
        raise HTTPException(
            status_code=500,
            detail=f"Feature mismatch: model expects {model.n_features_in_}, got {X.shape[1]}.",
        )

    proba     = model.predict_proba(X)[0]
    up_prob   = float(proba[1])
    down_prob = float(proba[0])
    threshold = UP_THRESHOLDS.get(symbol, 0.5)
    direction = "UP" if up_prob >= threshold else "DOWN"
    confidence = up_prob if direction == "UP" else down_prob

    return PredictionOut(
        symbol       = symbol,
        direction    = direction,
        confidence   = round(confidence, 4),
        up_prob      = round(up_prob, 4),
        down_prob    = round(down_prob, 4),
        threshold    = threshold,
        predicted_at = datetime.utcnow().isoformat(),
    )


@app.get("/debug/cache")
def debug_cache():
    """Shows Redis connection status and which prediction keys exist."""
    redis_url_set = bool(os.getenv("REDIS_URL"))
    r = _get_redis()
    if r is None:
        return {"redis_connected": False, "redis_url_set": redis_url_set, "keys": []}
    try:
        keys = r.keys("prediction:*")
        values = {k: r.get(k) for k in keys}
        return {"redis_connected": True, "redis_url_set": redis_url_set, "keys": values}
    except Exception as e:
        return {"redis_connected": False, "error": str(e)}


@app.get("/market-signals")
def market_signals():
    """
    Returns the latest Fear & Greed value, BTC/ETH funding rates, and
    BTC/ETH price ratio change — displayed as market context on the dashboard.
    """
    import os
    import psycopg2

    DATABASE_URL = os.environ.get("DATABASE_URL")
    if not DATABASE_URL:
        raise HTTPException(status_code=503, detail="DATABASE_URL not configured")

    try:
        conn = psycopg2.connect(DATABASE_URL, sslmode="require")
        cur  = conn.cursor()

        # Fear & Greed — latest value
        cur.execute("""
            SELECT date, value FROM fear_greed_index
            ORDER BY date DESC LIMIT 1
        """)
        fg_row = cur.fetchone()
        fear_greed = {"date": str(fg_row[0]), "value": fg_row[1]} if fg_row else None

        # Funding rates — latest for BTC and ETH
        funding = {}
        for sym in ("BTC", "ETH"):
            cur.execute("""
                SELECT date, rate_avg FROM funding_rates
                WHERE symbol = %s ORDER BY date DESC LIMIT 1
            """, (sym,))
            row = cur.fetchone()
            if row:
                funding[sym] = {"date": str(row[0]), "rate_avg": row[1]}

        # BTC/ETH ratio — last 8 days to compute 7d change
        cur.execute("""
            SELECT symbol, fetched_at::date AS d, AVG(close_usd) AS price
            FROM crypto_prices
            WHERE symbol IN ('BTC', 'ETH')
              AND interval_minutes = 30
              AND fetched_at >= NOW() - INTERVAL '10 days'
            GROUP BY symbol, d
            ORDER BY symbol, d
        """)
        rows = cur.fetchall()
        cur.close()
        conn.close()

        # Build per-symbol price series
        from collections import defaultdict
        prices_by_sym: dict = defaultdict(dict)
        for sym, d, p in rows:
            prices_by_sym[sym][d] = p

        ratio_7d_change = None
        btc_prices = sorted(prices_by_sym.get("BTC", {}).items())
        eth_prices = sorted(prices_by_sym.get("ETH", {}).items())
        if len(btc_prices) >= 2 and len(eth_prices) >= 2:
            # Align on common dates
            btc_dict = dict(btc_prices)
            eth_dict = dict(eth_prices)
            common   = sorted(set(btc_dict) & set(eth_dict))
            if len(common) >= 2:
                ratio_now  = btc_dict[common[-1]] / eth_dict[common[-1]]
                ratio_prev = btc_dict[common[0]]  / eth_dict[common[0]]
                ratio_7d_change = round((ratio_now - ratio_prev) / ratio_prev * 100, 2)

        return {
            "fear_greed":      fear_greed,
            "funding_rates":   funding,
            "ratio_7d_change": ratio_7d_change,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/precompute")
def trigger_precompute():
    """
    Manually triggers prediction precomputation and caches results in Redis.
    Call this after retraining a model (POST /reload → POST /predict/precompute).
    """
    if not models:
        raise HTTPException(status_code=503, detail="No models loaded. POST /reload first.")
    results = precompute_predictions()
    return {
        "computed": list(results.keys()),
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/analyze")
async def analyze(request: Request):
    """
    Accepts a JSON body: {"question": "...", "history": [...]}
    Gathers all live market context, then streams an OpenAI response.
    history: list of {role: "user"|"assistant", content: "..."}
    """
    import openai
    from fastapi.responses import StreamingResponse

    body = await request.json()
    question = body.get("question", "").strip()
    history   = body.get("history", [])

    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY not configured")

    # ── Gather live context ────────────────────────────────────────────────────
    r = _get_redis()
    context_parts = []

    # ML predictions
    for symbol in SYMBOLS:
        for horizon in HORIZONS:
            key = f"{symbol}_{horizon}d"
            cached = r.get(_pred_cache_key(symbol, horizon)) if r else None
            if cached:
                p = json.loads(cached)
                context_parts.append(
                    f"{symbol} {horizon}D forecast: {p['direction']} "
                    f"(confidence {p['confidence']*100:.1f}%, "
                    f"UP {p['up_prob']*100:.1f}% / DOWN {p['down_prob']*100:.1f}%)"
                )

    # Market signals from DB
    try:
        import psycopg2
        db_url = os.environ.get("DATABASE_URL")
        if db_url:
            conn = psycopg2.connect(db_url, sslmode="require")
            cur  = conn.cursor()

            cur.execute("SELECT date, value FROM fear_greed_index ORDER BY date DESC LIMIT 1")
            fg = cur.fetchone()
            if fg:
                v = fg[1]
                label = ("Extreme Fear" if v <= 20 else "Fear" if v <= 40 else
                         "Neutral" if v <= 60 else "Greed" if v <= 80 else "Extreme Greed")
                context_parts.append(f"Fear & Greed index: {v:.0f}/100 ({label}) as of {fg[0]}")

            for sym in ("BTC", "ETH"):
                cur.execute(
                    "SELECT date, rate_avg FROM funding_rates WHERE symbol=%s ORDER BY date DESC LIMIT 1",
                    (sym,)
                )
                fr = cur.fetchone()
                if fr:
                    context_parts.append(
                        f"{sym} funding rate: {fr[1]*100:.4f}% daily avg as of {fr[0]}"
                    )

            # Recent price action
            for sym in ("BTC", "ETH"):
                cur.execute("""
                    SELECT DATE(fetched_at) as d, AVG(close_usd) as price
                    FROM crypto_prices
                    WHERE symbol=%s AND interval_minutes=30
                      AND fetched_at >= NOW() - INTERVAL '8 days'
                    GROUP BY d ORDER BY d DESC LIMIT 7
                """, (sym,))
                rows = cur.fetchall()
                if rows:
                    prices = [(str(r[0]), round(r[1], 2)) for r in rows]
                    context_parts.append(f"{sym} recent closes (newest first): {prices}")

            cur.close()
            conn.close()
    except Exception as e:
        context_parts.append(f"(DB context unavailable: {e})")

    context_block = "\n".join(f"- {p}" for p in context_parts)

    system_prompt = f"""You are a professional crypto market analyst assistant built into a trading dashboard.
You have access to real-time ML model predictions and live market data.

CURRENT MARKET CONTEXT:
{context_block}

IMPORTANT RULES:
- Always reference the specific data above when making assessments
- Be concise but thorough — bullet points where appropriate
- Always end with a risk disclaimer
- Never give specific price targets
- Acknowledge uncertainty — crypto is volatile
- The ML model predictions are probabilistic, not guarantees
"""

    messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-10:]:   # keep last 10 turns for context
        if msg.get("role") in ("user", "assistant") and msg.get("content"):
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": question})

    client = openai.AsyncOpenAI(api_key=api_key)

    async def stream_response():
        try:
            stream = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                stream=True,
                temperature=0.4,
                max_tokens=1024,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as e:
            yield f"\n\n[Error: {e}]"

    return StreamingResponse(stream_response(), media_type="text/plain")


@app.post("/trade/chat")
async def trade_chat(request: Request):
    """
    Trading-aware chat endpoint.
    Detects trading intent → analyses → proposes order with full reasoning.
    Returns streaming text + optionally a pending_order JSON at the end
    (delimited by \\n---ORDER_JSON---\\n so the frontend can parse it).

    Body: {"message": "...", "history": [...], "confirm": false}
    If confirm=true and history contains a pending order → executes it.
    """
    import openai as _openai
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "trading"))

    from fastapi.responses import StreamingResponse

    body    = await request.json()
    message = body.get("message", "").strip()
    history = body.get("history", [])
    confirm = body.get("confirm", False)

    if not message:
        raise HTTPException(status_code=400, detail="message is required")

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY not configured")

    # ── Gather market context (same as /analyze) ───────────────────────────────
    r = _get_redis()
    context_parts = []
    for symbol in SYMBOLS:
        for horizon in HORIZONS:
            cached = r.get(_pred_cache_key(symbol, horizon)) if r else None
            if cached:
                p = json.loads(cached)
                context_parts.append(
                    f"{symbol} {horizon}D: {p['direction']} "
                    f"(UP {p['up_prob']*100:.1f}% / DOWN {p['down_prob']*100:.1f}%)"
                )

    try:
        import psycopg2
        db_url = os.environ.get("DATABASE_URL")
        if db_url:
            conn = psycopg2.connect(db_url, sslmode="require")
            cur  = conn.cursor()

            # Fear & Greed
            cur.execute("SELECT date, value FROM fear_greed_index ORDER BY date DESC LIMIT 1")
            fg = cur.fetchone()
            if fg:
                v = fg[1]
                label = ("Extreme Fear" if v<=20 else "Fear" if v<=40 else
                         "Neutral" if v<=60 else "Greed" if v<=80 else "Extreme Greed")
                context_parts.append(f"Fear & Greed: {v:.0f}/100 ({label}) as of {fg[0]}")

            # Funding rates
            for sym in ("BTC", "ETH"):
                cur.execute(
                    "SELECT date, rate_avg FROM funding_rates WHERE symbol=%s ORDER BY date DESC LIMIT 1", (sym,)
                )
                fr = cur.fetchone()
                if fr:
                    context_parts.append(f"{sym} funding rate: {fr[1]*100:.4f}%/day as of {fr[0]}")

            # Recent price action (7 days)
            for sym in ("BTC", "ETH"):
                cur.execute("""
                    SELECT DATE(fetched_at), AVG(close_usd) FROM crypto_prices
                    WHERE symbol=%s AND interval_minutes=30
                      AND fetched_at >= NOW() - INTERVAL '7 days'
                    GROUP BY DATE(fetched_at) ORDER BY 1 DESC LIMIT 7
                """, (sym,))
                rows = cur.fetchall()
                if rows:
                    price_str = " → ".join([f"${round(r[1],0):,.0f}" for r in rows])
                    context_parts.append(f"{sym} recent prices (newest first): {price_str}")

            # Open positions with live Binance data
            cur.execute("""
                SELECT symbol, side, entry_price, quantity, stop_loss, take_profit,
                       leverage, opened_at
                FROM trades WHERE status='OPEN'
            """)
            open_pos = cur.fetchall()
            if open_pos:
                # Enrich with Binance mark price
                try:
                    import sys as _sys
                    _sys.path.insert(0, str(Path(__file__).parent.parent / "trading"))
                    from engine import get_client, _get_binance_positions
                    _bn_client = get_client()
                    _bn_pos    = _get_binance_positions(_bn_client)
                except Exception:
                    _bn_pos = {}

                from datetime import datetime as _dt, timezone as _tz2
                now = _dt.now(_tz2.utc)
                for p in open_pos:
                    sym, side, entry, qty, sl, tp, _, opened_at = p
                    if opened_at.tzinfo is None:
                        opened_at = opened_at.replace(tzinfo=_tz2.utc)
                    hours_open = (now - opened_at).total_seconds() / 3600
                    bn = _bn_pos.get(sym, {})
                    mark = bn.get("mark_price")
                    upnl = bn.get("unrealized_pnl")
                    if mark and upnl is not None:
                        sl_dist = abs(mark - sl) / mark * 100
                        tp_dist = abs(tp - mark) / mark * 100
                        context_parts.append(
                            f"Open position: {sym} {'LONG' if side=='BUY' else 'SHORT'} {qty} | "
                            f"Entry ${entry:,.2f} | Mark ${mark:,.2f} | "
                            f"Unrealized P&L: {upnl:+.2f} USDT | "
                            f"SL ${sl:,.2f} ({sl_dist:.1f}% away) | TP ${tp:,.2f} ({tp_dist:.1f}% away) | "
                            f"Open {hours_open:.1f}h"
                        )
                    else:
                        context_parts.append(
                            f"Open position: {sym} {'LONG' if side=='BUY' else 'SHORT'} {qty} | "
                            f"Entry ${entry:,.2f} | SL ${sl:,.2f} | TP ${tp:,.2f} | "
                            f"Open {hours_open:.1f}h"
                        )
            else:
                context_parts.append("Open positions: None")

            # Trade history — last 15 closed trades
            cur.execute("""
                SELECT symbol, side, entry_price, exit_price, pnl_usdt, pnl_pct,
                       close_reason,
                       EXTRACT(EPOCH FROM (closed_at - opened_at))/3600 AS duration_h,
                       closed_at
                FROM trades
                WHERE status='CLOSED' AND pnl_usdt IS NOT NULL
                ORDER BY closed_at DESC LIMIT 15
            """)
            closed_rows = cur.fetchall()
            if closed_rows:
                context_parts.append("Recent trade history (newest first):")
                for h in closed_rows:
                    sym, side, entry, _, pnl_u, pnl_pct, reason, dur, closed = h
                    context_parts.append(
                        f"  {sym} {'LONG' if side=='BUY' else 'SHORT'} | "
                        f"P&L: {pnl_u:+.2f} USDT ({float(pnl_pct):+.2f}%) | "
                        f"held {float(dur):.1f}h | closed: {reason} | {str(closed)[:10]}"
                    )
            else:
                context_parts.append("Trade history: No closed trades yet")

            # Account performance stats
            cur.execute("""
                SELECT
                    COUNT(*) AS total,
                    SUM(CASE WHEN pnl_usdt > 0 THEN 1 ELSE 0 END) AS wins,
                    COALESCE(SUM(pnl_usdt), 0) AS total_pnl,
                    COALESCE(AVG(pnl_usdt), 0) AS avg_pnl,
                    COALESCE(MAX(pnl_usdt), 0) AS best,
                    COALESCE(MIN(pnl_usdt), 0) AS worst
                FROM trades WHERE status='CLOSED' AND pnl_usdt IS NOT NULL
            """)
            stats = cur.fetchone()
            if stats and stats[0]:
                total, wins, total_pnl, avg_pnl, best, worst = stats
                losses = total - wins
                win_rate = wins / total * 100 if total else 0
                context_parts.append(
                    f"Account stats: {total} trades | Win rate {win_rate:.0f}% ({wins}W/{losses}L) | "
                    f"Total P&L {total_pnl:+.2f} USDT | Avg {avg_pnl:+.2f} | "
                    f"Best {best:+.2f} | Worst {worst:+.2f}"
                )

            # Last autonomous engine decisions (4h cycle)
            cur.execute("""
                SELECT decided_at, decision FROM llm_decisions
                ORDER BY decided_at DESC LIMIT 3
            """)
            llm_rows = cur.fetchall()
            if llm_rows:
                context_parts.append("Autonomous engine's recent decisions (every 4h):")
                for decided_at, dec in llm_rows:
                    summary = []
                    for item in dec.get("decisions", []):
                        summary.append(
                            f"{item['symbol']}={item['action']} "
                            f"(conf {item.get('confidence_score', 0):.0%})"
                        )
                    context_parts.append(
                        f"  {str(decided_at)[:16]} | {', '.join(summary)} | "
                        f"\"{dec.get('market_summary', '')}\""
                    )

            cur.close()
            conn.close()
    except Exception as e:
        context_parts.append(f"(DB unavailable: {e})")

    context_block = "\n".join(f"- {p}" for p in context_parts)

    system_prompt = f"""You are an expert crypto trading assistant with the ability to place real orders on Binance Testnet.

CURRENT MARKET DATA:
{context_block}

YOUR CAPABILITIES:
- You can analyse the market and answer questions
- You can propose and execute trades on the user's behalf
- You use 5x leverage on USDT-margined futures (BTC and ETH only)

WHEN THE USER WANTS TO PLACE A TRADE:
1. First provide a thorough analysis covering:
   - Does the ML model support this trade?
   - What does Fear & Greed suggest?
   - What do funding rates tell us?
   - What is the recent price trend?
   - What are the main risks?
   - Bull case and bear case
2. Then propose specific order parameters:
   - Entry: market or specific price
   - Stop loss: specific price and % from entry
   - Take profit: specific price and % from entry
   - Position size: small/normal/large (0.5x/1x/1.5x)
3. End your message with EXACTLY this block (fill in real numbers):

---PROPOSED_ORDER---
{{"symbol":"BTC","side":"BUY","entry_type":"market","stop_loss_pct":0.03,"take_profit_pct":0.06,"size_multiplier":1.0,"reasoning":"brief reason"}}
---END_ORDER---

IMPORTANT:
- Only include the ORDER block if user is explicitly asking to place a trade
- stop_loss_pct: 0.01-0.05, take_profit_pct: 0.02-0.10
- side: "BUY" for long, "SELL" for short
- Always warn about risks before proposing
- Never place a trade the user didn't ask for
"""

    messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-12:]:
        if msg.get("role") in ("user", "assistant") and msg.get("content"):
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": message})

    oai_client = _openai.AsyncOpenAI(api_key=api_key)

    async def stream_response():
        try:
            stream = await oai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                stream=True,
                temperature=0.3,
                max_tokens=1200,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as e:
            yield f"\n\n[Error: {e}]"

    return StreamingResponse(stream_response(), media_type="text/plain")


@app.post("/trade/execute")
async def trade_execute(request: Request):
    """
    Executes a previously proposed order after user confirmation.
    Body: {"order": {"symbol":..., "side":..., "stop_loss_pct":..., "take_profit_pct":..., "size_multiplier":...}}
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "trading"))

    body  = await request.json()
    order = body.get("order")
    if not order:
        raise HTTPException(status_code=400, detail="order is required")

    try:
        from engine import (
            get_client, get_conn, save_trade, calc_quantity,
            LEVERAGE,
        )

        client = get_client()
        symbol = order["symbol"].upper()
        side   = order["side"].upper()   # BUY or SELL
        sl_pct = float(order.get("stop_loss_pct",  0.03))
        tp_pct = float(order.get("take_profit_pct", 0.06))
        mult   = float(order.get("size_multiplier", 1.0))

        # Get balance
        account  = client.account()
        bal_info = next((a for a in account["assets"] if a["asset"] == "USDT"), None)
        balance  = float(bal_info["walletBalance"]) if bal_info else 1000.0

        # Set leverage + get price
        sym_pair = f"{symbol}USDT"
        client.change_leverage(symbol=sym_pair, leverage=LEVERAGE)
        ticker      = client.ticker_price(symbol=sym_pair)
        entry_price = float(ticker["price"])

        quantity = calc_quantity(balance, mult, entry_price, symbol)
        if quantity <= 0:
            raise HTTPException(status_code=400, detail="Position size too small")

        if side == "BUY":
            stop_loss   = round(entry_price * (1 - sl_pct), 2)
            take_profit = round(entry_price * (1 + tp_pct), 2)
        else:
            stop_loss   = round(entry_price * (1 + sl_pct), 2)
            take_profit = round(entry_price * (1 - tp_pct), 2)

        # Place order
        result = client.new_order(
            symbol=sym_pair, side=side, type="MARKET", quantity=quantity,
        )
        order_id    = str(result.get("orderId", ""))
        actual_price = float(result.get("avgPrice", entry_price)) or entry_price

        trade_id = save_trade({
            "symbol":           symbol,
            "side":             side,
            "entry_price":      actual_price,
            "quantity":         quantity,
            "stop_loss":        stop_loss,
            "take_profit":      take_profit,
            "confidence":       mult / 2.0,
            "horizon":          1,
            "binance_order_id": order_id,
            "notes":            f"Manual via AI chat: {order.get('reasoning','')}",
        })

        # Place native SL/TP on Binance — fires even if engine is offline
        from engine import _place_native_sl_tp
        _place_native_sl_tp(client, symbol, side, stop_loss, take_profit)

        return {
            "success":     True,
            "trade_id":    trade_id,
            "symbol":      symbol,
            "side":        side,
            "entry_price": actual_price,
            "quantity":    quantity,
            "stop_loss":   stop_loss,
            "take_profit": take_profit,
            "order_id":    order_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reload")
def reload_models(
    symbol: Optional[str] = Query(
        default=None,
        description="Symbol to reload (BTC or ETH). Omit to reload all.",
    ),
):
    """
    Busts the Redis cache and re-downloads the model from DagsHub MLflow.
    Use this after running train.py so the live service picks up the new model
    without a process restart.

    Examples:
      POST /reload            — reload both BTC and ETH
      POST /reload?symbol=BTC — reload only BTC
    """
    global models
    bust_cache(symbol)

    fresh = load_all_models()
    if not fresh:
        raise HTTPException(
            status_code=503,
            detail=(
                "No models could be loaded after reload. "
                "Check DagsHub for a FINISHED training run and confirm "
                "MLFLOW_TRACKING_URI and credentials are set in .env."
            ),
        )

    if symbol:
        sym = symbol.upper()
        sym_keys = [k for k in fresh if k.startswith(sym + "_")]
        if not sym_keys:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Reload succeeded but no models found for {sym}. "
                    f"Run train.py for {sym} and try again."
                ),
            )
        for k in sym_keys:
            models[k] = fresh[k]
        return {"reloaded": sym_keys, "timestamp": datetime.utcnow().isoformat()}

    models = fresh
    return {
        "reloaded":  list(models.keys()),
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/positions/sync")
def positions_sync():
    """
    Fetches live open position data from Binance Testnet for BTC and ETH.
    Returns mark price, real entry, actual quantity, and unrealized PnL
    directly from Binance — not from the local DB.

    The dashboard uses this to show accurate real-time P&L instead of
    relying on stale DB values that may not match what was actually filled.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "trading"))

    try:
        from engine import get_client, SYMBOLS
        client = get_client()

        positions = {}
        for sym in SYMBOLS:
            sym_pair = f"{sym}USDT"
            risk = client.get_position_risk(symbol=sym_pair)
            for pos in risk:
                amt = float(pos.get("positionAmt", 0))
                if amt == 0:
                    continue   # no open position for this symbol
                positions[sym] = {
                    "symbol":             sym,
                    "position_amt":       amt,
                    "entry_price":        float(pos.get("entryPrice", 0)),
                    "mark_price":         float(pos.get("markPrice", 0)),
                    "unrealized_pnl":     float(pos.get("unRealizedProfit", 0)),
                    "liquidation_price":  float(pos.get("liquidationPrice", 0)),
                    "leverage":           int(pos.get("leverage", 5)),
                    "margin_type":        pos.get("marginType", "cross"),
                }
        return positions

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Binance sync failed: {e}")


@app.post("/cache/bust")
def cache_bust(
    symbol: Optional[str] = Query(
        default=None,
        description="Symbol to bust (BTC or ETH). Omit to bust all.",
    ),
):
    """
    Expires the Redis cache entry without reloading.
    Next restart or POST /reload will pull a fresh model from DagsHub.
    Useful for forcing a version rollback — bust cache, then restart.
    """
    bust_cache(symbol)
    return {
        "busted":    symbol.upper() if symbol else "ALL",
        "timestamp": datetime.utcnow().isoformat(),
    }
