"""
model_store.py — Model loading with Redis cache + DagsHub MLflow fallback.

Load priority:
  1. Upstash Redis  (~2ms)  — warm start
  2. DagsHub MLflow (~8s)   — pulls the custom joblib bundle artifact
  3. Local disk             — offline fallback (models/btc_model.pkl)

NOTE: train.py saves a custom bundle dict via joblib, not a raw MLflow model.
      The bundle shape is:
        {
          "symbol": "BTC",
          "model":  <XGBClassifier>,
          "features": [...],
          "metrics": {...},
          "run_id": "...",
          ...
        }
      We extract bundle["model"] for inference.
"""
import io
import os
import logging
import joblib
import mlflow
import mlflow.artifacts
import redis
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

EXPERIMENT_NAME   = "trading-signals"
REDIS_TTL_SECONDS = 86_400        # 24h
SYMBOLS           = ["BTC", "ETH", "SOL", "XRP"]
HORIZONS          = [1, 7]        # days — one model trained per horizon
MODEL_DIR         = Path(__file__).parent / "models"

# Tried in order: custom joblib bundle first, raw MLflow model as fallback
BUNDLE_ARTIFACT_NAMES = ["{symbol}-bundle", "{symbol}-model"]


def _redis_url():   return os.getenv("REDIS_URL", "")
def _mlflow_uri():  return os.getenv("MLFLOW_TRACKING_URI", "")
def _mlflow_user(): return os.getenv("MLFLOW_TRACKING_USERNAME", "")
def _mlflow_pw():   return os.getenv("MLFLOW_TRACKING_PASSWORD", "")


_uri = _mlflow_uri()
if _uri:
    os.environ["MLFLOW_TRACKING_USERNAME"] = _mlflow_user()
    os.environ["MLFLOW_TRACKING_PASSWORD"] = _mlflow_pw()
    mlflow.set_tracking_uri(_uri)
    log.info(f"MLflow URI: {_uri[:60]}")
else:
    log.warning("MLFLOW_TRACKING_URI not set — DagsHub loading disabled")


def _redis_client() -> Optional[redis.Redis]:
    url = _redis_url()
    if not url:
        log.warning("REDIS_URL not set — skipping Redis cache")
        return None
    if url.startswith("redis://"):
        url = "rediss://" + url[len("redis://"):]
    try:
        r = redis.from_url(url, decode_responses=False, socket_timeout=3)
        r.ping()
        return r
    except Exception as e:
        log.warning(f"Redis unreachable: {e}")
        return None


def _model_key(symbol: str, horizon: int) -> str:
    """Canonical key used for both the in-memory dict and Redis."""
    return f"{symbol.upper()}_{horizon}d"


def _redis_key(symbol: str, horizon: int) -> str:
    return f"ml:model:{symbol.upper()}:{horizon}d"


def _trained_at_key(symbol: str, horizon: int) -> str:
    return f"ml:trained_at:{symbol.upper()}:{horizon}d"


def _load_from_redis(symbol: str, horizon: int, r: redis.Redis):
    key = _model_key(symbol, horizon)
    try:
        blob = r.get(_redis_key(symbol, horizon))
        if blob is None:
            log.info(f"[{key}] Redis cache miss")
            return None, None
        model = joblib.load(io.BytesIO(blob))
        trained_at = r.get(_trained_at_key(symbol, horizon))
        if isinstance(trained_at, bytes):
            trained_at = trained_at.decode()
        log.info(f"[{key}] Loaded from Redis cache")
        return model, trained_at
    except Exception as e:
        log.warning(f"[{key}] Redis read failed: {e}")
        return None, None


def _save_to_redis(symbol: str, horizon: int, model, r: redis.Redis, trained_at: Optional[str] = None) -> None:
    key = _model_key(symbol, horizon)
    try:
        buf = io.BytesIO()
        joblib.dump(model, buf)
        size_kb = len(buf.getvalue()) / 1024
        r.setex(_redis_key(symbol, horizon), REDIS_TTL_SECONDS, buf.getvalue())
        if trained_at:
            r.setex(_trained_at_key(symbol, horizon), REDIS_TTL_SECONDS, trained_at)
        log.info(f"[{key}] Cached in Redis ({size_kb:.0f} KB, TTL=24h)")
    except Exception as e:
        log.warning(f"[{key}] Redis write failed (non-fatal): {e}")


def _extract_model_from_artifact(artifact_dir: Path, symbol: str):
    """
    Handles two artifact formats that train.py produces:

    Format A — custom joblib bundle ({symbol}-bundle):
      artifact_dir/
        btc_model.pkl   ← joblib dict with keys: model, features, metrics, ...
      We load the dict and return dict["model"] (the raw XGBClassifier).

    Format B — MLflow xgboost model ({symbol}-model):
      artifact_dir/
        MLmodel
        model.xgb
      We use mlflow.xgboost.load_model() on the directory.
    """
    pkl_files = list(artifact_dir.rglob("*.pkl"))
    if pkl_files:
        bundle = joblib.load(pkl_files[0])
        if isinstance(bundle, dict) and "model" in bundle:
            log.info(f"[{symbol}] Extracted XGBClassifier from custom bundle dict")
            return bundle["model"], bundle.get("trained_at_utc")
        log.info(f"[{symbol}] Loaded raw model from .pkl artifact")
        return bundle, None

    mlmodel_file = artifact_dir / "MLmodel"
    if mlmodel_file.exists():
        model = mlflow.xgboost.load_model(str(artifact_dir))
        log.info(f"[{symbol}] Loaded MLflow XGBoost model from artifact dir")
        return model, None

    raise RuntimeError(
        f"[{symbol}] Could not find .pkl or MLmodel in artifact dir: {artifact_dir}"
    )


def _load_from_dagshub(symbol: str, horizon: int = 1):
    """
    Loads the model for a given symbol+horizon from DagsHub.

    Priority:
      1. @champion alias on registered model  (btc-direction-7d@champion)
         — set manually on DagsHub UI or promoted automatically by train.py
      2. Most recent FINISHED run matching the run name pattern
         — fallback when no registered model / alias exists yet

    Run name format: {symbol}-direction-{horizon}d  (e.g. BTC-direction-7d)
    Falls back to legacy 'direction-24h' name for horizon=1.
    """
    uri = _mlflow_uri()
    if not uri:
        raise RuntimeError("MLFLOW_TRACKING_URI not set")
    os.environ["MLFLOW_TRACKING_USERNAME"] = _mlflow_user()
    os.environ["MLFLOW_TRACKING_PASSWORD"] = _mlflow_pw()
    mlflow.set_tracking_uri(uri)

    client = mlflow.tracking.MlflowClient()
    key = _model_key(symbol, horizon)

    # Priority 1: @champion alias on registered model
    model_name = f"{symbol.lower()}-direction-{horizon}d"
    try:
        mv = client.get_model_version_by_alias(model_name, "champion")
        if mv and mv.run_id:
            log.info(f"[{key}] Found @champion alias → version={mv.version} run={mv.run_id[:8]}")
            # Determine which bundle artifact this run has
            artifacts = client.list_artifacts(mv.run_id)
            artifact_names = {a.path for a in artifacts}
            target_artifact = None
            for tmpl in BUNDLE_ARTIFACT_NAMES:
                name = tmpl.replace("{symbol}", symbol)
                if name in artifact_names:
                    target_artifact = name
                    break
            if target_artifact:
                log.info(f"[{key}] Downloading @champion artifact '{target_artifact}'...")
                artifact_dir = mlflow.artifacts.download_artifacts(
                    run_id=mv.run_id,
                    artifact_path=target_artifact,
                )
                return _extract_model_from_artifact(Path(artifact_dir), symbol)  # (model, trained_at)
    except Exception as e:
        log.info(f"[{key}] No @champion alias found ({e}) — falling back to run search")

    # Priority 2: most recent FINISHED run by run name
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        raise RuntimeError(f"Experiment '{EXPERIMENT_NAME}' not found on DagsHub")

    run_name_candidates = [f"{symbol}-direction-{horizon}d"]

    runs = []
    for run_name in run_name_candidates:
        found = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string=f"tags.mlflow.runName = '{run_name}'",
            order_by=["start_time DESC"],
            max_results=20,
        )
        runs.extend(found)
        if runs:
            break

    target_run_id = None
    target_artifact = None
    for run in runs:
        if run.info.status != "FINISHED":
            continue
        artifacts = client.list_artifacts(run.info.run_id)
        artifact_names = {a.path for a in artifacts}
        if not artifact_names:
            continue
        for tmpl in BUNDLE_ARTIFACT_NAMES:
            name = tmpl.replace("{symbol}", symbol)
            if name in artifact_names:
                target_run_id = run.info.run_id
                target_artifact = name
                break
        if target_run_id:
            break

    if not target_run_id:
        raise RuntimeError(
            f"[{key}] No FINISHED run with artifacts found. "
            f"Run train.py to upload a model to DagsHub."
        )

    log.info(f"[{symbol}] Downloading artifact '{target_artifact}' from run {target_run_id[:8]}...")
    artifact_dir = mlflow.artifacts.download_artifacts(
        run_id=target_run_id,
        artifact_path=target_artifact,
    )

    model, trained_at = _extract_model_from_artifact(Path(artifact_dir), symbol)
    log.info(f"[{symbol}] Loaded from DagsHub ✓")
    return model, trained_at


def _load_from_disk(symbol: str):
    path = MODEL_DIR / f"{symbol.lower()}_model.pkl"
    if not path.exists():
        raise RuntimeError(f"No local model at {path}")

    data = joblib.load(path)

    if isinstance(data, dict) and "model" in data:
        log.info(f"[{symbol}] Loaded bundle dict from disk, extracting model")
        return data["model"], data.get("trained_at_utc")

    log.info(f"[{symbol}] Loaded from disk ({path.name})")
    return data, None


def load_all_models() -> tuple[dict[str, object], dict[str, str]]:
    """
    Loads models for every (symbol, horizon) combination.
    Returns (models_dict, trained_at_dict).
    Keys: "BTC_1d", "BTC_7d", "ETH_1d", "ETH_7d"
    Load priority: Redis → DagsHub → local disk (horizon=1 only for disk fallback)
    """
    r = _redis_client()
    loaded: dict[str, object] = {}
    trained_at: dict[str, str] = {}

    for symbol in SYMBOLS:
        for horizon in HORIZONS:
            key = _model_key(symbol, horizon)
            model = None
            ts = None

            if r is not None:  # 1. Redis
                model, ts = _load_from_redis(symbol, horizon, r)

            if model is None and _mlflow_uri():  # 2. DagsHub
                try:
                    model, ts = _load_from_dagshub(symbol, horizon)
                except Exception as e:
                    log.warning(f"[{key}] DagsHub load failed: {e}")

            if model is None and horizon == 1:  # 3. local disk (horizon=1 only)
                try:
                    model, ts = _load_from_disk(symbol)
                except Exception as e:
                    log.warning(f"[{key}] Disk fallback failed: {e}")

            if model is None:
                log.error(f"[{key}] All sources failed — skipping")
                continue

            if r is not None:
                _save_to_redis(symbol, horizon, model, r, ts)

            loaded[key] = model
            if ts:
                trained_at[key] = ts
            log.info(f"[{key}] Ready ✓")

    return loaded, trained_at


def bust_cache(symbol: Optional[str] = None) -> None:
    """Deletes Redis cache key(s) for all horizons so next load pulls fresh from DagsHub."""
    r = _redis_client()
    if r is None:
        log.warning("Redis not available — nothing to bust")
        return
    targets = [symbol.upper()] if symbol else SYMBOLS
    for s in targets:
        for horizon in HORIZONS:
            key = _model_key(s, horizon)
            deleted = r.delete(_redis_key(s, horizon))
            log.info(f"[{key}] Redis cache {'busted' if deleted else 'was already empty'}")