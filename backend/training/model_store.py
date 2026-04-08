"""
model_store.py â€” Model loading with Redis cache + DagsHub MLflow fallback.

Load priority:
  1. Upstash Redis  (~2ms)  â€” warm start
  2. DagsHub MLflow (~8s)   â€” pulls the custom joblib bundle artifact
  3. Local disk             â€” offline fallback (models/btc_model.pkl)

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
MODEL_DIR         = Path(__file__).parent / "models"
FEATURE_TRAIN_TIMEFRAME = os.getenv("FEATURE_TRAIN_TIMEFRAME", "1D").strip().upper()
MODEL_NAME_SUFFIX = os.getenv("MODEL_NAME_SUFFIX", "").strip().lower()

# Tried in order: custom joblib bundle first, raw MLflow model as fallback
BUNDLE_ARTIFACT_NAMES = ["{symbol}-bundle", "{symbol}-model"]


def _parse_horizon_tokens(raw: str) -> list[str]:
    allowed = {"4h", "1d"}
    tokens: list[str] = []
    for p in raw.split(","):
        t = p.strip().lower()
        if not t:
            continue
        if len(t) < 2 or t[-1] not in {"d", "h"} or not t[:-1].isdigit():
            raise ValueError(f"Invalid horizon token '{t}'. Use forms like 1d,4h")
        if t in allowed:
            tokens.append(t)
    if not tokens:
        tokens = ["4h", "1d"]
    out: list[str] = []
    for t in tokens:
        if t not in out:
            out.append(t)
    return out


HORIZONS = _parse_horizon_tokens(os.getenv("MODEL_HORIZONS", "4h,1d"))


def _model_variant_suffix() -> str:
    if MODEL_NAME_SUFFIX:
        return f"-{MODEL_NAME_SUFFIX}"
    if FEATURE_TRAIN_TIMEFRAME in {"", "1D", "1DAY", "DAILY"}:
        return ""
    return f"-{FEATURE_TRAIN_TIMEFRAME.lower()}"


def _registry_model_candidates(symbol: str, horizon_token: str) -> list[str]:
    candidates: list[str] = []
    suffix = _model_variant_suffix()
    if suffix:
        candidates.append(f"{symbol.lower()}-direction-{horizon_token}{suffix}")
    candidates.append(f"{symbol.lower()}-direction-{horizon_token}")
    if horizon_token == "1d":
        candidates.append(f"{symbol.lower()}-direction-24h")
    out: list[str] = []
    for c in candidates:
        if c not in out:
            out.append(c)
    return out


def _run_name_candidates(symbol: str, horizon_token: str) -> list[str]:
    candidates: list[str] = []
    suffix = _model_variant_suffix()
    if suffix:
        candidates.append(f"{symbol}-direction-{horizon_token}{suffix}")
    candidates.append(f"{symbol}-direction-{horizon_token}")
    out: list[str] = []
    for c in candidates:
        if c not in out:
            out.append(c)
    return out


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
    log.warning("MLFLOW_TRACKING_URI not set â€” DagsHub loading disabled")


def _redis_client() -> Optional[redis.Redis]:
    url = _redis_url()
    if not url:
        log.warning("REDIS_URL not set â€” skipping Redis cache")
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


def _model_key(symbol: str, horizon_token: str) -> str:
    """Canonical key used for both the in-memory dict and Redis."""
    return f"{symbol.upper()}_{horizon_token}"


def _redis_key(symbol: str, horizon_token: str) -> str:
    return f"ml:model:{symbol.upper()}:{horizon_token}"


def _trained_at_key(symbol: str, horizon_token: str) -> str:
    return f"ml:trained_at:{symbol.upper()}:{horizon_token}"


def _load_from_redis(symbol: str, horizon_token: str, r: redis.Redis):
    key = _model_key(symbol, horizon_token)
    try:
        blob = r.get(_redis_key(symbol, horizon_token))
        if blob is None:
            log.info(f"[{key}] Redis cache miss")
            return None, None
        model = joblib.load(io.BytesIO(blob))
        trained_at = r.get(_trained_at_key(symbol, horizon_token))
        if isinstance(trained_at, bytes):
            trained_at = trained_at.decode()
        log.info(f"[{key}] Loaded from Redis cache")
        return model, trained_at
    except Exception as e:
        log.warning(f"[{key}] Redis read failed: {e}")
        return None, None


def _save_to_redis(symbol: str, horizon_token: str, model, r: redis.Redis, trained_at: Optional[str] = None) -> None:
    key = _model_key(symbol, horizon_token)
    try:
        buf = io.BytesIO()
        joblib.dump(model, buf)
        size_kb = len(buf.getvalue()) / 1024
        r.setex(_redis_key(symbol, horizon_token), REDIS_TTL_SECONDS, buf.getvalue())
        if trained_at:
            r.setex(_trained_at_key(symbol, horizon_token), REDIS_TTL_SECONDS, trained_at)
        log.info(f"[{key}] Cached in Redis ({size_kb:.0f} KB, TTL=24h)")
    except Exception as e:
        log.warning(f"[{key}] Redis write failed (non-fatal): {e}")


def _extract_model_from_artifact(artifact_dir: Path, symbol: str):
    """
    Handles two artifact formats that train.py produces:

    Format A â€” custom joblib bundle ({symbol}-bundle):
      artifact_dir/
        btc_model.pkl   â† joblib dict with keys: model, features, metrics, ...
      We load the dict and return dict["model"] (the raw XGBClassifier).

    Format B â€” MLflow xgboost model ({symbol}-model):
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


def _load_from_dagshub(symbol: str, horizon_token: str = "1d"):
    """
    Loads the model for a given symbol+horizon from DagsHub.

    Priority:
      1. @champion alias on registered model  (btc-direction-4h@champion)
         â€” set manually on DagsHub UI or promoted automatically by train.py
      2. Most recent FINISHED run matching the run name pattern
         â€” fallback when no registered model / alias exists yet

    Run name format: {symbol}-direction-{horizon_token} (e.g. BTC-direction-4h)
    Falls back to legacy 'direction-24h' name for horizon_token='1d'.
    """
    uri = _mlflow_uri()
    if not uri:
        raise RuntimeError("MLFLOW_TRACKING_URI not set")
    os.environ["MLFLOW_TRACKING_USERNAME"] = _mlflow_user()
    os.environ["MLFLOW_TRACKING_PASSWORD"] = _mlflow_pw()
    mlflow.set_tracking_uri(uri)

    client = mlflow.tracking.MlflowClient()
    key = _model_key(symbol, horizon_token)

    # Priority 1: @champion alias on registered model (variant first, then legacy)
    champion_checked = []
    for model_name in _registry_model_candidates(symbol, horizon_token):
        champion_checked.append(model_name)
        try:
            mv = client.get_model_version_by_alias(model_name, "champion")
            if not mv or not mv.run_id:
                continue
            log.info(f"[{key}] Found @champion alias on {model_name} -> version={mv.version} run={mv.run_id[:8]}")
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
        except Exception:
            continue
    log.info(f"[{key}] No @champion alias found in: {champion_checked} - falling back to run search")

    # Priority 2: most recent FINISHED run by run name
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        raise RuntimeError(f"Experiment '{EXPERIMENT_NAME}' not found on DagsHub")

    run_name_candidates = _run_name_candidates(symbol, horizon_token)

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
    log.info(f"[{symbol}] Loaded from DagsHub âœ“")
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
    Keys: "BTC_4h", "BTC_1d", "ETH_4h", "ETH_1d"
    Load priority: Redis â†’ DagsHub â†’ local disk (horizon=1 only for disk fallback)
    """
    r = _redis_client()
    loaded: dict[str, object] = {}
    trained_at: dict[str, str] = {}

    for symbol in SYMBOLS:
        for horizon_token in HORIZONS:
            key = _model_key(symbol, horizon_token)
            model = None
            ts = None

            if r is not None:  # 1. Redis
                model, ts = _load_from_redis(symbol, horizon_token, r)

            if model is None and _mlflow_uri():  # 2. DagsHub
                try:
                    model, ts = _load_from_dagshub(symbol, horizon_token)
                except Exception as e:
                    log.warning(f"[{key}] DagsHub load failed: {e}")

            if model is None and horizon_token == "1d":  # 3. local disk (legacy 1d only)
                try:
                    model, ts = _load_from_disk(symbol)
                except Exception as e:
                    log.warning(f"[{key}] Disk fallback failed: {e}")

            if model is None:
                log.error(f"[{key}] All sources failed â€” skipping")
                continue

            if r is not None:
                _save_to_redis(symbol, horizon_token, model, r, ts)

            loaded[key] = model
            if ts:
                trained_at[key] = ts
            log.info(f"[{key}] Ready âœ“")

    return loaded, trained_at


def bust_cache(symbol: Optional[str] = None) -> None:
    """Deletes Redis cache key(s) for all horizons so next load pulls fresh from DagsHub."""
    r = _redis_client()
    if r is None:
        log.warning("Redis not available â€” nothing to bust")
        return
    targets = [symbol.upper()] if symbol else SYMBOLS
    for s in targets:
        for horizon_token in HORIZONS:
            key = _model_key(s, horizon_token)
            deleted = r.delete(_redis_key(s, horizon_token))
            log.info(f"[{key}] Redis cache {'busted' if deleted else 'was already empty'}")
