"""
check_pipeline.py — Run this to find out exactly where your model is.

Usage (from project root):
    python check_pipeline.py

It checks every layer in order and prints a clear status table.
No services need to be running — it queries DagsHub, Redis, and disk directly.
"""
import os
import io
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# ── Colours ────────────────────────────────────────────────────────────────────
G  = "\033[32m"   # green
R  = "\033[31m"   # red
Y  = "\033[33m"   # yellow
C  = "\033[36m"   # cyan
DIM = "\033[2m"
NC = "\033[0m"

def ok(msg):   print(f"  {G}✓{NC}  {msg}")
def err(msg):  print(f"  {R}✗{NC}  {msg}")
def warn(msg): print(f"  {Y}!{NC}  {msg}")
def info(msg): print(f"  {C}·{NC}  {DIM}{msg}{NC}")
def hdr(msg):  print(f"\n{C}━━━ {msg} ━━━{NC}")

SYMBOLS = ["BTC", "ETH"]

# ══════════════════════════════════════════════════════════════════════════════
# 1. Check environment variables
# ══════════════════════════════════════════════════════════════════════════════
hdr("1. Environment variables")

required = {
    "MLFLOW_TRACKING_URI":      os.getenv("MLFLOW_TRACKING_URI"),
    "MLFLOW_TRACKING_USERNAME": os.getenv("MLFLOW_TRACKING_USERNAME"),
    "MLFLOW_TRACKING_PASSWORD": os.getenv("MLFLOW_TRACKING_PASSWORD"),
    "REDIS_URL":                os.getenv("REDIS_URL"),
    "DATABASE_URL":             os.getenv("DATABASE_URL"),
    "PYTHON_ML_URL":            os.getenv("PYTHON_ML_URL", "http://localhost:8000"),
}

env_ok = True
for key, val in required.items():
    if val:
        # Mask secrets — show first 8 chars only
        display = val[:8] + "..." if len(val) > 8 else val
        ok(f"{key} = {display}")
    else:
        err(f"{key} is NOT SET in .env")
        env_ok = False

if not env_ok:
    print(f"\n{R}Fix missing env vars in .env before continuing.{NC}")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════════
# 2. Check DagsHub / MLflow
# ══════════════════════════════════════════════════════════════════════════════
hdr("2. DagsHub MLflow")

try:
    import mlflow

    os.environ["MLFLOW_TRACKING_USERNAME"] = required["MLFLOW_TRACKING_USERNAME"]
    os.environ["MLFLOW_TRACKING_PASSWORD"] = required["MLFLOW_TRACKING_PASSWORD"]
    mlflow.set_tracking_uri(required["MLFLOW_TRACKING_URI"])

    client = mlflow.tracking.MlflowClient()

    print()
    info("Checking registered models on DagsHub...")

    found_any = False

    for symbol in SYMBOLS:
        model_name = f"{symbol.lower()}-direction-24h"

        try:
            versions = client.search_model_versions(f"name='{model_name}'")
        except Exception as e:
            warn(f"Could not query registered model '{model_name}': {e}")
            continue

        if not versions:
            warn(f"No registered model found: {model_name}")
            continue

        found_any = True
        ok(f"Registered model exists: {model_name} ({len(versions)} version(s))")

        for v in sorted(versions, key=lambda x: int(x.version), reverse=True)[:5]:
            alias_text = ""
            try:
                mv = client.get_model_version(model_name, v.version)
                aliases = getattr(mv, "aliases", [])
                if aliases:
                    alias_text = f" aliases={aliases}"
            except Exception:
                pass

            print(
                f"      version={v.version} "
                f"stage={getattr(v, 'current_stage', 'None')} "
                f"run_id={v.run_id[:8]}...{alias_text}"
            )

    if not found_any:
        warn("No registered models found on DagsHub.")

except ImportError:
    err("mlflow not installed. Run: pip install mlflow")
except Exception as e:
    err(f"DagsHub connection failed: {e}")
    warn("Check MLFLOW_TRACKING_URI and credentials in .env")

# ══════════════════════════════════════════════════════════════════════════════
# 3. Check Redis cache
# ══════════════════════════════════════════════════════════════════════════════
hdr("3. Redis (Upstash) model cache")

try:
    import redis as redis_lib

    r = redis_lib.from_url(required["REDIS_URL"], decode_responses=False)
    r.ping()
    ok("Upstash Redis reachable")

    for symbol in SYMBOLS:
        key  = f"ml:model:{symbol}"
        blob = r.get(key)
        ttl  = r.ttl(key)
        if blob is None:
            warn(f"{symbol}: NOT in Redis cache  (key={key})")
            info(  "Will be populated automatically on next FastAPI startup")
        else:
            size_kb = len(blob) / 1024
            ttl_h   = ttl // 3600
            ok(f"{symbol}: cached in Redis  size={size_kb:.0f}KB  TTL={ttl_h}h remaining")

            # Try deserialising to confirm it's a valid model
            try:
                import joblib
                model = joblib.load(io.BytesIO(blob))
                ok(f"  Redis blob deserialises OK → {type(model).__name__}  n_features={model.n_features_in_}")
            except Exception as de:
                err(f"  Redis blob is CORRUPT — joblib.load failed: {de}")
                warn("  → Run POST /cache/bust then POST /reload to refresh")

except ImportError:
    err("redis not installed.  Run:  pip install redis")
except Exception as e:
    err(f"Redis connection failed: {e}")
    warn("→ Check REDIS_URL in .env")

# ══════════════════════════════════════════════════════════════════════════════
# 4. Check local disk (models/ directory)
# ══════════════════════════════════════════════════════════════════════════════
hdr("4. Local disk (models/ directory)")

model_dir = Path(__file__).parent / "models"
if not model_dir.exists():
    warn(f"models/ directory does not exist at {model_dir}")
    info("It will be created by train.py — this is fine if you're using DagsHub/Redis")
else:
    for symbol in SYMBOLS:
        path = model_dir / f"{symbol.lower()}_model.pkl"
        if path.exists():
            size_kb = path.stat().st_size / 1024
            import datetime
            mtime = datetime.datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            ok(f"{symbol}: {path.name}  size={size_kb:.0f}KB  saved={mtime}")
        else:
            warn(f"{symbol}: {path.name} not found — no local fallback")

# ══════════════════════════════════════════════════════════════════════════════
# 5. Check FastAPI service
# ══════════════════════════════════════════════════════════════════════════════
hdr("5. FastAPI ML service")

try:
    import urllib.request, json as _json

    url = required["PYTHON_ML_URL"].rstrip("/") + "/health"
    try:
        with urllib.request.urlopen(url, timeout=4) as resp:
            body = _json.loads(resp.read())
        loaded = body.get("models_loaded", [])
        if loaded:
            ok(f"FastAPI running at {required['PYTHON_ML_URL']}")
            ok(f"Models loaded in memory: {', '.join(loaded)}")
        else:
            err("FastAPI is running but NO models are loaded")
            warn("→ Check FastAPI startup logs — model_store.load_all_models() may have failed")
    except OSError:
        warn(f"FastAPI is NOT running at {required['PYTHON_ML_URL']}")
        info("Start it with:  uvicorn main:app --port 8000 --reload")
        info("(This is fine — service just needs to be started before serving)")
except Exception as e:
    warn(f"Could not check FastAPI: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# 6. Summary + what to do next
# ══════════════════════════════════════════════════════════════════════════════
hdr("Summary & next steps")

print(f"""
{C}Your model lives in THREE places (in priority order):{NC}

  1. {G}Upstash Redis{NC}     — fast cache, populated on FastAPI startup
                      key: ml:model:BTC  /  ml:model:ETH

  2. {G}DagsHub MLflow{NC}    — source of truth after each training run
                      {required['MLFLOW_TRACKING_URI']}

  3. {G}Local disk{NC}        — offline fallback written by train.py
                      models/btc_model.pkl  /  models/eth_model.pkl

{Y}If you have NO model anywhere yet:{NC}
  → Run training:   python ml/training/train.py
  → This writes to disk AND logs to DagsHub in one step.
  → FastAPI will pick it up automatically on startup.

{Y}If you have a model on DagsHub but FastAPI isn't loading it:{NC}
  → Start FastAPI:  uvicorn main:app --port 8000 --reload
  → Force reload:   curl -X POST http://localhost:8000/reload

{Y}If Redis cache is stale after retraining:{NC}
  → Bust cache:     curl -X POST http://localhost:8000/cache/bust
  → Reload:         curl -X POST http://localhost:8000/reload
""")