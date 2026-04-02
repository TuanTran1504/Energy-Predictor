import os
import json
import shutil
from pathlib import Path
import sys
import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
from datetime import datetime, UTC
# -----------------------------------------------------------------------------
# Path resolution
# -----------------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()


def find_backend_dir(start: Path) -> Path:
    """
    Walk upward until we find a folder that looks like the backend root.
    We expect it to contain api/ and training/ or ml/.
    """
    for candidate in [start.parent] + list(start.parents):
        has_api = (candidate / "api").exists()
        has_training = (candidate / "training").exists()
        has_ml = (candidate / "ml").exists()
        if has_api and (has_training or has_ml):
            return candidate

    # Fallback: old assumption
    return start.parent.parent.parent


BACKEND_DIR = find_backend_dir(THIS_FILE)
PROJECT_ROOT = BACKEND_DIR.parent

# Put dotenv lookup on backend/.env first, then project root /.env as fallback
backend_env = BACKEND_DIR / ".env"
project_env = PROJECT_ROOT / ".env"
if backend_env.exists():
    load_dotenv(backend_env)
else:
    load_dotenv(project_env)

# Support both backend/training/train.py and backend/ml/training/train.py layouts
TRAINING_PARENT = THIS_FILE.parent.parent
if str(TRAINING_PARENT) not in sys.path:
    sys.path.append(str(TRAINING_PARENT))
SYMBOLS = ["BTC", "ETH"]
MIN_PROMOTION_IMPROVEMENT = 0.0

REGISTER_MODELS = os.getenv("MLFLOW_REGISTER_MODELS", "false").strip().lower() in {
    "1", "true", "yes", "y"
}

PROMOTION_METRIC = os.getenv("MODEL_PROMOTION_METRIC", "accuracy").strip().lower()

SYNC_CHAMPION_TO_LOCAL = os.getenv("SYNC_CHAMPION_TO_LOCAL", "false").strip().lower() in {
    "1", "true", "yes", "y"
}

LOCAL_MODEL_SYNC_DIR = os.getenv(
    "LOCAL_MODEL_SYNC_DIR",
    str(BACKEND_DIR / "api" / "models"),
).strip()

def sync_model_bundle_from_dagshub(
    symbol: str,
    local_dir: str,
    tracking_uri: str,
    username: str,
    password: str,
    alias: str = "champion",
    version: str | None = None,
) -> dict:
    """
    Sync one model bundle from DagsHub MLflow to local serving directory.

    Uses either:
      - a registry alias, e.g. alias="champion"
      - or an explicit registry version, e.g. version="2"

    Downloads the custom "{SYMBOL}-bundle" artifact, not the raw MLflow model,
    because local inference expects the custom joblib bundle format.
    """
    symbol = symbol.upper()
    model_name = f"{symbol.lower()}-direction-24h"
    artifact_path = f"{symbol}-bundle"

    os.environ["MLFLOW_TRACKING_USERNAME"] = username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = password
    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()

    if version is not None:
        mv = client.get_model_version(name=model_name, version=str(version))
        selected_by = f"version={version}"
    else:
        mv = client.get_model_version_by_alias(name=model_name, alias=alias)
        selected_by = f"alias={alias}"

    run_id = mv.run_id
    if not run_id:
        raise RuntimeError(
            f"Registered model {model_name} ({selected_by}) has no run_id attached."
        )

    local_dir_path = Path(local_dir)
    local_dir_path.mkdir(parents=True, exist_ok=True)

    target_pkl = local_dir_path / f"{symbol.lower()}_model.pkl"
    metadata_path = local_dir_path / f"{symbol.lower()}_model.sync.json"

    if metadata_path.exists():
        try:
            current = json.loads(metadata_path.read_text(encoding="utf-8"))
            if current.get("version") == str(mv.version):
                return {
                    "symbol": symbol,
                    "model_name": model_name,
                    "selected_by": selected_by,
                    "version": str(mv.version),
                    "run_id": run_id,
                    "artifact_path": artifact_path,
                    "local_model_path": str(target_pkl),
                    "skipped": True,
                }
        except Exception:
            pass

    download_dir = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path=artifact_path,
    )

    download_dir = Path(download_dir)
    pkl_files = list(download_dir.rglob("*.pkl"))
    if not pkl_files:
        raise FileNotFoundError(
            f"No .pkl file found inside artifact '{artifact_path}' for run_id={run_id}"
        )

    source_pkl = pkl_files[0]
    shutil.copy2(source_pkl, target_pkl)

    metadata = {
        "symbol": symbol,
        "model_name": model_name,
        "selected_by": selected_by,
        "version": str(mv.version),
        "run_id": run_id,
        "artifact_path": artifact_path,
        "local_model_path": str(target_pkl),
        "synced_at_utc": datetime.now(UTC).isoformat(),
    }

    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata

def main():
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    if mlflow_uri:
        os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv(
            "MLFLOW_TRACKING_USERNAME", ""
        )
        os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv(
            "MLFLOW_TRACKING_PASSWORD", ""
        )
        mlflow.set_tracking_uri(mlflow_uri)

    mlflow.set_experiment("trading-signals")

    print(f"Resolved BACKEND_DIR:            {BACKEND_DIR}")
    print(f"Local sync enabled:             {SYNC_CHAMPION_TO_LOCAL}")
    print(f"Local sync dir:                 {LOCAL_MODEL_SYNC_DIR}")
    print(f"Model registration enabled:     {REGISTER_MODELS}")
    print(f"Promotion metric:               {PROMOTION_METRIC}")

    results = {}

    for symbol in SYMBOLS:
        with mlflow.start_run(run_name=f"{symbol}-direction-24h") as run:
            sync_info = sync_model_bundle_from_dagshub(
                    symbol=symbol,
                    local_dir=LOCAL_MODEL_SYNC_DIR,
                    tracking_uri=os.environ["MLFLOW_TRACKING_URI"],
                    username=os.environ["MLFLOW_TRACKING_USERNAME"],
                    password=os.environ["MLFLOW_TRACKING_PASSWORD"],
                    alias="champion",
                )
            print(f"[OK] Synced new champion locally for {symbol}: {sync_info}")



if __name__ == "__main__":
    main()