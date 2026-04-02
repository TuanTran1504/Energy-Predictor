import sys
import os
import json
import joblib
import tempfile
import traceback
import shutil
from pathlib import Path
from datetime import datetime, UTC

import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import mlflow.xgboost

from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from dotenv import load_dotenv


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
load_dotenv(project_env)   # root .env first (lowest priority)
load_dotenv(backend_env)   # backend/.env second — overrides root for duplicates

# Support both backend/training/train.py and backend/ml/training/train.py layouts
TRAINING_PARENT = THIS_FILE.parent.parent
if str(TRAINING_PARENT) not in sys.path:
    sys.path.append(str(TRAINING_PARENT))

from feature_engineering import build_features, get_feature_columns  # noqa: E402


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
SYMBOLS = ["BTC", "ETH"]
MIN_PROMOTION_IMPROVEMENT = 0.0
LOOKAHEAD_DAYS = int(os.getenv("LOOKAHEAD_DAYS", "1"))

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


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def train_model(symbol: str) -> dict:
    """
    Train one symbol model using a strict temporal split.
    """
    print(f"\n{'=' * 50}")
    print(f"Training {symbol} direction model...")

    df = build_features(symbol, lookahead=LOOKAHEAD_DAYS)
    feature_cols = get_feature_columns()

    X = df[feature_cols].fillna(0)
    y = df["target"]

    print(f"Features: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")

    split_idx = int(len(X) * 0.8)
    X_train = X.iloc[:split_idx]
    X_val = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_val = y.iloc[split_idx:]

    print(
        f"Train: {len(X_train)} rows "
        f"({df['date'].iloc[0].date()} to {df['date'].iloc[split_idx - 1].date()})"
    )
    print(
        f"Val:   {len(X_val)} rows "
        f"({df['date'].iloc[split_idx].date()} to {df['date'].iloc[-1].date()})"
    )

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, zero_division=0)
    recall = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)

    print("\nValidation Results:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_val, y_pred, target_names=["DOWN", "UP"], zero_division=0
        )
    )

    importance = pd.Series(
        model.feature_importances_,
        index=feature_cols,
    ).sort_values(ascending=False)

    print("\nTop 10 features:")
    print(importance.head(10).to_string())

    baseline_acc = float(y_val.mean())
    improvement = accuracy - baseline_acc

    print(f"\nBaseline (always UP): {baseline_acc:.4f}")
    print(f"Model improvement: {improvement * 100:+.1f}%")

    return {
        "symbol": symbol,
        "model": model,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "baseline": float(baseline_acc),
        "improvement": float(improvement),
        "features": feature_cols,
        "X_train": X_train,
        "X_val": X_val,
        "y_val": y_val,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "importance": importance,
        "trained_at_utc": datetime.now(UTC).isoformat(),
    }


# -----------------------------------------------------------------------------
# Bundle creation
# -----------------------------------------------------------------------------
def make_model_bundle(result: dict, run_id: str) -> dict:
    """
    Create the custom serving bundle expected by local inference.
    """
    return {
        "symbol": result["symbol"],
        "model": result["model"],
        "features": result["features"],
        "metrics": {
            "accuracy": result["accuracy"],
            "precision": result["precision"],
            "recall": result["recall"],
            "f1": result["f1"],
            "baseline": result["baseline"],
            "improvement": result["improvement"],
        },
        "run_id": run_id,
        "trained_at_utc": result["trained_at_utc"],
        "threshold": 0.5,
        "model_type": type(result["model"]).__name__,
    }


# -----------------------------------------------------------------------------
# Champion comparison
# -----------------------------------------------------------------------------
def get_metric_value_from_run(client: MlflowClient, run_id: str, metric_name: str) -> float:
    run = client.get_run(run_id)
    value = run.data.metrics.get(metric_name)
    if value is None:
        return float("-inf")
    return float(value)


def should_promote_over_champion(
    client: MlflowClient,
    model_name: str,
    candidate_metric_value: float,
    metric_name: str = PROMOTION_METRIC,
) -> bool:
    """
    Compare the candidate against the current registry champion.
    If no champion exists yet, promote.
    """
    try:
        champion_mv = client.get_model_version_by_alias(model_name, "champion")
    except Exception:
        return True

    if not champion_mv or not champion_mv.run_id:
        return True

    current_value = get_metric_value_from_run(client, champion_mv.run_id, metric_name)
    return candidate_metric_value > (current_value + MIN_PROMOTION_IMPROVEMENT)


# -----------------------------------------------------------------------------
# Local sync
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# MLflow / DagsHub logging
# -----------------------------------------------------------------------------
def log_result_to_mlflow(result: dict, run_id: str) -> dict:
    """
    Log params, metrics, metadata, MLflow model, and custom bundle artifact.
    """
    symbol = result["symbol"]

    mlflow.log_params(
        {
            "symbol": symbol,
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.05,
            "horizon_days": LOOKAHEAD_DAYS,
            "n_features": len(result["features"]),
            "train_size": len(result["X_train"]),
            "val_size": len(result["X_val"]),
            "promotion_metric": PROMOTION_METRIC,
        }
    )

    mlflow.log_metrics(
        {
            "accuracy": result["accuracy"],
            "precision": result["precision"],
            "recall": result["recall"],
            "f1": result["f1"],
            "baseline": result["baseline"],
            "improvement": result["improvement"],
        }
    )

    for feat, imp in result["importance"].head(25).items():
        mlflow.log_metric(f"importance_{feat}", float(imp))

    metadata = {
        "symbol": symbol,
        "features": result["features"],
        "metrics": {
            "accuracy": result["accuracy"],
            "precision": result["precision"],
            "recall": result["recall"],
            "f1": result["f1"],
            "baseline": result["baseline"],
            "improvement": result["improvement"],
        },
        "trained_at_utc": result["trained_at_utc"],
        "model_type": type(result["model"]).__name__,
        "backend_dir": str(BACKEND_DIR),
    }

    tmp_meta = BACKEND_DIR / f"{symbol.lower()}_metadata.json"
    with open(tmp_meta, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    try:
        mlflow.log_artifact(str(tmp_meta), artifact_path=f"{symbol}-metadata")
        print(f"[OK] Logged metadata artifact for {symbol}")
    except Exception:
        print(f"[FAIL] Could not log metadata artifact for {symbol}")
        traceback.print_exc()
        raise
    finally:
        tmp_meta.unlink(missing_ok=True)

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(
            {
                "symbol": symbol,
                "status": "artifact smoke test ok",
                "trained_at_utc": result["trained_at_utc"],
            },
            f,
            indent=2,
        )
        smoke_path = f.name

    try:
        mlflow.log_artifact(smoke_path, artifact_path=f"{symbol}-smoke")
        print(f"[OK] Logged smoke artifact for {symbol}")
    except Exception:
        print(f"[FAIL] Could not log smoke artifact for {symbol}")
        traceback.print_exc()
        raise
    finally:
        Path(smoke_path).unlink(missing_ok=True)

    x_example = result["X_val"].head(5).copy()
    for col in x_example.columns:
        if pd.api.types.is_integer_dtype(x_example[col]):
            x_example[col] = x_example[col].astype("float64")

    y_example = result["model"].predict(x_example)
    signature = infer_signature(x_example, y_example)

    try:
        model_info = mlflow.xgboost.log_model(
            xgb_model=result["model"],
            artifact_path=f"{symbol}-model",
            input_example=x_example,
            signature=signature,
            pip_requirements=[
                f"xgboost=={xgb.__version__}",
                f"mlflow=={mlflow.__version__}",
                f"pandas=={pd.__version__}",
                f"numpy=={np.__version__}",
                "scikit-learn",
                "joblib",
            ],
        )
        print(f"[OK] Logged MLflow model for {symbol}: {model_info.model_uri}")
    except Exception:
        print(f"[FAIL] Could not log MLflow model for {symbol}")
        traceback.print_exc()
        raise

    # Log the custom serving bundle to DagsHub only
    try:
        bundle = make_model_bundle(result, run_id)
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / f"{symbol.lower()}_model.pkl"
            joblib.dump(bundle, bundle_path)
            mlflow.log_artifact(str(bundle_path), artifact_path=f"{symbol}-bundle")
            print(
                f"[OK] Logged custom bundle artifact for {symbol}: "
                f"{symbol}-bundle/{bundle_path.name}"
            )
    except Exception:
        print(f"[FAIL] Could not log custom bundle artifact for {symbol}")
        traceback.print_exc()
        raise

    return {
        "bundle_artifact_path": f"{symbol}-bundle",
        "model_artifact_path": f"{symbol}-model",
    }


def register_logged_model(run_id: str, symbol: str, result: dict) -> dict:
    """
    Register every candidate version.
    Move alias 'champion' only if candidate is better than current champion.
    """
    if not REGISTER_MODELS:
        return {
            "registered": False,
            "promoted_to_champion": False,
            "model_name": None,
            "version": None,
        }

    client = MlflowClient()
    model_name = f"{symbol.lower()}-direction-{LOOKAHEAD_DAYS}d"
    artifact_path = f"{symbol}-model"

    try:
        client.create_registered_model(model_name)
    except Exception:
        pass

    mv = client.create_model_version(
        name=model_name,
        source=f"runs:/{run_id}/{artifact_path}",
        run_id=run_id,
    )

    candidate_metric_value = float(result[PROMOTION_METRIC])

    promoted_to_champion = should_promote_over_champion(
        client=client,
        model_name=model_name,
        candidate_metric_value=candidate_metric_value,
        metric_name=PROMOTION_METRIC,
    )

    if promoted_to_champion:
        client.set_registered_model_alias(model_name, "champion", mv.version)
        print(
            f"[OK] Promoted {model_name} version={mv.version} to @champion "
            f"based on {PROMOTION_METRIC}={candidate_metric_value:.4f}"
        )
    else:
        print(
            f"[OK] Registered {model_name} version={mv.version}, "
            f"but did not move @champion "
            f"(candidate {PROMOTION_METRIC}={candidate_metric_value:.4f} was not better)"
        )

    return {
        "registered": True,
        "promoted_to_champion": promoted_to_champion,
        "model_name": model_name,
        "version": str(mv.version),
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
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
        with mlflow.start_run(run_name=f"{symbol}-direction-{LOOKAHEAD_DAYS}d") as run:
            result = train_model(symbol)
            results[symbol] = result

            artifact_info = log_result_to_mlflow(
                result=result,
                run_id=run.info.run_id,
            )

            reg_info = register_logged_model(
                run_id=run.info.run_id,
                symbol=symbol,
                result=result,
            )

            mlflow.log_params(
                {
                    "bundle_artifact_path": artifact_info["bundle_artifact_path"],
                    "model_artifact_path": artifact_info["model_artifact_path"],
                    "registered": str(reg_info["registered"]).lower(),
                    "promoted_to_champion": str(reg_info["promoted_to_champion"]).lower(),
                    "registered_model_name": reg_info["model_name"] or "",
                    "registered_model_version": reg_info["version"] or "",
                }
            )

            if reg_info["promoted_to_champion"] and SYNC_CHAMPION_TO_LOCAL:
                sync_info = sync_model_bundle_from_dagshub(
                    symbol=symbol,
                    local_dir=LOCAL_MODEL_SYNC_DIR,
                    tracking_uri=os.environ["MLFLOW_TRACKING_URI"],
                    username=os.environ["MLFLOW_TRACKING_USERNAME"],
                    password=os.environ["MLFLOW_TRACKING_PASSWORD"],
                    alias="champion",
                )
                print(f"[OK] Synced new champion locally for {symbol}: {sync_info}")

    print(f"\n{'=' * 50}")
    print("TRAINING COMPLETE")
    print(f"{'=' * 50}")
    for symbol, result in results.items():
        improvement = (result["accuracy"] - result["baseline"]) * 100
        print(
            f"{symbol}: {result['accuracy'] * 100:.1f}% accuracy "
            f"({improvement:+.1f}% vs baseline)"
        )

    return results


if __name__ == "__main__":
    main()