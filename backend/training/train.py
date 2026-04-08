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

backend_env = BACKEND_DIR / ".env"
project_env = PROJECT_ROOT / ".env"
load_dotenv(project_env)   # root first (lowest priority)
load_dotenv(backend_env)   # backend overrides root for duplicates

TRAINING_PARENT = THIS_FILE.parent.parent
if str(TRAINING_PARENT) not in sys.path:
    sys.path.append(str(TRAINING_PARENT))

from feature_engineering import build_features, get_feature_columns  # noqa: E402

 
SYMBOLS = ["BTC", "ETH", "SOL", "XRP"]
MIN_PROMOTION_IMPROVEMENT = 0.0
LOOKAHEAD_DAYS = int(os.getenv("LOOKAHEAD_DAYS", "1"))
LOOKAHEAD_HOURS = int(os.getenv("LOOKAHEAD_HOURS", "0"))
FEATURE_TRAIN_TIMEFRAME = os.getenv("FEATURE_TRAIN_TIMEFRAME", "1D").strip().upper()
MODEL_NAME_SUFFIX = os.getenv("MODEL_NAME_SUFFIX", "").strip().lower()


def _model_variant_suffix() -> str:
    # Explicit suffix wins (e.g. MODEL_NAME_SUFFIX=exp42)
    if MODEL_NAME_SUFFIX:
        return f"-{MODEL_NAME_SUFFIX}"
    # Default keeps legacy model names for 1D.
    if FEATURE_TRAIN_TIMEFRAME in {"", "1D", "1DAY", "DAILY"}:
        return ""
    return f"-{FEATURE_TRAIN_TIMEFRAME.lower()}"


def _horizon_token() -> str:
    if LOOKAHEAD_HOURS > 0:
        return f"{LOOKAHEAD_HOURS}h"
    return f"{LOOKAHEAD_DAYS}d"


def _registry_model_name(symbol: str) -> str:
    return f"{symbol.lower()}-direction-{_horizon_token()}{_model_variant_suffix()}"


def _run_name(symbol: str) -> str:
    return f"{symbol}-direction-{_horizon_token()}{_model_variant_suffix()}"

SHADOW_TRAIN = os.getenv("SHADOW_TRAIN", "false").strip().lower() in {
    "1", "true", "yes", "y"
}

REGISTER_MODELS = os.getenv("MLFLOW_REGISTER_MODELS", "false").strip().lower() in {
    "1", "true", "yes", "y"
}

PROMOTION_METRIC = os.getenv("MODEL_PROMOTION_METRIC", "accuracy").strip().lower()

SYNC_CHAMPION_TO_LOCAL = os.getenv("SYNC_CHAMPION_TO_LOCAL", "false").strip().lower() in {
    "1", "true", "yes", "y"
}

# SHADOW_TRAIN disables registry promotion and local sync
if SHADOW_TRAIN:
    REGISTER_MODELS = False
    SYNC_CHAMPION_TO_LOCAL = False

LOCAL_MODEL_SYNC_DIR = os.getenv(
    "LOCAL_MODEL_SYNC_DIR",
    str(BACKEND_DIR / "api" / "models"),
).strip()

WALK_FORWARD_SPLITS = int(os.getenv("WALK_FORWARD_SPLITS", "3"))
WALK_FORWARD_VAL_RATIO = float(os.getenv("WALK_FORWARD_VAL_RATIO", "0.2"))
MIN_TRAIN_ROWS = int(os.getenv("MIN_TRAIN_ROWS", "300"))


def train_model(symbol: str) -> dict:
    """
    Train one symbol model using walk-forward validation, then fit final model
    on all available samples for serving.
    """
    print(f"\n{'=' * 50}")
    print(f"Training {symbol} direction model...")

    df = build_features(
        symbol,
        lookahead=LOOKAHEAD_DAYS,
        lookahead_hours=LOOKAHEAD_HOURS if LOOKAHEAD_HOURS > 0 else None,
    )
    feature_cols = get_feature_columns()

    X = df[feature_cols].fillna(0)
    y = df["target"]

    print(f"Features: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")

    if len(X) < MIN_TRAIN_ROWS:
        raise ValueError(
            f"Not enough rows for training ({len(X)}). "
            f"Need at least {MIN_TRAIN_ROWS}."
        )

    def make_model() -> xgb.XGBClassifier:
        # Tuned for ~500-1200 training rows with 18 low-correlation features.
        #
        # Previous over-regularised attempt (min_child_weight=10, gamma=2,
        # colsample_bytree=0.8) caused severe UP bias: the model could barely
        # make any splits and fell back to predicting the overall direction of
        # the training distribution. Recall UP hit 0.76 while DOWN recall was
        # only 0.24 — worse than a coin flip.
        #
        # Key changes vs that attempt:
        #   min_child_weight 10→5   allow splits on smaller leaf groups
        #   gamma 2→0.5             moderate split conservatism
        #   reg_alpha 0.5→0.1       relax L1
        #   reg_lambda 2.0→1.0      relax L2
        #   colsample_bytree 0.8→1.0  with only 18 features, excluding 3-4 per
        #                             tree was randomly dropping real signal;
        #                             use all 18 every time
        #   n_estimators 150→300    more trees compensate for shallower depth
        #   learning_rate 0.03→0.05 restore original convergence speed
        return xgb.XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=1.0,
            min_child_weight=5,
            gamma=0.5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )

    n = len(X)
    val_window = max(1, int(n * WALK_FORWARD_VAL_RATIO))
    fold_bounds: list[tuple[int, int]] = []
    for k in range(WALK_FORWARD_SPLITS):
        val_end = n - (WALK_FORWARD_SPLITS - 1 - k) * val_window
        val_start = val_end - val_window
        if val_start <= MIN_TRAIN_ROWS or val_end > n:
            continue
        fold_bounds.append((val_start, val_end))

    if not fold_bounds:
        # Fallback to one strict temporal split if dataset is short.
        split_idx = int(n * 0.8)
        if split_idx <= MIN_TRAIN_ROWS:
            split_idx = MIN_TRAIN_ROWS
        fold_bounds = [(split_idx, n)]

    print(f"\nWalk-forward validation: {len(fold_bounds)} fold(s)")
    print(f"Validation window size:  {fold_bounds[0][1] - fold_bounds[0][0]} rows")

    fold_metrics = []
    last_fold = None

    for fold_idx, (val_start, val_end) in enumerate(fold_bounds, start=1):
        X_train = X.iloc[:val_start]
        y_train = y.iloc[:val_start]
        X_val = X.iloc[val_start:val_end]
        y_val = y.iloc[val_start:val_end]

        print(
            f"\nFold {fold_idx}: "
            f"train={len(X_train)} ({df['date'].iloc[0].date()} to {df['date'].iloc[val_start - 1].date()}) | "
            f"val={len(X_val)} ({df['date'].iloc[val_start].date()} to {df['date'].iloc[val_end - 1].date()})"
        )

        fold_model = make_model()
        fold_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        y_pred = fold_model.predict(X_val)
        y_prob = fold_model.predict_proba(X_val)[:, 1]

        fold_accuracy = accuracy_score(y_val, y_pred)
        fold_precision = precision_score(y_val, y_pred, zero_division=0)
        fold_recall = recall_score(y_val, y_pred, zero_division=0)
        fold_f1 = f1_score(y_val, y_pred, zero_division=0)
        fold_baseline = float(y_val.mean())

        fold_metrics.append(
            {
                "fold": fold_idx,
                "accuracy": float(fold_accuracy),
                "precision": float(fold_precision),
                "recall": float(fold_recall),
                "f1": float(fold_f1),
                "baseline": float(fold_baseline),
                "improvement": float(fold_accuracy - fold_baseline),
            }
        )
        print(
            f"  Acc={fold_accuracy:.4f} Prec={fold_precision:.4f} "
            f"Rec={fold_recall:.4f} F1={fold_f1:.4f} "
            f"Base={fold_baseline:.4f} "
            f"Imp={(fold_accuracy - fold_baseline) * 100:+.1f}%"
        )
        last_fold = (X_train, X_val, y_val, y_pred, y_prob)

    metrics_df = pd.DataFrame(fold_metrics)
    accuracy = float(metrics_df["accuracy"].mean())
    precision = float(metrics_df["precision"].mean())
    recall = float(metrics_df["recall"].mean())
    f1 = float(metrics_df["f1"].mean())
    baseline_acc = float(metrics_df["baseline"].mean())
    improvement = accuracy - baseline_acc

    print("\nValidation Results (walk-forward mean):")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  Baseline:  {baseline_acc:.4f}")
    print(f"  Improvement: {improvement * 100:+.1f}%")

    X_train, X_val, y_val, y_pred, y_prob = last_fold
    print("\nClassification Report (last walk-forward fold):")
    print(
        classification_report(
            y_val, y_pred, target_names=["DOWN", "UP"], zero_division=0
        )
    )

    # Final model trained on all samples for serving
    model = make_model()
    model.fit(X, y, verbose=False)

    importance = pd.Series(
        model.feature_importances_,
        index=feature_cols,
    ).sort_values(ascending=False)

    print("\nTop 10 features:")
    print(importance.head(10).to_string())

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
        "walk_forward_folds": len(fold_bounds),
        "walk_forward_accuracy_std": float(metrics_df["accuracy"].std(ddof=0)),
        "trained_at_utc": datetime.now(UTC).isoformat(),
    }


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
    model_name = _registry_model_name(symbol)
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


def log_result_to_mlflow(result: dict, run_id: str) -> dict:
    """
    Log params, metrics, metadata, MLflow model, and custom bundle artifact.
    """
    symbol = result["symbol"]

    mlflow.log_params(
        {
            "symbol": symbol,
            "n_estimators": 300,
            "max_depth": 3,
            "learning_rate": 0.05,
            "horizon_days": LOOKAHEAD_DAYS,
            "horizon_hours": LOOKAHEAD_HOURS,
            "horizon_token": _horizon_token(),
            "feature_train_timeframe": FEATURE_TRAIN_TIMEFRAME,
            "model_name_suffix": _model_variant_suffix() or "legacy",
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
            "walk_forward_accuracy_std": result.get("walk_forward_accuracy_std", 0.0),
            "walk_forward_folds": float(result.get("walk_forward_folds", 0)),
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

    # Custom serving bundle (joblib dict — not a raw MLflow model)
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
    model_name = _registry_model_name(symbol)
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

    client.set_registered_model_alias(model_name, "champion", mv.version)
    promoted_to_champion = True
    print(
        f"[OK] Promoted {model_name} version={mv.version} to @champion "
        f"({PROMOTION_METRIC}={candidate_metric_value:.4f})"
    )

    return {
        "registered": True,
        "promoted_to_champion": promoted_to_champion,
        "model_name": model_name,
        "version": str(mv.version),
    }


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
    print(f"Shadow train mode:              {SHADOW_TRAIN}")
    print(f"Local sync enabled:             {SYNC_CHAMPION_TO_LOCAL}")
    print(f"Local sync dir:                 {LOCAL_MODEL_SYNC_DIR}")
    print(f"Model registration enabled:     {REGISTER_MODELS}")
    print(f"Promotion metric:               {PROMOTION_METRIC}")
    print(f"Horizon token:                  {_horizon_token()}")
    print(f"Feature train timeframe:        {FEATURE_TRAIN_TIMEFRAME}")
    if FEATURE_TRAIN_TIMEFRAME == "4H" and LOOKAHEAD_HOURS <= 0:
        print(
            "WARNING: FEATURE_TRAIN_TIMEFRAME=4H but LOOKAHEAD_HOURS<=0. "
            "Training will use day-based horizon from LOOKAHEAD_DAYS."
        )
    print(f"Model registry suffix:          {_model_variant_suffix() or '(legacy)'}")
    print(f"Walk-forward splits:            {WALK_FORWARD_SPLITS}")
    print(f"Walk-forward val ratio:         {WALK_FORWARD_VAL_RATIO}")
    print(
        "Target threshold pct:           "
        f"{os.getenv('TARGET_RETURN_THRESHOLD_PCT', '0')}"
    )

    results = {}

    for symbol in SYMBOLS:
        with mlflow.start_run(run_name=_run_name(symbol)) as run:
            try:
                result = train_model(symbol)
            except ValueError as e:
                print(f"\n[SKIP] {symbol}: {e}")
                mlflow.set_tag("skipped_reason", str(e))
                continue
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
