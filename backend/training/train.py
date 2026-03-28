"""
Trains XGBoost BTC/ETH direction classifier.
Predicts: will price be UP or DOWN in 24 hours?

Run locally:
    python ml/training/train.py

Run on Modal:
    modal run backend/ml/training/modal_train.py
"""
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

from feature_engineering import build_features, get_feature_columns
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")

import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.xgboost


def train_model(symbol: str) -> dict:
    """
    Trains XGBoost classifier for one symbol.
    Uses time-series cross validation — no data leakage.
    Returns metrics dict.
    """
    print(f"\n{'='*50}")
    print(f"Training {symbol} direction model...")

    # ── Load features ──────────────────────────────────────────────────────
    df = build_features(symbol)
    feature_cols = get_feature_columns()

    X = df[feature_cols].fillna(0)
    y = df["target"]

    print(f"Features: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")

    # ── Time series split — no data leakage ────────────────────────────────
    # Always train on past, validate on future
    split_idx = int(len(X) * 0.8)
    X_train = X.iloc[:split_idx]
    X_val   = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_val   = y.iloc[split_idx:]

    print(f"Train: {len(X_train)} rows ({X_train.index[0]} to {X_train.index[-1]})")
    print(f"Val:   {len(X_val)} rows ({X_val.index[0]} to {X_val.index[-1]})")

    # ── Train XGBoost ──────────────────────────────────────────────────────
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
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # ── Evaluate ───────────────────────────────────────────────────────────
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    accuracy = accuracy_score(y_val, y_pred)

    print(f"\nValidation Results:")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"\nClassification Report:")
    print(classification_report(y_val, y_pred,
                                target_names=["DOWN", "UP"]))

    # Feature importance
    importance = pd.Series(
        model.feature_importances_,
        index=feature_cols
    ).sort_values(ascending=False)

    print(f"\nTop 10 features:")
    print(importance.head(10).to_string())

    # ── Baseline comparison ────────────────────────────────────────────────
    # Baseline = always predict UP (crypto has upward bias)
    baseline_acc = y_val.mean()
    print(f"\nBaseline (always UP): {baseline_acc:.4f}")
    print(f"Model improvement: {(accuracy - baseline_acc)*100:+.1f}%")

    return {
        "symbol":   symbol,
        "model":    model,
        "accuracy": accuracy,
        "baseline": float(baseline_acc),
        "features": feature_cols,
        "X_val":    X_val,
        "y_val":    y_val,
        "y_pred":   y_pred,
        "importance": importance,
    }


def save_model(result: dict, run_id: str):
    """Saves model to disk for serving."""
    import joblib
    symbol = result["symbol"]
    model_dir = Path(__file__).parent.parent.parent / "models"
    model_dir.mkdir(exist_ok=True)

    model_path = model_dir / f"{symbol.lower()}_model.pkl"
    joblib.dump(result["model"], model_path)
    print(f"Model saved to {model_path}")
    return str(model_path)


def main():
    # ── MLflow setup ───────────────────────────────────────────────────────
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "")
    if mlflow_uri:
        mlflow.set_tracking_uri(mlflow_uri)
        os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv(
            "MLFLOW_TRACKING_USERNAME", ""
        )
        os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv(
            "MLFLOW_TRACKING_PASSWORD", ""
        )

    mlflow.set_experiment("trading-signals")

    results = {}

    for symbol in ["BTC", "ETH"]:
        with mlflow.start_run(run_name=f"{symbol}-direction-24h"):
            result = train_model(symbol)
            results[symbol] = result

            # Log to MLflow
            mlflow.log_params({
                "symbol":       symbol,
                "n_estimators": 200,
                "max_depth":    4,
                "learning_rate": 0.05,
                "horizon_hrs":  24,
                "n_features":   len(result["features"]),
                "train_size":   len(result["X_val"]),
            })

            mlflow.log_metrics({
                "accuracy":    result["accuracy"],
                "baseline":    result["baseline"],
                "improvement": result["accuracy"] - result["baseline"],
            })

            # Log feature importances
            for feat, imp in result["importance"].items():
                mlflow.log_metric(f"importance_{feat}", float(imp))

            # Save model
            mlflow.xgboost.log_model(result["model"], f"{symbol}-model")

            # Save locally too
            save_model(result, mlflow.active_run().info.run_id)

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print("TRAINING COMPLETE")
    print(f"{'='*50}")
    for symbol, result in results.items():
        improvement = (result["accuracy"] - result["baseline"]) * 100
        print(f"{symbol}: {result['accuracy']*100:.1f}% accuracy "
              f"({improvement:+.1f}% vs baseline)")

    return results


if __name__ == "__main__":
    main()