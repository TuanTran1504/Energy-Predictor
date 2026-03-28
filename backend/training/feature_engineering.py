"""
Feature engineering for BTC/ETH 24h direction prediction.

For each day in crypto_prices, computes:
  - Price momentum features (3, 7, 14 day returns)
  - Volume trend
  - Brent crude signal
  - USD/VND signal  
  - Shock event proximity
  - Macro event proximity
  - Calendar features

Target: 1 if next day close > today close, 0 otherwise (UP/DOWN)
"""
import pandas as pd
import numpy as np
import psycopg2
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent.parent.parent / ".env")
DATABASE_URL = os.getenv("DATABASE_URL")


def load_crypto_prices(symbol: str) -> pd.DataFrame:
    conn = psycopg2.connect(DATABASE_URL)
    df = pd.read_sql("""
        SELECT 
            DATE(fetched_at) as date,
            open_usd, high_usd, low_usd, 
            close_usd, volume_usd, change_pct
        FROM crypto_prices
        WHERE symbol = %s
        ORDER BY fetched_at ASC
    """, conn, params=(symbol,))
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_vnd_rates() -> pd.DataFrame:
    conn = psycopg2.connect(DATABASE_URL)
    df = pd.read_sql("""
        SELECT 
            DATE(fetched_at) as date,
            usd_to_vnd
        FROM vnd_rates
        ORDER BY fetched_at ASC
    """, conn)
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    # One row per day — take last reading per day
    df = df.groupby("date")["usd_to_vnd"].last().reset_index()
    return df


def load_shock_events() -> pd.DataFrame:
    conn = psycopg2.connect(DATABASE_URL)
    df = pd.read_sql("""
        SELECT 
            DATE(event_date) as date,
            severity,
            oil_impact,
            peak_score,
            btc_impact_24h,
            eth_impact_24h
        FROM shock_events
        WHERE verified = TRUE
        ORDER BY event_date ASC
    """, conn)
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    severity_map = {"LOW": 1, "ELEVATED": 2, "HIGH": 3}
    df["severity_score"] = df["severity"].map(severity_map).fillna(0)
    return df


def load_macro_events() -> pd.DataFrame:
    conn = psycopg2.connect(DATABASE_URL)
    df = pd.read_sql("""
        SELECT
            DATE(event_date) as date,
            event_type,
            actual,
            btc_impact_24h,
            eth_impact_24h
        FROM macro_events
        WHERE verified = TRUE
        ORDER BY event_date ASC
    """, conn)
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    return df


def compute_shock_features(date: pd.Timestamp,
                           shock_df: pd.DataFrame) -> dict:
    """
    For a given date, look back 30 days for shock events.
    Returns features capturing shock regime state.
    """
    window_start = date - timedelta(days=30)
    recent = shock_df[
        (shock_df["date"] <= date) &
        (shock_df["date"] >= window_start)
    ]

    if recent.empty:
        return {
            "shock_active":        0,
            "days_since_shock":    30,
            "max_shock_severity":  0,
            "shock_oil_impact":    0.0,
            "shock_peak_score":    0.0,
        }

    latest = recent.iloc[-1]
    days_since = (date - latest["date"]).days

    return {
        "shock_active":       1,
        "days_since_shock":   days_since,
        "max_shock_severity": int(recent["severity_score"].max()),
        "shock_oil_impact":   float(recent["oil_impact"].sum()),
    }


def compute_macro_features(date: pd.Timestamp,
                            macro_df: pd.DataFrame) -> dict:
    """
    For a given date, look back 7 days for macro events.
    Fed decisions and CPI prints strongly move crypto.
    """
    window_start = date - timedelta(days=7)
    recent = macro_df[
        (macro_df["date"] <= date) &
        (macro_df["date"] >= window_start)
    ]

    fed_recent = recent[recent["event_type"] == "FED_RATE"]
    cpi_recent = recent[recent["event_type"] == "CPI"]
    nfp_recent = recent[recent["event_type"] == "NFP"]

    # Was the Fed dovish (cut) or hawkish (hike)?
    fed_dovish = 0
    fed_hawkish = 0
    if not fed_recent.empty:
        actual = str(fed_recent.iloc[-1]["actual"] or "")
        if "HOLD" in actual.upper():
            pass  # neutral
        elif "-" in actual:
            fed_dovish = 1  # rate cut = bullish crypto
        elif "+" in actual:
            fed_hawkish = 1  # rate hike = bearish crypto

    return {
        "fed_event_last_7d":  int(len(fed_recent) > 0),
        "fed_dovish":         fed_dovish,
        "fed_hawkish":        fed_hawkish,
        "cpi_event_last_7d":  int(len(cpi_recent) > 0),
        "nfp_event_last_7d":  int(len(nfp_recent) > 0),
    }


def build_features(symbol: str) -> pd.DataFrame:
    """
    Builds the complete feature matrix for one symbol (BTC or ETH).
    Each row = one trading day.
    Target = 1 if next day close > today close, 0 otherwise.
    """
    print(f"Loading data for {symbol}...")
    crypto_df = load_crypto_prices(symbol)
    vnd_df    = load_vnd_rates()
    shock_df  = load_shock_events()

    # Load macro events safely
    try:
        macro_df = load_macro_events()
    except Exception as e:
        print(f"Warning: macro events load failed: {e}")
        macro_df = pd.DataFrame(columns=["date", "event_type", "actual"])

    print(f"  Crypto rows: {len(crypto_df)}")
    print(f"  VND rows:    {len(vnd_df)}")
    print(f"  Shock rows:  {len(shock_df)}")
    print(f"  Macro rows:  {len(macro_df)}")

    # Merge VND rates into crypto df
    crypto_df = crypto_df.merge(vnd_df, on="date", how="left")
    crypto_df["usd_to_vnd"] = crypto_df["usd_to_vnd"].ffill()

    rows = []

    for i in range(14, len(crypto_df) - 1):
        today    = crypto_df.iloc[i]
        tomorrow = crypto_df.iloc[i + 1]
        date     = today["date"]

        # ── Price momentum features ────────────────────────────────────────
        close_today = float(today["close_usd"])

        # Returns over different windows
        ret_1d  = float(today["change_pct"]) / 100
        ret_3d  = (close_today / float(crypto_df.iloc[i-2]["close_usd"]) - 1)
        ret_7d  = (close_today / float(crypto_df.iloc[i-6]["close_usd"]) - 1)
        ret_14d = (close_today / float(crypto_df.iloc[i-13]["close_usd"]) - 1)

        # High-low range (volatility proxy)
        hl_range = (float(today["high_usd"]) - float(today["low_usd"])) / close_today

        # Volume trend (is volume increasing?)
        vol_today = float(today["volume_usd"]) if today["volume_usd"] else 0
        vol_7d_avg = crypto_df.iloc[i-6:i]["volume_usd"].mean()
        vol_trend = (vol_today / vol_7d_avg - 1) if vol_7d_avg > 0 else 0

        # Price position within recent range
        high_14d = crypto_df.iloc[i-13:i+1]["high_usd"].max()
        low_14d  = crypto_df.iloc[i-13:i+1]["low_usd"].min()
        price_position = ((close_today - low_14d) / (high_14d - low_14d)
                          if high_14d > low_14d else 0.5)

        # ── VND features ──────────────────────────────────────────────────
        vnd_today = float(today["usd_to_vnd"]) if pd.notna(today["usd_to_vnd"]) else 25000
        vnd_7d_ago = float(crypto_df.iloc[i-6]["usd_to_vnd"]) if pd.notna(crypto_df.iloc[i-6]["usd_to_vnd"]) else vnd_today
        vnd_change_7d = (vnd_today - vnd_7d_ago) / vnd_7d_ago if vnd_7d_ago > 0 else 0

        # ── Event features ────────────────────────────────────────────────
        shock_feats = compute_shock_features(date, shock_df)
        macro_feats = compute_macro_features(date, macro_df)

        # ── Calendar features ─────────────────────────────────────────────
        calendar = {
            "day_of_week": date.dayofweek,   # 0=Mon, 6=Sun
            "month":       date.month,
            "is_weekend":  int(date.dayofweek >= 5),
            "is_month_end": int(date.day >= 28),
        }

        # ── Target ────────────────────────────────────────────────────────
        # 1 = UP (next close > today close)
        # 0 = DOWN or flat
        target = int(float(tomorrow["close_usd"]) > close_today)

        row = {
            "date":           date,
            "symbol":         symbol,
            "close_usd":      close_today,

            # Price features
            "ret_1d":         ret_1d,
            "ret_3d":         ret_3d,
            "ret_7d":         ret_7d,
            "ret_14d":        ret_14d,
            "hl_range":       hl_range,
            "vol_trend":      vol_trend,
            "price_position": price_position,

            # VND features
            "vnd_rate":       vnd_today,
            "vnd_change_7d":  vnd_change_7d,

            # Shock features
            **shock_feats,

            # Macro features
            **macro_feats,

            # Calendar features
            **calendar,

            # Target
            "target": target,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"  Feature matrix: {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"  Target distribution: {df['target'].value_counts().to_dict()}")
    return df


def get_feature_columns() -> list:
    return [
        "ret_1d", "ret_3d", "ret_7d", "ret_14d",
        "hl_range", "vol_trend", "price_position",
        "vnd_rate", "vnd_change_7d",
        "shock_active", "days_since_shock",
        "max_shock_severity", "shock_oil_impact",
        "fed_event_last_7d", "fed_dovish", "fed_hawkish",
        "cpi_event_last_7d", "nfp_event_last_7d",
        "day_of_week", "month", "is_weekend", "is_month_end",
    ]


if __name__ == "__main__":
    for symbol in ["BTC", "ETH"]:
        print(f"\n{'='*50}")
        df = build_features(symbol)
        print(f"\nSample features (last 3 rows):")
        print(df[get_feature_columns() + ["target"]].tail(3).to_string())
        print(f"\nFeature correlations with target:")
        correlations = df[get_feature_columns() + ["target"]].corr()["target"].drop("target")
        print(correlations.sort_values(ascending=False).to_string())