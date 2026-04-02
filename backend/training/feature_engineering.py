"""
Feature engineering for BTC/ETH 24h direction prediction.

For each day in crypto_prices, computes:
  - Price momentum features (1, 3, 7, 14 day returns)
  - Volatility / volume / price position
  - Shock event proximity
  - Macro event proximity (Fed, CPI, NFP)
  - Fear & Greed index (sentiment)
  - Funding rate (derivatives market positioning)
  - BTC/ETH ratio change (dominance proxy)
  - Calendar features

Target: 1 if next day close > today close, 0 otherwise (UP/DOWN)
"""
import json
import os
import urllib.request
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")
DATABASE_URL = os.getenv("DATABASE_URL")


@contextmanager
def get_conn():
    conn = psycopg2.connect(DATABASE_URL)
    try:
        yield conn
    finally:
        conn.close()


def load_crypto_prices(symbol: str) -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql("""
            SELECT
                DATE(fetched_at) as date,
                open_usd, high_usd, low_usd,
                close_usd, volume_usd, change_pct
            FROM crypto_prices
            WHERE symbol = %s
            ORDER BY fetched_at ASC
        """, conn, params=(symbol,))
    df["date"] = pd.to_datetime(df["date"])
    # Aggregate 30-min candles to daily OHLCV correctly:
    # open=first, high=max, low=min, close=last, volume=sum
    df = df.groupby("date").agg(
        open_usd   = ("open_usd",   "first"),
        high_usd   = ("high_usd",   "max"),
        low_usd    = ("low_usd",    "min"),
        close_usd  = ("close_usd",  "last"),
        volume_usd = ("volume_usd", "sum"),
    ).reset_index()
    df["change_pct"] = (
        (df["close_usd"] - df["open_usd"]) / df["open_usd"] * 100
    ).round(4)
    return df


def load_shock_events() -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql("""
            SELECT
                DATE(event_date) as date,
                severity,
                oil_impact,
                peak_score
            FROM shock_events
            ORDER BY event_date ASC
        """, conn)
    df["date"] = pd.to_datetime(df["date"])
    severity_map = {"LOW": 1, "ELEVATED": 2, "HIGH": 3}
    df["severity_score"] = df["severity"].map(severity_map).fillna(0)
    return df


def load_macro_events() -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql("""
            SELECT
                DATE(event_date) as date,
                event_type,
                actual
            FROM macro_events
            ORDER BY event_date ASC
        """, conn)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_fear_greed() -> pd.DataFrame:
    """
    Load Fear & Greed index from DB (populated by schedule_fear_greed.py).
    Falls back to live API fetch if the table is empty.
    Returns daily values normalised to 0-1.
    """
    try:
        with get_conn() as conn:
            df = pd.read_sql("""
                SELECT date, value AS fear_greed
                FROM fear_greed_index
                ORDER BY date ASC
            """, conn)
        df["date"] = pd.to_datetime(df["date"])
        df["fear_greed"] = df["fear_greed"] / 100.0
        if not df.empty:
            print(f"  Fear & Greed rows (DB): {len(df)}")
            return df
    except Exception as e:
        print(f"Warning: Fear & Greed DB read failed: {e}")

    # Fallback — hit the API directly if DB is empty
    print("  Fear & Greed DB empty — falling back to live API fetch")
    try:
        url = "https://api.alternative.me/fng/?limit=500&format=json"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())["data"]
        rows = [
            {
                "date":       pd.Timestamp(int(e["timestamp"]), unit="s").normalize(),
                "fear_greed": int(e["value"]) / 100.0,
            }
            for e in data
        ]
        df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
        print(f"  Fear & Greed rows (API fallback): {len(df)}")
        return df
    except Exception as e:
        print(f"Warning: Fear & Greed API fallback failed: {e} — using neutral defaults")
        return pd.DataFrame(columns=["date", "fear_greed"])


def load_funding_rates(symbol: str) -> pd.DataFrame:
    """
    Load funding rates from DB (populated by schedule_funding_rates.py).
    Falls back to live Binance API fetch if the table is empty.
    Returns daily average funding rates.
    """
    try:
        with get_conn() as conn:
            df = pd.read_sql("""
                SELECT date, rate_avg AS funding_rate_avg
                FROM funding_rates
                WHERE symbol = %s
                ORDER BY date ASC
            """, conn, params=(symbol,))
        df["date"] = pd.to_datetime(df["date"])
        if not df.empty:
            print(f"  Funding rate rows ({symbol}, DB): {len(df)}")
            return df
    except Exception as e:
        print(f"Warning: Funding rate DB read failed for {symbol}: {e}")

    # Fallback — hit Bybit directly if DB is empty (Binance blocks US servers)
    print(f"  Funding rate DB empty for {symbol} — falling back to Bybit API")
    try:
        bybit_sym = f"{symbol}USDT"
        url = (
            f"https://api.bybit.com/v5/market/funding/history"
            f"?category=linear&symbol={bybit_sym}&limit=1000"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            result = json.loads(r.read())
        entries = result.get("result", {}).get("list", [])
        rows = [
            {
                "date":         pd.Timestamp(int(e["fundingRateTimestamp"]), unit="ms").normalize(),
                "funding_rate": float(e["fundingRate"]),
            }
            for e in entries
        ]
        df = pd.DataFrame(rows)
        df = df.groupby("date")["funding_rate"].mean().reset_index()
        df.columns = ["date", "funding_rate_avg"]
        print(f"  Funding rate rows ({symbol}, Bybit fallback): {len(df)}")
        return df
    except Exception as e:
        print(f"Warning: Funding rate Bybit fallback failed for {symbol}: {e} — using neutral defaults")
        return pd.DataFrame(columns=["date", "funding_rate_avg"])


def compute_shock_features(date: pd.Timestamp, shock_df: pd.DataFrame) -> dict:
    window_start = date - timedelta(days=30)
    recent = shock_df[
        (shock_df["date"] <= date) &
        (shock_df["date"] >= window_start)
    ]

    if recent.empty:
        return {
            "shock_active":       0,
            "days_since_shock":   30,
            "max_shock_severity": 0,
            "shock_oil_impact":   0.0,
        }

    latest = recent.iloc[-1]
    days_since = (date - latest["date"]).days

    return {
        "shock_active":       1,
        "days_since_shock":   days_since,
        "max_shock_severity": int(recent["severity_score"].max()),
        "shock_oil_impact":   float(recent["oil_impact"].sum()),
    }


def compute_macro_features(date: pd.Timestamp, macro_df: pd.DataFrame) -> dict:
    window_start = date - timedelta(days=7)
    recent = macro_df[
        (macro_df["date"] <= date) &
        (macro_df["date"] >= window_start)
    ]

    fed_recent = recent[recent["event_type"] == "FED_RATE"]
    cpi_recent = recent[recent["event_type"] == "CPI"]
    nfp_recent = recent[recent["event_type"] == "NFP"]

    fed_dovish = 0
    fed_hawkish = 0
    if not fed_recent.empty:
        actual = str(fed_recent.iloc[-1]["actual"] or "")
        if "-" in actual:
            fed_dovish = 1
        elif "+" in actual:
            fed_hawkish = 1

    return {
        "fed_event_last_7d": int(len(fed_recent) > 0),
        "fed_dovish":        fed_dovish,
        "fed_hawkish":       fed_hawkish,
        "cpi_event_last_7d": int(len(cpi_recent) > 0),
        "nfp_event_last_7d": int(len(nfp_recent) > 0),
    }


def build_features(symbol: str, lookahead: int = 1) -> pd.DataFrame:
    """
    Builds the complete feature matrix for one symbol (BTC or ETH).
    Each row = one trading day.
    Target = 1 if close N days ahead > today close, 0 otherwise.
    lookahead=1 → next-day direction (default)
    lookahead=7 → 7-day direction
    """
    print(f"Loading data for {symbol}...")

    # ── Price data ─────────────────────────────────────────────────────────────
    crypto_df  = load_crypto_prices(symbol)
    other_sym  = "ETH" if symbol == "BTC" else "BTC"
    other_df   = load_crypto_prices(other_sym)[["date", "close_usd"]].rename(
        columns={"close_usd": "other_close_usd"}
    )
    crypto_df = crypto_df.merge(other_df, on="date", how="left")
    crypto_df["other_close_usd"] = crypto_df["other_close_usd"].ffill().fillna(1.0)

    # ── External signals ───────────────────────────────────────────────────────
    fear_greed_df = load_fear_greed()
    funding_df    = load_funding_rates(symbol)
    shock_df      = load_shock_events()

    try:
        macro_df = load_macro_events()
    except Exception as e:
        print(f"Warning: macro events load failed: {e}")
        macro_df = pd.DataFrame(columns=["date", "event_type", "actual"])

    # Merge external signals — forward-fill gaps (weekends, missing days)
    if not fear_greed_df.empty:
        crypto_df = crypto_df.merge(fear_greed_df, on="date", how="left")
        crypto_df["fear_greed"] = crypto_df["fear_greed"].ffill().fillna(0.5)
    else:
        crypto_df["fear_greed"] = 0.5

    if not funding_df.empty:
        crypto_df = crypto_df.merge(funding_df, on="date", how="left")
        crypto_df["funding_rate_avg"] = crypto_df["funding_rate_avg"].ffill().fillna(0.0)
    else:
        crypto_df["funding_rate_avg"] = 0.0

    print(f"  Crypto rows:   {len(crypto_df)}")
    print(f"  Shock rows:    {len(shock_df)}")
    print(f"  Macro rows:    {len(macro_df)}")

    rows = []

    for i in range(14, len(crypto_df) - lookahead):
        today    = crypto_df.iloc[i]
        future   = crypto_df.iloc[i + lookahead]
        date     = today["date"]

        close_today = float(today["close_usd"])

        # ── Price momentum ─────────────────────────────────────────────────────
        ret_1d  = float(today["change_pct"]) / 100
        ret_3d  = close_today / float(crypto_df.iloc[i - 3]["close_usd"]) - 1
        ret_7d  = close_today / float(crypto_df.iloc[i - 7]["close_usd"]) - 1
        ret_14d = close_today / float(crypto_df.iloc[i - 14]["close_usd"]) - 1

        # ── Volatility / volume / range ────────────────────────────────────────
        hl_range   = (float(today["high_usd"]) - float(today["low_usd"])) / close_today
        vol_today  = float(today["volume_usd"]) if today["volume_usd"] else 0
        vol_7d_avg = crypto_df.iloc[i - 7:i]["volume_usd"].astype(float).mean()
        vol_trend  = (vol_today / vol_7d_avg - 1) if vol_7d_avg > 0 else 0

        high_14d = float(crypto_df.iloc[i - 14:i + 1]["high_usd"].max())
        low_14d  = float(crypto_df.iloc[i - 14:i + 1]["low_usd"].min())
        price_position = (
            (close_today - low_14d) / (high_14d - low_14d)
            if high_14d > low_14d else 0.5
        )

        # ── Fear & Greed ───────────────────────────────────────────────────────
        # Captures crowd sentiment — extreme fear/greed signals potential reversals.
        # 0 = extreme fear, 1 = extreme greed. Normalised from 0-100.
        fg_today  = float(today["fear_greed"])
        fg_7d_avg = float(crypto_df.iloc[i - 7:i + 1]["fear_greed"].mean())
        fear_greed_feats = {
            "fear_greed":         fg_today,
            "fear_greed_7d_avg":  fg_7d_avg,
            # 1 when index is in extreme zone (< 20 fear or > 80 greed)
            "fear_greed_extreme": int(fg_today < 0.2 or fg_today > 0.8),
        }

        # ── Funding rate ───────────────────────────────────────────────────────
        # Positive = longs paying (overleveraged long → likely correction).
        # Negative = shorts paying (squeeze risk → price may pump).
        fr_today  = float(today["funding_rate_avg"])
        fr_7d_avg = float(crypto_df.iloc[i - 7:i + 1]["funding_rate_avg"].mean())
        funding_feats = {
            "funding_rate_avg":      fr_today,
            "funding_rate_7d_avg":   fr_7d_avg,
            # > 0.01% per 8h = longs significantly overleveraged
            "funding_extreme_long":  int(fr_today > 0.0001),
            # < -0.01% per 8h = shorts significantly overleveraged
            "funding_extreme_short": int(fr_today < -0.0001),
        }

        # ── BTC/ETH ratio ──────────────────────────────────────────────────────
        # Rising ratio = BTC outperforming ETH (dominance rising) → ETH may lag.
        # Falling ratio = ETH outperforming (alt season) → ETH may surge.
        other_today  = float(today["other_close_usd"])
        other_7d_ago = float(crypto_df.iloc[i - 7]["other_close_usd"])
        ratio_today  = close_today / other_today if other_today > 0 else 1.0
        ratio_7d_ago = (
            float(crypto_df.iloc[i - 7]["close_usd"]) / other_7d_ago
            if other_7d_ago > 0 else ratio_today
        )
        btc_eth_ratio_7d_change = (
            (ratio_today - ratio_7d_ago) / ratio_7d_ago
            if ratio_7d_ago != 0 else 0.0
        )

        # ── Shock & macro events ───────────────────────────────────────────────
        shock_feats = compute_shock_features(date, shock_df)
        macro_feats = compute_macro_features(date, macro_df)

        # ── Calendar ───────────────────────────────────────────────────────────
        calendar = {
            "day_of_week":  date.dayofweek,
            "month":        date.month,
            "is_weekend":   int(date.dayofweek >= 5),
            "is_month_end": int(date.day >= 28),
        }

        target = int(float(future["close_usd"]) > close_today)

        row = {
            "date":      date,
            "symbol":    symbol,
            "close_usd": close_today,

            "ret_1d":         ret_1d,
            "ret_3d":         ret_3d,
            "ret_7d":         ret_7d,
            "ret_14d":        ret_14d,
            "hl_range":       hl_range,
            "vol_trend":      vol_trend,
            "price_position": price_position,

            **fear_greed_feats,
            **funding_feats,
            "btc_eth_ratio_7d_change": btc_eth_ratio_7d_change,

            **shock_feats,
            **macro_feats,
            **calendar,

            "target": target,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"  Feature matrix: {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"  Target distribution: {df['target'].value_counts().to_dict()}")
    return df


def get_feature_columns() -> list:
    return [
        # Price momentum
        "ret_1d", "ret_3d", "ret_7d", "ret_14d",
        # Volatility / volume / range
        "hl_range", "vol_trend", "price_position",
        # Sentiment
        "fear_greed", "fear_greed_7d_avg", "fear_greed_extreme",
        # Derivatives positioning
        "funding_rate_avg", "funding_rate_7d_avg",
        "funding_extreme_long", "funding_extreme_short",
        # Market structure
        "btc_eth_ratio_7d_change",
        # Geopolitical shocks
        "shock_active", "days_since_shock",
        "max_shock_severity", "shock_oil_impact",
        # Macro events
        "fed_event_last_7d", "fed_dovish", "fed_hawkish",
        "cpi_event_last_7d", "nfp_event_last_7d",
        # Calendar
        "day_of_week", "month", "is_weekend", "is_month_end",
    ]


if __name__ == "__main__":
    for symbol in ["BTC", "ETH"]:
        print(f"\n{'=' * 50}")
        df = build_features(symbol)
        print(f"\nSample features (last 3 rows):")
        print(df[get_feature_columns() + ["target"]].tail(3).to_string())
        print(f"\nFeature correlations with target:")
        correlations = df[get_feature_columns() + ["target"]].corr()["target"].drop("target")
        print(correlations.sort_values(ascending=False).to_string())
