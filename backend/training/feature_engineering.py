"""
Feature engineering for BTC/ETH 24h direction prediction.

For each day in crypto_prices, computes:
  - Price momentum features (1, 3, 7, 14 day returns)
  - Volatility / volume / price position
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


def load_macro_features(dates: pd.Series) -> pd.DataFrame:
    """
    As-of join of macro_releases onto training dates.

    Contract:
    - actual_value / surprise_value are only visible on dates >=
      actual_first_available_at (i.e. after the data was published).
    - expected_value is visible from release_time_utc (pre-announcement).
    - Values are forward-filled between release dates.
    - Returns neutral defaults if the table is empty or unreachable.
    """
    base = pd.DataFrame({"date": pd.to_datetime(dates)}).sort_values("date").reset_index(drop=True)

    try:
        with get_conn() as conn:
            rel_df = pd.read_sql(
                """
                SELECT
                    event_name,
                    release_time_utc::date                                       AS release_date,
                    COALESCE(actual_first_available_at, release_time_utc)::date  AS actual_date,
                    expected_value,
                    actual_value,
                    previous_value,
                    surprise_value
                FROM macro_releases
                WHERE event_name IN ('FED_RATE', 'CPI', 'NFP')
                ORDER BY release_time_utc ASC
                """,
                conn,
            )
    except Exception as exc:
        print(f"Warning: macro_releases read failed: {exc}")
        rel_df = pd.DataFrame()

    result = base.copy()

    if rel_df.empty:
        result["macro_fed_rate"] = 5.25
        result["macro_days_since_fed"] = 365
        result["macro_cpi_surprise"] = 0.0
        result["macro_nfp_surprise"] = 0.0
        print("  Macro features: table empty — using neutral defaults")
        return result

    rel_df["release_date"] = pd.to_datetime(rel_df["release_date"])
    rel_df["actual_date"] = pd.to_datetime(rel_df["actual_date"])

    # ── FED rate ───────────────────────────────────────────────────────────────
    fed = (
        rel_df[rel_df["event_name"] == "FED_RATE"]
        .sort_values("release_date")
        .drop_duplicates("release_date")
    )
    if not fed.empty:
        m = pd.merge_asof(result, fed, left_on="date", right_on="release_date", direction="backward")
        # FED actual is always stamped at release_time — no pre/post split needed.
        result["macro_fed_rate"] = m["actual_value"].ffill().fillna(5.25).values
        result["macro_days_since_fed"] = (
            (m["date"] - m["actual_date"]).dt.days.clip(0, 365).fillna(365).values
        )
        print(f"  Macro FED rows (DB): {len(fed)}")
    else:
        result["macro_fed_rate"] = 5.25
        result["macro_days_since_fed"] = 365

    # ── CPI ────────────────────────────────────────────────────────────────────
    cpi = (
        rel_df[rel_df["event_name"] == "CPI"]
        .sort_values("release_date")
        .drop_duplicates("release_date")
    )
    if not cpi.empty:
        m = pd.merge_asof(result, cpi, left_on="date", right_on="release_date", direction="backward")
        actual_known = m["date"] >= m["actual_date"]
        result["macro_cpi_surprise"] = (
            m["surprise_value"].where(actual_known).ffill().fillna(0.0).values
        )
        print(f"  Macro CPI rows (DB): {len(cpi)}")
    else:
        result["macro_cpi_surprise"] = 0.0

    # ── NFP ────────────────────────────────────────────────────────────────────
    nfp = (
        rel_df[rel_df["event_name"] == "NFP"]
        .sort_values("release_date")
        .drop_duplicates("release_date")
    )
    if not nfp.empty:
        m = pd.merge_asof(result, nfp, left_on="date", right_on="release_date", direction="backward")
        actual_known = m["date"] >= m["actual_date"]
        result["macro_nfp_surprise"] = (
            m["surprise_value"].where(actual_known).ffill().fillna(0.0).values
        )
        print(f"  Macro NFP rows (DB): {len(nfp)}")
    else:
        result["macro_nfp_surprise"] = 0.0

    return result


def load_etf_flow_features(dates: pd.Series) -> pd.DataFrame:
    """
    As-of join of ETF net flows onto training dates.

    Returns the 7-day rolling sum of aggregate net inflows (all tickers)
    known at or before each training date. Forward-fills weekends / gaps.
    Returns 0 if the table is empty or unreachable.
    """
    base = pd.DataFrame({"date": pd.to_datetime(dates)}).sort_values("date").reset_index(drop=True)

    try:
        with get_conn() as conn:
            df = pd.read_sql(
                """
                SELECT flow_date AS date, SUM(net_flow_usd) AS net_flow_usd
                FROM etf_flows
                GROUP BY flow_date
                ORDER BY flow_date ASC
                """,
                conn,
            )
    except Exception as exc:
        print(f"Warning: etf_flows read failed: {exc}")
        base["macro_etf_flow_7d"] = 0.0
        return base

    if df.empty:
        base["macro_etf_flow_7d"] = 0.0
        return base

    df["date"] = pd.to_datetime(df["date"])

    # Build a dense daily series and compute 7-day rolling sum.
    full_range = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    daily = df.set_index("date").reindex(full_range, fill_value=0.0).reset_index()
    daily.columns = ["date", "net_flow_usd"]
    daily["etf_flow_7d"] = daily["net_flow_usd"].rolling(7, min_periods=1).sum()

    # As-of merge: for each training date, pick the latest flow_date <= date.
    merged = pd.merge_asof(
        base,
        daily[["date", "etf_flow_7d"]].sort_values("date"),
        on="date",
        direction="backward",
    )
    merged["macro_etf_flow_7d"] = merged["etf_flow_7d"].ffill().fillna(0.0)
    print(f"  ETF flow rows (DB): {len(df)}")
    return merged[["date", "macro_etf_flow_7d"]]


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def compute_technical_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    close = out["close_usd"].astype(float)
    high = out["high_usd"].astype(float)
    low = out["low_usd"].astype(float)
    volume = out["volume_usd"].astype(float)

    out["daily_return"] = close.pct_change().fillna(0.0)
    out["volatility_3d"] = out["daily_return"].rolling(3, min_periods=3).std()
    out["volatility_7d"] = out["daily_return"].rolling(7, min_periods=7).std()
    out["volatility_14d"] = out["daily_return"].rolling(14, min_periods=14).std()

    out["rsi_7"] = compute_rsi(close, window=7)
    out["rsi_14"] = compute_rsi(close, window=14)

    out["ema_12"] = close.ewm(span=12, adjust=False, min_periods=12).mean()
    out["ema_26"] = close.ewm(span=26, adjust=False, min_periods=26).mean()
    out["ema_50"] = close.ewm(span=50, adjust=False, min_periods=50).mean()

    out["macd_line"] = out["ema_12"] - out["ema_26"]
    out["macd_signal"] = out["macd_line"].ewm(span=9, adjust=False, min_periods=9).mean()
    out["macd_hist"] = out["macd_line"] - out["macd_signal"]

    stoch_low = low.rolling(14, min_periods=14).min()
    stoch_high = high.rolling(14, min_periods=14).max()
    stoch_range = (stoch_high - stoch_low).replace(0, pd.NA)
    out["stoch_k"] = ((close - stoch_low) / stoch_range * 100).fillna(50.0)
    out["stoch_d"] = out["stoch_k"].rolling(3, min_periods=3).mean().fillna(50.0)

    for window in (7, 14, 30, 50, 100, 200):
        out[f"ma_{window}"] = close.rolling(window, min_periods=window).mean()

    out["ema_12_slope_3"] = out["ema_12"].pct_change(3)
    out["ema_26_slope_3"] = out["ema_26"].pct_change(3)

    out["volume_change"] = volume.pct_change()
    out["volume_ma_7"] = volume.rolling(7, min_periods=7).mean()
    out["volume_ma_30"] = volume.rolling(30, min_periods=30).mean()

    return out


def build_features(symbol: str, lookahead: int = 1) -> pd.DataFrame:
    """
    Builds the complete feature matrix for one symbol (BTC or ETH).
    Each row = one trading day.
    Target = 1 if close N days ahead > today close, 0 otherwise.
    lookahead=1 → next-day direction (default)
    lookahead=7 → 7-day direction
    """
    print(f"Loading data for {symbol}...")
    target_threshold_pct = float(os.getenv("TARGET_RETURN_THRESHOLD_PCT", "0"))
    target_threshold = target_threshold_pct / 100.0
    dropped_neutral = 0

    # ── Price data ─────────────────────────────────────────────────────────────
    crypto_df  = load_crypto_prices(symbol)
    other_sym  = "ETH" if symbol == "BTC" else "BTC"
    other_df   = load_crypto_prices(other_sym)[["date", "close_usd"]].rename(
        columns={"close_usd": "other_close_usd"}
    )
    crypto_df = crypto_df.merge(other_df, on="date", how="left")
    crypto_df["other_close_usd"] = crypto_df["other_close_usd"].ffill().fillna(1.0)
    crypto_df = compute_technical_columns(crypto_df)

    # ── External signals ───────────────────────────────────────────────────────
    fear_greed_df = load_fear_greed()
    funding_df    = load_funding_rates(symbol)

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

    # ── Macro releases (as-of join) ────────────────────────────────────────────
    macro_df = load_macro_features(crypto_df["date"])
    crypto_df = crypto_df.merge(macro_df, on="date", how="left")
    macro_defaults = {
        "macro_fed_rate": 5.25,
        "macro_days_since_fed": 365,
        "macro_cpi_surprise": 0.0,
        "macro_nfp_surprise": 0.0,
    }
    for col, default in macro_defaults.items():
        if col in crypto_df.columns:
            crypto_df[col] = crypto_df[col].ffill().fillna(default)

    # ── ETF flows (as-of rolling 7-day sum) ───────────────────────────────────
    etf_df = load_etf_flow_features(crypto_df["date"])
    crypto_df = crypto_df.merge(etf_df, on="date", how="left")
    crypto_df["macro_etf_flow_7d"] = crypto_df["macro_etf_flow_7d"].ffill().fillna(0.0)

    print(f"  Crypto rows:   {len(crypto_df)}")

    rows = []

    for i in range(200, len(crypto_df) - lookahead):
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

        volatility_3d = float(today["volatility_3d"]) if pd.notna(today["volatility_3d"]) else 0.0
        volatility_7d = float(today["volatility_7d"]) if pd.notna(today["volatility_7d"]) else 0.0
        volatility_14d = float(today["volatility_14d"]) if pd.notna(today["volatility_14d"]) else 0.0

        rsi_7 = float(today["rsi_7"]) if pd.notna(today["rsi_7"]) else 50.0
        rsi_14 = float(today["rsi_14"]) if pd.notna(today["rsi_14"]) else 50.0

        macd_line = float(today["macd_line"]) if pd.notna(today["macd_line"]) else 0.0
        macd_signal = float(today["macd_signal"]) if pd.notna(today["macd_signal"]) else 0.0
        macd_hist = float(today["macd_hist"]) if pd.notna(today["macd_hist"]) else 0.0
        stoch_k = float(today["stoch_k"]) if pd.notna(today["stoch_k"]) else 50.0
        stoch_d = float(today["stoch_d"]) if pd.notna(today["stoch_d"]) else 50.0

        ma_7_rel = (close_today / float(today["ma_7"]) - 1) if pd.notna(today["ma_7"]) and float(today["ma_7"]) > 0 else 0.0
        ma_14_rel = (close_today / float(today["ma_14"]) - 1) if pd.notna(today["ma_14"]) and float(today["ma_14"]) > 0 else 0.0
        ma_30_rel = (close_today / float(today["ma_30"]) - 1) if pd.notna(today["ma_30"]) and float(today["ma_30"]) > 0 else 0.0
        ma_50_rel = (close_today / float(today["ma_50"]) - 1) if pd.notna(today["ma_50"]) and float(today["ma_50"]) > 0 else 0.0
        ma_100_rel = (close_today / float(today["ma_100"]) - 1) if pd.notna(today["ma_100"]) and float(today["ma_100"]) > 0 else 0.0
        ma_200_rel = (close_today / float(today["ma_200"]) - 1) if pd.notna(today["ma_200"]) and float(today["ma_200"]) > 0 else 0.0

        ema_12_rel = (close_today / float(today["ema_12"]) - 1) if pd.notna(today["ema_12"]) and float(today["ema_12"]) > 0 else 0.0
        ema_26_rel = (close_today / float(today["ema_26"]) - 1) if pd.notna(today["ema_26"]) and float(today["ema_26"]) > 0 else 0.0
        ema_50_rel = (close_today / float(today["ema_50"]) - 1) if pd.notna(today["ema_50"]) and float(today["ema_50"]) > 0 else 0.0
        ema_12_slope_3 = float(today["ema_12_slope_3"]) if pd.notna(today["ema_12_slope_3"]) else 0.0
        ema_26_slope_3 = float(today["ema_26_slope_3"]) if pd.notna(today["ema_26_slope_3"]) else 0.0

        volume_change = float(today["volume_change"]) if pd.notna(today["volume_change"]) else 0.0
        volume_ma7 = float(today["volume_ma_7"]) if pd.notna(today["volume_ma_7"]) else 0.0
        volume_ma30 = float(today["volume_ma_30"]) if pd.notna(today["volume_ma_30"]) else 0.0
        volume_ma7_ratio = (vol_today / volume_ma7 - 1) if volume_ma7 > 0 else 0.0
        volume_ma30_ratio = (vol_today / volume_ma30 - 1) if volume_ma30 > 0 else 0.0
        volume_spike_2x = int(volume_ma7 > 0 and vol_today > 2 * volume_ma7)

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

        # ── Regime ────────────────────────────────────────────────────────────
        # bull_regime: 1 when price is above the 200-day MA (bull market),
        # 0 otherwise (bear market). The single most powerful regime filter
        # in technical analysis — separates two fundamentally different
        # distributions of daily returns.
        ma200 = today["ma_200"]
        bull_regime = int(pd.notna(ma200) and float(ma200) > 0 and close_today > float(ma200))

        # ── Macro releases (as-of) ────────────────────────────────────────────
        # Fed rate level: higher rates → tighter liquidity → crypto headwind.
        # Days since last Fed change: freshness of the most recent decision.
        # CPI/NFP surprise: positive = above consensus → risk-off pressure.
        # ETF flow 7d: sustained inflows → demand pressure for BTC spot.
        macro_feats = {
            "macro_fed_rate":       float(today["macro_fed_rate"])       if pd.notna(today["macro_fed_rate"])       else 5.25,
            "macro_days_since_fed": float(today["macro_days_since_fed"]) if pd.notna(today["macro_days_since_fed"]) else 365.0,
            "macro_cpi_surprise":   float(today["macro_cpi_surprise"])   if pd.notna(today["macro_cpi_surprise"])   else 0.0,
            "macro_nfp_surprise":   float(today["macro_nfp_surprise"])   if pd.notna(today["macro_nfp_surprise"])   else 0.0,
            "macro_etf_flow_7d":    float(today["macro_etf_flow_7d"])    if pd.notna(today["macro_etf_flow_7d"])    else 0.0,
        }

        # ── Calendar ───────────────────────────────────────────────────────────
        calendar = {
            "day_of_week":  date.dayofweek,
            "month":        date.month,
            "is_weekend":   int(date.dayofweek >= 5),
            "is_month_end": int(date.day >= 28),
        }

        future_return = float(future["close_usd"]) / close_today - 1.0
        if target_threshold > 0 and abs(future_return) <= target_threshold:
            dropped_neutral += 1
            continue
        target = int(future_return > 0)

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
            "volatility_3d": volatility_3d,
            "volatility_7d": volatility_7d,
            "volatility_14d": volatility_14d,
            "rsi_7": rsi_7,
            "rsi_14": rsi_14,
            "macd_line": macd_line,
            "macd_signal": macd_signal,
            "macd_hist": macd_hist,
            "stoch_k": stoch_k,
            "stoch_d": stoch_d,
            "ma_7_rel": ma_7_rel,
            "ma_14_rel": ma_14_rel,
            "ma_30_rel": ma_30_rel,
            "ma_50_rel": ma_50_rel,
            "ma_100_rel": ma_100_rel,
            "ma_200_rel": ma_200_rel,
            "ema_12_rel": ema_12_rel,
            "ema_26_rel": ema_26_rel,
            "ema_50_rel": ema_50_rel,
            "ema_12_slope_3": ema_12_slope_3,
            "ema_26_slope_3": ema_26_slope_3,
            "volume_change": volume_change,
            "volume_ma7_ratio": volume_ma7_ratio,
            "volume_ma30_ratio": volume_ma30_ratio,
            "volume_spike_2x": volume_spike_2x,

            **fear_greed_feats,
            **funding_feats,
            "btc_eth_ratio_7d_change": btc_eth_ratio_7d_change,
            "bull_regime": bull_regime,

            **macro_feats,

            **calendar,

            "target": target,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if target_threshold > 0:
        print(
            f"  Target threshold: +/-{target_threshold_pct:.2f}% "
            f"(dropped {dropped_neutral} neutral rows)"
        )
    print(f"  Feature matrix: {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"  Target distribution: {df['target'].value_counts().to_dict()}")
    return df


def get_feature_columns() -> list:
    # Kept to ~18 features that are mutually low-correlation.
    # Dropping redundant price-derived indicators (7 MAs + 3 EMAs + 2 RSIs +
    # MACD line/signal + stoch_d + volatility_3d/14d + etc.) was the main fix
    # for the flat-importance / near-random-accuracy problem: XGBoost spreads
    # splits evenly across correlated features and learns nothing robust.
    # Each feature here contributes a genuinely independent source of information.
    return [
        # Price momentum — 3 timeframes; ret_1d dropped (too noisy solo)
        "ret_3d", "ret_7d", "ret_14d",
        # Volatility regime — single mid-term window
        "volatility_7d",
        # Price position within recent range (summarises all the MA/EMA signals
        # without needing 10 separate collinear MA columns)
        "price_position",
        # High-low range as a fraction of close (intraday uncertainty)
        "hl_range",
        # Volume confirmation
        "vol_trend",
        # Momentum — one normalised oscillator + one trend-cross indicator
        "rsi_14", "macd_hist",
        # Regime — above/below 200-day MA; single most powerful regime filter
        "bull_regime",
        # Market structure — BTC/ETH relative strength
        "btc_eth_ratio_7d_change",
        # Sentiment — unique external signal (not derived from price)
        "fear_greed",
        # Derivatives positioning — unique external signal
        "funding_rate_avg", "funding_extreme_long",
        # Macro — unique external signals (as-of joined, no lookahead)
        "macro_fed_rate", "macro_cpi_surprise", "macro_nfp_surprise",
        # Calendar — day-of-week effect is real in crypto
        "day_of_week",
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
