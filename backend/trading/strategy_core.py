"""
strategy_core.py — Technical analysis, gate logic, and scoring.

Shared between the live engine (engine_v2.py) and the backtester (backtest.py).
No exchange calls here — only pure DataFrame computation.

Public API:
  compute_indicators(df_h1, df_m15, df_m5)  -> dict
  check_macro_bias(ml_pred, fear_greed, funding_rate, symbol) -> (bool, str)
  check_technical_gates(context) -> (bool, str)
  find_sr_levels(df_h1, current_price, df_m15) -> dict
  compute_score(df_m15, context) -> (int, list[str])
  classify_trend(gap_pct, ema34, ema89, threshold, atr_pct) -> str
  detect_bb_mean_reversion(df_m5, idx, context) -> dict | None
"""

import numpy as np
import pandas as pd

H1_TREND_GAP      = 0.8     # % EMA34/89 gap to call H1 trend
M15_TREND_GAP     = 0.4     # % EMA34/89 gap to call M15 trend
BTC_TREND_GAP     = 1.0     # % EMA34/89 gap to call BTC trend
ATR_VOLATILE_PCT  = 0.8     # H1 ATR % threshold for volatile range
FUNDING_THRESHOLD = 0.05    # % daily funding — above = crowded longs
SCORE_THRESHOLD   = 3       # min score out of 5 to call AI
ML_CONFIDENCE_MIN = 0.55    # min ML confidence to use as bias

SL_MAX_PCT = 0.008    # 0.8% hard max SL distance
SL_MIN_PCT = 0.002    # 0.2% hard min SL distance
MIN_RR     = 1.5      # minimum reward-to-risk ratio


def classify_trend(gap_pct: float, ema34: float, ema89: float,
                   threshold: float, atr_pct: float = 0.0) -> str:
    """
    UPTREND / DOWNTREND / VOLATILE_RANGE / SIDEWAY
    Gap >= threshold → directional trend.
    Gap < threshold + high ATR → volatile range (tradeable).
    Gap < threshold + low ATR  → flat sideway (skip).
    """
    if gap_pct >= threshold:
        return "UPTREND" if ema34 > ema89 else "DOWNTREND"
    if atr_pct >= ATR_VOLATILE_PCT:
        return "VOLATILE_RANGE"
    return "SIDEWAY"


def compute_indicators(df_h1: pd.DataFrame, df_m15: pd.DataFrame,
                       df_m5: pd.DataFrame) -> dict:
    """
    Computes all indicators needed for gates + scoring + chart.
    Returns a flat context dict (no DataFrames — pass those separately).
    """
    # H1
    df_h1 = df_h1.copy()
    df_h1["ema34"] = df_h1["close"].ewm(span=34, adjust=False).mean()
    df_h1["ema89"] = df_h1["close"].ewm(span=89, adjust=False).mean()

    df_h1["tr"] = np.maximum(
        df_h1["high"] - df_h1["low"],
        np.maximum(
            abs(df_h1["high"] - df_h1["close"].shift(1)),
            abs(df_h1["low"]  - df_h1["close"].shift(1)),
        )
    )
    df_h1["atr_h1"] = df_h1["tr"].ewm(span=14, adjust=False).mean()

    h1_ema34    = df_h1["ema34"].iloc[-1]
    h1_ema89    = df_h1["ema89"].iloc[-1]
    h1_close    = df_h1["close"].iloc[-1]
    h1_atr      = df_h1["atr_h1"].iloc[-1]
    h1_atr_pct  = h1_atr / h1_close * 100
    h1_gap      = abs(h1_ema34 - h1_ema89) / h1_ema89 * 100
    h1_trend    = classify_trend(h1_gap, h1_ema34, h1_ema89,
                                 H1_TREND_GAP, h1_atr_pct)

    # M15
    df_m15 = df_m15.copy()
    df_m15["ema34"] = df_m15["close"].ewm(span=34, adjust=False).mean()
    df_m15["ema89"] = df_m15["close"].ewm(span=89, adjust=False).mean()

    df_m15["tr"] = np.maximum(
        df_m15["high"] - df_m15["low"],
        np.maximum(
            abs(df_m15["high"] - df_m15["close"].shift(1)),
            abs(df_m15["low"]  - df_m15["close"].shift(1)),
        )
    )
    df_m15["atr"] = df_m15["tr"].ewm(span=14, adjust=False).mean()

    m15_ema34  = df_m15["ema34"].iloc[-1]
    m15_ema89  = df_m15["ema89"].iloc[-1]
    m15_gap    = abs(m15_ema34 - m15_ema89) / m15_ema89 * 100
    m15_trend  = classify_trend(m15_gap, m15_ema34, m15_ema89,
                                M15_TREND_GAP, h1_atr_pct)
    atr_m15    = df_m15["atr"].iloc[-1]

    delta  = df_m15["close"].diff()
    gain   = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
    loss   = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
    rs     = gain / loss.replace(0, np.nan)
    rsi    = (100 - 100 / (1 + rs)).iloc[-1]

    # ADX (M15) — simplified directional movement
    df_m15["dm_plus"]  = np.where(
        (df_m15["high"].diff() > -df_m15["low"].diff()) &
        (df_m15["high"].diff() > 0),
        df_m15["high"].diff(), 0
    )
    df_m15["dm_minus"] = np.where(
        (-df_m15["low"].diff() > df_m15["high"].diff()) &
        (-df_m15["low"].diff() > 0),
        -df_m15["low"].diff(), 0
    )
    atr_s   = df_m15["tr"].ewm(span=14, adjust=False).mean()
    di_plus  = 100 * df_m15["dm_plus"].ewm(span=14, adjust=False).mean() / atr_s
    di_minus = 100 * df_m15["dm_minus"].ewm(span=14, adjust=False).mean() / atr_s
    dx       = 100 * abs(di_plus - di_minus) / (di_plus + di_minus).replace(0, np.nan)
    adx      = dx.ewm(span=14, adjust=False).mean().iloc[-1]

    direction = np.sign(df_m15["close"].diff()).fillna(0)
    obv       = (direction * df_m15["volume"]).cumsum()
    obv_slope = float(obv.iloc[-1] - obv.iloc[-6])

    sma20      = df_m15["close"].rolling(20).mean()
    std20      = df_m15["close"].rolling(20).std()
    upper_bb   = sma20 + 2 * std20
    lower_bb   = sma20 - 2 * std20
    bb_width   = ((upper_bb - lower_bb) / sma20)
    close_now  = df_m15["close"].iloc[-1]
    bb_breakout = close_now > upper_bb.iloc[-1] or close_now < lower_bb.iloc[-1]
    bb_squeeze  = bb_width.iloc[-1] < bb_width.rolling(20).mean().iloc[-1]

    vol_ma      = df_m15["volume"].rolling(20).mean().iloc[-1]
    vol_spike   = df_m15["volume"].iloc[-1] > vol_ma * 1.5

    current_price = float(df_m5["close"].iloc[-1])

    is_range = (h1_trend == "VOLATILE_RANGE" or m15_trend == "VOLATILE_RANGE")
    is_aligned = (h1_trend == m15_trend and
                  h1_trend not in ("SIDEWAY", "VOLATILE_RANGE"))

    if is_range:
        market_mode = "VOLATILE_RANGE"
    else:
        market_mode = h1_trend

    return {
        "h1_trend":      h1_trend,
        "m15_trend":     m15_trend,
        "market_mode":   market_mode,
        "is_aligned":    is_aligned,
        "is_range":      is_range,
        "h1_gap":        round(h1_gap, 3),
        "m15_gap":       round(m15_gap, 3),
        "h1_atr_pct":    round(h1_atr_pct, 4),
        "atr_m15":       round(atr_m15, 4),
        "rsi":           round(float(rsi), 2),
        "adx":           round(float(adx), 2),
        "obv_slope":     round(obv_slope, 2),
        "bb_breakout":   bool(bb_breakout),
        "bb_squeeze":    bool(bb_squeeze),
        "vol_spike":     bool(vol_spike),
        "vol_ma":        round(float(vol_ma), 2),
        "current_price": current_price,
        "_df_h1_ema34":  round(float(h1_ema34), 4),
        "_df_h1_ema89":  round(float(h1_ema89), 4),
    }


def find_sr_levels(df_h1: pd.DataFrame, current_price: float,
                   df_m15: pd.DataFrame = None) -> dict:
    """
    Swing high/low S/R from H1 (last 50 bars) supplemented by M15 swing levels.
    Returns nearest resistance above and support below current price.
    """
    df = df_h1.tail(50).reset_index(drop=True)
    W  = 5
    swing_highs, swing_lows = [], []

    for i in range(W, len(df) - W):
        window_h = df["high"].iloc[i - W: i + W + 1]
        window_l = df["low"].iloc[i - W: i + W + 1]
        if df["high"].iloc[i] == window_h.max():
            swing_highs.append(float(df["high"].iloc[i]))
        if df["low"].iloc[i] == window_l.min():
            swing_lows.append(float(df["low"].iloc[i]))

    if df_m15 is not None and len(df_m15) >= 30:
        m15 = df_m15.tail(40).reset_index(drop=True)
        W2  = 3
        for i in range(W2, len(m15) - W2):
            if m15["high"].iloc[i] == m15["high"].iloc[i - W2: i + W2 + 1].max():
                swing_highs.append(float(m15["high"].iloc[i]))
            if m15["low"].iloc[i] == m15["low"].iloc[i - W2: i + W2 + 1].min():
                swing_lows.append(float(m15["low"].iloc[i]))

    resistances = sorted(h for h in set(swing_highs) if h > current_price * 1.002)
    supports    = sorted(
        (lv for lv in set(swing_lows) if lv < current_price * 0.998), reverse=True
    )

    return {
        "resistance": round(resistances[0], 2) if resistances else round(current_price * 1.015, 2),
        "support":    round(supports[0],    2) if supports    else round(current_price * 0.985, 2),
    }


def compute_score(context: dict) -> tuple[int, list[str]]:
    """
    5-point quantitative score. Called after compute_indicators().
    Returns (score, details_list).
    """
    score   = 0
    details = []
    h1      = context["h1_trend"]
    is_range = context["is_range"]

    # Bollinger Bands (+2 breakout / +1 squeeze)
    if context["bb_breakout"]:
        score += 2
        details.append("BB breakout (+2)")
    elif context["bb_squeeze"]:
        score += 1
        details.append("BB squeeze (+1)")

    # ADX
    if context["adx"] > 20:
        score += 1
        details.append(f"ADX={context['adx']:.1f} (+1)")

    # RSI aligned with trend
    rsi = context["rsi"]
    if (h1 == "UPTREND" and rsi > 55) or (h1 == "DOWNTREND" and rsi < 45):
        score += 1
        details.append(f"RSI={rsi:.1f} (+1)")
    elif is_range and (rsi > 60 or rsi < 40):
        score += 1
        details.append(f"RSI={rsi:.1f} extreme in range (+1)")

    # OBV slope
    obv = context["obv_slope"]
    if (h1 == "UPTREND" and obv > 0) or (h1 == "DOWNTREND" and obv < 0):
        score += 1
        details.append("OBV aligned (+1)")
    elif is_range and abs(obv) > 0:
        score += 1
        details.append("OBV momentum (+1)")

    # Volume spike
    if context["vol_spike"]:
        score += 1
        details.append("Vol spike 1.5x (+1)")

    return score, details


def check_macro_bias(ml_direction: str, ml_confidence: float,
                     fear_greed: int, funding_rate: float,
                     symbol: str) -> tuple[bool, str, str]:
    """
    Returns (pass, reason, allowed_direction).
    allowed_direction: "BUY", "SELL", or "BOTH"
    """
    allowed = "BOTH"

    if ml_confidence >= ML_CONFIDENCE_MIN:
        allowed = "BUY" if ml_direction == "UP" else "SELL"

    # Fear & Greed: extreme values are contrarian signals
    if fear_greed is not None:
        if fear_greed <= 15:
            allowed = "BUY"   # extreme fear = contrarian long
        elif fear_greed >= 85:
            allowed = "SELL"  # extreme greed = contrarian short

    if funding_rate > FUNDING_THRESHOLD and allowed in ("BUY", "BOTH"):
        if allowed == "BUY":
            return False, f"Funding {funding_rate:+.4f}% — longs saturated", allowed
        allowed = "SELL"

    if funding_rate < -FUNDING_THRESHOLD and allowed in ("SELL", "BOTH"):
        if allowed == "SELL":
            return False, f"Funding {funding_rate:+.4f}% — shorts saturated", allowed
        allowed = "BUY"

    return True, "OK", allowed


def check_technical_gates(context: dict) -> tuple[bool, str]:
    """
    Runs all technical gates in order.
    Returns (pass, reason_for_failure).
    """
    h1    = context["h1_trend"]
    m15   = context["m15_trend"]
    score = context["score"]
    is_range = context["is_range"]

    if h1 == "SIDEWAY" and m15 == "SIDEWAY":  # pure flat — no edge
        return False, "GATE1: both H1+M15 SIDEWAY"

    # Trend misalignment check (skipped in range mode)
    if not is_range:
        if (h1 not in ("SIDEWAY", "VOLATILE_RANGE") and
                m15 not in ("SIDEWAY", "VOLATILE_RANGE") and
                h1 != m15):
            return False, f"GATE2: H1={h1} vs M15={m15} misaligned"

    threshold = SCORE_THRESHOLD
    if score < threshold:
        return False, f"GATE3: score {score}/{threshold} insufficient"

    # Range mode: only trade when price is near S/R
    if is_range:
        price = context["current_price"]
        sr    = context.get("sr", {})
        if sr:
            dist_r = abs(sr["resistance"] - price) / price * 100
            dist_s = abs(price - sr["support"])    / price * 100
            edge   = context.get("h1_atr_pct", 1.0)  # ATR % as proximity threshold
            if dist_r > edge and dist_s > edge:
                return False, f"GATE4: price in middle of range (R:{dist_r:.2f}% S:{dist_s:.2f}%)"

    return True, "OK"


def get_range_bias(context: dict) -> str:
    """Returns 'NEAR_SUPPORT', 'NEAR_RESISTANCE', or 'MIDDLE'."""
    price = context["current_price"]
    sr    = context.get("sr", {})
    if not sr:
        return "MIDDLE"
    dist_r = abs(sr["resistance"] - price) / price * 100
    dist_s = abs(price - sr["support"])    / price * 100
    if dist_s <= dist_r:
        return "NEAR_SUPPORT"
    return "NEAR_RESISTANCE"


def detect_candle_pattern(df: pd.DataFrame, idx: int) -> dict:
    """
    Rule-based pattern detection on a single candle at df.iloc[idx].
    Returns {"pattern": str, "direction": "BUY"|"SELL"|"NONE"}.

    Patterns detected:
      bullish_pinbar, bearish_pinbar,
      bullish_engulfing, bearish_engulfing,
      none
    """
    if idx < 1 or idx >= len(df):
        return {"pattern": "none", "direction": "NONE"}

    row  = df.iloc[idx]
    prev = df.iloc[idx - 1]

    o, h, lo, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
    body     = abs(c - o)
    total    = h - lo
    wick_up  = h - max(o, c)
    wick_dn  = min(o, c) - lo

    if total == 0:
        return {"pattern": "none", "direction": "NONE"}

    if (wick_dn >= 2 * body and  # bullish pinbar
            c > (lo + total * 0.5) and
            wick_up <= wick_dn * 0.4):
        return {"pattern": "bullish_pinbar", "direction": "BUY"}

    if (wick_up >= 2 * body and  # bearish pinbar
            c < (lo + total * 0.5) and
            wick_dn <= wick_up * 0.4):
        return {"pattern": "bearish_pinbar", "direction": "SELL"}

    p_o = float(prev["open"])
    p_c = float(prev["close"])
    if (c > o and p_c < p_o and           # current green, prev red
            o <= p_c and c >= p_o):        # body engulfs prev body
        return {"pattern": "bullish_engulfing", "direction": "BUY"}

    if (c < o and p_c > p_o and           # bearish engulfing
            o >= p_c and c <= p_o):
        return {"pattern": "bearish_engulfing", "direction": "SELL"}

    return {"pattern": "none", "direction": "NONE"}


def detect_bb_mean_reversion(df_m5: pd.DataFrame, idx: int, context: dict) -> dict | None:
    """
    Setup E: Mean reversion when price reaches Bollinger Band extreme in SIDEWAY mode.

    Entry conditions (ALL required):
      BUY  — close ≤ lower BB  AND  RSI < 30
      SELL — close ≥ upper BB  AND  RSI > 70

    SL: beyond the band extreme + 0.2% buffer (capped at SL_MAX_PCT).
    TP: SMA20 midline (the reversion target).

    Only fires when context["market_mode"] == "SIDEWAY".
    Returns signal dict {"signal", "entry", "sl", "tp", "rr", "setup"} or None.
    """
    if context.get("market_mode") != "SIDEWAY":
        return None
    if idx < 20:
        return None

    slice_df = df_m5.iloc[max(0, idx - 29): idx + 1].copy()
    if len(slice_df) < 20:
        return None

    closes = slice_df["close"]
    sma20  = closes.rolling(20).mean().iloc[-1]
    std20  = closes.rolling(20).std().iloc[-1]

    if pd.isna(sma20) or pd.isna(std20) or std20 == 0:
        return None

    upper_bb  = sma20 + 2 * std20
    lower_bb  = sma20 - 2 * std20
    close_now = float(df_m5["close"].iloc[idx])
    high_now  = float(df_m5["high"].iloc[idx])
    low_now   = float(df_m5["low"].iloc[idx])
    rsi       = context.get("rsi", 50)

    near_lower = close_now <= lower_bb * 1.005  # within 0.5% of lower band
    near_upper = close_now >= upper_bb * 0.995  # within 0.5% of upper band

    context["_bb_debug"] = {
        "price": round(close_now, 4),
        "upper_bb": round(upper_bb, 4),
        "lower_bb": round(lower_bb, 4),
        "sma20": round(float(sma20), 4),
        "rsi": round(rsi, 1),
        "near_lower": near_lower,
        "near_upper": near_upper,
    }

    if near_lower and rsi < 35:  # BUY: near lower BB + RSI oversold
        raw_sl  = min(low_now, lower_bb) - close_now * 0.002
        sl_dist = close_now - raw_sl

        if sl_dist / close_now < SL_MIN_PCT:
            return None
        if sl_dist / close_now > SL_MAX_PCT:
            raw_sl  = close_now * (1 - SL_MAX_PCT)
            sl_dist = close_now - raw_sl

        # TP at upper BB (full range target) — better R:R than SMA20 on tight bands
        tp = float(upper_bb)
        if tp <= close_now:
            return None

        rr = (tp - close_now) / sl_dist
        if rr < MIN_RR:
            return None

        return {
            "signal": "BUY",
            "entry":  round(close_now, 4),
            "sl":     round(raw_sl, 4),
            "tp":     round(tp, 4),
            "rr":     round(rr, 2),
            "setup":  "setup_E_bb_mean_reversion",
        }

    if near_upper and rsi > 65:  # SELL: near upper BB + RSI overbought
        raw_sl  = max(high_now, upper_bb) + close_now * 0.002
        sl_dist = raw_sl - close_now

        if sl_dist / close_now < SL_MIN_PCT:
            return None
        if sl_dist / close_now > SL_MAX_PCT:
            raw_sl  = close_now * (1 + SL_MAX_PCT)
            sl_dist = raw_sl - close_now

        # TP at lower BB (full range target) — better R:R than SMA20 on tight bands
        tp = float(lower_bb)
        if tp >= close_now:
            return None

        rr = (close_now - tp) / sl_dist
        if rr < MIN_RR:
            return None

        return {
            "signal": "SELL",
            "entry":  round(close_now, 4),
            "sl":     round(raw_sl, 4),
            "tp":     round(tp, 4),
            "rr":     round(rr, 2),
            "setup":  "setup_E_bb_mean_reversion",
        }

    return None
