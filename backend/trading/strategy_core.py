"""
strategy_core.py - Technical analysis, gate logic, and trade planning.

Shared between the live engine and the backtester.
No exchange calls here - only pure DataFrame computation.

Public API:
  compute_indicators(df_h1, df_m15, df_m5) -> dict
  check_macro_bias(ml_pred, fear_greed, funding_rate, symbol) -> (bool, str, str)
  check_technical_gates(context) -> (bool, str)
  find_sr_levels(df_h1, current_price, df_m15) -> dict
  build_trade_plan(signal, setup_name, context, df_m5) -> (dict | None, str)
  validate_ai_trade_decision(decision, context, df_m5) -> (bool, str)
  compute_score(context) -> (int, list[str])
  classify_trend(gap_pct, ema34, ema89, threshold, atr_pct, adx) -> str
  detect_bb_mean_reversion(df_m5, idx, context) -> dict | None
"""

import os

import numpy as np
import pandas as pd

H1_TREND_GAP = 0.8
M15_TREND_GAP = 0.4
BTC_TREND_GAP = 1.0
ATR_VOLATILE_PCT = 0.8
FUNDING_THRESHOLD = 0.05
SCORE_THRESHOLD = 3
ML_CONFIDENCE_MIN = 0.55

SL_MAX_PCT = 0.008
SL_MIN_PCT = 0.002
MIN_RR = 1.5

ADX_TREND_THRESHOLD = 45
BREAKOUT_CONFIRM_ATR_MULT = 0.20
BREAKOUT_CONFIRM_PCT = 0.0008
ROLLOVER_ATR_MULT = 0.75


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return int(default)


def classify_trend(gap_pct: float, ema34: float, ema89: float,
                   threshold: float, atr_pct: float = 0.0,
                   adx: float = 0.0) -> str:
    """
    UPTREND / DOWNTREND / VOLATILE_RANGE / SIDEWAY
    """
    if gap_pct >= threshold:
        return "UPTREND" if ema34 > ema89 else "DOWNTREND"
    if adx >= ADX_TREND_THRESHOLD:
        return "UPTREND" if ema34 > ema89 else "DOWNTREND"
    if atr_pct >= ATR_VOLATILE_PCT:
        return "VOLATILE_RANGE"
    return "SIDEWAY"


def _is_strong_breakout_context(setup_code: str, context: dict,
                                entry: float, atr: float,
                                box_low: float, box_high: float,
                                signal: str) -> bool:
    """
    Allows TP2 only when breakout quality is genuinely strong.
    """
    if setup_code != "B":
        return False

    adx = float(context.get("adx") or 0.0)
    score = int(context.get("score") or 0)
    vol_spike = bool(context.get("vol_spike"))
    if adx < 25 or score < 4 or not vol_spike:
        return False

    breakout_buffer = max(entry * BREAKOUT_CONFIRM_PCT, atr * BREAKOUT_CONFIRM_ATR_MULT)
    price = float(context.get("current_price") or entry)
    if signal == "BUY":
        return price >= (box_high + breakout_buffer)
    return price <= (box_low - breakout_buffer)


def detect_m15_rollover(raw_trend: str, df_m15: pd.DataFrame,
                        atr: float, rsi: float) -> tuple[bool, str]:
    """
    Detects when an M15 trend still qualifies by slow EMA rules but has already
    deteriorated into a short-term rollover.
    """
    if raw_trend not in ("UPTREND", "DOWNTREND") or len(df_m15) < 5:
        return False, ""

    closes = df_m15["close"].astype(float)
    highs = df_m15["high"].astype(float)
    lows = df_m15["low"].astype(float)
    ema34_series = closes.ewm(span=34, adjust=False).mean()
    ema89_series = closes.ewm(span=89, adjust=False).mean()

    last_close = float(closes.iloc[-1])
    ema34_last = float(ema34_series.iloc[-1])
    ema34_prev2 = float(ema34_series.iloc[-3])
    ema89_last = float(ema89_series.iloc[-1])
    three_bar_move = float(closes.iloc[-1] - closes.iloc[-4])

    if raw_trend == "UPTREND":
        below_ema34 = last_close < ema34_last
        below_ema89 = last_close < ema89_last
        ema_slope_down = ema34_last <= ema34_prev2
        lower_highs = bool(highs.iloc[-1] < highs.iloc[-2] <= highs.iloc[-3])
        sharp_drop = atr > 0 and three_bar_move <= -(atr * ROLLOVER_ATR_MULT)
        weak_rsi = rsi < 48

        if below_ema89 and sharp_drop:
            return True, (
                f"M15 uptrend rolling over: close {last_close:.4f} below EMA89 "
                f"{ema89_last:.4f} after {abs(three_bar_move):.4f} drop"
            )
        if below_ema34 and (ema_slope_down or lower_highs) and (sharp_drop or weak_rsi):
            return True, (
                f"M15 uptrend rolling over: close {last_close:.4f} below EMA34 "
                f"{ema34_last:.4f} with weakening structure"
            )
        return False, ""

    above_ema34 = last_close > ema34_last
    above_ema89 = last_close > ema89_last
    ema_slope_up = ema34_last >= ema34_prev2
    higher_lows = bool(lows.iloc[-1] > lows.iloc[-2] >= lows.iloc[-3])
    sharp_rally = atr > 0 and three_bar_move >= atr * ROLLOVER_ATR_MULT
    strong_rsi = rsi > 52

    if above_ema89 and sharp_rally:
        return True, (
            f"M15 downtrend rolling over: close {last_close:.4f} above EMA89 "
            f"{ema89_last:.4f} after {abs(three_bar_move):.4f} rally"
        )
    if above_ema34 and (ema_slope_up or higher_lows) and (sharp_rally or strong_rsi):
        return True, (
            f"M15 downtrend rolling over: close {last_close:.4f} above EMA34 "
            f"{ema34_last:.4f} with improving structure"
        )
    return False, ""


def compute_indicators(df_h1: pd.DataFrame, df_m15: pd.DataFrame,
                       df_m5: pd.DataFrame) -> dict:
    """
    Computes all indicators needed for gates + scoring + chart.
    Returns a flat context dict.
    """
    df_h1 = df_h1.copy()
    df_h1["ema34"] = df_h1["close"].ewm(span=34, adjust=False).mean()
    df_h1["ema89"] = df_h1["close"].ewm(span=89, adjust=False).mean()

    df_h1["tr"] = np.maximum(
        df_h1["high"] - df_h1["low"],
        np.maximum(
            abs(df_h1["high"] - df_h1["close"].shift(1)),
            abs(df_h1["low"] - df_h1["close"].shift(1)),
        ),
    )
    df_h1["atr_h1"] = df_h1["tr"].ewm(span=14, adjust=False).mean()

    h1_ema34 = df_h1["ema34"].iloc[-1]
    h1_ema89 = df_h1["ema89"].iloc[-1]
    h1_close = df_h1["close"].iloc[-1]
    h1_atr = df_h1["atr_h1"].iloc[-1]
    h1_atr_pct = h1_atr / h1_close * 100
    h1_gap = abs(h1_ema34 - h1_ema89) / h1_ema89 * 100

    df_m15 = df_m15.copy()
    df_m15["ema34"] = df_m15["close"].ewm(span=34, adjust=False).mean()
    df_m15["ema89"] = df_m15["close"].ewm(span=89, adjust=False).mean()
    df_m15["tr"] = np.maximum(
        df_m15["high"] - df_m15["low"],
        np.maximum(
            abs(df_m15["high"] - df_m15["close"].shift(1)),
            abs(df_m15["low"] - df_m15["close"].shift(1)),
        ),
    )
    df_m15["atr"] = df_m15["tr"].ewm(span=14, adjust=False).mean()

    df_m15["dm_plus"] = np.where(
        (df_m15["high"].diff() > -df_m15["low"].diff()) &
        (df_m15["high"].diff() > 0),
        df_m15["high"].diff(),
        0,
    )
    df_m15["dm_minus"] = np.where(
        (-df_m15["low"].diff() > df_m15["high"].diff()) &
        (-df_m15["low"].diff() > 0),
        -df_m15["low"].diff(),
        0,
    )
    atr_s = df_m15["tr"].ewm(span=14, adjust=False).mean()
    di_plus = 100 * df_m15["dm_plus"].ewm(span=14, adjust=False).mean() / atr_s
    di_minus = 100 * df_m15["dm_minus"].ewm(span=14, adjust=False).mean() / atr_s
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus).replace(0, np.nan)
    adx = dx.ewm(span=14, adjust=False).mean().iloc[-1]

    h1_trend = classify_trend(h1_gap, h1_ema34, h1_ema89, H1_TREND_GAP, h1_atr_pct, adx)

    m15_ema34 = df_m15["ema34"].iloc[-1]
    m15_ema89 = df_m15["ema89"].iloc[-1]
    m15_gap = abs(m15_ema34 - m15_ema89) / m15_ema89 * 100
    raw_m15_trend = classify_trend(m15_gap, m15_ema34, m15_ema89, M15_TREND_GAP, h1_atr_pct, adx)
    atr_m15 = df_m15["atr"].iloc[-1]

    delta = df_m15["close"].diff()
    gain = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = (100 - 100 / (1 + rs)).iloc[-1]

    m15_rollover, m15_rollover_reason = detect_m15_rollover(
        raw_m15_trend, df_m15, float(atr_m15), float(rsi)
    )
    m15_trend = raw_m15_trend

    direction = np.sign(df_m15["close"].diff()).fillna(0)
    obv = (direction * df_m15["volume"]).cumsum()
    obv_slope = float(obv.iloc[-1] - obv.iloc[-6])

    sma20 = df_m15["close"].rolling(20).mean()
    std20 = df_m15["close"].rolling(20).std()
    upper_bb = sma20 + 2 * std20
    lower_bb = sma20 - 2 * std20
    bb_width = (upper_bb - lower_bb) / sma20
    close_now = df_m15["close"].iloc[-1]
    bb_breakout = close_now > upper_bb.iloc[-1] or close_now < lower_bb.iloc[-1]
    bb_squeeze = bb_width.iloc[-1] < bb_width.rolling(20).mean().iloc[-1]

    vol_ma = df_m15["volume"].rolling(20).mean().iloc[-1]
    vol_spike = df_m15["volume"].iloc[-1] > vol_ma * 1.5

    current_price = float(df_m5["close"].iloc[-1])

    is_range = (h1_trend == "VOLATILE_RANGE" or m15_trend == "VOLATILE_RANGE")
    is_aligned = (h1_trend == m15_trend and h1_trend not in ("SIDEWAY", "VOLATILE_RANGE"))
    primary_trend = m15_trend
    market_mode = "VOLATILE_RANGE" if is_range else primary_trend

    return {
        "h1_trend": h1_trend,
        "m15_trend": m15_trend,
        "m15_trend_raw": raw_m15_trend,
        "primary_trend": primary_trend,
        "market_mode": market_mode,
        "is_aligned": is_aligned,
        "is_range": is_range,
        "trend_rollover": bool(m15_rollover),
        "trend_rollover_reason": m15_rollover_reason,
        "h1_gap": round(h1_gap, 3),
        "m15_gap": round(m15_gap, 3),
        "h1_atr_pct": round(h1_atr_pct, 4),
        "atr_m15": round(atr_m15, 4),
        "rsi": round(float(rsi), 2),
        "adx": round(float(adx), 2),
        "obv_slope": round(obv_slope, 2),
        "bb_breakout": bool(bb_breakout),
        "bb_squeeze": bool(bb_squeeze),
        "vol_spike": bool(vol_spike),
        "vol_ma": round(float(vol_ma), 2),
        "current_price": current_price,
        "_df_h1_ema34": round(float(h1_ema34), 4),
        "_df_h1_ema89": round(float(h1_ema89), 4),
    }


def find_sr_levels(df_h1: pd.DataFrame, current_price: float,
                   df_m15: pd.DataFrame = None) -> dict:
    """
    Swing high/low S/R from H1 supplemented by M15 swing levels.
    Returns nearest levels plus a few additional levels for trade planning.
    """
    df = df_h1.tail(50).reset_index(drop=True)
    width = 5
    swing_highs, swing_lows = [], []

    for i in range(width, len(df) - width):
        window_h = df["high"].iloc[i - width:i + width + 1]
        window_l = df["low"].iloc[i - width:i + width + 1]
        if df["high"].iloc[i] == window_h.max():
            swing_highs.append(float(df["high"].iloc[i]))
        if df["low"].iloc[i] == window_l.min():
            swing_lows.append(float(df["low"].iloc[i]))

    if df_m15 is not None and len(df_m15) >= 30:
        m15 = df_m15.tail(40).reset_index(drop=True)
        width_m15 = 3
        for i in range(width_m15, len(m15) - width_m15):
            high_window = m15["high"].iloc[i - width_m15:i + width_m15 + 1]
            low_window = m15["low"].iloc[i - width_m15:i + width_m15 + 1]
            if m15["high"].iloc[i] == high_window.max():
                swing_highs.append(float(m15["high"].iloc[i]))
            if m15["low"].iloc[i] == low_window.min():
                swing_lows.append(float(m15["low"].iloc[i]))

    resistances = sorted(h for h in set(swing_highs) if h > current_price * 1.002)
    supports = sorted((lv for lv in set(swing_lows) if lv < current_price * 0.998), reverse=True)

    resistance_levels = [round(x, 2) for x in resistances[:5]]
    support_levels = [round(x, 2) for x in supports[:5]]

    if not resistance_levels:
        resistance_levels = [round(current_price * 1.015, 2)]
    if not support_levels:
        support_levels = [round(current_price * 0.985, 2)]

    return {
        "resistance": resistance_levels[0],
        "support": support_levels[0],
        "resistance_levels": resistance_levels,
        "support_levels": support_levels,
    }


def compute_score(context: dict) -> tuple[int, list[str]]:
    """
    5-point quantitative score.
    """
    score = 0
    details = []
    trend = context.get("primary_trend") or context["m15_trend"]
    is_range = context["is_range"]

    if context["bb_breakout"]:
        score += 2
        details.append("BB breakout (+2)")
    elif context["bb_squeeze"]:
        score += 1
        details.append("BB squeeze (+1)")

    if context["adx"] > 20:
        score += 1
        details.append(f"ADX={context['adx']:.1f} (+1)")

    rsi = context["rsi"]
    if (trend == "UPTREND" and rsi > 55) or (trend == "DOWNTREND" and rsi < 45):
        score += 1
        details.append(f"RSI={rsi:.1f} (+1)")
    elif is_range and (rsi > 60 or rsi < 40):
        score += 1
        details.append(f"RSI={rsi:.1f} extreme in range (+1)")

    obv = context["obv_slope"]
    if (trend == "UPTREND" and obv > 0) or (trend == "DOWNTREND" and obv < 0):
        score += 1
        details.append("OBV aligned (+1)")
    elif is_range and abs(obv) > 0:
        score += 1
        details.append("OBV momentum (+1)")

    if context["vol_spike"]:
        score += 1
        details.append("Vol spike 1.5x (+1)")

    if context.get("trend_rollover"):
        details.append("M15 rollover caution")

    return score, details


def check_macro_bias(ml_direction: str, ml_confidence: float,
                     fear_greed: int, funding_rate: float,
                     symbol: str) -> tuple[bool, str, str]:
    """
    Returns advisory-only macro context.
    Market mode remains the primary directional filter.
    """
    notes = []

    if ml_direction:
        notes.append(f"ML={ml_direction}({ml_confidence:.0%}) advisory")

    if fear_greed is not None:
        if fear_greed <= 15:
            notes.append(f"F&G={fear_greed} extreme fear")
        elif fear_greed >= 85:
            notes.append(f"F&G={fear_greed} extreme greed")
        else:
            notes.append(f"F&G={fear_greed}")

    if funding_rate is not None:
        if funding_rate > FUNDING_THRESHOLD:
            notes.append(f"funding {funding_rate:+.4f}% crowded longs")
        elif funding_rate < -FUNDING_THRESHOLD:
            notes.append(f"funding {funding_rate:+.4f}% crowded shorts")
        else:
            notes.append(f"funding {funding_rate:+.4f}%")

    return True, " | ".join(notes) if notes else "advisory only", "BOTH"


def check_technical_gates(context: dict) -> tuple[bool, str]:
    """
    Runs all technical gates in order.
    """
    primary = context.get("primary_trend") or context["m15_trend"]
    score = context["score"]
    is_range = context["is_range"]
    allow_sideway = _env_bool("TECH_ALLOW_SIDEWAY", False)

    if primary == "SIDEWAY" and not allow_sideway:
        return False, "GATE1: M15 SIDEWAY"

    if not is_range:
        if primary == "VOLATILE_RANGE":
            return False, "GATE2: M15 VOLATILE_RANGE"

    base_threshold = max(1, _env_int("TECH_SCORE_THRESHOLD", SCORE_THRESHOLD))
    range_threshold = max(1, _env_int("TECH_SCORE_THRESHOLD_RANGE", base_threshold))
    non_btc_high_adx_threshold = max(1, _env_int("TECH_SCORE_THRESHOLD_NON_BTC_HIGH_ADX", 2))
    adx_relax_level = _env_float("TECH_ADX_RELAX_LEVEL", 45.0)
    rollover_threshold = max(1, _env_int("TECH_ROLLOVER_SCORE_THRESHOLD", 1))

    threshold = range_threshold if is_range else base_threshold
    if (
        not str(context.get("symbol", "BTC")).startswith("BTC")
        and context.get("adx", 0) > adx_relax_level
    ):
        threshold = min(threshold, non_btc_high_adx_threshold)
    if context.get("trend_rollover"):
        threshold = min(threshold, rollover_threshold)
    if score < threshold:
        return False, f"GATE3: score {score}/{threshold} insufficient"

    if is_range:
        price = context["current_price"]
        sr = context.get("sr", {})
        if sr:
            dist_r = abs(sr["resistance"] - price) / price * 100
            dist_s = abs(price - sr["support"]) / price * 100
            edge = context.get("h1_atr_pct", 1.0)
            if dist_r > edge and dist_s > edge:
                return False, f"GATE4: price in middle of range (R:{dist_r:.2f}% S:{dist_s:.2f}%)"

    return True, "OK"


def get_range_bias(context: dict) -> str:
    """Returns 'NEAR_SUPPORT', 'NEAR_RESISTANCE', or 'MIDDLE'."""
    price = context["current_price"]
    sr = context.get("sr", {})
    if not sr:
        return "MIDDLE"
    dist_r = abs(sr["resistance"] - price) / price * 100
    dist_s = abs(price - sr["support"]) / price * 100
    if dist_s <= dist_r:
        return "NEAR_SUPPORT"
    return "NEAR_RESISTANCE"


def detect_candle_pattern(df: pd.DataFrame, idx: int) -> dict:
    """
    Rule-based pattern detection on a single candle at df.iloc[idx].
    """
    if idx < 1 or idx >= len(df):
        return {"pattern": "none", "direction": "NONE"}

    row = df.iloc[idx]
    prev = df.iloc[idx - 1]

    open_price = float(row["open"])
    high_price = float(row["high"])
    low_price = float(row["low"])
    close_price = float(row["close"])
    body = abs(close_price - open_price)
    total = high_price - low_price
    wick_up = high_price - max(open_price, close_price)
    wick_down = min(open_price, close_price) - low_price

    if total == 0:
        return {"pattern": "none", "direction": "NONE"}

    if (
        wick_down >= 2 * body
        and close_price > (low_price + total * 0.5)
        and wick_up <= wick_down * 0.4
    ):
        return {"pattern": "bullish_pinbar", "direction": "BUY"}

    if (
        wick_up >= 2 * body
        and close_price < (low_price + total * 0.5)
        and wick_down <= wick_up * 0.4
    ):
        return {"pattern": "bearish_pinbar", "direction": "SELL"}

    prev_open = float(prev["open"])
    prev_close = float(prev["close"])
    if (
        close_price > open_price
        and prev_close < prev_open
        and open_price <= prev_close
        and close_price >= prev_open
    ):
        return {"pattern": "bullish_engulfing", "direction": "BUY"}

    if (
        close_price < open_price
        and prev_close > prev_open
        and open_price >= prev_close
        and close_price <= prev_open
    ):
        return {"pattern": "bearish_engulfing", "direction": "SELL"}

    return {"pattern": "none", "direction": "NONE"}


def detect_bb_mean_reversion(df_m5: pd.DataFrame, idx: int, context: dict) -> dict | None:
    """
    Setup E: mean reversion when price reaches a Bollinger Band extreme in SIDEWAY mode.
    """
    if context.get("market_mode") != "SIDEWAY":
        return None
    if idx < 20:
        return None

    slice_df = df_m5.iloc[max(0, idx - 29):idx + 1].copy()
    if len(slice_df) < 20:
        return None

    closes = slice_df["close"]
    sma20 = closes.rolling(20).mean().iloc[-1]
    std20 = closes.rolling(20).std().iloc[-1]
    if pd.isna(sma20) or pd.isna(std20) or std20 == 0:
        return None

    upper_bb = sma20 + 2 * std20
    lower_bb = sma20 - 2 * std20
    close_now = float(df_m5["close"].iloc[idx])
    high_now = float(df_m5["high"].iloc[idx])
    low_now = float(df_m5["low"].iloc[idx])
    rsi = context.get("rsi", 50)

    band_width = upper_bb - lower_bb
    if band_width <= 0:
        return None
    quarter = band_width * 0.25
    near_lower = close_now <= lower_bb + quarter
    near_upper = close_now >= upper_bb - quarter
    setup_e_min_rr = 1.0
    setup_e_sl_min = 0.001

    context["_bb_debug"] = {
        "price": round(close_now, 4),
        "upper_bb": round(upper_bb, 4),
        "lower_bb": round(lower_bb, 4),
        "sma20": round(float(sma20), 4),
        "rsi": round(rsi, 1),
        "near_lower": near_lower,
        "near_upper": near_upper,
    }

    if near_lower and rsi < 35:
        raw_sl = min(low_now, lower_bb) - close_now * 0.001
        sl_dist = close_now - raw_sl
        if sl_dist / close_now < setup_e_sl_min:
            raw_sl = close_now * (1 - setup_e_sl_min)
            sl_dist = close_now - raw_sl
        if sl_dist / close_now > SL_MAX_PCT:
            raw_sl = close_now * (1 - SL_MAX_PCT)
            sl_dist = close_now - raw_sl

        tp = float(sma20)
        if tp <= close_now:
            return None

        rr = (tp - close_now) / sl_dist
        context["_bb_debug"]["rr"] = round(rr, 2)
        context["_bb_debug"]["side"] = "BUY"
        if rr < setup_e_min_rr:
            return None

        return {
            "signal": "BUY",
            "entry": round(close_now, 4),
            "sl": round(raw_sl, 4),
            "tp": round(tp, 4),
            "rr": round(rr, 2),
            "setup": "setup_E_bb_mean_reversion",
        }

    if near_upper and rsi > 65:
        raw_sl = max(high_now, upper_bb) + close_now * 0.001
        sl_dist = raw_sl - close_now
        if sl_dist / close_now < setup_e_sl_min:
            raw_sl = close_now * (1 + setup_e_sl_min)
            sl_dist = raw_sl - close_now
        if sl_dist / close_now > SL_MAX_PCT:
            raw_sl = close_now * (1 + SL_MAX_PCT)
            sl_dist = raw_sl - close_now

        tp = float(sma20)
        if tp >= close_now:
            return None

        rr = (close_now - tp) / sl_dist
        context["_bb_debug"]["rr"] = round(rr, 2)
        context["_bb_debug"]["side"] = "SELL"
        if rr < setup_e_min_rr:
            return None

        return {
            "signal": "SELL",
            "entry": round(close_now, 4),
            "sl": round(raw_sl, 4),
            "tp": round(tp, 4),
            "rr": round(rr, 2),
            "setup": "setup_E_bb_mean_reversion",
        }

    return None


def validate_ai_trade_decision(decision: dict, context: dict,
                               df_m5: pd.DataFrame = None) -> tuple[bool, str]:
    """
    Confirms the AI signal follows the current market structure.
    """
    signal = str(decision.get("signal", "")).upper()
    if signal not in ("BUY", "SELL"):
        return False, f"invalid signal {signal!r}"

    primary = context.get("primary_trend") or context.get("m15_trend", "")
    if primary == "UPTREND" and signal != "BUY":
        return False, f"M15={primary} requires BUY"
    if primary == "DOWNTREND" and signal != "SELL":
        return False, f"M15={primary} requires SELL"

    setup_code = _setup_code(decision.get("analysis", {}).get("setup_identified", ""))
    if context.get("trend_rollover") and setup_code in ("A", "B"):
        return False, (
            f"M15 rollover: continuation {setup_code} blocked; "
            f"wait for Setup C or stabilization"
        )
    if setup_code == "C":
        setup_c_ok, setup_c_reason = _setup_c_reversal_ok(signal, context, df_m5)
        if not setup_c_ok:
            return False, setup_c_reason

    timing_ok, timing_reason = _m5_entry_timing_ok(signal, context, df_m5, setup_code=setup_code)
    if not timing_ok:
        return False, timing_reason

    return True, "OK"


def _setup_code(setup_name: str) -> str:
    text = str(setup_name or "").strip().lower()
    if "setup " in text:
        return text.split("setup ", 1)[1][:1].upper()
    if "setup_" in text:
        return text.split("setup_", 1)[1][:1].upper()
    return text[:1].upper() if text else "?"


def _bb_snapshot(df_m5: pd.DataFrame) -> dict:
    closes = df_m5["close"].astype(float)
    if len(closes) < 20:
        last = float(closes.iloc[-1])
        return {"sma20": last, "upper_bb": last, "lower_bb": last}

    sma20 = closes.rolling(20).mean().iloc[-1]
    std20 = closes.rolling(20).std().iloc[-1]
    if pd.isna(sma20) or pd.isna(std20):
        last = float(closes.iloc[-1])
        return {"sma20": last, "upper_bb": last, "lower_bb": last}

    return {
        "sma20": float(sma20),
        "upper_bb": float(sma20 + 2 * std20),
        "lower_bb": float(sma20 - 2 * std20),
    }


def _dedupe_levels(values: list[float], reverse: bool = False) -> list[float]:
    cleaned = []
    for value in values:
        if value is None or not np.isfinite(value):
            continue
        rounded = round(float(value), 6)
        if rounded not in cleaned:
            cleaned.append(rounded)
    return sorted(cleaned, reverse=reverse)


def _recent_box_levels(df_m5: pd.DataFrame, window: int = 6,
                       exclude_last: int = 0) -> tuple[float, float]:
    source = df_m5.iloc[:-exclude_last] if exclude_last > 0 and len(df_m5) > exclude_last else df_m5
    box = source.tail(window) if not source.empty else df_m5.tail(window)
    return float(box["low"].min()), float(box["high"].max())


def _pullback_cluster_levels(df_m5: pd.DataFrame, signal: str, lookback: int = 12) -> tuple[float, float]:
    cluster = df_m5.tail(lookback).copy()
    if signal == "BUY":
        red = cluster[cluster["close"] < cluster["open"]]
        lows = red["low"] if not red.empty else cluster["low"]
        return float(lows.min()), float(cluster["high"].max())
    green = cluster[cluster["close"] > cluster["open"]]
    highs = green["high"] if not green.empty else cluster["high"]
    return float(cluster["low"].min()), float(highs.max())


def _setup_c_reversal_ok(signal: str, context: dict,
                         df_m5: pd.DataFrame | None) -> tuple[bool, str]:
    """
    Setup C gets a little more freedom, but only when the reversal candle is
    genuinely strong.
    """
    if df_m5 is None or len(df_m5) < 5:
        return False, "Setup C needs more M5 candles"

    closes = df_m5["close"].astype(float)
    ema34_series = closes.ewm(span=34, adjust=False).mean()
    ema89_series = closes.ewm(span=89, adjust=False).mean()
    latest = df_m5.iloc[-1]
    latest_close = float(latest["close"])
    latest_open = float(latest["open"])
    latest_volume = float(latest.get("volume", 0.0))
    prev_vol_mean = float(df_m5.tail(5).iloc[:-1]["volume"].mean())
    atr = float(context.get("atr_m15") or context.get("atr") or 0.0)
    pattern = detect_candle_pattern(df_m5, len(df_m5) - 1)
    recent_move = float(closes.iloc[-2] - closes.iloc[-4])
    recent_low = float(df_m5.tail(4)["low"].min())
    recent_high = float(df_m5.tail(4)["high"].max())

    candle_range = float(latest["high"] - latest["low"])
    body = abs(latest_close - latest_open)
    lower_wick = min(latest_open, latest_close) - float(latest["low"])
    upper_wick = float(latest["high"]) - max(latest_open, latest_close)
    body_ratio = body / candle_range if candle_range > 0 else 0.0
    relaxed_setup_c = _env_bool("SETUP_C_RELAXED", False)
    soft_body_ratio = _env_float("SETUP_C_SOFT_BODY_RATIO", 0.45)
    volume_mult = _env_float("SETUP_C_VOLUME_MULT", 1.00 if relaxed_setup_c else 1.10)
    fake_move_atr_mult = _env_float("SETUP_C_FAKE_MOVE_ATR_MULT", 0.20 if relaxed_setup_c else 0.35)

    if signal == "BUY":
        strong_engulf = pattern["pattern"] == "bullish_engulfing"
        strong_pin = (
            pattern["pattern"] == "bullish_pinbar"
            and lower_wick >= body * 2.5
            and latest_close > latest_open
        )
        soft_body = (
            relaxed_setup_c
            and latest_close > latest_open
            and body_ratio >= soft_body_ratio
        )
        if not (strong_engulf or strong_pin or soft_body):
            return False, "Setup C BUY needs bullish engulfing or strong rejection pinbar"
        if latest_close <= float(ema34_series.iloc[-1]):
            return False, "Setup C BUY needs close back above M5 EMA34"
        if latest_volume <= prev_vol_mean * volume_mult:
            return False, "Setup C BUY needs reversal volume above recent candles"
        if atr > 0 and recent_move > -(atr * fake_move_atr_mult):
            return False, "Setup C BUY needs a real fake-drop before reversal"
        if latest_close <= recent_low:
            return False, "Setup C BUY reversal has not reclaimed local structure"
        return True, "OK"

    strong_engulf = pattern["pattern"] == "bearish_engulfing"
    strong_pin = (
        pattern["pattern"] == "bearish_pinbar"
        and upper_wick >= body * 2.5
        and latest_close < latest_open
    )
    soft_body = (
        relaxed_setup_c
        and latest_close < latest_open
        and body_ratio >= soft_body_ratio
    )
    if not (strong_engulf or strong_pin or soft_body):
        return False, "Setup C SELL needs bearish engulfing or strong rejection pinbar"
    if latest_close >= float(ema34_series.iloc[-1]):
        return False, "Setup C SELL needs close back below M5 EMA34"
    if latest_volume <= prev_vol_mean * volume_mult:
        return False, "Setup C SELL needs reversal volume above recent candles"
    if atr > 0 and recent_move < atr * fake_move_atr_mult:
        return False, "Setup C SELL needs a real fake-pump before reversal"
    if latest_close >= recent_high:
        return False, "Setup C SELL reversal has not lost local structure"
    return True, "OK"


def _m5_entry_timing_ok(signal: str, context: dict,
                        df_m5: pd.DataFrame | None,
                        setup_code: str = "?") -> tuple[bool, str]:
    """
    Fast veto so higher-timeframe trend does not force entries during an active
    5m counter-impulse.
    """
    if df_m5 is None or len(df_m5) < 6:
        return True, "OK"

    closes = df_m5["close"].astype(float)
    ema34_series = closes.ewm(span=34, adjust=False).mean()
    ema34_m5 = float(ema34_series.iloc[-1])
    ema34_prev = float(ema34_series.iloc[-2])
    latest = df_m5.iloc[-1]
    latest_open = float(latest["open"])
    latest_close = float(latest["close"])
    latest_volume = float(latest.get("volume", 0.0))
    prev_close = float(closes.iloc[-2])
    atr = float(context.get("atr_m15") or context.get("atr") or 0.0)
    box_low, box_high = _recent_box_levels(df_m5, window=6, exclude_last=1)

    recent = df_m5.tail(4).copy().reset_index(drop=True)
    three_bar_move = float(recent["close"].iloc[-1] - recent["close"].iloc[0])
    pre_reclaim_move = float(closes.iloc[-2] - closes.iloc[-4])
    prev_vol_mean = float(df_m5.tail(6).iloc[:-1]["volume"].mean())
    vol_expanding = latest_volume > prev_vol_mean * 1.10 if prev_vol_mean > 0 else False

    latest_bearish = latest_close < latest_open
    latest_bullish = latest_close > latest_open
    prior_below_ema = prev_close <= ema34_prev
    prior_above_ema = prev_close >= ema34_prev

    if signal == "BUY":
        if latest_bearish and latest_close < ema34_m5:
            return False, f"M5 bearish impulse: close {latest_close:.4f} below EMA34 {ema34_m5:.4f}"
        if atr > 0 and three_bar_move <= -atr and latest_close <= ema34_m5:
            return False, f"M5 bearish impulse: 3-candle drop {abs(three_bar_move):.4f} >= ATR {atr:.4f}"
        if (
            atr > 0
            and pre_reclaim_move <= -(atr * 0.75)
            and latest_bullish
            and latest_close > ema34_m5
            and prior_below_ema
            and setup_code != "C"
        ):
            return False, (
                f"M5 flush recovery not confirmed: fresh reclaim above EMA34 {ema34_m5:.4f} "
                f"after {abs(pre_reclaim_move):.4f} drop; wait for hold"
            )
        if latest_close < box_low:
            return False, f"M5 lost micro support {box_low:.4f}"
        if vol_expanding and latest_bearish and latest_close <= ema34_m5:
            return False, "M5 sell pressure expanding on the latest candle"
        return True, "OK"

    if latest_bullish and latest_close > ema34_m5:
        return False, f"M5 bullish impulse: close {latest_close:.4f} above EMA34 {ema34_m5:.4f}"
    if atr > 0 and three_bar_move >= atr and latest_close >= ema34_m5:
        return False, f"M5 bullish impulse: 3-candle rally {abs(three_bar_move):.4f} >= ATR {atr:.4f}"
    if (
        atr > 0
        and pre_reclaim_move >= atr * 0.75
        and latest_bearish
        and latest_close < ema34_m5
        and prior_above_ema
        and setup_code != "C"
    ):
        return False, (
            f"M5 squeeze-down not confirmed: fresh reclaim below EMA34 {ema34_m5:.4f} "
            f"after {abs(pre_reclaim_move):.4f} rally; wait for hold"
        )
    if latest_close > box_high:
        return False, f"M5 broke above micro resistance {box_high:.4f}"
    if vol_expanding and latest_bullish and latest_close >= ema34_m5:
        return False, "M5 buy pressure expanding on the latest candle"
    return True, "OK"


def _setup_b_breakout_gate(signal: str, entry: float, atr: float,
                           box_low: float, box_high: float,
                           df_m5: pd.DataFrame) -> tuple[bool, str]:
    if df_m5.empty:
        return False, "missing M5 candles for breakout validation"

    last = df_m5.iloc[-1]
    last_open = float(last["open"])
    last_close = float(last["close"])
    breakout_buffer = max(entry * BREAKOUT_CONFIRM_PCT, atr * BREAKOUT_CONFIRM_ATR_MULT)

    if signal == "BUY":
        breakout_level = box_high + breakout_buffer
        if last_close <= last_open:
            return False, f"breakout candle close {last_close:.4f} is not bullish"
        if last_close <= breakout_level:
            return False, (
                f"breakout close {last_close:.4f} not above box {box_high:.4f} "
                f"+ buffer {breakout_buffer:.4f}"
            )
        return True, ""

    breakout_level = box_low - breakout_buffer
    if last_close >= last_open:
        return False, f"breakout candle close {last_close:.4f} is not bearish"
    if last_close >= breakout_level:
        return False, (
            f"breakout close {last_close:.4f} not below box {box_low:.4f} "
            f"- buffer {breakout_buffer:.4f}"
        )
    return True, ""


def build_trade_plan(signal: str, setup_name: str, context: dict,
                     df_m5: pd.DataFrame) -> tuple[dict | None, str]:
    """
    Deterministically builds entry, stop loss, and take profit from structure.
    Gemini decides whether a setup is valid; Python owns the trade levels.
    """
    signal = str(signal or "").upper()
    if signal not in ("BUY", "SELL"):
        return None, f"invalid signal {signal!r}"

    entry = float(context.get("current_price") or float(df_m5["close"].iloc[-1]))
    if entry <= 0:
        return None, "missing entry price"

    setup_code = _setup_code(setup_name)
    market_mode = str(context.get("market_mode", "") or "")
    is_setup_e = setup_code == "E" or market_mode == "SIDEWAY"
    is_range = setup_code == "D" or bool(context.get("is_range"))

    atr = float(context.get("atr_m15") or context.get("atr") or 0.0)
    sr = context.get("sr") or {}
    support = sr.get("support")
    resistance = sr.get("resistance")
    support_levels = _dedupe_levels(
        list(sr.get("support_levels") or []) + ([support] if support is not None else []),
        reverse=True,
    )
    resistance_levels = _dedupe_levels(
        list(sr.get("resistance_levels") or []) + ([resistance] if resistance is not None else []),
    )

    m5_window = df_m5.tail(8).copy()
    recent_low = float(m5_window["low"].min())
    recent_high = float(m5_window["high"].max())
    pullback_low, pullback_high = _pullback_cluster_levels(df_m5, signal)
    box_low, box_high = _recent_box_levels(df_m5, exclude_last=1)
    ema89 = float(df_m5["close"].astype(float).ewm(span=89, adjust=False).mean().iloc[-1])
    bb = _bb_snapshot(df_m5)

    min_rr_default = max(0.5, _env_float("TRADE_MIN_RR", MIN_RR))
    min_rr_range = max(0.5, _env_float("TRADE_MIN_RR_RANGE", min_rr_default))
    min_rr_setup_e = max(0.5, _env_float("SETUP_E_MIN_RR", 1.0))
    fee_cost = entry * 0.0018
    if is_setup_e:
        min_rr = min_rr_setup_e
        base_buffer = max(entry * 0.0010, atr * 0.10)
        min_risk = max(entry * 0.0010, atr * 0.25)
    elif is_range:
        min_rr = min_rr_range
        base_buffer = max(entry * 0.0015, atr * 0.20)
        min_risk = max(entry * SL_MIN_PCT, atr * 0.35)
    else:
        min_rr = min_rr_default
        base_buffer = max(entry * 0.0015, atr * 0.25)
        min_risk = max(entry * SL_MIN_PCT, atr * 0.55)

    if setup_code == "B":
        breakout_ok, breakout_reason = _setup_b_breakout_gate(
            signal, entry, atr, box_low, box_high, df_m5
        )
        if not breakout_ok:
            return None, breakout_reason

    if signal == "BUY":
        if setup_code == "A":
            stop_refs = [pullback_low, support, ema89]
        elif setup_code == "B":
            stop_refs = [box_low, support, ema89]
        elif setup_code == "C":
            stop_refs = [pullback_low, recent_low, support, ema89]
        elif setup_code == "D":
            stop_refs = [support, recent_low]
        elif is_setup_e:
            stop_refs = [bb["lower_bb"], recent_low]
        else:
            stop_refs = [pullback_low, recent_low, support, ema89, bb["lower_bb"]]
        stop_candidates = _dedupe_levels([x for x in stop_refs if x is not None and x < entry], reverse=True)
        if not stop_candidates:
            return None, "no valid bullish invalidation below entry"

        stop_anchor = stop_candidates[-1] if len(stop_candidates) > 1 else stop_candidates[0]
        raw_sl = stop_anchor - base_buffer
        risk = entry - raw_sl
        if risk < min_risk:
            raw_sl = entry - min_risk

        target_candidates = []
        if is_setup_e:
            target_candidates.extend([bb["sma20"], bb["upper_bb"]])
        elif setup_code == "B":
            target_candidates.extend([recent_high, box_high])
        else:
            target_candidates.extend([recent_high])
        target_candidates.extend(resistance_levels[:5])
        target_levels = _dedupe_levels([x for x in target_candidates if x > entry])
        if not target_levels:
            return None, "no valid bullish target above entry"

        sl = raw_sl
        risk = entry - sl
        net_risk = risk + fee_cost
        tp1 = target_levels[0]
        tp2 = target_levels[1] if len(target_levels) > 1 else target_levels[0]
        strong_breakout = _is_strong_breakout_context(
            setup_code, context, entry, atr, box_low, box_high, signal
        )

        def _buy_rr(target: float) -> tuple[float, float]:
            gross_reward = target - entry
            net_reward = gross_reward - fee_cost
            rr = net_reward / net_risk if net_risk > 0 else 0.0
            gross_rr = gross_reward / risk if risk > 0 else 0.0
            return rr, gross_rr

        tp1_rr, tp1_gross_rr = _buy_rr(tp1)
        tp2_rr, tp2_gross_rr = _buy_rr(tp2)

        chosen_tp = None
        best_target = tp1
        best_rr = tp1_rr
        best_gross_rr = tp1_gross_rr
        target_mode = "TP1"

        if tp1_rr >= min_rr:
            chosen_tp = tp1
        elif strong_breakout:
            farther_targets = target_levels[1:] if len(target_levels) > 1 else []
            for target in farther_targets:
                rr, gross_rr = _buy_rr(target)
                if rr >= min_rr:
                    chosen_tp = target
                    best_target = target
                    best_rr = rr
                    best_gross_rr = gross_rr
                    target_mode = "TP2"
                    break
                if rr > best_rr:
                    best_target = target
                    best_rr = rr
                    best_gross_rr = gross_rr
        if chosen_tp is None:
            return None, (
                f"TP1 RR {tp1_rr:.2f} < {min_rr:.2f}"
                + (f"; TP2 RR {tp2_rr:.2f}" if len(target_levels) > 1 else "")
                + f" (tp1={tp1:.4f})"
            )

    else:
        if setup_code == "A":
            stop_refs = [pullback_high, resistance, ema89]
        elif setup_code == "B":
            stop_refs = [box_high, resistance, ema89]
        elif setup_code == "C":
            stop_refs = [pullback_high, recent_high, resistance, ema89]
        elif setup_code == "D":
            stop_refs = [resistance, recent_high]
        elif is_setup_e:
            stop_refs = [bb["upper_bb"], recent_high]
        else:
            stop_refs = [pullback_high, recent_high, resistance, ema89, bb["upper_bb"]]
        stop_candidates = _dedupe_levels([x for x in stop_refs if x is not None and x > entry])
        if not stop_candidates:
            return None, "no valid bearish invalidation above entry"

        stop_anchor = stop_candidates[-1] if len(stop_candidates) > 1 else stop_candidates[0]
        raw_sl = stop_anchor + base_buffer
        risk = raw_sl - entry
        if risk < min_risk:
            raw_sl = entry + min_risk

        target_candidates = []
        if is_setup_e:
            target_candidates.extend([bb["sma20"], bb["lower_bb"]])
        elif setup_code == "B":
            target_candidates.extend([recent_low, box_low])
        else:
            target_candidates.extend([recent_low])
        target_candidates.extend(support_levels[:5])
        target_levels = _dedupe_levels([x for x in target_candidates if x < entry], reverse=True)
        if not target_levels:
            return None, "no valid bearish target below entry"

        sl = raw_sl
        risk = sl - entry
        net_risk = risk + fee_cost
        tp1 = target_levels[0]
        tp2 = target_levels[1] if len(target_levels) > 1 else target_levels[0]
        strong_breakout = _is_strong_breakout_context(
            setup_code, context, entry, atr, box_low, box_high, signal
        )

        def _sell_rr(target: float) -> tuple[float, float]:
            gross_reward = entry - target
            net_reward = gross_reward - fee_cost
            rr = net_reward / net_risk if net_risk > 0 else 0.0
            gross_rr = gross_reward / risk if risk > 0 else 0.0
            return rr, gross_rr

        tp1_rr, tp1_gross_rr = _sell_rr(tp1)
        tp2_rr, tp2_gross_rr = _sell_rr(tp2)

        chosen_tp = None
        best_target = tp1
        best_rr = tp1_rr
        best_gross_rr = tp1_gross_rr
        target_mode = "TP1"

        if tp1_rr >= min_rr:
            chosen_tp = tp1
        elif strong_breakout:
            farther_targets = target_levels[1:] if len(target_levels) > 1 else []
            for target in farther_targets:
                rr, gross_rr = _sell_rr(target)
                if rr >= min_rr:
                    chosen_tp = target
                    best_target = target
                    best_rr = rr
                    best_gross_rr = gross_rr
                    target_mode = "TP2"
                    break
                if rr > best_rr:
                    best_target = target
                    best_rr = rr
                    best_gross_rr = gross_rr
        if chosen_tp is None:
            return None, (
                f"TP1 RR {tp1_rr:.2f} < {min_rr:.2f}"
                + (f"; TP2 RR {tp2_rr:.2f}" if len(target_levels) > 1 else "")
                + f" (tp1={tp1:.4f})"
            )

    final_risk_pct = abs(entry - sl) / entry
    if final_risk_pct > SL_MAX_PCT:
        return None, f"stop distance {final_risk_pct*100:.3f}% exceeds max {SL_MAX_PCT*100:.2f}%"

    plan = {
        "entry_price": round(entry, 6),
        "stop_loss": round(sl, 6),
        "take_profit": round(chosen_tp, 6),
        "take_profit_1": round(tp1, 6),
        "take_profit_2": round(tp2, 6),
        "target_mode": target_mode,
        "rr": round(best_rr, 2),
        "gross_rr": round(best_gross_rr, 2),
        "tp1_rr": round(tp1_rr, 2),
        "tp2_rr": round(tp2_rr, 2),
        "min_rr": round(min_rr, 2),
        "stop_anchor": round(stop_anchor, 6),
        "target_anchor": round(best_target, 6),
        "strong_breakout": bool(strong_breakout),
        "estimated_cost": round(fee_cost, 6),
    }
    reason = (
        f"stop@{plan['stop_anchor']} TP1@{plan['take_profit_1']} TP2@{plan['take_profit_2']} "
        f"selected={plan['target_mode']} RRnet={plan['rr']:.2f} RRgross={plan['gross_rr']:.2f} "
        f"RR1={plan['tp1_rr']:.2f} RR2={plan['tp2_rr']:.2f} min={plan['min_rr']:.2f}"
    )
    return plan, reason
