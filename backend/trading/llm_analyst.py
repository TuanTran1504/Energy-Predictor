"""
llm_analyst.py — Gemini Flash vision + language analyst.

Sends the M5 chart image + quantitative context to Gemini Flash.
Returns a structured trading decision dict.

Requires:
  GOOGLE_API_KEY in environment (Google AI Studio key)

Model: gemini-2.0-flash  (vision + text, cheap, fast)
API:   OpenAI-compatible endpoint provided by Google
"""

import json
import os

import pandas as pd
from openai import OpenAI

def _get_client() -> OpenAI:
    return OpenAI(
        api_key=os.environ["GOOGLE_API_KEY"],
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

MODEL = "gemini-2.5-flash-lite"


PATTERN_LIBRARY = {

    "setup_A_buy": """
=== SETUP A — EMA PULLBACK BUY (UPTREND) ===
Context: EMA34 (green) is ABOVE EMA89 (orange). Gap is wide or stable.

REQUIRED on M5 chart — ALL must be true:
[1] PULLBACK: Price has retraced from higher levels and is touching or near EMA34 or EMA89.
    "Near" = candle body or lower wick within 0.3% of the EMA line.
[2] SIGNAL CANDLE — one of:
    Option A (Bullish Pinbar): lower wick ≥ 2× body, body closes in upper 50% of full candle range, short upper wick.
    Option B (Bullish Engulfing): current green candle body fully engulfs previous red candle body.
[3] VOLUME: signal candle volume > average of previous 3 candles (visually taller bar).
[4] ENTRY: at signal candle close. SL below signal candle low minus 0.25%.
    TP toward nearest H1 resistance (red dashed line).

FALSE SIGNALS — output WAIT if:
  - Signal candle has small body, no clear shape (Doji / Spinning Top)
  - EMAs are converging (gap closing)
  - Volume is lower than surrounding candles
  - Price has broken below EMA89 for more than 3 consecutive candles
""",

    "setup_A_sell": """
=== SETUP A — EMA PULLBACK SELL (DOWNTREND) ===
Context: EMA34 (green) is BELOW EMA89 (orange). Gap is wide or stable.

REQUIRED on M5 chart — ALL must be true:
[1] RALLY PULLBACK: Price has rallied from lower levels and is touching or near EMA34 or EMA89.
[2] SIGNAL CANDLE — one of:
    Option A (Bearish Pinbar): upper wick ≥ 2× body, body closes in lower 50%, short lower wick.
    Option B (Bearish Engulfing): current red candle body fully engulfs previous green candle body.
[3] VOLUME: signal candle volume > average of previous 3 candles.
[4] ENTRY: at signal candle close. SL above signal candle high plus 0.25%.
    TP toward nearest H1 support (green dashed line).

FALSE SIGNALS — output WAIT if:
  - No clear upper wick, body dominates candle (> 70%)
  - EMAs converging
  - Low volume on signal candle
  - Price broken above EMA89 for more than 3 consecutive candles
""",

    "setup_B_buy": """
=== SETUP B — BREAKOUT BUY (UPTREND, HIGH SCORE) ===
Context: Strong uptrend, EMA gap widening.

REQUIRED on M5 chart — ALL must be true:
[1] CONSOLIDATION BOX: 3–8 candles moving sideways forming a visible box.
    ALL candle bodies in the box must be ABOVE EMA89 (clear gap — no touches).
    Volume during consolidation is LOW (bars visually shorter than normal).
[2] BREAKOUT CANDLE: Green candle closes ABOVE the box top.
    Body ≥ 60% of full candle range. Short upper wick (≤ 30% of body). Closes near high.
[3] VOLUME: Breakout candle volume ≥ 1.5× the blue Volume MA line. Must spike visibly.
[4] ENTRY: at breakout candle close. SL below box bottom minus 0.25%.
    TP at next H1 resistance.

FALSE SIGNALS — output WAIT if:
  - Breakout candle has long upper wick (> 50% of body): rejection at top
  - Volume does not spike clearly
  - Any consolidation candle touched EMA89
  - Fewer than 3 candles in consolidation
""",

    "setup_B_sell": """
=== SETUP B — BREAKDOWN SELL (DOWNTREND, HIGH SCORE) ===
Context: Strong downtrend, EMA gap widening downward.

REQUIRED on M5 chart — ALL must be true:
[1] CONSOLIDATION BOX: 3–8 sideways candles. ALL candle bodies BELOW EMA89 (clear gap).
    Volume during consolidation is LOW.
[2] BREAKDOWN CANDLE: Red candle closes BELOW the box bottom.
    Body ≥ 60% of full candle range. Short lower wick (≤ 30% of body). Closes near low.
[3] VOLUME: Breakdown candle volume ≥ 1.5× Volume MA. Clear spike.
[4] ENTRY: at breakdown candle close. SL above box top plus 0.25%.
    TP at next H1 support.

FALSE SIGNALS — output WAIT if:
  - Long lower wick on breakdown candle
  - Volume does not spike
  - Consolidation candle touched EMA89 from below
""",

    "setup_C_buy": """
=== SETUP C — FAKE DROP BUY (UPTREND) ===
Context: Overall uptrend. Price briefly fakes a breakdown then reverses hard.

SEQUENCE — all steps in order:
[STEP 1] FAKE DROP: 1–3 red candles drop below EMA34 or EMA89.
  Volume during drop is LOW (bars smaller than or equal to Volume MA line).
  Price does NOT go more than 0.5% below EMA89.
[STEP 2] REVERSAL CANDLE: A strong Bullish Pinbar appears.
  Lower wick ≥ 2.5× body. Body closes in top 1/3 of full candle range.
  Candle low is AT or BELOW EMA34/89. Close is ABOVE EMA34.
  OR: Bullish Engulfing that engulfs the entire fake drop cluster.
[STEP 3] REVERSAL VOLUME: Volume on the reversal candle is clearly HIGHER than during the fake drop.
[STEP 4] QUALITY FILTER: Prefer only strong reversals.
  Best case = Bullish Engulfing with a decisive close back above EMA34.
  Acceptable alternative = strong Bullish Pinbar / rejection wick with clearly higher volume.
[4] ENTRY: at reversal candle close. SL below fake drop low minus 0.25%.
    TP at next H1 resistance.

FALSE SIGNALS — output WAIT if:
  - Volume during the drop is HIGH or spikes: real breakdown, not fake
  - Price drops more than 1% below EMA89
  - Reversal candle is weak, small-bodied, or closes only marginally above EMA34
  - Reversal candle does not have a long lower wick and is not a clear bullish engulfing
  - Next candle after reversal closes lower (trap failed)
""",

    "setup_C_sell": """
=== SETUP C — FAKE PUMP SELL (DOWNTREND) ===
Context: Overall downtrend. Price briefly fakes a breakout then drops hard.

SEQUENCE — all steps in order:
[STEP 1] FAKE PUMP: 1–3 green candles rally above EMA34 or EMA89.
  Volume during rally is LOW.
  Price does NOT go more than 0.5% above EMA89.
[STEP 2] REVERSAL CANDLE: A strong Bearish Pinbar appears.
  Upper wick ≥ 2.5× body. Body closes in bottom 1/3 of full candle range.
  Candle high is AT or ABOVE EMA34/89. Close is BELOW EMA34.
  OR: Bearish Engulfing that engulfs the entire fake pump cluster.
[STEP 3] REVERSAL VOLUME: Clearly higher than during the fake pump.
[STEP 4] QUALITY FILTER: Prefer only strong reversals.
  Best case = Bearish Engulfing with a decisive close back below EMA34.
  Acceptable alternative = strong Bearish Pinbar / rejection wick with clearly higher volume.
[4] ENTRY: at reversal candle close. SL above fake pump high plus 0.25%.
    TP at next H1 support.

FALSE SIGNALS — output WAIT if:
  - Reversal candle is weak, small-bodied, or closes only marginally below EMA34
  - Reversal candle does not have a long upper wick and is not a clear bearish engulfing
  - Next candle after reversal closes higher (trap failed)
""",

    "setup_D_buy": """
=== SETUP D — RANGE BOUNCE BUY (VOLATILE RANGE) ===
Context: EMA34 ≈ EMA89 (intertwined). Price oscillating between H1 S/R levels.

REQUIRED:
[1] Price is near H1 SUPPORT (green dashed line) — within 0.3% of it.
[2] SIGNAL CANDLE at or near support:
    Option A: Bullish Pinbar with lower wick piercing the support level.
    Option B: Bullish Engulfing at the support zone.
    Option C: Second touch of support (double bottom) — price bouncing off same level twice.
[3] VOLUME: Higher than the 3 previous candles.
[4] ENTRY: at signal candle close (must be ABOVE the support line).
    SL below support minus 0.25%. TP toward H1 resistance. R:R ≥ 1.5.

FALSE SIGNALS — output WAIT if:
  - Price has broken clearly through support (more than 3 candles held below)
  - Signal candle has no clear lower wick
  - Low volume at the support level
  - Price is in the middle of the range (far from both S and R)
""",

    "setup_D_sell": """
=== SETUP D — RANGE BOUNCE SELL (VOLATILE RANGE) ===
Context: EMA34 ≈ EMA89 (intertwined). Price oscillating between H1 S/R levels.

REQUIRED:
[1] Price is near H1 RESISTANCE (red dashed line) — within 0.3%.
[2] SIGNAL CANDLE at or near resistance:
    Option A: Bearish Pinbar with upper wick piercing the resistance level.
    Option B: Bearish Engulfing at the resistance zone.
    Option C: Second touch of resistance (double top).
[3] VOLUME: Higher than the 3 previous candles.
[4] ENTRY: at signal candle close (must be BELOW resistance).
    SL above resistance plus 0.25%. TP toward H1 support. R:R ≥ 1.5.
""",

    "setup_E_buy": """
=== SETUP E — BB MEAN REVERSION BUY (SIDEWAY) ===
Context: EMA34 and EMA89 are FLAT and INTERTWINED. Market has no directional trend.
Price has extended to the LOWER Bollinger Band (white dashed) with RSI deeply oversold.

REQUIRED — ALL must be true:
[1] BAND TOUCH: Current candle close is AT or BELOW the lower Bollinger Band (white dashed).
    The candle low or body should visually touch or pierce the lower band.
[2] RSI OVERSOLD: Quantitative data shows RSI < 30. Confirm this is below the neutral zone.
[3] CANDLE CONFIRMATION (any one):
    Option A: Bullish Pinbar — lower wick ≥ 2× body, closes in upper half of range.
    Option B: Bullish Engulfing — current green candle engulfs previous red candle.
    Option C: Rejection wick — long lower wick shows rejection at the band level.
[4] VOLUME: Signal candle has equal or higher volume than the 2 preceding candles.
[5] ENTRY: at signal candle close. SL below candle low minus 0.2% buffer.
    TP = upper Bollinger Band (the opposite band — full range target).

FALSE SIGNALS — output WAIT if:
  - Candle closes well BELOW the lower band (momentum breakdown, not reversal)
  - Volume is clearly falling (no buying interest at the band)
  - RSI is between 30–50 (not truly oversold)
  - EMA34 is trending sharply downward (not flat sideway)
  - SL distance exceeds 0.8% of entry price
  - R:R < 1.5 after applying SMA20 as TP cap
""",

    "setup_E_sell": """
=== SETUP E — BB MEAN REVERSION SELL (SIDEWAY) ===
Context: EMA34 and EMA89 are FLAT and INTERTWINED. Market has no directional trend.
Price has extended to the UPPER Bollinger Band (white dashed) with RSI deeply overbought.

REQUIRED — ALL must be true:
[1] BAND TOUCH: Current candle close is AT or ABOVE the upper Bollinger Band (white dashed).
    The candle high or body should visually touch or pierce the upper band.
[2] RSI OVERBOUGHT: Quantitative data shows RSI > 70. Confirm this is above the neutral zone.
[3] CANDLE CONFIRMATION (any one):
    Option A: Bearish Pinbar — upper wick ≥ 2× body, closes in lower half of range.
    Option B: Bearish Engulfing — current red candle engulfs previous green candle.
    Option C: Rejection wick — long upper wick shows rejection at the band level.
[4] VOLUME: Signal candle has equal or higher volume than the 2 preceding candles.
[5] ENTRY: at signal candle close. SL above candle high plus 0.2% buffer.
    TP = lower Bollinger Band (the opposite band — full range target).

FALSE SIGNALS — output WAIT if:
  - Candle closes well ABOVE the upper band (momentum breakout, not reversal)
  - Volume is clearly falling (no selling interest at the band)
  - RSI is between 50–70 (not truly overbought)
  - EMA34 is trending sharply upward (not flat sideway)
  - SL distance exceeds 0.8% of entry price
  - R:R < 1.5 after applying SMA20 as TP cap
""",

    "wait": """
=== MANDATORY WAIT CONDITIONS ===
Choose WAIT if ANY of the following:
- Signal candle shape is ambiguous (Doji, Spinning Top, small body)
- Volume on signal candle is LOWER than surrounding candles
- Both required conditions present but R:R < 1.5 after applying H1 S/R cap
- EMAs are choppy (crossing back and forth in last 10 candles)
- Price is in the middle of a range with no clear S/R nearby
- You can see only PART of the required setup (e.g. nice candle but wrong volume)

RULE: When in doubt → WAIT. A missed trade doesn't lose money. A bad trade does.
""",
}


def _build_prompt(context: dict) -> tuple[str, str]:
    """Returns (system_prompt, user_text)."""

    h1          = context.get("h1_trend", "?")
    primary     = context.get("primary_trend", context.get("m15_trend", "?"))
    market_mode = context.get("market_mode", primary)
    trend_rollover = bool(context.get("trend_rollover"))
    trend_rollover_reason = context.get("trend_rollover_reason", "")
    score       = context.get("score", 0)
    adx         = context.get("adx", 0)
    rsi         = context.get("rsi", 0)
    atr         = context.get("atr_m15", 0)
    sr          = context.get("sr", {})
    allowed_dir = context.get("allowed_direction", "BOTH")
    range_bias  = context.get("range_bias", "MIDDLE")
    candle_text = context.get("candle_summary", "")
    symbol      = context.get("symbol", "BTC/USDT")
    symbol_base = str(symbol).split("/", 1)[0].upper()
    ml_dir      = context.get("ml_direction", "—")
    ml_conf     = context.get("ml_confidence", 0)

    if market_mode == "VOLATILE_RANGE":
        if range_bias == "NEAR_SUPPORT":
            patterns = [PATTERN_LIBRARY["setup_D_buy"], PATTERN_LIBRARY["wait"]]
            bias_note = "Price is NEAR SUPPORT → look for BUY bounce only."
        elif range_bias == "NEAR_RESISTANCE":
            patterns = [PATTERN_LIBRARY["setup_D_sell"], PATTERN_LIBRARY["wait"]]
            bias_note = "Price is NEAR RESISTANCE → look for SELL bounce only."
        else:
            patterns = [PATTERN_LIBRARY["wait"]]
            bias_note = "Price is in the middle of the range → WAIT."
    elif market_mode == "SIDEWAY" or primary == "SIDEWAY":
        # Setup E: direction driven by RSI extreme at BB
        if rsi < 40:
            patterns = [PATTERN_LIBRARY["setup_E_buy"], PATTERN_LIBRARY["wait"]]
            bias_note = f"SIDEWAY + RSI={rsi:.1f} oversold → look for BUY mean reversion to SMA20."
        elif rsi > 60:
            patterns = [PATTERN_LIBRARY["setup_E_sell"], PATTERN_LIBRARY["wait"]]
            bias_note = f"SIDEWAY + RSI={rsi:.1f} overbought → look for SELL mean reversion to SMA20."
        else:
            patterns = [PATTERN_LIBRARY["wait"]]
            bias_note = f"SIDEWAY + RSI={rsi:.1f} neutral — no edge. WAIT."
    elif primary == "UPTREND":
        if trend_rollover:
            patterns = [PATTERN_LIBRARY["setup_C_buy"], PATTERN_LIBRARY["wait"]]
            bias_note = (
                f"M15 uptrend is rolling over — do NOT use continuation buys. "
                f"Only consider strong Setup C reversals or WAIT. {trend_rollover_reason}"
            )
        elif score >= 4:
            patterns = [PATTERN_LIBRARY["setup_B_buy"], PATTERN_LIBRARY["wait"]]
            bias_note = f"M15 uptrend strong — breakout BUYs allowed. H1 context: {h1}. Allowed direction: {allowed_dir}."
        else:
            patterns = [PATTERN_LIBRARY["setup_A_buy"],
                        PATTERN_LIBRARY["setup_C_buy"],
                        PATTERN_LIBRARY["wait"]]
            bias_note = f"M15 uptrend — only BUY setups. H1 context: {h1}. Allowed direction: {allowed_dir}."
    elif primary == "DOWNTREND":
        if trend_rollover:
            patterns = [PATTERN_LIBRARY["setup_C_sell"], PATTERN_LIBRARY["wait"]]
            bias_note = (
                f"M15 downtrend is rolling over — do NOT use continuation sells. "
                f"Only consider strong Setup C reversals or WAIT. {trend_rollover_reason}"
            )
        elif score >= 4:
            patterns = [PATTERN_LIBRARY["setup_B_sell"], PATTERN_LIBRARY["wait"]]
            bias_note = f"M15 downtrend strong — breakdown SELLs allowed. H1 context: {h1}. Allowed direction: {allowed_dir}."
        else:
            patterns = [PATTERN_LIBRARY["setup_A_sell"],
                        PATTERN_LIBRARY["setup_C_sell"],
                        PATTERN_LIBRARY["wait"]]
            bias_note = f"M15 downtrend — only SELL setups. H1 context: {h1}. Allowed direction: {allowed_dir}."
    else:
        patterns = [PATTERN_LIBRARY["wait"]]
        bias_note = "Trend unclear → WAIT."

    try:
        trend_min_rr = max(0.5, float(os.getenv("TRADE_MIN_RR", "1.5")))
    except (TypeError, ValueError):
        trend_min_rr = 1.5
    min_rr = 1.0 if symbol_base in ("SOL", "XRP") else trend_min_rr
    rr_str = str(min_rr)
    patterns = [p.replace("R:R ≥ 1.5", f"R:R ≥ {rr_str}")
                 .replace("R:R < 1.5", f"R:R < {rr_str}") for p in patterns]
    combined = "\n\n" + ("=" * 60 + "\n").join(patterns)

    system_prompt = f"""You are a quantitative trading analyst combining chart pattern recognition with hard data.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUANTITATIVE DATA (Python-confirmed — trust completely)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Symbol        : {symbol}
Market Mode   : {market_mode}
Primary Trend : {primary} (M15)
H1 Trend      : {h1}
Quant Score   : {score}/5
ADX (M15)     : {adx:.1f}
RSI (M15)     : {rsi:.1f}
ATR (M15)     : {atr:.4f}
H1 Resistance : {sr.get('resistance', 'N/A')}
H1 Support    : {sr.get('support', 'N/A')}
ML Model      : {ml_dir} confidence={ml_conf:.0%}
Bias directive: {bias_note}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LAST 5 M5 CANDLES (newest last)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{candle_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PATTERN LIBRARY — read carefully before analysing the chart
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{combined}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHART LEGEND
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EMA 34  = bright GREEN line
EMA 89  = ORANGE line
Vol MA  = BLUE line on volume panel
Red dashed horizontal = H1 Resistance
Green dashed horizontal = H1 Support
White dashed bands = Bollinger Bands

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SL / TP MATH RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TREND setups (A/B/C/D):
  SL buffer for this symbol = {"0.50%" if symbol_base in ("SOL", "XRP") else "0.25%"} (volatile coins need wider SL to survive wicks).
  BUY : SL = signal candle low − (entry × {"0.005" if symbol_base in ("SOL", "XRP") else "0.0025"}). Need |entry−SL| ≥ {atr:.4f}.
        TP must be BELOW resistance = {sr.get('resistance','?')}.
  SELL: SL = signal candle high + (entry × {"0.005" if symbol_base in ("SOL", "XRP") else "0.0025"}). Need |entry−SL| ≥ {atr:.4f}.
        TP must be ABOVE support = {sr.get('support','?')}.

SIDEWAY Setup E (mean reversion):
  BUY : SL below lower Bollinger Band minus 0.2% buffer. TP = upper Bollinger Band.
  SELL: SL above upper Bollinger Band plus 0.2% buffer. TP = lower Bollinger Band.

Minimum R:R = {1.0 if symbol_base in ("SOL", "XRP") else trend_min_rr} for this symbol. If R:R below minimum → WAIT.

MATH CHECK (enforce):
  BUY:  TP > entry > SL
  SELL: SL > entry > TP

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR TASK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Look at BOTH M5 chart panes carefully.
2. Use the wide M5 context pane to judge structure, box quality, nearby resistance/support, and whether a breakout is clean or fragile.
3. Use the close M5 execution pane to judge the signal candle, wick/body quality, and immediate confirmation.
4. Match each condition in the Pattern Library against what you see.
5. Return ONLY valid JSON - no markdown, no explanation outside JSON.

OUTPUT FORMAT:
{{
  "analysis": {{
    "setup_identified": "Setup A/B/C/D/None",
    "ema_check":    "EMA34 [above/below/intertwined] EMA89. Gap [widening/narrowing/stable].",
    "price_action": "Describe signal candle: type, colour, wick vs body ratio.",
    "volume_check": "Signal candle volume [above/below] Vol MA. Pass/Fail.",
    "rr_check":     "Entry=X, SL=X, TP=X, risk=X, reward=X, R:R=X. Pass/Fail.",
    "pattern_match": "Which conditions passed. Which failed."
  }},
  "signal":       "BUY" | "SELL" | "WAIT",
  "reason":       "Brief explanation in English.",
  "seek_entry_low":  number | null,
  "seek_entry_high": number | null,
  "seek_entry_basis": "Brief note describing the preferred buy/sell area or trigger.",
  "entry_price":  number,
  "stop_loss":    number,
  "take_profit":  number
}}
"""

    user_text = (
        f"Here is the {symbol} dual-view M5 chart generated from live Binance data. "
        f"The TOP pane is a wider 5-minute context view showing more history for structure and breakout quality. "
        f"The LOWER panes are the close-up 5-minute execution view with price, RSI, and volume for signal confirmation. "
        f"Chart includes candlesticks, EMA34 (green), EMA89 (orange), Volume MA (blue), Bollinger Bands, and H1 S/R levels. "
        f"Analyse according to the system prompt instructions and use both panes together."
    )

    return system_prompt, user_text


def _build_candle_summary(df_m5: pd.DataFrame) -> str:
    lines = []
    for _, row in df_m5.tail(5).iterrows():
        o, h, lo, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
        color    = "green" if c >= o else "red"
        body     = abs(c - o)
        wick_up  = h - max(o, c)
        wick_dn  = min(o, c) - lo
        vol      = float(row["volume"])
        lines.append(
            f"  {color:5s}  O={o:.2f} H={h:.2f} L={lo:.2f} C={c:.2f}  "
            f"body={body:.2f} wick↑={wick_up:.2f} wick↓={wick_dn:.2f}  vol={vol:.0f}"
        )
    return "\n".join(lines)


def ask_gemini(chart_b64: str, context: dict, df_m5) -> dict | None:
    """
    Sends chart image + context to Gemini Flash.
    Returns parsed decision dict or None on failure.
    """
    if not chart_b64:
        return None

    context["candle_summary"] = _build_candle_summary(df_m5)
    system_prompt, user_text  = _build_prompt(context)
    system_prompt += """

IMPORTANT OVERRIDE:
- Market mode is the primary directional filter. Treat ML as side information only.
- Do not call BUY during an active 5m bearish impulse, even if M15 is still uptrend.
- Do not call SELL during an active 5m bullish impulse, even if M15 is still downtrend.
- If the latest 5m candle closes against the trade and through EMA34 / micro structure, prefer WAIT.
- After a sharp 5m flush, do not call BUY on the first reclaim candle alone; prefer WAIT until price reclaims EMA34 / micro structure and holds.
- After a sharp 5m squeeze-up, do not call SELL on the first breakdown candle alone; prefer WAIT until price loses EMA34 / micro structure and holds below it.
- Do not reject a valid setup because of stop loss, take profit, or R:R math.
- Python will calculate entry, stop loss, take profit, and final R:R after your decision.
- You should decide only whether the chart shows a valid BUY, SELL, or WAIT setup.
- Recommend a seek-entry zone only as an advisory area to monitor, not as a hard executable order.
- If BUY/SELL is valid, set seek_entry_low/high to a realistic pullback, retest, or trigger area visible on M5.
- If no clear area exists yet, choose WAIT and set seek-entry fields to null.
- If you mention rr_check, say that Python will calculate levels after the decision.
"""

    content_payload = [
        {"type": "text", "text": user_text},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{chart_b64}"}},
    ]

    import time as _time
    last_err = None
    for attempt in range(1, 4):
        try:
            client   = _get_client()
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": content_payload},
                ],
                max_tokens=2048,
                temperature=0.1,
                response_format={"type": "json_object"},
                timeout=60,
            )
            raw    = response.choices[0].message.content
            result = _safe_parse(raw)

            if "analysis" in result:
                a = result["analysis"]
                print(f"    [AI] setup={a.get('setup_identified')}  signal={result.get('signal')}")
                print(f"    [AI] EMA  : {a.get('ema_check','')}")
                print(f"    [AI] PA   : {a.get('price_action','')}")
                print(f"    [AI] VOL  : {a.get('volume_check','')}")
                print(f"    [AI] R:R  : {a.get('rr_check','')}")
                print(f"    [AI] Match: {a.get('pattern_match','')}")
                print(
                    "    [AI] Zone : "
                    f"{result.get('seek_entry_low')} - {result.get('seek_entry_high')} "
                    f"({result.get('seek_entry_basis', '')})"
                )
                print(f"    [AI] Reason: {result.get('reason','')}")

            result["seek_entry_low"] = _to_float_or_none(result.get("seek_entry_low"))
            result["seek_entry_high"] = _to_float_or_none(result.get("seek_entry_high"))
            if (
                result["seek_entry_low"] is not None
                and result["seek_entry_high"] is not None
                and result["seek_entry_low"] > result["seek_entry_high"]
            ):
                result["seek_entry_low"], result["seek_entry_high"] = (
                    result["seek_entry_high"],
                    result["seek_entry_low"],
                )

            return result

        except Exception as e:
            last_err = e
            print(f"[llm_analyst] Gemini attempt {attempt}/3 failed: {e}")
            if attempt < 3:
                _time.sleep(5)

    print(f"[llm_analyst] Gemini all retries exhausted: {last_err}")
    return None


def _safe_parse(raw: str) -> dict:
    import re
    text = raw.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text  = parts[1] if len(parts) > 1 else text
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    last = text.rfind("}")
    if last != -1:
        text = text[: last + 1]
    first = text.find("{")
    if first > 0:
        text = text[first:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    text = re.sub(r",\s*([}\]])", r"\1", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    raise ValueError(f"Cannot parse Gemini JSON (len={len(raw)}): {raw[:300]!r}")


def _to_float_or_none(value):
    if value in (None, "", "null"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None



