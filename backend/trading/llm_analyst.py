"""
llm_analyst.py — Gemini Flash vision + language analyst.

Sends the execution chart image + quantitative context to Gemini Flash.
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
from strategy_core import M15_TREND_GAP

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
    "Near" = candle body or lower wick within 1.5% of the EMA line.
[2] SIGNAL PRIORITY — rank these from strongest to weakest:
    Highest priority: Bullish Engulfing or strong bullish rejection close from the EMA zone.
    Medium priority: Bullish Pinbar with clear lower wick rejection.
    Lower priority: Any other obvious bullish reaction candle that closes back upward from the EMA zone.
[3] VOLUME: signal candle volume > average of previous 5 candles is preferred, but slightly-below-average volume can be forgiven if the reaction candle is very clear and the EMA pullback is clean.
[4] ENTRY:
    If the signal candle is a bullish pinbar, do NOT enter immediately.
    Wait for the NEXT candle to break and CLOSE above the pinbar high before entry.
    If the signal candle is a bullish engulfing, entry at signal candle close is acceptable.
    Exact SL/TP math is owned by Python and may use extra ATR / structure buffer.
    TP should still have room toward H1 resistance (red dashed line).

FALSE SIGNALS — output WAIT if:
  - Signal candle has small body, no clear rejection, and no decisive close back upward
  - Bullish pinbar forms inside chop / a tiny sideways cluster / a small consolidation box
  - EMAs are rapidly converging (gap closing more than 30% in the last 5 candles)
  - Volume is clearly weak AND the price reaction is also weak
  - Price has broken below EMA89 for more than 3 consecutive candles
""",

    "setup_A_sell": """
=== SETUP A — EMA PULLBACK SELL (DOWNTREND) ===
Context: EMA34 (green) is BELOW EMA89 (orange). Gap is wide or stable.

REQUIRED on M5 chart — ALL must be true:
[1] RALLY PULLBACK: Price has rallied from lower levels and is touching or near EMA34 or EMA89.
    "Near" = candle body or upper wick within 1.5% of the EMA line.
[2] SIGNAL PRIORITY — rank these from strongest to weakest:
    Highest priority: Bearish Engulfing or strong bearish rejection close from the EMA zone.
    Medium priority: Bearish Pinbar with clear upper wick rejection.
    Lower priority: Any other obvious bearish reaction candle that closes back downward from the EMA zone.
[3] VOLUME: signal candle volume > average of previous 5 candles is preferred, but slightly-below-average volume can be forgiven if the reaction candle is very clear and the EMA pullback is clean.
[4] ENTRY:
    If the signal candle is a bearish pinbar, do NOT enter immediately.
    Wait for the NEXT candle to break and CLOSE below the pinbar low before entry.
    If the signal candle is a bearish engulfing, entry at signal candle close is acceptable.
    Exact SL/TP math is owned by Python and may use extra ATR / structure buffer.
    TP should still have room toward H1 support (green dashed line).

FALSE SIGNALS — output WAIT if:
  - Signal candle has small body, no clear rejection, and no decisive close back downward
  - Bearish pinbar forms inside chop / a tiny sideways cluster / a small consolidation box
  - EMAs are rapidly converging (gap closing more than 30% in the last 5 candles)
  - Volume is clearly weak AND the price reaction is also weak
  - Price broken above EMA89 for more than 3 consecutive candles
""",

    "setup_B_buy": """
=== SETUP B — BREAKOUT ACCEPTANCE BUY (UPTREND) ===
Context: Market is trending up and price is trying to start a fresh upward expansion.

REQUIRED:
[1] Price breaks ABOVE a recent local structure high / small consolidation high / minor resistance.
[2] SIGNAL PRIORITY:
    Highest priority: Breakout candle closes strongly near its highs with a clear body.
    Medium priority: Breakout candle closes above the level with decent body but not perfect range expansion.
    Lower priority: Breakout is acceptable even if volume is only average, as long as the close outside the level is obvious.
[3] Breakout candle body matters more than wick shape. Prefer real closes through the level, not just a wick above resistance.
[4] Breakout candle volume should be at or above recent average, but average volume can be forgiven if the close outside the level is very clear.
[5] ENTRY: enter only if the breakout candle CLOSES above the broken level.

FALSE SIGNALS — output WAIT if:
  - Price only wicks above resistance and closes back inside the range
  - Breakout candle body is weak or small
  - Breakout candle closes far from its high
  - Breakout happens directly into obvious higher-timeframe resistance with no room
""",

    "setup_B_sell": """
=== SETUP B — BREAKOUT ACCEPTANCE SELL (DOWNTREND) ===
Context: Market is trending down and price is trying to start a fresh downward expansion.

REQUIRED:
[1] Price breaks BELOW a recent local structure low / small consolidation low / minor support.
[2] SIGNAL PRIORITY:
    Highest priority: Breakdown candle closes strongly near its lows with a clear body.
    Medium priority: Breakdown candle closes below the level with decent body but not perfect expansion.
    Lower priority: Breakdown is acceptable even if volume is only average, as long as the close outside the level is obvious.
[3] Breakdown candle body matters more than wick shape. Prefer real closes through the level, not just a wick below support.
[4] Breakdown candle volume should be at or above recent average, but average volume can be forgiven if the close outside the level is very clear.
[5] ENTRY: enter only if the breakdown candle CLOSES below the broken level.

FALSE SIGNALS — output WAIT if:
  - Price only wicks below support and closes back inside the range
  - Breakdown candle body is weak or small
  - Breakdown candle closes far from its low
  - Breakdown happens directly into obvious higher-timeframe support with no room
""",

    "setup_C_buy": """
=== SETUP C — BREAKOUT RETEST HOLD BUY (UPTREND) ===
Context: Price already broke higher and is retesting the broken level from above.

REQUIRED:
[1] A prior breakout above recent structure is visible.
[2] Price revisits that broken resistance / breakout level.
[3] SIGNAL PRIORITY:
    Highest priority: Retest holds and the confirmation candle closes bullish back above the level.
    Medium priority: The retest is slightly messy, but price still clearly defends the level and closes back above it.
    Lower priority: Exact candle shape is less important than a clean hold and a clear close back upward.
[4] The confirmation candle closes bullish and back above the retest area.
[5] ENTRY: enter only after the retest-hold candle CLOSES back above the broken level.

FALSE SIGNALS — output WAIT if:
  - There is no clear prior breakout
  - Price falls back deeply into the old range
  - The retest candle closes weakly or indecisively
  - Retest happens inside messy chop with no clear level
""",

    "setup_C_sell": """
=== SETUP C — BREAKOUT RETEST FAIL SELL (DOWNTREND) ===
Context: Price already broke lower and is retesting the broken level from below.

REQUIRED:
[1] A prior breakdown below recent structure is visible.
[2] Price revisits that broken support / breakdown level.
[3] SIGNAL PRIORITY:
    Highest priority: Retest fails and the confirmation candle closes bearish back below the level.
    Medium priority: The retest is slightly messy, but price still clearly rejects the level and closes back below it.
    Lower priority: Exact candle shape is less important than a clean failure and a clear close back downward.
[4] The confirmation candle closes bearish and back below the retest area.
[5] ENTRY: enter only after the retest-fail candle CLOSES back below the broken level.

FALSE SIGNALS — output WAIT if:
  - There is no clear prior breakdown
  - Price reclaims deeply back into the old range
  - The retest candle closes weakly or indecisively
  - Retest happens inside messy chop with no clear level
""",

    "setup_D_buy": """
=== SETUP D — RANGE BOUNCE BUY (VOLATILE RANGE) ===
Context: EMA34 ≈ EMA89 (intertwined). Price oscillating between H1 S/R levels.

REQUIRED:
[1] Price is near H1 SUPPORT (green dashed line) — within 1.5% of it.
[2] SIGNAL PRIORITY at or near support:
    Highest priority: Bullish engulfing or strong bullish rejection close from support.
    Medium priority: Double bottom / second touch of support with a bounce.
    Lower priority: Bullish pinbar with clear lower wick rejection.
[3] VOLUME: Higher than the average of the 5 previous candles is preferred, but slightly-below-average volume can be forgiven if the support reaction is very obvious.
[4] ENTRY:
    If using a bullish pinbar, wait for the NEXT candle to break and CLOSE above the pinbar high before entry.
    Otherwise entry at signal candle close is acceptable.
    Entry must still be ABOVE the support line.
    Exact SL/TP math is owned by Python and may use extra ATR / structure buffer.
    TP should still point toward H1 resistance. (Python will verify R:R post-decision.)

FALSE SIGNALS — output WAIT if:
  - Price has broken clearly through support (more than 3 candles held below)
  - Reaction happens in the middle of the range instead of at a clear support edge
  - Signal candle has no clear bullish reaction close
  - Volume is clearly weak AND the support reaction is also weak
""",

    "setup_D_sell": """
=== SETUP D — RANGE BOUNCE SELL (VOLATILE RANGE) ===
Context: EMA34 ≈ EMA89 (intertwined). Price oscillating between H1 S/R levels.

REQUIRED:
[1] Price is near H1 RESISTANCE (red dashed line) — within 1.5%.
[2] SIGNAL PRIORITY at or near resistance:
    Highest priority: Bearish engulfing or strong bearish rejection close from resistance.
    Medium priority: Double top / second touch of resistance with rejection.
    Lower priority: Bearish pinbar with clear upper wick rejection.
[3] VOLUME: Higher than the average of the 5 previous candles is preferred, but slightly-below-average volume can be forgiven if the resistance reaction is very obvious.
[4] ENTRY:
    If using a bearish pinbar, wait for the NEXT candle to break and CLOSE below the pinbar low before entry.
    Otherwise entry at signal candle close is acceptable.
    Entry must still be BELOW resistance.
    Exact SL/TP math is owned by Python and may use extra ATR / structure buffer.
    TP should still point toward H1 support. (Python will verify R:R post-decision.)

FALSE SIGNALS — output WAIT if:
  - Price has broken clearly through resistance (more than 3 candles held above)
  - Reaction happens in the middle of the range instead of at a clear resistance edge
  - Signal candle has no clear bearish reaction close
  - Volume is clearly weak AND the resistance reaction is also weak
""",

    "wait": """
=== WAIT CONDITIONS ===
Choose WAIT if ANY of the following:
- Signal candle shape is clearly ambiguous (Doji, Spinning Top with tiny body)
- Volume on signal candle is notably below surrounding candles
- EMAs are choppy (crossing back and forth in last 5 candles)
- Pinbar appears inside a small consolidation instead of at a meaningful reaction point
- Price is in the middle of a range with no clear S/R nearby

RULE: When clearly ambiguous → WAIT. Borderline setups with good structure are acceptable.
""",
}


def _build_prompt(context: dict) -> tuple[str, str]:
    """Returns (system_prompt, user_text)."""

    h1          = context.get("h1_trend", "?")
    primary     = context.get("primary_trend", context.get("m15_trend", "?"))
    market_mode = context.get("market_mode", primary)
    rsi         = context.get("rsi", 0)
    atr         = context.get("atr_m15", 0)
    sr          = context.get("sr", {})
    nearest_ssl = context.get("nearest_ssl")
    nearest_bsl = context.get("nearest_bsl")
    chart_box_support = context.get("chart_box_support", sr.get("support"))
    chart_box_resistance = context.get("chart_box_resistance", sr.get("resistance"))
    chart_box_state = str(context.get("chart_box_state", "UNKNOWN"))
    chart_box_window = int(context.get("chart_box_window", 0) or 0)
    chart_box_skip = int(context.get("chart_box_skip", 0) or 0)
    range_bias  = context.get("range_bias", "MIDDLE")
    candle_text = context.get("candle_summary", "")
    symbol      = context.get("symbol", "BTC/USDT")
    symbol_base = str(symbol).split("/", 1)[0].upper()
    m15_gap     = float(context.get("m15_gap", 0))
    exec_tf     = str(context.get("llm_exec_tf", "5m")).lower()
    exec_tf_label = exec_tf.upper()

    # How far is the current gap from the classification threshold?
    gap_margin = abs(m15_gap - M15_TREND_GAP) / M15_TREND_GAP if M15_TREND_GAP > 0 else 1.0
    is_borderline = (
        gap_margin < 0.25
        and primary in ("UPTREND", "DOWNTREND")
    )
    if m15_gap > M15_TREND_GAP * 1.5:
        gap_strength = "STRONG"
    elif is_borderline:
        gap_strength = "BORDERLINE"
    else:
        gap_strength = "MODERATE"

    def _fmt_level(v):
        return f"{float(v):.4f}" if v is not None else "N/A"

    chart_box_state_label = chart_box_state.replace("_", " ")
    if chart_box_state == "ABOVE_BOX":
        box_note = (
            "Price is already above the historical H1 box. Prefer continuation or retest-hold BUY ideas, "
            "not blind box-top fades."
        )
    elif chart_box_state == "BELOW_BOX":
        box_note = (
            "Price is already below the historical H1 box. Prefer continuation or retest-fail SELL ideas, "
            "not blind box-bottom bounces."
        )
    else:
        box_note = (
            "Price is still inside the historical H1 box, so edge rejection logic is still valid."
        )

    if market_mode in ("VOLATILE_RANGE", "SIDEWAY") or primary in ("VOLATILE_RANGE", "SIDEWAY"):
        if range_bias == "NEAR_SUPPORT":
            patterns = [PATTERN_LIBRARY["setup_D_buy"], PATTERN_LIBRARY["wait"]]
            bias_note = (
                f"RANGE market — price near support ({sr.get('support')}). "
                f"SIDEWAY/RANGE mode is VALID for trading — evaluate Setup D BUY. "
                f"Do NOT use 'market is sideways' as a reason to WAIT."
            )
        elif range_bias == "NEAR_RESISTANCE":
            patterns = [PATTERN_LIBRARY["setup_D_sell"], PATTERN_LIBRARY["wait"]]
            bias_note = (
                f"RANGE market — price near resistance ({sr.get('resistance')}). "
                f"SIDEWAY/RANGE mode is VALID for trading — evaluate Setup D SELL. "
                f"Do NOT use 'market is sideways' as a reason to WAIT."
            )
        else:
            patterns = [PATTERN_LIBRARY["setup_D_buy"], PATTERN_LIBRARY["setup_D_sell"], PATTERN_LIBRARY["wait"]]
            bias_note = (
                f"RANGE market — price between support ({sr.get('support')}) and resistance ({sr.get('resistance')}). "
                f"Check chart: if closer to support look for Setup D BUY, if closer to resistance look for Setup D SELL. "
                f"Do NOT use 'price in middle of range' as a standalone reason to WAIT."
            )
    elif primary == "UPTREND":
        if is_borderline:
            patterns = [
                PATTERN_LIBRARY["setup_A_buy"],
                PATTERN_LIBRARY["setup_B_buy"],
                PATTERN_LIBRARY["setup_C_buy"],
                PATTERN_LIBRARY["setup_D_buy"],
                PATTERN_LIBRARY["wait"],
            ]
            bias_note = (
                f"M15 uptrend (BORDERLINE — EMA gap {m15_gap:.3f}% is near threshold {M15_TREND_GAP:.1f}%). "
                f"Prefer Setup A pullback BUY, Setup B breakout BUY, or Setup C retest-hold BUY. "
                f"If EMAs look intertwined on the chart rather than clearly separated, evaluate Setup D BUY instead. "
                f"H1: {h1}."
            )
        else:
            patterns = [
                PATTERN_LIBRARY["setup_A_buy"],
                PATTERN_LIBRARY["setup_B_buy"],
                PATTERN_LIBRARY["setup_C_buy"],
                PATTERN_LIBRARY["wait"],
            ]
            bias_note = (
                f"M15 uptrend ({gap_strength} — EMA gap {m15_gap:.3f}%) — "
                f"look for Setup A pullback BUY, Setup B breakout BUY, or Setup C retest-hold BUY. H1: {h1}."
            )
    elif primary == "DOWNTREND":
        if is_borderline:
            patterns = [
                PATTERN_LIBRARY["setup_A_sell"],
                PATTERN_LIBRARY["setup_B_sell"],
                PATTERN_LIBRARY["setup_C_sell"],
                PATTERN_LIBRARY["setup_D_sell"],
                PATTERN_LIBRARY["wait"],
            ]
            bias_note = (
                f"M15 downtrend (BORDERLINE — EMA gap {m15_gap:.3f}% is near threshold {M15_TREND_GAP:.1f}%). "
                f"Prefer Setup A rally SELL, Setup B breakdown SELL, or Setup C retest-fail SELL. "
                f"If EMAs look intertwined on the chart rather than clearly separated, evaluate Setup D SELL instead. "
                f"H1: {h1}."
            )
        else:
            patterns = [
                PATTERN_LIBRARY["setup_A_sell"],
                PATTERN_LIBRARY["setup_B_sell"],
                PATTERN_LIBRARY["setup_C_sell"],
                PATTERN_LIBRARY["wait"],
            ]
            bias_note = (
                f"M15 downtrend ({gap_strength} — EMA gap {m15_gap:.3f}%) — "
                f"look for Setup A rally SELL, Setup B breakdown SELL, or Setup C retest-fail SELL. H1: {h1}."
            )
    else:
        patterns = [
            PATTERN_LIBRARY["setup_A_buy"],
            PATTERN_LIBRARY["setup_A_sell"],
            PATTERN_LIBRARY["setup_B_buy"],
            PATTERN_LIBRARY["setup_B_sell"],
            PATTERN_LIBRARY["setup_C_buy"],
            PATTERN_LIBRARY["setup_C_sell"],
            PATTERN_LIBRARY["wait"],
        ]
        bias_note = f"Trend unclear — check chart for strongest directional signal. H1: {h1}."

    allowed_direction = context.get("allowed_direction", "BOTH")
    fear_greed_val = context.get("fear_greed")
    fg_constraint = ""
    if allowed_direction == "UP":
        patterns = [p for p in patterns if "SELL" not in p and "sell" not in p]
        if not any("buy" in p.lower() or "BUY" in p for p in patterns):
            patterns = [PATTERN_LIBRARY["setup_D_buy"], PATTERN_LIBRARY["wait"]]
        fg_constraint = (
            f"\n⚠️  DIRECTION CONSTRAINT — HARD RULE ⚠️\n"
            f"Fear & Greed = {fear_greed_val} (EXTREME FEAR). Market is in capitulation.\n"
            f"You MUST only look for LONG / BUY opportunities.\n"
            f"Do NOT suggest SELL or SHORT under any circumstances.\n"
            f"If no valid BUY setup exists, output WAIT.\n"
        )
    elif allowed_direction == "DOWN":
        patterns = [p for p in patterns if "BUY" not in p and "buy" not in p]
        if not any("sell" in p.lower() or "SELL" in p for p in patterns):
            patterns = [PATTERN_LIBRARY["setup_D_sell"], PATTERN_LIBRARY["wait"]]
        fg_constraint = (
            f"\n⚠️  DIRECTION CONSTRAINT — HARD RULE ⚠️\n"
            f"Fear & Greed = {fear_greed_val} (EXTREME GREED). Market is overheated.\n"
            f"You MUST only look for SHORT / SELL opportunities.\n"
            f"Do NOT suggest BUY or LONG under any circumstances.\n"
            f"If no valid SELL setup exists, output WAIT.\n"
        )

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
{fg_constraint}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUANTITATIVE DATA (Python-confirmed — trust completely)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Symbol        : {symbol}
Market Mode   : {market_mode}
Primary Trend : {primary} (M15)
H1 Trend      : {h1}
RSI (M15)     : {rsi:.1f}
ATR (M15)     : {atr:.4f}
M15 EMA Gap   : {m15_gap:.3f}% (threshold {M15_TREND_GAP:.1f}%, strength: {gap_strength})
H1 Box High   : {_fmt_level(chart_box_resistance)}
H1 Box Low    : {_fmt_level(chart_box_support)}
H1 Box State  : {chart_box_state_label}
Box Source    : built from prior {chart_box_window} H1 candles after skipping latest {chart_box_skip}
Nearest SSL   : {_fmt_level(nearest_ssl)}
Nearest BSL   : {_fmt_level(nearest_bsl)}
Bias directive: {bias_note}
Box note      : {box_note}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LAST 5 {exec_tf_label} CANDLES (newest last)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{candle_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PATTERN LIBRARY — read carefully before analysing the chart
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{combined}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SIGN RANKING RULES — KEEP THIS SIMPLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Highest-priority signs:
- Clear location at the correct edge: EMA pullback, breakout level, retest level, or H1 support/resistance.
- Decisive close in the expected direction.
- Clean structure: breakout accepted, retest held/failed, or obvious bounce/rejection.

Medium-priority signs:
- Engulfing candle.
- Double touch / double top / double bottom.
- Strong rejection candle.

Lower-priority signs:
- Textbook pinbar shape.
- Slightly above-average volume.
- Perfect candle color or perfect wick ratio.

Forgivable misses:
- Volume is only average, but the level reaction and close are very clear.
- Candle is not a perfect textbook pinbar, but the rejection/engulfing/retest behavior is obvious.
- Retest or range reaction is slightly messy, but the final close clearly confirms the direction.

Do NOT forgive:
- Wrong location.
- No clear directional close.
- Breakout that is only a wick.
- Retest that closes back the wrong way.
- Obvious chop with no edge.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHART LEGEND
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EMA 34  = bright GREEN line
EMA 89  = ORANGE line
Vol MA  = BLUE line on volume panel
Red dashed horizontal = H1 historical box high
Green dashed horizontal = H1 historical box low
The TOP pane has NO decision box. Use it only for execution timing.
Cyan shaded area on H1 = historical decision box built from older H1 candles
Inside H1 Decision Box:
  - Near top edge: look for rejection SELL
  - Near bottom edge: look for rejection BUY
Historical Box logic:
  - ABOVE BOX = breakout already happened. Prefer continuation or retest-hold, not blind fade.
  - BELOW BOX = breakdown already happened. Prefer continuation or retest-fail, not blind fade.
  - INSIDE BOX = range rejection logic is still valid.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SL / TP MATH RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TREND setups (A/B/C/D):
  Python owns the exact SL/TP math using structure, ATR floor, and support/resistance buffer.
  BUY : make sure the chart still leaves realistic room toward resistance = {sr.get('resistance','?')}.
  SELL: make sure the chart still leaves realistic room toward support = {sr.get('support','?')}.

NOTE: Python will calculate and enforce the final stop loss, take profit, and R:R after your signal decision. Do NOT use R:R math as a reason to WAIT.

DIRECTION CHECK only (enforce):
  BUY:  TP > entry > SL
  SELL: SL > entry > TP

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR TASK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Review the {exec_tf_label} execution pane for signal candle quality, wick/body structure, and immediate confirmation.
2. Review the H1 context pane for the historical decision box:
   decide whether price is still ranging inside it or has already accepted above/below it.
3. Use RSI + volume panels to confirm timing quality and momentum/range context.
4. You must identify potential entries based on the patterns above. If no clear pattern is visible, you can still call BUY or SELL if the chart structure and signal quality are good — you do not need to match every single condition perfectly to call a valid signal. Use your judgement to evaluate the overall quality of the setup.
5. Match each condition in the Pattern Library against what you see.
6. If setup is not ready yet, return WAIT with a brief reason.
7. Return ONLY valid JSON - no markdown, no explanation outside JSON.

OUTPUT FORMAT:
{{
  "analysis": {{
    "setup_identified": "Setup A/B/C/D/None",
    "ema_check":    "EMA34 [above/below/intertwined] EMA89. Gap [widening/narrowing/stable].",
    "price_action": "Describe signal candle: type, colour, wick vs body ratio.",
    "volume_check": "Signal candle volume [above/below] Vol MA. Pass/Fail.",
    "rr_check":     "Python handles R:R. Just note estimated entry, SL, TP levels from the chart.",
    "pattern_match": "Which conditions passed. Which failed."
  }},
  "signal":       "BUY" | "SELL" | "WAIT",
  "reason":       "Brief explanation in English.",
  "entry_price":  number,
  "stop_loss":    number,
  "take_profit":  number
}}
"""

    user_text = (
        f"Here is the {symbol} chart generated from live Binance data. "
        f"The TOP pane is {exec_tf_label} execution view for entry timing. "
        f"The TOP pane does not include a decision box. "
        f"The SECOND pane is H1 context view with a cyan historical Decision Box. "
        f"The lower panels are {exec_tf_label} RSI and {exec_tf_label} volume. "
        f"Chart includes candlesticks, EMA34 (green), EMA89 (orange), and Volume MA (blue). "
        f"Analyse according to the system prompt and use both timeframes together."
    )

    return system_prompt, user_text


def _build_candle_summary(df_exec: pd.DataFrame) -> str:
    lines = []
    for _, row in df_exec.tail(5).iterrows():
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


def ask_gemini(chart_b64: str, context: dict, df_exec) -> dict | None:
    """
    Sends chart image + context to Gemini Flash.
    Returns parsed decision dict or None on failure.
    """
    if not chart_b64:
        return None

    context["candle_summary"] = _build_candle_summary(df_exec)
    system_prompt, user_text  = _build_prompt(context)
    system_prompt += """

IMPORTANT OVERRIDE:
- Do not call BUY during an active execution-timeframe bearish impulse
- Do not call SELL during an active execution-timeframe bullish impulse
- Use liquidity hints as secondary timing filter: prefer BUY after sweep/reclaim near SSL, prefer SELL after sweep/reject near BSL.
- Do not reject a valid setup because of stop loss, take profit, or R:R math.
- Python will calculate entry, stop loss, take profit, and final R:R after your decision.
- You should decide only whether the chart shows a valid BUY, SELL, or WAIT setup.
- If you mention rr_check, say that Python will calculate levels after the decision.
- CRITICAL: A pinbar by itself is not enough. Prefer pinbars only when they happen at a meaningful edge and the next candle confirms the move.
- CRITICAL: Reject pinbars that form inside chop or a tiny consolidation cluster, even if their volume is slightly above average.
- CRITICAL: Do NOT output WAIT solely because "market is sideways", "market mode is range", or "price is in the middle of the box". These are descriptive labels, not rejection criteria. Setup D exists specifically for sideways/range markets - if a D setup is visible, take it.
- CRITICAL: Do NOT output WAIT solely because the market label says SIDEWAY or VOLATILE_RANGE. Evaluate the actual chart for signal candle quality, volume, and S/R proximity.
- CRITICAL: If price is already ABOVE the historical H1 box, do not short just because price is near the box top. Only fade a clear failed breakout.
- CRITICAL: If price is already BELOW the historical H1 box, do not buy just because price is near the box bottom. Only fade a clear failed breakdown.
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
                print(f"    [AI] Reason: {result.get('reason','')}")

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




