"""
engine_v2.py — 5-minute LLM + Vision trading engine for BTC/ETH.

Strategy layers:
  1. Macro bias  — ML model direction + Fear&Greed + funding rate
  2. Technical gates — H1/M15 trend, score ≥ 3/5, range position
  3. BTC correlation filter — don't trade against BTC trend
  4. Gemini Flash vision — chart pattern confirmation (Setup A/B/C/D)
  5. Execution — Binance Futures market order + native SL/TP

Run modes:
  python engine_v2.py --loop      # continuous 5-min cycle
  python engine_v2.py --once      # single cycle and exit
  python engine_v2.py --dry-run   # single cycle, no order execution

Logs written to backend/trading/logs/
"""

import argparse
import asyncio
import json
import os
import time
from datetime import datetime, UTC
from pathlib import Path

import psycopg2
import redis
from binance.um_futures import UMFutures
from dotenv import load_dotenv

from chart_gen      import generate_chart
from llm_analyst    import ask_gemini
from strategy_core  import (
    compute_indicators, compute_score, find_sr_levels,
    check_macro_bias, check_technical_gates, get_range_bias,
    detect_bb_mean_reversion,
)
from trade_logger   import (
    get_logger, log_cycle_start, log_gate_pass, log_gate_fail,
    log_ai_request, log_ai_response, log_trade_open, log_trade_close,
    log_skip, log_error, log_cycle_summary,
)

# ── Env ────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT / ".env")

log = get_logger()

# ── Config ─────────────────────────────────────────────────────────────────────
API_KEY      = os.getenv("BINANCE_FUTURES_API_KEY", "")
API_SECRET   = os.getenv("BINANCE_FUTURES_SECRET_KEY", "")
DATABASE_URL = os.getenv("DATABASE_URL", "")
REDIS_URL    = os.getenv("REDIS_URL", "")

SYMBOLS        = ["BTC", "ETH"]
LEVERAGE       = 5
STOP_LOSS_PCT  = 0.008    # 0.8% hard max SL
SL_MIN_PCT     = 0.002    # 0.2% hard min SL
TAKE_PROFIT_MIN_RR = 1.5
POSITION_RISK_PCT  = 0.01  # risk 1% of balance per trade
CYCLE_INTERVAL     = 5 * 60  # 5 minutes

TESTNET_BASE = "https://testnet.binancefuture.com"


# ── Binance client ─────────────────────────────────────────────────────────────
def get_client() -> UMFutures:
    return UMFutures(key=API_KEY, secret=API_SECRET, base_url=TESTNET_BASE)


# ── Redis ──────────────────────────────────────────────────────────────────────
def _get_redis():
    url = REDIS_URL
    if not url:
        return None
    if url.startswith("redis://"):
        url = "rediss://" + url[8:]
    try:
        r = redis.from_url(url, decode_responses=True, socket_timeout=3)
        r.ping()
        return r
    except Exception:
        return None


def get_ml_predictions() -> dict:
    """Pull latest ML predictions from Redis cache."""
    try:
        r = _get_redis()
        if r is None:
            return {}
        result = {}
        for sym in SYMBOLS:
            for h in [1, 7]:
                val = r.get(f"prediction:{sym}:{h}d")
                if val:
                    result[f"{sym}_{h}d"] = json.loads(val)
        return result
    except Exception as e:
        log.warning(f"Redis ML read failed: {e}")
        return {}


def get_market_signals() -> dict:
    """Pull fear/greed + funding rates from DB."""
    signals = {}
    try:
        with psycopg2.connect(DATABASE_URL, sslmode="require") as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT value FROM fear_greed_index ORDER BY date DESC LIMIT 1"
                )
                row = cur.fetchone()
                if row:
                    signals["fear_greed"] = int(row[0])

                for sym in SYMBOLS:
                    cur.execute(
                        "SELECT rate_avg FROM funding_rates WHERE symbol=%s "
                        "ORDER BY date DESC LIMIT 1",
                        (sym,)
                    )
                    row = cur.fetchone()
                    if row:
                        signals[f"{sym.lower()}_funding"] = float(row[0])
    except Exception as e:
        log.warning(f"DB market signals failed: {e}")
    return signals


# ── Binance data fetch ─────────────────────────────────────────────────────────
import pandas as pd


def fetch_ohlcv(client: UMFutures, symbol: str, interval: str,
                limit: int = 200) -> pd.DataFrame:
    """Fetch klines and return OHLCV DataFrame. Drops the last (incomplete) candle."""
    raw = client.klines(
        symbol=f"{symbol}USDT", interval=interval, limit=limit
    )
    df = pd.DataFrame(raw, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col])
    df["timestamp"] = pd.to_numeric(df["timestamp"])
    return df.iloc[:-1].reset_index(drop=True)   # drop incomplete candle


def get_account_balance(client: UMFutures) -> float:
    try:
        account = client.account()
        for asset in account.get("assets", []):
            if asset["asset"] == "USDT":
                return float(asset["walletBalance"])
    except Exception as e:
        log.warning(f"Balance fetch failed: {e}")
    return 0.0


def get_open_position(client: UMFutures, symbol: str) -> dict | None:
    """Returns position dict if open, else None."""
    try:
        risk = client.get_position_risk(symbol=f"{symbol}USDT")
        for pos in risk:
            amt = float(pos.get("positionAmt", 0))
            if abs(amt) > 0:
                return {
                    "symbol":      symbol,
                    "side":        "BUY" if amt > 0 else "SELL",
                    "amount":      abs(amt),
                    "entry_price": float(pos.get("entryPrice", 0)),
                    "mark_price":  float(pos.get("markPrice", 0)),
                    "upnl":        float(pos.get("unRealizedProfit", 0)),
                }
    except Exception as e:
        log.warning(f"Position fetch failed for {symbol}: {e}")
    return None


# ── Position sizing ────────────────────────────────────────────────────────────
def calc_quantity(balance: float, entry: float, sl: float, symbol: str) -> float:
    risk_usdt     = balance * POSITION_RISK_PCT
    sl_distance   = abs(entry - sl)
    if sl_distance == 0:
        return 0.0
    position_value = (risk_usdt / sl_distance) * entry
    position_value = min(position_value, balance * 0.10 * LEVERAGE)
    qty = position_value / entry
    return round(qty, 3) if symbol == "BTC" else round(qty, 2)


# ── Order execution ────────────────────────────────────────────────────────────
def execute_trade(client: UMFutures, symbol: str, decision: dict,
                  balance: float, context: dict, dry_run: bool = False) -> bool:
    """
    Places market entry + native SL/TP orders.
    Returns True if trade was opened.
    """
    signal = decision.get("signal")
    if signal not in ("BUY", "SELL"):
        return False

    try:
        ai_sl   = float(decision.get("stop_loss", 0))
        ai_tp   = float(decision.get("take_profit", 0))
        reason  = decision.get("reason", "")
        setup   = decision.get("analysis", {}).get("setup_identified", "?")
    except (TypeError, ValueError) as e:
        log_error(f"[{symbol}] Invalid SL/TP from AI", e)
        return False

    # Get current price
    try:
        ticker = client.ticker_price(symbol=f"{symbol}USDT")
        entry  = float(ticker["price"])
    except Exception as e:
        log_error(f"[{symbol}] Ticker fetch failed", e)
        return False

    # Direction sanity check
    if signal == "BUY" and not (ai_tp > entry > ai_sl):
        log_gate_fail("DIRECTION", f"BUY needs TP({ai_tp}) > entry({entry}) > SL({ai_sl})", symbol)
        return False
    if signal == "SELL" and not (ai_sl > entry > ai_tp):
        log_gate_fail("DIRECTION", f"SELL needs SL({ai_sl}) > entry({entry}) > TP({ai_tp})", symbol)
        return False

    # SL distance check
    risk_pct = abs(entry - ai_sl) / entry * 100
    if risk_pct < SL_MIN_PCT * 100:
        log_gate_fail("SL_MIN", f"SL {risk_pct:.3f}% < min {SL_MIN_PCT*100:.2f}%", symbol)
        return False
    if risk_pct > STOP_LOSS_PCT * 100:
        log_gate_fail("SL_MAX", f"SL {risk_pct:.3f}% > max {STOP_LOSS_PCT*100:.2f}%", symbol)
        return False

    # R:R check
    risk   = abs(entry - ai_sl)
    reward = abs(ai_tp - entry)
    rr     = reward / risk if risk > 0 else 0
    if rr < TAKE_PROFIT_MIN_RR:
        log_gate_fail("RR", f"R:R={rr:.2f} < {TAKE_PROFIT_MIN_RR}", symbol)
        return False

    qty = calc_quantity(balance, entry, ai_sl, symbol)
    if qty <= 0:
        log_gate_fail("QTY", f"Quantity too small for balance {balance:.2f}", symbol)
        return False

    log.info(
        f"  [EXEC] {signal} {symbol} | entry≈{entry} SL={ai_sl} TP={ai_tp} "
        f"qty={qty} R:R={rr:.2f} | dry_run={dry_run}"
    )

    if dry_run:
        log_trade_open(symbol, signal, entry, ai_sl, ai_tp, rr, setup, reason,
                       trade_id="DRY_RUN", context=context)
        return True

    side     = "BUY" if signal == "BUY" else "SELL"
    pos_side = "LONG" if signal == "BUY" else "SHORT"
    close_side = "SELL" if signal == "BUY" else "BUY"
    sym_pair = f"{symbol}USDT"

    try:
        # Set leverage
        client.change_leverage(symbol=sym_pair, leverage=LEVERAGE)

        # Entry
        order = client.new_order(
            symbol=sym_pair, side=side, type="MARKET", quantity=qty,
        )
        order_id    = str(order.get("orderId", ""))
        actual_price = float(order.get("avgPrice", entry)) or entry
        log.info(f"  [EXEC] Entry filled @ {actual_price} orderId={order_id}")

        time.sleep(0.5)

        # Native SL
        try:
            client.new_order(
                symbol=sym_pair, side=close_side, type="STOP_MARKET",
                stopPrice=str(round(ai_sl, 2)),
                closePosition=True, timeInForce="GTE_GTC",
            )
            log.info(f"  [EXEC] SL order placed @ {ai_sl}")
        except Exception as e:
            log_error(f"[{symbol}] SL placement failed — SET MANUALLY @ {ai_sl}", e)

        # Native TP
        try:
            client.new_order(
                symbol=sym_pair, side=close_side, type="TAKE_PROFIT_MARKET",
                stopPrice=str(round(ai_tp, 2)),
                closePosition=True, timeInForce="GTE_GTC",
            )
            log.info(f"  [EXEC] TP order placed @ {ai_tp}")
        except Exception as e:
            log_error(f"[{symbol}] TP placement failed — SET MANUALLY @ {ai_tp}", e)

        log_trade_open(symbol, signal, actual_price, ai_sl, ai_tp, rr,
                       setup, reason, trade_id=order_id, context=context)
        return True

    except Exception as e:
        log_error(f"[{symbol}] Order execution failed", e)
        return False


# ── Single symbol cycle ────────────────────────────────────────────────────────
def run_symbol_cycle(client: UMFutures, symbol: str,
                     ml_preds: dict, market_signals: dict,
                     balance: float, dry_run: bool = False):

    log.info(f"\n{'─'*50}")
    log.info(f"  CYCLE  {symbol}  {datetime.now(UTC).strftime('%H:%M:%S UTC')}")

    # ── Gate 0: open position? ─────────────────────────────────────────────────
    pos = get_open_position(client, symbol)
    if pos:
        upnl_pct = pos["upnl"] / (pos["entry_price"] * pos["amount"]) * 100
        log.info(
            f"  [SKIP] {symbol} has open {pos['side']} position | "
            f"entry={pos['entry_price']} mark={pos['mark_price']} "
            f"uPnL={upnl_pct:+.2f}%"
        )
        return

    # ── Fetch OHLCV ───────────────────────────────────────────────────────────
    try:
        df_h1  = fetch_ohlcv(client, symbol, "1h",  200)
        df_m15 = fetch_ohlcv(client, symbol, "15m", 100)
        df_m5  = fetch_ohlcv(client, symbol, "5m",  100)
        # BTC as correlation filter
        df_btc_h1 = fetch_ohlcv(client, "BTC", "1h", 100) if symbol != "BTC" else df_h1
    except Exception as e:
        log_error(f"[{symbol}] OHLCV fetch failed", e)
        return

    # ── Compute indicators ────────────────────────────────────────────────────
    ctx = compute_indicators(df_h1, df_m15, df_m5)
    ctx["symbol"] = f"{symbol}/USDT"

    # BTC trend check
    if symbol != "BTC":
        btc_ctx = compute_indicators(df_btc_h1, df_m15, df_m5)
        btc_trend = btc_ctx["h1_trend"]
        ctx["btc_trend"] = btc_trend
    else:
        ctx["btc_trend"] = ctx["h1_trend"]

    # ── S/R levels ────────────────────────────────────────────────────────────
    ctx["sr"] = find_sr_levels(df_h1, ctx["current_price"], df_m15)

    # ── Score ─────────────────────────────────────────────────────────────────
    score, score_details = compute_score(ctx)
    ctx["score"]        = score
    ctx["score_details"] = score_details

    log_cycle_start(symbol, ctx["market_mode"], score)
    log.info(f"  H1={ctx['h1_trend']} M15={ctx['m15_trend']} BTC={ctx.get('btc_trend','?')}")
    log.info(f"  S/R  R={ctx['sr']['resistance']} S={ctx['sr']['support']}")
    log.info(f"  Score {score}/5: {', '.join(score_details) or 'none'}")
    log.info(f"  ADX={ctx['adx']:.1f} RSI={ctx['rsi']:.1f} ATR={ctx['atr_m15']:.4f}")

    # ── Macro bias gate ───────────────────────────────────────────────────────
    ml_key      = f"{symbol}_1d"
    ml_pred     = ml_preds.get(ml_key, {})
    ml_dir      = ml_pred.get("direction", "")
    ml_conf     = float(ml_pred.get("confidence", 0))
    fear_greed  = market_signals.get("fear_greed")
    funding     = market_signals.get(f"{symbol.lower()}_funding", 0)

    ctx["ml_direction"]  = ml_dir
    ctx["ml_confidence"] = ml_conf
    ctx["fear_greed"]    = fear_greed
    ctx["funding_rate"]  = funding

    macro_ok, macro_reason, allowed_dir = check_macro_bias(
        ml_dir, ml_conf, fear_greed, funding, symbol
    )
    ctx["allowed_direction"] = allowed_dir

    if not macro_ok:
        log_gate_fail("MACRO", macro_reason, symbol, ctx)
        return
    log_gate_pass("MACRO", f"allowed={allowed_dir} ML={ml_dir}({ml_conf:.0%}) F&G={fear_greed}")

    # ── BTC correlation gate ──────────────────────────────────────────────────
    if symbol != "BTC":
        btc_trend = ctx.get("btc_trend", "")
        h1        = ctx["h1_trend"]
        if (h1 == "UPTREND" and btc_trend == "DOWNTREND") or \
           (h1 == "DOWNTREND" and btc_trend == "UPTREND"):
            log_gate_fail("BTC_CORR", f"BTC={btc_trend} conflicts with {symbol}={h1}", symbol, ctx)
            return
        log_gate_pass("BTC_CORR", f"BTC={btc_trend}")

    # ── Technical gates (SIDEWAY bypasses to Setup E path) ───────────────────
    is_sideway = ctx["market_mode"] == "SIDEWAY"

    if is_sideway:
        log.info("  SIDEWAY mode → Setup E (BB mean reversion) pre-check...")
        mr_signal = detect_bb_mean_reversion(df_m5, len(df_m5) - 1, ctx)
        if mr_signal is None:
            log_gate_fail("SETUP_E", "No BB extreme + RSI signal (price not at band or RSI neutral)", symbol, ctx)
            return
        log_gate_pass("SETUP_E", f"BB MR: {mr_signal['signal']} entry={mr_signal['entry']} "
                                  f"SL={mr_signal['sl']} TP={mr_signal['tp']} R:R={mr_signal['rr']}")
        ctx["setup_e_signal"] = mr_signal
    else:
        tech_ok, tech_reason = check_technical_gates(ctx)
        if not tech_ok:
            log_gate_fail("TECHNICAL", tech_reason, symbol, ctx)
            return
        log_gate_pass("TECHNICAL", tech_reason)

        # ── Range bias ────────────────────────────────────────────────────────
        if ctx["is_range"]:
            ctx["range_bias"] = get_range_bias(ctx)
            log.info(f"  Range bias: {ctx['range_bias']}")
            if ctx["range_bias"] == "MIDDLE":
                log_gate_fail("RANGE_POSITION", "price in middle of range — no edge", symbol, ctx)
                return

    # ── All gates passed — generate chart + call Gemini ───────────────────────
    log.info("  All gates passed → generating chart...")
    chart_b64 = generate_chart(df_m5, ctx)

    if not chart_b64:
        log_error(f"[{symbol}] Chart generation failed")
        return

    log_ai_request(symbol, ctx.get("market_mode", "?"))
    decision = ask_gemini(chart_b64, ctx, df_m5)

    if decision is None:
        log_error(f"[{symbol}] Gemini returned no response")
        return

    log_ai_response(decision)
    signal = decision.get("signal", "WAIT")

    if signal == "WAIT":
        log.info(f"  [AI] WAIT — {decision.get('reason','')}")
        log_skip("AI_WAIT", decision.get("reason", ""), ctx, decision)
        log_cycle_summary(symbol, "WAIT", False, balance)
        return

    # ── For Setup E: override AI prices with math-validated levels ────────────
    if is_sideway and "setup_e_signal" in ctx:
        mr = ctx["setup_e_signal"]
        if signal == mr["signal"]:   # AI agrees with direction
            decision["stop_loss"]   = mr["sl"]
            decision["take_profit"] = mr["tp"]
            decision["entry_price"] = mr["entry"]
            log.info(f"  [SETUP_E] Using pre-computed SL={mr['sl']} TP={mr['tp']} R:R={mr['rr']}")

    # ── Execute ────────────────────────────────────────────────────────────────
    executed = execute_trade(client, symbol, decision, balance, ctx, dry_run=dry_run)
    log_cycle_summary(symbol, signal, executed, balance)


# ── Main loop ──────────────────────────────────────────────────────────────────
def run_once(dry_run: bool = False):
    log.info("=" * 60)
    log.info(f"ENGINE V2  {'DRY RUN' if dry_run else 'LIVE'}  {datetime.now(UTC).isoformat()}")
    log.info("=" * 60)

    client         = get_client()
    ml_preds       = get_ml_predictions()
    market_signals = get_market_signals()
    balance        = get_account_balance(client)

    log.info(f"Balance: {balance:.2f} USDT | ML preds loaded: {list(ml_preds.keys())}")
    log.info(f"Fear&Greed={market_signals.get('fear_greed','?')} "
             f"BTC funding={market_signals.get('btc_funding',0):+.4f}% "
             f"ETH funding={market_signals.get('eth_funding',0):+.4f}%")

    for symbol in SYMBOLS:
        try:
            run_symbol_cycle(client, symbol, ml_preds, market_signals, balance, dry_run)
        except Exception as e:
            log_error(f"[{symbol}] Unhandled cycle error", e)


def run_loop(dry_run: bool = False):
    log.info("Engine V2 started — 5-min cycle loop")
    while True:
        try:
            run_once(dry_run=dry_run)
        except Exception as e:
            log_error("Loop-level error", e)
        log.info(f"  Sleeping {CYCLE_INTERVAL}s until next cycle...")
        time.sleep(CYCLE_INTERVAL)


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Energy Forecaster Trading Engine V2")
    parser.add_argument("--loop",    action="store_true", help="Run continuously every 5 min")
    parser.add_argument("--once",    action="store_true", help="Run one cycle and exit")
    parser.add_argument("--dry-run", action="store_true", help="No real orders — log only")
    args = parser.parse_args()

    if not API_KEY or not API_SECRET:
        print("ERROR: BINANCE_FUTURES_API_KEY / BINANCE_FUTURES_SECRET_KEY not set")
        exit(1)

    if args.loop:
        run_loop(dry_run=args.dry_run)
    else:
        run_once(dry_run=args.dry_run)
