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
import json
import math
import os
import time
import threading
from datetime import datetime, timezone
UTC = timezone.utc
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
    validate_ai_trade_decision, build_trade_plan,
)
from trade_logger   import (
    get_logger, log_cycle_start, log_gate_pass, log_gate_fail,
    log_ai_request, log_ai_response, log_trade_open, log_trade_close,
    log_skip, log_error, log_cycle_summary,
)

ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT / ".env")

log = get_logger()

API_KEY      = os.getenv("BINANCE_FUTURES_API_KEY", "")
API_SECRET   = os.getenv("BINANCE_FUTURES_SECRET_KEY", "")
DATABASE_URL = os.getenv("DATABASE_URL", "")
REDIS_URL    = os.getenv("REDIS_URL", "")
USE_FVG_FILTER = os.getenv("USE_FVG_FILTER", "0") == "1"

SYMBOLS        = ["BTC", "ETH", "SOL", "XRP"]
LEVERAGE       = 5
try:
    DEFAULT_MIN_NOTIONAL = max(
        0.0,
        float(os.getenv("BINANCE_TESTNET_MIN_NOTIONAL", os.getenv("BINANCE_MIN_NOTIONAL", "100"))),
    )
except (TypeError, ValueError):
    DEFAULT_MIN_NOTIONAL = 100.0

# Binance Futures quantity step sizes (decimal precision per coin)
QTY_PRECISION  = {
    "BTC": 3,   # step 0.001
    "ETH": 2,   # step 0.01
    "SOL": 1,   # step 0.1
    "XRP": 0,   # step 1 (integer lots)
}
STOP_LOSS_PCT  = 0.008    # 0.8% hard max SL
SL_MIN_PCT     = 0.002    # 0.2% hard min SL
try:
    SL_ATR_MULTIPLIER = max(0.0, float(os.getenv("SL_ATR_MULTIPLIER", "1.8")))
except (TypeError, ValueError):
    SL_ATR_MULTIPLIER = 1.8
try:
    SL_SR_BUFFER_ATR_MULT = max(0.0, float(os.getenv("SL_SR_BUFFER_ATR_MULT", "0.25")))
except (TypeError, ValueError):
    SL_SR_BUFFER_ATR_MULT = 0.25
TAKE_PROFIT_MIN_RR    = 1.5
SETUP_E_MIN_RR        = 1.0   # lower bar for BB mean reversion (tighter TP target)
POSITION_RISK_PCT  = 0.01  # risk 1% of balance per trade
MAX_POSITION_FRACTION = 0.10  # max 10% of balance * leverage per position
MAX_POSITION_FRACTION_BY_SYMBOL = {
    "BTC": 0.35,  # allow smaller accounts to reach Binance min notional
}
CYCLE_INTERVAL     = 5 * 60  # 5 minutes

TESTNET_BASE    = "https://testnet.binancefuture.com"
MONITOR_INTERVAL = 5   # seconds between DB↔Binance sync checks
ALLOWED_ML_HORIZONS = {"4h", "1d"}
_SYMBOL_MIN_NOTIONAL_CACHE: dict[str, float] = {}


def _normalize_horizon_token(token: str) -> str:
    t = str(token).strip().lower()
    if len(t) < 2 or t[-1] not in {"d", "h"} or not t[:-1].isdigit():
        raise ValueError(f"Invalid horizon token '{token}'. Use values like 1d, 4h.")
    return f"{int(t[:-1])}{t[-1]}"


def _parse_horizon_tokens(raw: str) -> list[str]:
    tokens: list[str] = []
    for part in raw.split(","):
        p = part.strip()
        if not p:
            continue
        token = _normalize_horizon_token(p)
        if token not in ALLOWED_ML_HORIZONS:
            continue
        if token not in tokens:
            tokens.append(token)
    return tokens or ["4h", "1d"]


def _sanitize_horizon(token: str, fallback: str) -> str:
    return token if token in ALLOWED_ML_HORIZONS else fallback


ML_HORIZONS = _parse_horizon_tokens(os.getenv("ML_HORIZONS", "4h,1d"))
ML_PRIMARY_HORIZON = _sanitize_horizon(
    _normalize_horizon_token(os.getenv("ML_PRIMARY_HORIZON", ML_HORIZONS[0])),
    ML_HORIZONS[0],
)
if ML_PRIMARY_HORIZON not in ML_HORIZONS:
    ML_HORIZONS = [ML_PRIMARY_HORIZON] + ML_HORIZONS
ML_TREND_HORIZON = _sanitize_horizon(
    _normalize_horizon_token(os.getenv("ML_TREND_HORIZON", "1d")),
    "1d",
)
ML_REQUIRE_TREND_ALIGNMENT = os.getenv("ML_REQUIRE_TREND_ALIGNMENT", "1") == "1"
ML_CONFLICT_MODE = os.getenv("ML_CONFLICT_MODE", "higher_confidence").strip().lower()
if ML_CONFLICT_MODE not in {"skip", "higher_confidence"}:
    ML_CONFLICT_MODE = "higher_confidence"
try:
    ML_CONFLICT_MIN_RR = max(TAKE_PROFIT_MIN_RR, float(os.getenv("ML_CONFLICT_MIN_RR", "2.0")))
except (TypeError, ValueError):
    ML_CONFLICT_MIN_RR = 2.0


def get_client() -> UMFutures:
    return UMFutures(key=API_KEY, secret=API_SECRET, base_url=TESTNET_BASE)


def _get_conn():
    return psycopg2.connect(DATABASE_URL, sslmode="require")


def db_ensure_trades_table():
    """Create trades table if it doesn't exist, and add any missing columns (idempotent)."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id                BIGSERIAL PRIMARY KEY,
                    symbol            TEXT        NOT NULL,
                    side              TEXT        NOT NULL,
                    status            TEXT        NOT NULL DEFAULT 'OPEN',
                    entry_price       FLOAT,
                    exit_price        FLOAT,
                    quantity          FLOAT,
                    leverage          INTEGER     DEFAULT 5,
                    stop_loss         FLOAT,
                    take_profit       FLOAT,
                    pnl_usdt          FLOAT,
                    pnl_pct           FLOAT,
                    confidence        FLOAT,
                    horizon           INTEGER     DEFAULT 1,
                    binance_order_id  TEXT,
                    close_reason      TEXT,
                    setup             TEXT,
                    notes             TEXT,
                    opened_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    closed_at         TIMESTAMPTZ
                )
            """)
            # Add columns that may be missing from older table versions
            for col, definition in [
                ("setup", "TEXT"),
                ("notes", "TEXT"),
                ("confidence", "FLOAT"),
                ("binance_order_id", "TEXT"),
                ("close_reason", "TEXT"),
            ]:
                cur.execute(f"""
                    ALTER TABLE trades ADD COLUMN IF NOT EXISTS {col} {definition}
                """)
        conn.commit()
    finally:
        conn.close()


def db_create_pending(symbol: str, side: str, setup: str,
                      confidence: float = 0.0) -> int:
    """INSERT a PENDING record before placing the market order. Returns trade id."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO trades
                  (symbol, side, status, entry_price, quantity, leverage,
                   stop_loss, take_profit, confidence, setup, horizon)
                VALUES (%s,%s,'PENDING', 0, 0, %s, 0, 0, %s,%s, 1)
                RETURNING id
            """, (symbol, side, LEVERAGE, confidence, setup))
            tid = cur.fetchone()[0]
        conn.commit()
    finally:
        conn.close()
    return tid


def db_confirm_open(trade_id: int, entry_price: float, quantity: float,
                    stop_loss: float, take_profit: float,
                    order_id: str, notes: str):
    """UPDATE PENDING → OPEN once entry + SL/TP are all confirmed on exchange."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE trades SET
                    status='OPEN', entry_price=%s, quantity=%s,
                    stop_loss=%s, take_profit=%s,
                    binance_order_id=%s, notes=%s
                WHERE id=%s
            """, (entry_price, quantity, stop_loss, take_profit,
                  order_id, notes[:200], trade_id))
        conn.commit()
    finally:
        conn.close()


def db_cancel_pending(trade_id: int):
    """DELETE a PENDING record when entry or SL/TP placement failed."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM trades WHERE id=%s AND status='PENDING'",
                        (trade_id,))
        conn.commit()
    finally:
        conn.close()


def db_save_trade(symbol: str, side: str, entry_price: float, quantity: float,
                  stop_loss: float, take_profit: float, order_id: str,
                  setup: str, notes: str, confidence: float = 0.0) -> int:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO trades
                  (symbol, side, status, entry_price, quantity, leverage,
                   stop_loss, take_profit, confidence, binance_order_id, setup, notes)
                VALUES (%s,%s,'OPEN',%s,%s,%s,%s,%s,%s,%s,%s,%s)
                RETURNING id
            """, (symbol, side, entry_price, quantity, LEVERAGE,
                  stop_loss, take_profit, confidence, order_id, setup, notes))
            tid = cur.fetchone()[0]
        conn.commit()
    finally:
        conn.close()
    return tid


def db_close_trade(trade_id: int, exit_price: float, reason: str):
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT entry_price, quantity, side FROM trades WHERE id=%s",
                (trade_id,)
            )
            row = cur.fetchone()
            if not row:
                return
            entry, qty, side = row
            if side == "BUY":
                pnl_pct = (exit_price - entry) / entry * LEVERAGE
            else:
                pnl_pct = (entry - exit_price) / entry * LEVERAGE
            margin   = qty * entry / LEVERAGE
            pnl_usdt = pnl_pct * margin
            cur.execute("""
                UPDATE trades SET
                    status='CLOSED', exit_price=%s, pnl_usdt=%s,
                    pnl_pct=%s, close_reason=%s, closed_at=NOW()
                WHERE id=%s
            """, (exit_price, round(pnl_usdt, 4), round(pnl_pct * 100, 4),
                  reason, trade_id))
        conn.commit()
    finally:
        conn.close()
    log.info(f"[DB] Trade {trade_id} closed — {reason} @ {exit_price} pnl={pnl_pct*100:.2f}%")


def db_get_open_trades() -> list[dict]:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, symbol, side, entry_price, quantity,
                       stop_loss, take_profit, binance_order_id, opened_at
                FROM trades WHERE status='OPEN'
            """)
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]
    except Exception as e:
        log.warning(f"[DB] get_open_trades failed: {e}")
        return []
    finally:
        conn.close()


def db_cleanup_stale_pending(max_age_seconds: int = 120):
    """Delete PENDING records older than max_age_seconds (crash mid-execution)."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                DELETE FROM trades
                WHERE status='PENDING'
                  AND opened_at < NOW() - INTERVAL '%s seconds'
                RETURNING id, symbol
            """, (max_age_seconds,))
            rows = cur.fetchall()
            for tid, sym in rows:
                log.warning(f"[DB] Cleaned up stale PENDING record id={tid} symbol={sym}")
        conn.commit()
    except Exception as e:
        log.warning(f"[DB] cleanup_stale_pending failed: {e}")
    finally:
        conn.close()


def log_demo_trade_open(symbol: str, side: str, entry: float, sl: float, tp: float,
                        rr: float, setup: str, reason: str, trade_id: str | None = None,
                        context: dict = None):
    record = {
        "event": "OPEN",
        "ts": datetime.now(UTC).isoformat(),
        "id": trade_id,
        "symbol": symbol,
        "side": side,
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "rr": round(rr, 2),
        "setup": setup,
        "reason": reason,
        "score": context.get("score") if context else None,
        "mode": context.get("market_mode") if context else None,
        "funding": context.get("funding_rate") if context else None,
        "fear_greed": context.get("fear_greed") if context else None,
        "ml_direction": context.get("ml_direction") if context else None,
    }
    from trade_logger import _append_jsonl, _trade_log

    _trade_log.info(json.dumps(record))
    _append_jsonl(record)
    log.info(
        f"  [TRADE OPEN] {side} {symbol} | entry={entry} SL={sl} TP={tp} "
        f"R:R={rr:.2f} | setup={setup} | telegram=off (demo)"
    )


def _get_exit_price_from_binance(client: UMFutures, symbol: str) -> float | None:
    """Get fill price of the most recent closing trade from Binance account trades."""
    try:
        trades = client.get_account_trades(symbol=f"{symbol}USDT", limit=10)
        if trades:
            return float(trades[-1]["price"])
    except Exception as e:
        log.warning(f"[MONITOR] Could not fetch exit price for {symbol}: {e}")
    return None


_ORPHAN_SL_PCT = 0.005   # 0.5% default SL for recovered positions
_ORPHAN_TP_PCT = 0.010   # 1.0% default TP for recovered positions (R:R = 2)


def _recover_orphaned_positions(client: UMFutures, open_trades: list[dict]):
    """
    Scans Binance for open positions that have no matching OPEN record in the DB.
    Inserts a recovery record and places default SL/TP orders on Binance.
    """
    tracked_symbols = {t["symbol"] for t in open_trades}
    try:
        all_positions = client.get_position_risk()
    except Exception as e:
        log.warning(f"[MONITOR] Could not fetch Binance positions for orphan check: {e}")
        return

    for pos in all_positions:
        amt = float(pos.get("positionAmt", 0))
        if amt == 0:
            continue  # no position

        sym = pos["symbol"].replace("USDT", "")
        if sym in tracked_symbols:
            continue  # already tracked in DB

        side = "BUY" if amt > 0 else "SELL"
        entry = float(pos["entryPrice"])
        qty   = abs(amt)
        mark  = float(pos["markPrice"])

        # Compute default SL/TP from entry price
        if side == "BUY":
            sl = round(entry * (1 - _ORPHAN_SL_PCT), 2)
            tp = round(entry * (1 + _ORPHAN_TP_PCT), 2)
            close_side = "SELL"
        else:
            sl = round(entry * (1 + _ORPHAN_SL_PCT), 2)
            tp = round(entry * (1 - _ORPHAN_TP_PCT), 2)
            close_side = "BUY"

        log.warning(
            f"[MONITOR] Orphaned position detected: {sym} {side} qty={qty} "
            f"entry={entry} mark={mark} — recovering with SL={sl} TP={tp}"
        )

        sym_pair = f"{sym}USDT"

        # Place SL order on Binance
        sl_price = round(sl * (0.998 if close_side == "SELL" else 1.002), 2)
        try:
            client.new_order(
                symbol=sym_pair, side=close_side, type="STOP",
                stopPrice=str(sl), price=str(sl_price),
                quantity=qty, reduceOnly="true",
            )
            log.info(f"[MONITOR] Recovery SL placed @ {sl}")
        except Exception as e:
            log.warning(f"[MONITOR] Could not place recovery SL for {sym}: {e}")

        # Place TP order on Binance
        tp_price = round(tp * (0.999 if close_side == "SELL" else 1.001), 2)
        try:
            client.new_order(
                symbol=sym_pair, side=close_side, type="TAKE_PROFIT",
                stopPrice=str(tp), price=str(tp_price),
                quantity=qty, reduceOnly="true",
            )
            log.info(f"[MONITOR] Recovery TP placed @ {tp}")
        except Exception as e:
            log.warning(f"[MONITOR] Could not place recovery TP for {sym}: {e}")

        # Insert DB record with the computed SL/TP
        conn = _get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO trades
                      (symbol, side, status, entry_price, quantity, leverage,
                       stop_loss, take_profit, confidence, horizon)
                    VALUES (%s,%s,'OPEN',%s,%s,%s,%s,%s, 0, 1)
                """, (sym, side, entry, qty, LEVERAGE, sl, tp))
            conn.commit()
            log.info(f"[MONITOR] Recovery record inserted for {sym} {side} @ {entry}")
        except Exception as e:
            log.warning(f"[MONITOR] Failed to insert recovery record for {sym}: {e}")
        finally:
            conn.close()


def _monitor_loop(client: UMFutures):
    """
    1. Detects when a native Binance SL/TP has fired and records the close in DB.
    2. Detects positions open on Binance but missing from DB and auto-recovers them.
    Runs in a background daemon thread.
    """
    log.info("[MONITOR] Background sync thread started (60s interval)")
    while True:
        try:
            db_cleanup_stale_pending()
            open_trades = db_get_open_trades()

            # --- Phase 1: software SL/TP enforcement ---
            # Exchange-side conditional orders are not supported on this account.
            # The monitor enforces SL/TP by checking mark price every cycle.
            for trade in list(open_trades):
                sl = trade.get("stop_loss") or 0
                tp = trade.get("take_profit") or 0
                if not sl and not tp:
                    continue
                sym = trade["symbol"]
                try:
                    ticker = client.ticker_price(symbol=f"{sym}USDT")
                    mark = float(ticker["price"])
                except Exception:
                    continue
                trade_side = trade["side"]
                hit = None
                if trade_side == "BUY":
                    if sl and mark <= sl:
                        hit = "stop_loss"
                    elif tp and mark >= tp:
                        hit = "take_profit"
                else:
                    if sl and mark >= sl:
                        hit = "stop_loss"
                    elif tp and mark <= tp:
                        hit = "take_profit"
                if hit:
                    close_side = "SELL" if trade_side == "BUY" else "BUY"
                    qty = trade["quantity"]
                    log.warning(f"[MONITOR] {sym} {hit} hit (mark={mark}) — closing position")
                    try:
                        client.new_order(symbol=f"{sym}USDT", side=close_side,
                                         type="MARKET", quantity=qty)
                        db_close_trade(trade["id"], mark, hit)
                        open_trades = [t for t in open_trades if t["id"] != trade["id"]]
                    except Exception as e:
                        log.warning(f"[MONITOR] Failed to close {sym} on {hit}: {e}")

            # --- Phase 2: recover orphaned Binance positions ---
            _recover_orphaned_positions(client, open_trades)

            # --- Phase 3: close DB records when Binance position is gone ---
            for trade in open_trades:
                sym = trade["symbol"]
                pos = get_open_position(client, sym)
                if pos is not None:
                    continue  # still open — nothing to do

                # SL/TP fired or manually closed
                exit_price = _get_exit_price_from_binance(client, sym)
                if exit_price is None:
                    try:
                        ticker = client.ticker_price(symbol=f"{sym}USDT")
                        exit_price = float(ticker["price"])
                    except Exception:
                        continue

                sl = trade["stop_loss"]
                tp = trade["take_profit"]
                side = trade["side"]
                if side == "BUY":
                    reason = ("take_profit" if exit_price >= tp
                              else "stop_loss" if exit_price <= sl
                              else "native_close")
                else:
                    reason = ("take_profit" if exit_price <= tp
                              else "stop_loss" if exit_price >= sl
                              else "native_close")

                log.info(f"[MONITOR] {sym} closed on Binance → reason={reason} exit={exit_price}")
                db_close_trade(trade["id"], exit_price, reason)

        except Exception as e:
            log.warning(f"[MONITOR] Sync error: {e}")

        time.sleep(MONITOR_INTERVAL)


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
            for horizon_token in ML_HORIZONS:
                val = r.get(f"prediction:{sym}:{horizon_token}")
                if val:
                    result[f"{sym}_{horizon_token}"] = json.loads(val)
        return result
    except Exception as e:
        log.warning(f"Redis ML read failed: {e}")
        return {}


def get_market_signals() -> dict:
    """Pull fear/greed + funding rates from DB."""
    signals = {}
    conn = _get_conn()
    try:
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
    finally:
        conn.close()
    return signals


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
                wallet = float(asset["walletBalance"])
                unrealized = float(asset.get("unrealizedProfit", 0))
                return round(wallet + unrealized, 2)
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

def get_symbol_min_notional(client: UMFutures, symbol: str) -> float:
    cached = _SYMBOL_MIN_NOTIONAL_CACHE.get(symbol)
    if cached is not None:
        return cached

    sym_pair = f"{symbol}USDT"
    min_notional = DEFAULT_MIN_NOTIONAL
    try:
        info = client.exchange_info()
        for sym_info in info.get("symbols", []):
            if sym_info.get("symbol") != sym_pair:
                continue
            candidates: list[float] = []
            for filt in sym_info.get("filters", []):
                if filt.get("filterType") not in {"MIN_NOTIONAL", "NOTIONAL"}:
                    continue
                for key in ("notional", "minNotional"):
                    raw = filt.get(key)
                    try:
                        value = float(raw)
                    except (TypeError, ValueError):
                        continue
                    if value > 0:
                        candidates.append(value)
            if candidates:
                min_notional = max(candidates)
            break
    except Exception as e:
        log.warning(f"[{symbol}] exchange_info min notional lookup failed, using fallback {DEFAULT_MIN_NOTIONAL:.2f}: {e}")

    _SYMBOL_MIN_NOTIONAL_CACHE[symbol] = min_notional
    return min_notional


def calc_quantity(balance: float, entry: float, sl: float, symbol: str,
                  min_notional: float) -> float:
    risk_usdt     = balance * POSITION_RISK_PCT
    sl_distance   = abs(entry - sl)
    if sl_distance == 0:
        return 0.0
    max_fraction = MAX_POSITION_FRACTION_BY_SYMBOL.get(symbol, MAX_POSITION_FRACTION)
    position_value = (risk_usdt / sl_distance) * entry
    position_value = min(position_value, balance * max_fraction * LEVERAGE)
    if position_value < min_notional:
        return 0.0
    qty       = position_value / entry
    precision = QTY_PRECISION.get(symbol, 2)
    factor = 10 ** precision
    qty = math.floor(qty * factor) / factor
    if qty <= 0 or (qty * entry) < min_notional:
        return 0.0
    return qty


def min_balance_required_for_symbol(symbol: str, entry_price: float,
                                   min_notional: float) -> float:
    """
    Minimum balance needed so capped position size can still produce at least
    one exchange step-size unit (e.g., BTC 0.001).
    """
    precision = QTY_PRECISION.get(symbol, 2)
    min_qty_step = 10 ** (-precision)
    min_notional_for_step = entry_price * min_qty_step
    max_fraction = MAX_POSITION_FRACTION_BY_SYMBOL.get(symbol, MAX_POSITION_FRACTION)
    denom = max_fraction * LEVERAGE
    if denom <= 0:
        return float("inf")
    return max(min_notional_for_step, min_notional) / denom


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

    # Use price already fetched during analysis to avoid tick drift between analysis and execution
    entry = float(decision.get("entry_price") or context.get("current_price", 0))
    if entry == 0:
        try:
            ticker = client.ticker_price(symbol=f"{symbol}USDT")
            entry  = float(ticker["price"])
        except Exception as e:
            log_error(f"[{symbol}] Ticker fetch failed", e)
            return False

    if signal == "BUY" and not (ai_tp > entry > ai_sl):
        log_gate_fail("DIRECTION", f"BUY needs TP({ai_tp}) > entry({entry}) > SL({ai_sl})", symbol)
        return False
    if signal == "SELL" and not (ai_sl > entry > ai_tp):
        log_gate_fail("DIRECTION", f"SELL needs SL({ai_sl}) > entry({entry}) > TP({ai_tp})", symbol)
        return False

    risk_pct = abs(entry - ai_sl) / entry * 100
    if risk_pct < SL_MIN_PCT * 100:
        log_gate_fail("SL_MIN", f"SL {risk_pct:.3f}% < min {SL_MIN_PCT*100:.2f}%", symbol)
        return False
    if risk_pct > STOP_LOSS_PCT * 100:
        log_gate_fail("SL_MAX", f"SL {risk_pct:.3f}% > max {STOP_LOSS_PCT*100:.2f}%", symbol)
        return False

    risk   = abs(entry - ai_sl)
    reward  = abs(ai_tp - entry)
    rr      = float(decision.get("planned_rr") or (reward / risk if risk > 0 else 0))
    is_setup_e = "setup_e" in setup.lower() if setup else False
    min_rr  = SETUP_E_MIN_RR if is_setup_e else TAKE_PROFIT_MIN_RR
    if rr < min_rr:
        log_gate_fail("RR", f"R:R={rr:.2f} < {min_rr} ({'Setup E' if is_setup_e else 'standard'})", symbol)
        return False

    min_notional = get_symbol_min_notional(client, symbol)
    qty = calc_quantity(balance, entry, ai_sl, symbol, min_notional)
    if qty <= 0:
        min_balance = min_balance_required_for_symbol(symbol, entry, min_notional)
        log_gate_fail(
            "QTY",
            (
                f"Quantity too small or below min notional ${min_notional:.2f}. "
                f"balance={balance:.2f} requires>={min_balance:.2f} with leverage={LEVERAGE} "
                f"and cap={MAX_POSITION_FRACTION_BY_SYMBOL.get(symbol, MAX_POSITION_FRACTION):.2f}"
            ),
            symbol,
        )
        return False

    log.info(
        f"  [EXEC] {signal} {symbol} | entry≈{entry} SL={ai_sl} TP={ai_tp} "
        f"qty={qty} R:R={rr:.2f} | dry_run={dry_run}"
    )

    if dry_run:
        log_demo_trade_open(symbol, signal, entry, ai_sl, ai_tp, rr, setup, reason,
                            trade_id="DRY_RUN", context=context)
        return True

    side     = "BUY" if signal == "BUY" else "SELL"
    sym_pair = f"{symbol}USDT"
    confidence = context.get("ml_confidence", 0.0)

    # --- Step 1: reserve DB slot before touching the exchange ---
    pending_id = None
    try:
        pending_id = db_create_pending(symbol, signal, setup, confidence)
        log.info(f"  [EXEC] PENDING record created id={pending_id}")
    except Exception as e:
        log_error(f"[{symbol}] Could not create PENDING record — aborting", e)
        return False

    try:
        client.change_leverage(symbol=sym_pair, leverage=LEVERAGE)

        # --- Step 2: entry order ---
        order = client.new_order(
            symbol=sym_pair, side=side, type="MARKET", quantity=qty,
        )
        order_id     = str(order.get("orderId", ""))
        actual_price = float(order.get("avgPrice", entry)) or entry
        log.info(f"  [EXEC] Entry filled @ {actual_price} orderId={order_id}")

        # --- Step 3: confirm OPEN — SL/TP enforced by software monitor ---
        db_confirm_open(pending_id, actual_price, qty, ai_sl, ai_tp, order_id, reason)
        log_demo_trade_open(symbol, signal, actual_price, ai_sl, ai_tp, rr,
                            setup, reason, trade_id=order_id, context=context)
        return True

    except Exception as e:
        log_error(f"[{symbol}] Order execution failed", e)
        db_cancel_pending(pending_id)
        return False


def run_symbol_cycle(client: UMFutures, symbol: str,
                     ml_preds: dict, market_signals: dict,
                     balance: float, dry_run: bool = False):

    log.info(f"\n{'─'*50}")
    log.info(f"  CYCLE  {symbol}  {datetime.now(UTC).strftime('%H:%M:%S UTC')}")

    pos = get_open_position(client, symbol)
    if pos:
        upnl_pct = pos["upnl"] / (pos["entry_price"] * pos["amount"]) * 100
        log.info(
            f"  [SKIP] {symbol} has open {pos['side']} position | "
            f"entry={pos['entry_price']} mark={pos['mark_price']} "
            f"uPnL={upnl_pct:+.2f}%"
        )
        return

    try:
        df_h1  = fetch_ohlcv(client, symbol, "1h",  200)
        df_m15 = fetch_ohlcv(client, symbol, "15m", 100)
        df_m5  = fetch_ohlcv(client, symbol, "5m",  100)
        # BTC as correlation filter
        df_btc_h1 = fetch_ohlcv(client, "BTC", "1h", 100) if symbol != "BTC" else df_h1
        df_btc_m15 = fetch_ohlcv(client, "BTC", "15m", 100) if symbol != "BTC" else df_m15
    except Exception as e:
        log_error(f"[{symbol}] OHLCV fetch failed", e)
        return

    ctx = compute_indicators(df_h1, df_m15, df_m5)
    ctx["symbol"] = f"{symbol}/USDT"

    min_notional = get_symbol_min_notional(client, symbol)
    required_balance = min_balance_required_for_symbol(symbol, ctx["current_price"], min_notional)
    if balance < required_balance:
        reason = (
            f"balance {balance:.2f} below tradable minimum {required_balance:.2f} "
            f"for {symbol} at price {ctx['current_price']:.2f}"
        )
        log_gate_fail("BALANCE", reason, symbol, ctx)
        log_skip("BALANCE", reason, ctx, None)
        return

    # BTC trend check
    if symbol != "BTC":
        btc_ctx = compute_indicators(df_btc_h1, df_btc_m15, df_m5)
        btc_trend = btc_ctx["m15_trend"]
        ctx["btc_trend"] = btc_trend
    else:
        ctx["btc_trend"] = ctx["m15_trend"]

    ctx["sr"] = find_sr_levels(df_h1, ctx["current_price"], df_m15)

    ctx["use_fvg_scoring"] = USE_FVG_FILTER
    score, score_details = compute_score(ctx)
    ctx["score"]        = score
    ctx["score_details"] = score_details

    log_cycle_start(symbol, ctx["market_mode"], score)
    log.info(f"  H1={ctx['h1_trend']} M15={ctx['m15_trend']} BTC={ctx.get('btc_trend','?')}")
    log.info(f"  S/R  R={ctx['sr']['resistance']} S={ctx['sr']['support']}")
    log.info(f"  Score {score}/5: {', '.join(score_details) or 'none'}")
    log.info(f"  ADX={ctx['adx']:.1f} RSI={ctx['rsi']:.1f} ATR={ctx['atr_m15']:.4f}")

    ml_primary_token = ML_PRIMARY_HORIZON
    ml_pred = {}
    for horizon_token in [ML_PRIMARY_HORIZON] + [h for h in ML_HORIZONS if h != ML_PRIMARY_HORIZON]:
        candidate_key = f"{symbol}_{horizon_token}"
        candidate = ml_preds.get(candidate_key, {})
        if candidate:
            ml_pred = candidate
            ml_primary_token = horizon_token
            break

    ml_dir = ml_pred.get("direction", "")
    ml_conf = float(ml_pred.get("confidence", 0))

    trend_key = f"{symbol}_{ML_TREND_HORIZON}"
    trend_pred = ml_preds.get(trend_key, {})
    trend_dir = trend_pred.get("direction", "")
    trend_conf = float(trend_pred.get("confidence", 0))
    fear_greed  = market_signals.get("fear_greed")
    funding     = market_signals.get(f"{symbol.lower()}_funding", 0)

    ctx["ml_direction"]  = ml_dir
    ctx["ml_confidence"] = ml_conf
    ctx["ml_entry_horizon"] = ml_primary_token
    ctx["ml_trend_horizon"] = ML_TREND_HORIZON
    ctx["ml_trend_direction"] = trend_dir
    ctx["ml_trend_confidence"] = trend_conf
    ctx["ml_conflict_resolved"] = bool(ml_dir and trend_dir and ml_dir != trend_dir)
    ctx["ml_conflict_min_rr"] = 0.0
    ctx["fear_greed"]    = fear_greed
    ctx["funding_rate"]  = funding

    macro_ok, macro_reason, allowed_dir = check_macro_bias(
        ml_dir, ml_conf, fear_greed, funding, symbol
    )
    ctx["allowed_direction"] = allowed_dir

    if not macro_ok:
        log_gate_fail("MACRO", macro_reason, symbol, ctx)
        return
    trend_str = f" trend={ML_TREND_HORIZON}:{trend_dir}({trend_conf:.0%})" if trend_pred else " trend=n/a"
    conflict_str = " conflict=advisory_only" if ctx["ml_conflict_resolved"] else ""
    log_gate_pass(
        "MACRO",
        f"allowed={allowed_dir} ML={ml_primary_token}:{ml_dir}({ml_conf:.0%}){trend_str} F&G={fear_greed}{conflict_str}",
    )

    if symbol != "BTC":
        btc_trend = ctx.get("btc_trend", "")
        primary = ctx.get("primary_trend") or ctx["m15_trend"]
        if (primary == "UPTREND" and btc_trend == "DOWNTREND") or \
           (primary == "DOWNTREND" and btc_trend == "UPTREND"):
            log_gate_fail("BTC_CORR", f"BTC={btc_trend} conflicts with {symbol}={primary}", symbol, ctx)
            return
        log_gate_pass("BTC_CORR", f"BTC={btc_trend}")

    tech_ok, tech_reason = check_technical_gates(ctx)
    if not tech_ok:
        log_gate_fail("TECHNICAL", tech_reason, symbol, ctx)
        return
    log_gate_pass("TECHNICAL", tech_reason)

    if ctx["is_range"]:
        ctx["range_bias"] = get_range_bias(ctx)
        log.info(f"  Range bias: {ctx['range_bias']}")
        if ctx["range_bias"] == "MIDDLE":
            log_gate_fail("RANGE_POSITION", "price in middle of range — no edge", symbol, ctx)
            return

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
        log_cycle_summary(symbol, "WAIT", False, balance, ctx, decision)
        return

    ai_ok, ai_reason = validate_ai_trade_decision(decision, ctx, df_m5)
    if not ai_ok:
        log_gate_fail("AI_VALIDATION", ai_reason, symbol, ctx)
        log_skip("AI_INVALID", ai_reason, ctx, decision)
        log_cycle_summary(symbol, signal, False, balance, ctx, decision)
        return

    setup_name = decision.get("analysis", {}).get("setup_identified", "")
    plan, plan_reason = build_trade_plan(signal, setup_name, ctx, df_m5)
    if not plan:
        log_gate_fail("TRADE_PLAN", plan_reason, symbol, ctx)
        log_skip("TRADE_PLAN", plan_reason, ctx, decision)
        log_cycle_summary(symbol, signal, False, balance, ctx, decision)
        return

    decision["entry_price"] = plan["entry_price"]
    decision["stop_loss"] = plan["stop_loss"]
    decision["take_profit"] = plan["take_profit"]
    decision["planned_rr"] = plan["rr"]
    decision.setdefault("analysis", {})
    decision["analysis"]["rr_check"] = (
        f"Python planned Entry={plan['entry_price']} SL={plan['stop_loss']} "
        f"TP={plan['take_profit']} net R:R={plan['rr']:.2f} gross R:R={plan['gross_rr']:.2f}. Pass."
    )
    log.info(
        f"  [PLAN] entry={plan['entry_price']} SL={plan['stop_loss']} TP={plan['take_profit']} "
        f"netR:R={plan['rr']:.2f} grossR:R={plan['gross_rr']:.2f} | {plan_reason}"
    )

    executed = execute_trade(client, symbol, decision, balance, ctx, dry_run=dry_run)
    log_cycle_summary(symbol, signal, executed, balance, ctx, decision)


def run_once(dry_run: bool = False):
    log.info("=" * 60)
    variant = "FVG" if USE_FVG_FILTER else "BASELINE"
    log.info(f"ENGINE V2 {variant} {'DRY RUN' if dry_run else 'LIVE'}  {datetime.now(UTC).isoformat()}")
    log.info("=" * 60)

    client         = get_client()
    ml_preds       = get_ml_predictions()
    market_signals = get_market_signals()
    balance        = get_account_balance(client)

    log.info(f"Balance: {balance:.2f} USDT | ML preds loaded: {list(ml_preds.keys())}")
    log.info(
        f"ML mode: entry={ML_PRIMARY_HORIZON} trend={ML_TREND_HORIZON} "
        f"require_alignment={ML_REQUIRE_TREND_ALIGNMENT}"
    )
    funding_str = "  ".join(
        f"{s} funding={market_signals.get(f'{s.lower()}_funding', 0):+.4f}%"
        for s in SYMBOLS
    )
    log.info(f"Fear&Greed={market_signals.get('fear_greed','?')}  {funding_str}")

    for symbol in SYMBOLS:
        try:
            run_symbol_cycle(client, symbol, ml_preds, market_signals, balance, dry_run)
        except Exception as e:
            log_error(f"[{symbol}] Unhandled cycle error", e)


def run_loop(dry_run: bool = False):
    log.info("Engine V2 started — 5-min cycle loop")

    try:
        db_ensure_trades_table()
    except Exception as e:
        log.warning(f"DB table init failed (will retry): {e}")

    client_monitor = get_client()
    t = threading.Thread(target=_monitor_loop, args=(client_monitor,), daemon=True)
    t.start()

    while True:
        try:
            run_once(dry_run=dry_run)
        except Exception as e:
            log_error("Loop-level error", e)
        log.info(f"  Sleeping {CYCLE_INTERVAL}s until next cycle...")
        time.sleep(CYCLE_INTERVAL)


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
