"""
engine_live.py — Live account trading engine (BTC, ETH, SOL only).

Identical strategy to engine.py but:
  - Uses BINANCE_LIVE_API_KEY / BINANCE_LIVE_SECRET_KEY (real Binance, not testnet)
  - Trades BTC, ETH, SOL only (XRP excluded — min notional risk at small balance)
  - Exchange-side STOP_MARKET / TAKE_PROFIT_MARKET for SL/TP (works on live)
  - Software monitor as backup SL/TP in case exchange orders fail
  - account_type='live' on all DB records — separate from testnet trades
  - Min notional check: skips trades where order value is below Binance minimum
"""

import argparse
import hashlib
import hmac
import json
import math
import os
import time
import threading
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
UTC = timezone.utc
from pathlib import Path

import psycopg2
from psycopg2 import errors
from psycopg2 import OperationalError, InterfaceError
import redis
from binance.um_futures import UMFutures
from dotenv import load_dotenv

from chart_gen      import generate_chart
from llm_analyst    import ask_gemini
from strategy_core  import (
    compute_indicators, compute_score, find_sr_levels,
    check_macro_bias, check_technical_gates, get_range_bias,
    validate_ai_trade_decision, build_trade_plan, compute_max_stop_pct,
)
from trade_logger   import (
    get_logger, log_cycle_start, log_gate_pass, log_gate_fail,
    log_ai_request, log_ai_response, log_trade_open, log_trade_close,
    log_skip, log_error, log_cycle_summary,
)

ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT / ".env")

log = get_logger()

API_KEY      = os.getenv("BINANCE_LIVE_API_KEY", "")
API_SECRET   = os.getenv("BINANCE_LIVE_SECRET_KEY", "")
DATABASE_URL = os.getenv("DATABASE_URL", "")
REDIS_URL    = os.getenv("REDIS_URL", "")

ACCOUNT_TYPE   = "live"
TRADES_TABLE   = "trades_live"
SYMBOLS        = ["BTC", "ETH", "SOL"]
LEVERAGE       = 5
try:
    DEFAULT_MIN_NOTIONAL = max(
        0.0,
        float(os.getenv("BINANCE_LIVE_MIN_NOTIONAL", os.getenv("BINANCE_MIN_NOTIONAL", "100"))),
    )
except (TypeError, ValueError):
    DEFAULT_MIN_NOTIONAL = 100.0

QTY_PRECISION  = {
    "BTC": 3,
    "ETH": 2,
    "SOL": 1,
}
PRICE_PRECISION = {
    "BTC": 1,   # tick size 0.1  → e.g. 84234.5
    "ETH": 2,   # tick size 0.01 → e.g. 2113.95
    "SOL": 2,   # tick size 0.01 → e.g. 79.87
    "XRP": 4,   # tick size 0.0001 → e.g. 2.1234
}
STOP_LOSS_PCT         = 0.008
BREAKOUT_STOP_LOSS_PCT = 0.013
SL_MIN_PCT            = 0.002
try:
    SL_ATR_MULTIPLIER = max(0.0, float(os.getenv("SL_ATR_MULTIPLIER", "1.8")))
except (TypeError, ValueError):
    SL_ATR_MULTIPLIER = 1.8
try:
    SL_SR_BUFFER_ATR_MULT = max(0.0, float(os.getenv("SL_SR_BUFFER_ATR_MULT", "0.25")))
except (TypeError, ValueError):
    SL_SR_BUFFER_ATR_MULT = 0.25
try:
    TAKE_PROFIT_MIN_RR = max(0.5, float(os.getenv("TRADE_MIN_RR", "1.5")))
except (TypeError, ValueError):
    TAKE_PROFIT_MIN_RR = 1.5
try:
    RANGE_MIN_RR = max(0.5, float(os.getenv("TRADE_MIN_RR_RANGE", str(TAKE_PROFIT_MIN_RR))))
except (TypeError, ValueError):
    RANGE_MIN_RR = TAKE_PROFIT_MIN_RR
try:
    TP_EXTENSION_ATR_MULT = max(0.0, float(os.getenv("TP_EXTENSION_ATR_MULT", "0.15")))
except (TypeError, ValueError):
    TP_EXTENSION_ATR_MULT = 0.15
try:
    SETUP_E_MIN_RR = max(0.5, float(os.getenv("SETUP_E_MIN_RR", "1.0")))
except (TypeError, ValueError):
    SETUP_E_MIN_RR = 1.0
POSITION_RISK_PCT     = 0.01
MAX_POSITION_FRACTION = 0.10  # default cap: 10% of balance * leverage
MAX_POSITION_FRACTION_BY_SYMBOL = {
    "BTC": 0.35,  # higher cap so small balances can still reach Binance min notional
}
CYCLE_INTERVAL        = 5 * 60
MONITOR_INTERVAL      = 5
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
    return UMFutures(key=API_KEY, secret=API_SECRET)  # no base_url = live


def _algo_order(symbol: str, side: str, order_type: str, trigger_price: float,
                quantity: float | None = None) -> dict:
    """Place SL/TP via Binance Algo Order API (/fapi/v1/algoOrder).

    quantity=None  → closePosition=true (closes entire position — used for SL and single TP)
    quantity=N     → partial close of N units (used for staged TP1 / TP2)
    """
    base      = symbol.replace("USDT", "")
    precision = PRICE_PRECISION.get(base, 2)
    qty_prec  = QTY_PRECISION.get(base, 3)
    params = {
        "algoType":     "CONDITIONAL",
        "symbol":       symbol,
        "side":         side,
        "type":         order_type,
        "triggerPrice": f"{trigger_price:.{precision}f}",
        "workingType":  "MARK_PRICE",
        "timestamp":    int(time.time() * 1000),
    }
    if quantity is not None:
        params["quantity"] = f"{quantity:.{qty_prec}f}"
    else:
        params["closePosition"] = "true"
    query = urllib.parse.urlencode(params)
    sig   = hmac.new(API_SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()
    url   = f"https://fapi.binance.com/fapi/v1/algoOrder?{query}&signature={sig}"
    req   = urllib.request.Request(url, method="POST", headers={"X-MBX-APIKEY": API_KEY})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def _list_open_algo_orders(symbol: str) -> list:
    """Return open algo orders for a symbol from Binance."""
    params = {"symbol": symbol, "timestamp": int(time.time() * 1000)}
    query = urllib.parse.urlencode(params)
    sig   = hmac.new(API_SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()
    url   = f"https://fapi.binance.com/fapi/v1/openAlgoOrders?{query}&signature={sig}"
    req   = urllib.request.Request(url, headers={"X-MBX-APIKEY": API_KEY})
    with urllib.request.urlopen(req, timeout=10) as r:
        data = json.loads(r.read())
    return data.get("orders", [])


def _cancel_algo_order(algo_id: int) -> dict:
    """Cancel an algo order by algoId."""
    params = {"algoId": algo_id, "timestamp": int(time.time() * 1000)}
    query = urllib.parse.urlencode(params)
    sig   = hmac.new(API_SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()
    url   = f"https://fapi.binance.com/fapi/v1/algoOrder?{query}&signature={sig}"
    req   = urllib.request.Request(url, method="DELETE", headers={"X-MBX-APIKEY": API_KEY})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def _get_conn():
    return psycopg2.connect(
        DATABASE_URL,
        sslmode="require",
        connect_timeout=8,
        keepalives=1,
        keepalives_idle=30,
        keepalives_interval=10,
        keepalives_count=3,
    )


def _is_transient_db_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    transient_markers = (
        "ssl connection has been closed unexpectedly",
        "server closed the connection unexpectedly",
        "connection not open",
        "terminating connection due to administrator command",
        "could not receive data from server",
    )
    return any(marker in msg for marker in transient_markers)


def db_ensure_trades_table():
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {TRADES_TABLE} (
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
            for col, definition in [
                ("setup",            "TEXT"),
                ("notes",            "TEXT"),
                ("confidence",       "FLOAT"),
                ("binance_order_id", "TEXT"),
                ("close_reason",     "TEXT"),
                ("account_type",     "TEXT DEFAULT 'testnet'"),
                ("take_profit_2",    "FLOAT"),
            ]:
                cur.execute(f"""
                    ALTER TABLE {TRADES_TABLE} ADD COLUMN IF NOT EXISTS {col} {definition}
                """)
            cur.execute(f"""
                CREATE UNIQUE INDEX IF NOT EXISTS {TRADES_TABLE}_one_open_position_per_account_symbol_idx
                ON {TRADES_TABLE} (account_type, symbol)
                WHERE status = 'OPEN'
            """)
        conn.commit()
    finally:
        conn.close()


def db_create_pending(symbol: str, side: str, setup: str,
                      confidence: float = 0.0) -> int:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(f"""
                INSERT INTO {TRADES_TABLE}
                  (symbol, side, status, entry_price, quantity, leverage,
                   stop_loss, take_profit, confidence, setup, horizon, account_type)
                VALUES (%s,%s,'PENDING', 0, 0, %s, 0, 0, %s,%s, 1, %s)
                RETURNING id
            """, (symbol, side, LEVERAGE, confidence, setup, ACCOUNT_TYPE))
            tid = cur.fetchone()[0]
        conn.commit()
    finally:
        conn.close()
    return tid


def db_confirm_open(trade_id: int, entry_price: float, quantity: float,
                    stop_loss: float, take_profit: float, take_profit_2: float,
                    order_id: str, notes: str):
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(f"""
                UPDATE {TRADES_TABLE} SET
                    status='OPEN', entry_price=%s, quantity=%s,
                    stop_loss=%s, take_profit=%s, take_profit_2=%s,
                    binance_order_id=%s, notes=%s
                WHERE id=%s
            """, (entry_price, quantity, stop_loss, take_profit, take_profit_2,
                  order_id, notes[:200], trade_id))
        conn.commit()
    finally:
        conn.close()


def db_cancel_pending(trade_id: int):
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(f"DELETE FROM {TRADES_TABLE} WHERE id=%s AND status='PENDING'", (trade_id,))
        conn.commit()
    finally:
        conn.close()


def db_update_stop_loss(trade_id: int, new_sl: float):
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE {TRADES_TABLE} SET stop_loss=%s WHERE id=%s",
                (new_sl, trade_id),
            )
        conn.commit()
    finally:
        conn.close()


def db_close_trade(trade_id: int, exit_price: float, reason: str):
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT entry_price, quantity, side FROM {TRADES_TABLE} WHERE id=%s", (trade_id,)
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
            cur.execute(f"""
                UPDATE {TRADES_TABLE} SET
                    status='CLOSED', exit_price=%s, pnl_usdt=%s,
                    pnl_pct=%s, close_reason=%s, closed_at=NOW()
                WHERE id=%s
            """, (exit_price, round(pnl_usdt, 4), round(pnl_pct * 100, 4), reason, trade_id))
        conn.commit()
    finally:
        conn.close()
    log.info(f"[DB] Trade {trade_id} closed — {reason} @ {exit_price} pnl={pnl_pct*100:.2f}%")


def db_get_open_trades() -> list[dict]:
    retries = 2
    for attempt in range(retries + 1):
        conn = None
        try:
            conn = _get_conn()
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT id, symbol, side, entry_price, quantity,
                           stop_loss, take_profit, take_profit_2, binance_order_id, opened_at
                    FROM {TRADES_TABLE} WHERE status='OPEN' AND account_type=%s
                """, (ACCOUNT_TYPE,))
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, r)) for r in cur.fetchall()]
        except (OperationalError, InterfaceError) as e:
            if attempt < retries and _is_transient_db_error(e):
                wait_s = 0.25 * (attempt + 1)
                log.warning(
                    f"[DB] get_open_trades transient error ({e}); retrying in {wait_s:.2f}s "
                    f"({attempt + 1}/{retries})"
                )
                time.sleep(wait_s)
                continue
            log.warning(f"[DB] get_open_trades failed: {e}")
            return []
        except Exception as e:
            log.warning(f"[DB] get_open_trades failed: {e}")
            return []
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
    return []


def db_cleanup_stale_pending(max_age_seconds: int = 120):
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(f"""
                DELETE FROM {TRADES_TABLE}
                WHERE status='PENDING'
                  AND account_type=%s
                  AND opened_at < NOW() - INTERVAL '%s seconds'
                RETURNING id, symbol
            """, (ACCOUNT_TYPE, max_age_seconds))
            rows = cur.fetchall()
            for tid, sym in rows:
                log.warning(f"[DB] Cleaned up stale PENDING id={tid} symbol={sym}")
        conn.commit()
    except Exception as e:
        log.warning(f"[DB] cleanup_stale_pending failed: {e}")
    finally:
        conn.close()


def _get_exit_fill_from_binance(client: UMFutures, symbol: str, trade_side: str) -> dict | None:
    try:
        trades = client.get_account_trades(symbol=f"{symbol}USDT", limit=10)
        if not trades:
            return None

        closing_side = "SELL" if trade_side == "BUY" else "BUY"
        closing_fills = []
        for trade in reversed(trades):
            if trade.get("side") != closing_side:
                continue
            qty = abs(float(trade.get("qty", 0) or 0))
            price = float(trade.get("price", 0) or 0)
            if qty <= 0 or price <= 0:
                continue
            closing_fills.append(trade)
            # Stop once the fill set changes timestamp significantly.
            if len(closing_fills) >= 1:
                first_time = int(closing_fills[0].get("time", 0) or 0)
                current_time = int(trade.get("time", 0) or 0)
                if first_time and current_time and abs(first_time - current_time) > 5000:
                    closing_fills.pop()
                    break

        if not closing_fills:
            return None

        total_qty = sum(abs(float(t.get("qty", 0) or 0)) for t in closing_fills)
        if total_qty <= 0:
            return None

        weighted_exit = sum(
            float(t.get("price", 0) or 0) * abs(float(t.get("qty", 0) or 0))
            for t in closing_fills
        ) / total_qty
        realized_pnl = sum(float(t.get("realizedPnl", 0) or 0) for t in closing_fills)
        latest_time = max(int(t.get("time", 0) or 0) for t in closing_fills)
        return {
            "exit_price": weighted_exit,
            "qty": total_qty,
            "realized_pnl": realized_pnl,
            "time": latest_time,
            "fills": len(closing_fills),
        }
    except Exception as e:
        log.warning(f"[MONITOR] Could not fetch exit fills for {symbol}: {e}")
        return None


_ORPHAN_SL_PCT = 0.005
_ORPHAN_TP_PCT = 0.010
_POSITION_MISS_COUNTS: dict[int, int] = {}


def _recover_orphaned_positions(client: UMFutures, open_trades: list[dict]):
    tracked_symbols = {t["symbol"] for t in open_trades}
    try:
        all_positions = client.get_position_risk()
    except Exception as e:
        log.warning(f"[MONITOR] Could not fetch Binance positions: {e}")
        return

    for pos in all_positions:
        amt = float(pos.get("positionAmt", 0))
        if amt == 0:
            continue
        sym = pos["symbol"].replace("USDT", "")
        if sym not in SYMBOLS or sym in tracked_symbols:
            continue

        side  = "BUY" if amt > 0 else "SELL"
        entry = float(pos["entryPrice"])
        qty   = abs(amt)
        mark  = float(pos["markPrice"])

        if side == "BUY":
            sl = round(entry * (1 - _ORPHAN_SL_PCT), 2)
            tp = round(entry * (1 + _ORPHAN_TP_PCT), 2)
        else:
            sl = round(entry * (1 + _ORPHAN_SL_PCT), 2)
            tp = round(entry * (1 - _ORPHAN_TP_PCT), 2)

        log.warning(
            f"[MONITOR] Orphaned position: {sym} {side} qty={qty} "
            f"entry={entry} mark={mark} — recovering SL={sl} TP={tp}"
        )

        claimed_trade_id = None
        conn = _get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    INSERT INTO {TRADES_TABLE}
                      (symbol, side, status, entry_price, quantity, leverage,
                       stop_loss, take_profit, confidence, horizon, account_type)
                    VALUES (%s,%s,'OPEN',%s,%s,%s,%s,%s, 0, 1, %s)
                    RETURNING id
                """, (sym, side, entry, qty, LEVERAGE, sl, tp, ACCOUNT_TYPE))
                claimed_trade_id = cur.fetchone()[0]
            conn.commit()
            log.info(f"[MONITOR] Recovery record inserted for {sym} {side} @ {entry} id={claimed_trade_id}")
        except errors.UniqueViolation:
            conn.rollback()
            log.info(f"[MONITOR] Orphan {sym} already claimed by another monitor - skipping recovery insert")
            continue
        except Exception as e:
            conn.rollback()
            log.warning(f"[MONITOR] Failed to insert recovery record for {sym}: {e}")
            continue
        finally:
            conn.close()


def _monitor_loop(client: UMFutures):
    log.info("[MONITOR] Live monitor thread started (5s interval)")
    while True:
        try:
            db_cleanup_stale_pending()
            open_trades = db_get_open_trades()

            # --- Phase 1: software SL/TP backup ---
            for trade in list(open_trades):
                sl = trade.get("stop_loss") or 0
                tp = trade.get("take_profit") or 0
                tp2 = trade.get("take_profit_2") or 0
                # Staged exits (tp2 set and different from tp1) rely on exchange
                # reduceOnly orders for TP. Software backup only enforces SL to
                # avoid prematurely closing the remaining half at TP1 price.
                is_staged = bool(tp2 and abs(tp2 - tp) / max(tp, 1) > 0.0005)
                if not sl and not tp:
                    continue

                sym = trade["symbol"]
                try:
                    ticker = client.ticker_price(symbol=f"{sym}USDT")
                    mark = float(ticker["price"])
                except Exception:
                    continue

                # --- Break-even: move SL to entry once price reaches 1:1 profit ---
                entry = trade.get("entry_price") or 0
                if entry and sl:
                    be_applied = (
                        (trade.get("side") == "BUY"  and sl >= entry * 0.9999) or
                        (trade.get("side") == "SELL" and sl <= entry * 1.0001)
                    )
                    if not be_applied:
                        risk = abs(entry - sl)
                        be_trigger = (entry + risk * 0.5) if trade.get("side") == "BUY" else (entry - risk * 0.5)
                        be_reached = (
                            (trade.get("side") == "BUY"  and mark >= be_trigger) or
                            (trade.get("side") == "SELL" and mark <= be_trigger)
                        )
                        if be_reached:
                            log.info(f"[MONITOR] {sym} break-even triggered (mark={mark:.4f} be={be_trigger:.4f}) — moving SL to entry {entry}")
                            close_side = "SELL" if trade.get("side") == "BUY" else "BUY"
                            try:
                                algo_orders = _list_open_algo_orders(f"{sym}USDT")
                                for ao in algo_orders:
                                    if "STOP" in str(ao.get("type", "")).upper() and ao.get("side") == close_side:
                                        _cancel_algo_order(ao["algoId"])
                                        log.info(f"[MONITOR] {sym} cancelled SL algo {ao['algoId']}")
                                _algo_order(f"{sym}USDT", close_side, "STOP_MARKET", entry)
                                log.info(f"[MONITOR] {sym} new SL placed at entry {entry}")
                            except Exception as e:
                                log.warning(f"[MONITOR] {sym} break-even SL replace failed ({e}) — DB updated, software will enforce")
                            db_update_stop_loss(trade["id"], entry)
                            trade["stop_loss"] = entry
                            sl = entry

                trade_side = trade["side"]
                hit = None
                if trade_side == "BUY":
                    if sl and mark <= sl:
                        hit = "stop_loss"
                    elif not is_staged and tp and mark >= tp:
                        hit = "take_profit"
                else:
                    if sl and mark >= sl:
                        hit = "stop_loss"
                    elif not is_staged and tp and mark <= tp:
                        hit = "take_profit"

                if hit:
                    close_side = "SELL" if trade_side == "BUY" else "BUY"
                    qty = trade["quantity"]
                    log.warning(f"[MONITOR] {sym} {hit} hit (mark={mark}) - closing position")
                    try:
                        client.new_order(
                            symbol=f"{sym}USDT",
                            side=close_side,
                            type="MARKET",
                            quantity=qty,
                            reduceOnly="true",
                        )
                        db_close_trade(trade["id"], mark, hit)
                        open_trades = [t for t in open_trades if t["id"] != trade["id"]]
                    except Exception as e:
                        log.warning(f"[MONITOR] Failed to close {sym} on {hit}: {e}")

            # --- Phase 2: recover orphaned positions ---
            _recover_orphaned_positions(client, open_trades)

            # --- Phase 3: close DB records when Binance position is gone ---
            for trade in open_trades:
                sym = trade["symbol"]
                trade_id = trade["id"]
                pos = get_open_position(client, sym)
                if pos is not None:
                    _POSITION_MISS_COUNTS.pop(trade_id, None)
                    continue

                miss_count = _POSITION_MISS_COUNTS.get(trade_id, 0) + 1
                _POSITION_MISS_COUNTS[trade_id] = miss_count
                if miss_count < 3:
                    log.info(
                        f"[MONITOR] {sym} position miss {miss_count}/3 - waiting before DB close"
                    )
                    continue

                exit_fill = _get_exit_fill_from_binance(client, sym, trade["side"])
                if exit_fill is not None:
                    exit_price = float(exit_fill["exit_price"])
                    log.info(
                        f"[MONITOR] {sym} exit fills={exit_fill['fills']} "
                        f"qty={exit_fill['qty']:.6f} pnl={exit_fill['realized_pnl']:.4f}"
                    )
                else:
                    try:
                        ticker = client.ticker_price(symbol=f"{sym}USDT")
                        exit_price = float(ticker["price"])
                    except Exception:
                        continue
                sl   = trade["stop_loss"]
                tp   = trade["take_profit"]
                side = trade["side"]
                if side == "BUY":
                    reason = ("take_profit" if exit_price >= tp
                              else "stop_loss" if exit_price <= sl
                              else "native_close")
                else:
                    reason = ("take_profit" if exit_price <= tp
                              else "stop_loss" if exit_price >= sl
                              else "native_close")
                log.info(f"[MONITOR] {sym} closed → reason={reason} exit={exit_price}")
                db_close_trade(trade["id"], exit_price, reason)
                _POSITION_MISS_COUNTS.pop(trade_id, None)

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
    signals = {}
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT value FROM fear_greed_index ORDER BY date DESC LIMIT 1")
            row = cur.fetchone()
            if row:
                signals["fear_greed"] = int(row[0])
            for sym in SYMBOLS:
                cur.execute(
                    "SELECT rate_avg FROM funding_rates WHERE symbol=%s "
                    "ORDER BY date DESC LIMIT 1", (sym,)
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
    raw = client.klines(symbol=f"{symbol}USDT", interval=interval, limit=limit)
    df = pd.DataFrame(raw, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col])
    df["timestamp"] = pd.to_numeric(df["timestamp"])
    return df.iloc[:-1].reset_index(drop=True)


def _btc_macro_trend(df_4h: pd.DataFrame) -> str:
    """
    Returns 'BULL', 'BEAR', or 'NEUTRAL' based on BTC 4h EMA50.
      BULL:    close > EMA50 and EMA50 rising  → only block SHORTs on ETH/SOL
      BEAR:    close < EMA50 and EMA50 falling → only block LONGs  on ETH/SOL
      NEUTRAL: everything else                 → both directions allowed
    """
    if len(df_4h) < 55:
        return "NEUTRAL"
    close  = df_4h["close"].astype(float)
    ema50  = close.ewm(span=50, adjust=False).mean()
    last_close = float(close.iloc[-1])
    last_ema   = float(ema50.iloc[-1])
    slope      = float(ema50.iloc[-1]) - float(ema50.iloc[-3])  # 3-bar slope
    if last_close > last_ema and slope > 0:
        return "BULL"
    if last_close < last_ema and slope < 0:
        return "BEAR"
    return "NEUTRAL"


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
    try:
        risk = client.get_position_risk(symbol=f"{symbol}USDT")
        positions = risk if isinstance(risk, list) else [risk]
        for pos in positions:
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
    position_value = (risk_usdt / sl_distance) * entry
    max_fraction = MAX_POSITION_FRACTION_BY_SYMBOL.get(symbol, MAX_POSITION_FRACTION)
    position_value = min(position_value, balance * max_fraction * LEVERAGE)
    # Binance rejects orders below its minimum order notional.
    if position_value < min_notional:
        return 0.0
    qty       = position_value / entry
    precision = QTY_PRECISION.get(symbol, 2)
    factor = 10 ** precision
    qty = math.floor(qty * factor) / factor
    if qty <= 0 or (qty * entry) < min_notional:
        return 0.0
    return qty


def min_balance_required_for_symbol(symbol: str, entry_price: float, min_notional: float) -> float:
    precision = QTY_PRECISION.get(symbol, 2)
    min_qty_step = 10 ** (-precision)
    min_notional_for_step = entry_price * min_qty_step
    max_fraction = MAX_POSITION_FRACTION_BY_SYMBOL.get(symbol, MAX_POSITION_FRACTION)
    denom = max_fraction * LEVERAGE
    if denom <= 0:
        return float("inf")
    return max(min_notional, min_notional_for_step) / denom

def _signal_follows_trend(signal: str, context: dict) -> tuple[bool, str]:
    """Hard guard: AI signal must align with directional trend."""
    if signal not in ("BUY", "SELL"):
        return False, f"invalid signal {signal}"

    primary = context.get("primary_trend") or context.get("m15_trend", "")
    if primary == "UPTREND" and signal != "BUY":
        return False, f"M15={primary} requires BUY"
    if primary == "DOWNTREND" and signal != "SELL":
        return False, f"M15={primary} requires SELL"

    return True, "OK"


def execute_trade(client: UMFutures, symbol: str, decision: dict,
                  balance: float, context: dict, dry_run: bool = False) -> bool:
    signal = decision.get("signal")
    if signal not in ("BUY", "SELL"):
        return False

    try:
        ai_sl  = float(decision.get("stop_loss", 0))
        ai_tp  = float(decision.get("take_profit", 0))
        reason = decision.get("reason", "")
        setup  = decision.get("analysis", {}).get("setup_identified", "?")
    except (TypeError, ValueError) as e:
        log_error(f"[{symbol}] Invalid SL/TP from AI", e)
        return False

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
    max_stop_pct = compute_max_stop_pct(setup, context, entry=entry)
    if risk_pct > max_stop_pct * 100:
        log_gate_fail("SL_MAX", f"SL {risk_pct:.3f}% > max {max_stop_pct*100:.2f}%", symbol)
        return False

    risk      = abs(entry - ai_sl)
    reward    = abs(ai_tp - entry)
    rr        = float(decision.get("planned_rr") or (reward / risk if risk > 0 else 0))
    is_setup_e = "setup_e" in setup.lower() if setup else False
    is_range_setup = any(tag in setup.lower() for tag in ("setup d", "setup_d")) if setup else False
    if is_setup_e:
        min_rr = SETUP_E_MIN_RR
    elif is_range_setup:
        min_rr = RANGE_MIN_RR
    else:
        min_rr = TAKE_PROFIT_MIN_RR
    if rr < min_rr:
        log_gate_fail("RR", f"R:R={rr:.2f} < {min_rr}", symbol)
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

    # --- Staged exit: split position in half for TP1 and TP2 ---
    tp1_price = float(decision.get("take_profit_1") or ai_tp)
    tp2_price = float(decision.get("take_profit_2") or ai_tp)
    price_prec = PRICE_PRECISION.get(symbol, 2)
    qty_prec_factor = 10 ** QTY_PRECISION.get(symbol, 2)
    half_qty = math.floor((qty / 2) * qty_prec_factor) / qty_prec_factor
    staged = (
        half_qty > 0
        and (half_qty * entry) >= min_notional
        and abs(tp2_price - tp1_price) / entry > 0.0005  # TPs must differ by >0.05%
    )

    log.info(
        f"  [EXEC] {signal} {symbol} | entry≈{entry} SL={ai_sl} "
        f"TP1={tp1_price} TP2={tp2_price} qty={qty} half={half_qty} staged={staged} "
        f"R:R={rr:.2f} | dry_run={dry_run}"
    )

    if dry_run:
        log_trade_open(symbol, signal, entry, ai_sl, tp1_price if staged else ai_tp,
                       rr, setup, reason, trade_id="DRY_RUN", context=context)
        return True

    side       = "BUY" if signal == "BUY" else "SELL"
    close_side = "SELL" if signal == "BUY" else "BUY"
    sym_pair   = f"{symbol}USDT"
    confidence = context.get("ml_confidence", 0.0)

    # --- Step 1: reserve DB slot ---
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
        order = client.new_order(symbol=sym_pair, side=side, type="MARKET", quantity=qty)
        order_id     = str(order.get("orderId", ""))
        actual_price = float(order.get("avgPrice", entry)) or entry
        log.info(f"  [EXEC] Entry filled @ {actual_price} orderId={order_id}")

        time.sleep(0.5)

        # --- Step 3: SL via Algo Order API (closePosition=true covers full remaining qty) ---
        try:
            resp = _algo_order(sym_pair, close_side, "STOP_MARKET", ai_sl)
            log.info(f"  [EXEC] SL algo order placed @ {ai_sl} id={resp.get('algoId','?')}")
        except Exception as e:
            log.warning(f"  [EXEC] SL algo order failed ({e}) — software monitor will enforce")

        # --- Step 4: TP orders ---
        if staged:
            # TP1 — close first half at nearer target via Algo Order API
            try:
                resp = _algo_order(sym_pair, close_side, "TAKE_PROFIT_MARKET", tp1_price, quantity=half_qty)
                log.info(f"  [EXEC] TP1 algo order placed @ {tp1_price} qty={half_qty} id={resp.get('algoId','?')}")
            except Exception as e:
                log.warning(f"  [EXEC] TP1 order failed ({e}) — software monitor will enforce")
            # TP2 — close remaining half at farther target via Algo Order API
            try:
                resp = _algo_order(sym_pair, close_side, "TAKE_PROFIT_MARKET", tp2_price, quantity=half_qty)
                log.info(f"  [EXEC] TP2 algo order placed @ {tp2_price} qty={half_qty} id={resp.get('algoId','?')}")
            except Exception as e:
                log.warning(f"  [EXEC] TP2 order failed ({e}) — software monitor will enforce")
        else:
            # Fallback: single TP via Algo Order (position too small to split)
            try:
                resp = _algo_order(sym_pair, close_side, "TAKE_PROFIT_MARKET", ai_tp)
                log.info(f"  [EXEC] TP algo order placed @ {ai_tp} id={resp.get('algoId','?')}")
            except Exception as e:
                log.warning(f"  [EXEC] TP algo order failed ({e}) — software monitor will enforce")

        # --- Step 5: confirm OPEN ---
        db_tp1  = tp1_price if staged else ai_tp
        db_tp2  = tp2_price if staged else ai_tp
        db_confirm_open(pending_id, actual_price, qty, ai_sl, db_tp1, db_tp2, order_id, reason)
        log_trade_open(symbol, signal, actual_price, ai_sl, db_tp1, rr,
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
        log_skip(
            "OPEN_POSITION",
            (
                f"{symbol} has open {pos['side']} position | "
                f"entry={pos['entry_price']} mark={pos['mark_price']} "
                f"uPnL={upnl_pct:+.2f}%"
            ),
        )
        return

    try:
        df_h1  = fetch_ohlcv(client, symbol, "1h",  200)
        df_m15 = fetch_ohlcv(client, symbol, "15m", 100)
        df_m5  = fetch_ohlcv(client, symbol, "5m",  100)
        df_btc_h1  = fetch_ohlcv(client, "BTC", "1h",  100) if symbol != "BTC" else df_h1
        df_btc_m15 = fetch_ohlcv(client, "BTC", "15m", 100) if symbol != "BTC" else df_m15
        df_btc_4h  = fetch_ohlcv(client, "BTC", "4h",  100) if symbol != "BTC" else None
    except Exception as e:
        log_error(f"[{symbol}] OHLCV fetch failed", e)
        return

    ctx = compute_indicators(df_h1, df_m15, df_m5)
    ctx["symbol"] = f"{symbol}/USDT"

    if symbol != "BTC":
        btc_ctx        = compute_indicators(df_btc_h1, df_btc_m15, df_m5)
        btc_trend      = btc_ctx["m15_trend"]
        btc_macro      = _btc_macro_trend(df_btc_4h)
        ctx["btc_trend"]       = btc_trend
        ctx["btc_macro_trend"] = btc_macro
    else:
        ctx["btc_trend"]       = ctx["m15_trend"]
        ctx["btc_macro_trend"] = "NEUTRAL"  # BTC is not filtered against itself

    ctx["sr"] = find_sr_levels(df_h1, ctx["current_price"], df_m15)

    score, score_details = compute_score(ctx)
    ctx["score"]         = score
    ctx["score_details"] = score_details

    log_cycle_start(symbol, ctx["market_mode"], score)
    log.info(f"  H1={ctx['h1_trend']} M15={ctx['m15_trend']} BTC_M15={ctx.get('btc_trend','?')} BTC_4H={ctx.get('btc_macro_trend','?')}")
    log.info(f"  S/R  R={ctx['sr']['resistance']} S={ctx['sr']['support']}")
    log.info(
        "  PrevBox "
        f"state={ctx['sr'].get('breakout_state', 'INSIDE_PREV_BOX')} "
        f"R={ctx['sr'].get('prev_resistance')} S={ctx['sr'].get('prev_support')} "
        f"confirm={ctx['sr'].get('breakout_confirmed', False)} "
        f"bars_since={ctx['sr'].get('bars_since_breakout')}"
    )
    log.info(f"  Score {score}/5: {', '.join(score_details) or 'none'}")
    log.info(f"  RSI={ctx['rsi']:.1f} ATR={ctx['atr_m15']:.4f}")

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
    fear_greed = market_signals.get("fear_greed")
    funding    = market_signals.get(f"{symbol.lower()}_funding", 0)

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
    ctx["llm_exec_tf"]   = "15m"

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
        btc_macro = ctx.get("btc_macro_trend", "NEUTRAL")
        primary   = ctx.get("primary_trend") or ctx["m15_trend"]
        if btc_macro == "BEAR" and primary == "UPTREND":
            log_gate_fail(
                "BTC_MACRO",
                f"BTC 4h strongly BEAR (below EMA50, slope down) — blocking LONG on {symbol}",
                symbol, ctx,
            )
            return
        if btc_macro == "BULL" and primary == "DOWNTREND":
            log_gate_fail(
                "BTC_MACRO",
                f"BTC 4h strongly BULL (above EMA50, slope up) — blocking SHORT on {symbol}",
                symbol, ctx,
            )
            return
        log_gate_pass("BTC_MACRO", f"BTC 4h macro={btc_macro} | {symbol} trend={primary} — no conflict")

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
        rsi = ctx["rsi"]
        if ctx["range_bias"] == "NEAR_SUPPORT" and rsi > 55:
            log_gate_fail("RANGE_POSITION", f"near support but RSI={rsi:.1f} — sellers not exhausted", symbol, ctx)
            return
        if ctx["range_bias"] == "NEAR_RESISTANCE" and rsi < 45:
            log_gate_fail("RANGE_POSITION", f"near resistance but RSI={rsi:.1f} — buyers not exhausted", symbol, ctx)
            return

    log.info("  All gates passed → generating chart...")
    chart_b64 = generate_chart(df_m15, ctx, df_h1=df_h1)
    if not chart_b64:
        log_error(f"[{symbol}] Chart generation failed")
        return

    log_ai_request(symbol, ctx.get("market_mode", "?"))
    decision = ask_gemini(chart_b64, ctx, df_m15)
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

    ai_ok, ai_reason = validate_ai_trade_decision(decision, ctx, df_m15)
    if not ai_ok:
        log_gate_fail("AI_VALIDATION", ai_reason, symbol, ctx)
        log_skip("AI_INVALID", ai_reason, ctx, decision)
        log_cycle_summary(symbol, "WAIT", False, balance, ctx, decision)
        return

    setup_name = decision.get("analysis", {}).get("setup_identified", "")
    # Keep planning on the same 15m execution frame the LLM used for setup detection.
    plan, plan_reason = build_trade_plan(signal, setup_name, ctx, df_m15)
    if not plan:
        log_gate_fail("TRADE_PLAN", plan_reason, symbol, ctx)
        log_skip("TRADE_PLAN", plan_reason, ctx, decision)
        log_cycle_summary(symbol, "WAIT", False, balance, ctx, decision)
        return

    decision["entry_price"] = plan["entry_price"]
    decision["stop_loss"] = plan["stop_loss"]
    decision["take_profit"] = plan["take_profit"]
    decision["take_profit_1"] = plan["take_profit_1"]
    decision["take_profit_2"] = plan["take_profit_2"]
    decision["target_mode"] = plan["target_mode"]
    decision["planned_rr"] = plan["rr"]
    decision.setdefault("analysis", {})
    decision["analysis"]["rr_check"] = (
        f"Python planned Entry={plan['entry_price']} SL={plan['stop_loss']} "
        f"TP={plan['take_profit']} ({plan['target_mode']}, TP1={plan['take_profit_1']}, TP2={plan['take_profit_2']}) "
        f"net R:R={plan['rr']:.2f} gross R:R={plan['gross_rr']:.2f}. Pass."
    )
    log.info(
        f"  [PLAN] entry={plan['entry_price']} SL={plan['stop_loss']} TP={plan['take_profit']} "
        f"({plan['target_mode']}; TP1={plan['take_profit_1']} TP2={plan['take_profit_2']}) "
        f"netR:R={plan['rr']:.2f} grossR:R={plan['gross_rr']:.2f} | {plan_reason}"
    )

    executed = execute_trade(client, symbol, decision, balance, ctx, dry_run=dry_run)
    log_cycle_summary(symbol, signal, executed, balance, ctx, decision)


def run_once(dry_run: bool = False):
    log.info("=" * 60)
    log.info(f"ENGINE LIVE  {'DRY RUN' if dry_run else 'LIVE'}  {datetime.now(UTC).isoformat()}")
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
    log.info(
        "Filter mode: "
        f"TECH_ALLOW_SIDEWAY={os.getenv('TECH_ALLOW_SIDEWAY', '0')} "
        f"TECH_SCORE_THRESHOLD={os.getenv('TECH_SCORE_THRESHOLD', '3')} "
        f"TECH_SCORE_THRESHOLD_RANGE={os.getenv('TECH_SCORE_THRESHOLD_RANGE', os.getenv('TECH_SCORE_THRESHOLD', '3'))} "
        f"SETUP_C_RELAXED={os.getenv('SETUP_C_RELAXED', '0')} "
        f"SL_ATR_MULTIPLIER={SL_ATR_MULTIPLIER:.2f} "
        f"SL_SR_BUFFER_ATR_MULT={SL_SR_BUFFER_ATR_MULT:.2f} "
        f"TRADE_MIN_RR={TAKE_PROFIT_MIN_RR:.2f} "
        f"TRADE_MIN_RR_RANGE={RANGE_MIN_RR:.2f} "
        f"TP_EXTENSION_ATR_MULT={TP_EXTENSION_ATR_MULT:.2f} "
        f"SETUP_E_MIN_RR={SETUP_E_MIN_RR:.2f}"
    )
    funding_str = "  ".join(
        f"{s} funding={market_signals.get(f'{s.lower()}_funding', 0):+.4f}%"
        for s in SYMBOLS
    )
    log.info(f"Fear&Greed={market_signals.get('fear_greed','?')}  {funding_str}")

    latest_balance = balance
    for symbol in SYMBOLS:
        try:
            latest_balance = get_account_balance(client)
            run_symbol_cycle(client, symbol, ml_preds, market_signals, latest_balance, dry_run)
        except Exception as e:
            log_error(f"[{symbol}] Unhandled cycle error", e)


def run_loop(dry_run: bool = False):
    log.info("Engine LIVE started — 5-min cycle loop")

    try:
        db_ensure_trades_table()
    except Exception as e:
        log.warning(f"DB table init failed: {e}")

    client_monitor = get_client()
    t = threading.Thread(target=_monitor_loop, args=(client_monitor,), daemon=True)
    t.start()

    from trade_logger import _tg_send
    _tg_send("🟢 *Live engine started* — monitoring BTC / ETH / SOL")

    while True:
        try:
            run_once(dry_run=dry_run)
        except Exception as e:
            log_error("Loop-level error", e)
            _tg_send(f"⚠️ *Live engine error*: {e}")
        log.info(f"  Sleeping {CYCLE_INTERVAL}s until next cycle...")
        time.sleep(CYCLE_INTERVAL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Trading Engine — BTC/ETH/SOL")
    parser.add_argument("--loop",    action="store_true")
    parser.add_argument("--once",    action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not API_KEY or not API_SECRET:
        print("ERROR: BINANCE_LIVE_API_KEY / BINANCE_LIVE_SECRET_KEY not set in .env")
        exit(1)

    if args.loop:
        run_loop(dry_run=args.dry_run)
    else:
        run_once(dry_run=args.dry_run)
