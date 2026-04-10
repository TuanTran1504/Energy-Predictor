"""
engine_scalp.py — Pure technical scalping engine (no LLM).

Strategy: EMA Cross Momentum Scalp
  1. M5 trend filter  — EMA20/EMA50 direction
  2. M1 signal        — EMA9 crosses EMA21, confirmed by RSI + volume
  3. Execution        — ATR-anchored SL/TP, 1-min cycle, 5-min cooldown

Run modes:
  python engine_scalp.py --loop      # continuous 1-min cycle
  python engine_scalp.py --once      # single cycle and exit
  python engine_scalp.py --dry-run   # single cycle, no order execution

Logs written to backend/trading/logs/
"""

import argparse
import math
import os
import time
import threading
from datetime import datetime, timezone

UTC = timezone.utc
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from psycopg2 import OperationalError, InterfaceError
from binance.um_futures import UMFutures
from dotenv import load_dotenv

from trade_logger import get_logger, log_error

ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT / ".env")

log = get_logger()

API_KEY      = os.getenv("BINANCE_FUTURES_API_KEY", "")
API_SECRET   = os.getenv("BINANCE_FUTURES_SECRET_KEY", "")
DATABASE_URL = os.getenv("DATABASE_URL", "")

TESTNET_BASE = "https://testnet.binancefuture.com"

TRADES_TABLE = "trades_scalp"
ACCOUNT_TYPE = "scalp"

SYMBOLS        = ["BTC", "ETH", "SOL", "XRP"]
LEVERAGE       = 5
CYCLE_INTERVAL = 60   # seconds

QTY_PRECISION = {"BTC": 3, "ETH": 2, "SOL": 1, "XRP": 0}

POSITION_RISK_PCT     = 0.01
MAX_POSITION_FRACTION = 0.10
MAX_POSITION_FRACTION_BY_SYMBOL = {"BTC": 0.35}

try:
    DEFAULT_MIN_NOTIONAL = max(
        0.0,
        float(os.getenv("BINANCE_TESTNET_MIN_NOTIONAL", os.getenv("BINANCE_MIN_NOTIONAL", "100"))),
    )
except (TypeError, ValueError):
    DEFAULT_MIN_NOTIONAL = 100.0

MONITOR_INTERVAL  = 5    # seconds
COOLDOWN_SECONDS  = 300  # 5 min cooldown per symbol after a close

# ── Strategy parameters ───────────────────────────────────────────────────────
M1_FAST_EMA   = 9
M1_SLOW_EMA   = 21
M1_TREND_EMA  = 50
M5_FAST_EMA   = 20
M5_SLOW_EMA   = 50
RSI_LEN       = 7
ATR_LEN       = 7
VOL_MA_LEN    = 20

SL_BUFFER_ATR_MULT = 0.3   # SL = signal candle extreme ± ATR * 0.3
MIN_RR             = 1.5
MIN_ATR_PCT        = 0.05  # skip if ATR < 0.05% of price (dead market)

# ── In-memory state ───────────────────────────────────────────────────────────
_SYMBOL_MIN_NOTIONAL_CACHE: dict[str, float] = {}
_last_trade_close: dict[str, float] = {}   # symbol -> epoch timestamp


# ─────────────────────────────────────────────────────────────────────────────
# Exchange client
# ─────────────────────────────────────────────────────────────────────────────

def get_client() -> UMFutures:
    return UMFutures(key=API_KEY, secret=API_SECRET, base_url=TESTNET_BASE)


# ─────────────────────────────────────────────────────────────────────────────
# DB helpers  (scoped to this engine — no shared state with engine.py)
# ─────────────────────────────────────────────────────────────────────────────

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
    return any(m in msg for m in (
        "ssl connection has been closed unexpectedly",
        "server closed the connection unexpectedly",
        "connection not open",
        "terminating connection due to administrator command",
        "could not receive data from server",
    ))


def db_ensure_trades_table():
    """Create trades_scalp table if missing and add any new columns (idempotent)."""
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
                    account_type      TEXT        DEFAULT 'scalp',
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
                ("account_type",     "TEXT DEFAULT 'scalp'"),
            ]:
                cur.execute(f"""
                    ALTER TABLE {TRADES_TABLE} ADD COLUMN IF NOT EXISTS {col} {definition}
                """)
            cur.execute(f"""
                CREATE UNIQUE INDEX IF NOT EXISTS {TRADES_TABLE}_one_open_per_symbol_idx
                ON {TRADES_TABLE} (account_type, symbol)
                WHERE status = 'OPEN'
            """)
        conn.commit()
    finally:
        conn.close()


def db_create_pending(symbol: str, side: str, setup: str) -> int:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(f"""
                INSERT INTO {TRADES_TABLE}
                  (symbol, side, status, entry_price, quantity, leverage,
                   stop_loss, take_profit, confidence, setup, horizon, account_type)
                VALUES (%s,%s,'PENDING',0,0,%s,0,0,0,%s,1,%s)
                RETURNING id
            """, (symbol, side, LEVERAGE, setup, ACCOUNT_TYPE))
            tid = cur.fetchone()[0]
        conn.commit()
    finally:
        conn.close()
    return tid


def db_confirm_open(trade_id: int, entry: float, qty: float,
                    sl: float, tp: float, order_id: str, notes: str):
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(f"""
                UPDATE {TRADES_TABLE} SET
                  status='OPEN', entry_price=%s, quantity=%s,
                  stop_loss=%s, take_profit=%s,
                  binance_order_id=%s, notes=%s
                WHERE id=%s
            """, (entry, qty, sl, tp, order_id, notes[:200], trade_id))
        conn.commit()
    finally:
        conn.close()


def db_cancel_pending(trade_id: int):
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {TRADES_TABLE} WHERE id=%s AND status='PENDING'",
                (trade_id,),
            )
        conn.commit()
    finally:
        conn.close()


def db_close_trade(trade_id: int, exit_price: float, reason: str):
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT entry_price, quantity, side, symbol FROM {TRADES_TABLE} WHERE id=%s",
                (trade_id,),
            )
            row = cur.fetchone()
            if not row:
                return
            entry, qty, side, symbol = row
            pnl_pct = (
                (exit_price - entry) / entry if side == "BUY"
                else (entry - exit_price) / entry
            ) * LEVERAGE
            margin   = qty * entry / LEVERAGE
            pnl_usdt = pnl_pct * margin
            cur.execute(f"""
                UPDATE {TRADES_TABLE} SET
                  status='CLOSED', exit_price=%s, pnl_usdt=%s,
                  pnl_pct=%s, close_reason=%s, closed_at=NOW()
                WHERE id=%s
            """, (exit_price, round(pnl_usdt, 4), round(pnl_pct * 100, 4), reason, trade_id))
        conn.commit()
        _last_trade_close[symbol] = time.time()
        log.info(f"[DB] Trade {trade_id} closed — {reason} @ {exit_price}  pnl={pnl_pct*100:.2f}%")
    finally:
        conn.close()


def db_get_open_trades() -> list[dict]:
    retries = 2
    for attempt in range(retries + 1):
        conn = None
        try:
            conn = _get_conn()
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT id, symbol, side, entry_price, quantity,
                           stop_loss, take_profit, binance_order_id, opened_at
                    FROM {TRADES_TABLE} WHERE status='OPEN' AND account_type=%s
                """, (ACCOUNT_TYPE,))
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, r)) for r in cur.fetchall()]
        except (OperationalError, InterfaceError) as e:
            if attempt < retries and _is_transient_db_error(e):
                wait_s = 0.25 * (attempt + 1)
                log.warning(f"[DB] get_open_trades transient error, retry {attempt+1}/{retries} in {wait_s:.2f}s")
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
            for tid, sym in (cur.fetchall() or []):
                log.warning(f"[DB] Cleaned stale PENDING id={tid} {sym}")
        conn.commit()
    except Exception as e:
        log.warning(f"[DB] cleanup_stale_pending failed: {e}")
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Exchange helpers
# ─────────────────────────────────────────────────────────────────────────────

def fetch_ohlcv(client: UMFutures, symbol: str, interval: str,
                limit: int = 200) -> pd.DataFrame:
    """Fetch klines and drop the last (incomplete) candle."""
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


def get_account_balance(client: UMFutures) -> float:
    try:
        for asset in client.account().get("assets", []):
            if asset["asset"] == "USDT":
                return round(
                    float(asset["walletBalance"]) + float(asset.get("unrealizedProfit", 0)), 2
                )
    except Exception as e:
        log.warning(f"Balance fetch failed: {e}")
    return 0.0


def get_open_position(client: UMFutures, symbol: str) -> dict | None:
    try:
        for pos in client.get_position_risk(symbol=f"{symbol}USDT"):
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
    min_notional = DEFAULT_MIN_NOTIONAL
    try:
        for sym_info in client.exchange_info().get("symbols", []):
            if sym_info.get("symbol") != f"{symbol}USDT":
                continue
            candidates = []
            for filt in sym_info.get("filters", []):
                if filt.get("filterType") not in {"MIN_NOTIONAL", "NOTIONAL"}:
                    continue
                for key in ("notional", "minNotional"):
                    try:
                        v = float(filt.get(key, 0))
                        if v > 0:
                            candidates.append(v)
                    except (TypeError, ValueError):
                        pass
            if candidates:
                min_notional = max(candidates)
            break
    except Exception as e:
        log.warning(f"[{symbol}] min notional lookup failed: {e}")
    _SYMBOL_MIN_NOTIONAL_CACHE[symbol] = min_notional
    return min_notional


def calc_quantity(balance: float, entry: float, sl: float,
                  symbol: str, min_notional: float) -> float:
    sl_dist = abs(entry - sl)
    if sl_dist == 0:
        return 0.0
    risk_usdt = balance * POSITION_RISK_PCT
    max_frac  = MAX_POSITION_FRACTION_BY_SYMBOL.get(symbol, MAX_POSITION_FRACTION)
    pos_value = min((risk_usdt / sl_dist) * entry, balance * max_frac * LEVERAGE)
    if pos_value < min_notional:
        return 0.0
    qty    = pos_value / entry
    factor = 10 ** QTY_PRECISION.get(symbol, 2)
    qty    = math.floor(qty * factor) / factor
    if qty <= 0 or qty * entry < min_notional:
        return 0.0
    return qty


# ─────────────────────────────────────────────────────────────────────────────
# Scalp indicators
# ─────────────────────────────────────────────────────────────────────────────

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(span=period, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"]  - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Signal logic
# ─────────────────────────────────────────────────────────────────────────────

def compute_scalp_signal(df_m1: pd.DataFrame, df_m5: pd.DataFrame) -> dict:
    """
    EMA cross momentum scalp.

    Trend filter  : M5 EMA20 vs EMA50
    Entry trigger : M1 EMA9 crosses EMA21
    Confirmation  : RSI(7) in momentum zone, volume above MA(20)
    Dead market   : skip if ATR < MIN_ATR_PCT of price

    Returns dict with keys: signal, entry, sl, tp, rr, atr, reason
    """
    if len(df_m1) < M1_TREND_EMA + 5 or len(df_m5) < M5_SLOW_EMA + 5:
        return {"signal": "WAIT", "reason": "insufficient data"}

    # ── M5 trend ─────────────────────────────────────────────────────────────
    m5_close   = df_m5["close"].astype(float)
    m5_ema_fast = float(_ema(m5_close, M5_FAST_EMA).iloc[-1])
    m5_ema_slow = float(_ema(m5_close, M5_SLOW_EMA).iloc[-1])
    m5_trend   = "UP" if m5_ema_fast > m5_ema_slow else "DOWN"

    # ── M1 indicators ────────────────────────────────────────────────────────
    closes   = df_m1["close"].astype(float)
    highs    = df_m1["high"].astype(float)
    lows     = df_m1["low"].astype(float)
    volumes  = df_m1["volume"].astype(float)

    fast_ema  = _ema(closes, M1_FAST_EMA)
    slow_ema  = _ema(closes, M1_SLOW_EMA)
    trend_ema = _ema(closes, M1_TREND_EMA)
    rsi_s     = _rsi(closes, RSI_LEN)
    atr_s     = _atr(df_m1, ATR_LEN)
    vol_ma_s  = volumes.rolling(VOL_MA_LEN).mean()

    fast_now  = float(fast_ema.iloc[-1])
    fast_prev = float(fast_ema.iloc[-2])
    slow_now  = float(slow_ema.iloc[-1])
    slow_prev = float(slow_ema.iloc[-2])
    trend_now = float(trend_ema.iloc[-1])
    rsi_now   = float(rsi_s.iloc[-1])
    atr_now   = float(atr_s.iloc[-1])
    vol_now   = float(volumes.iloc[-1])
    vol_ma_now = float(vol_ma_s.iloc[-1]) if not np.isnan(vol_ma_s.iloc[-1]) else 0.0
    current   = float(closes.iloc[-1])
    sig_low   = float(lows.iloc[-1])
    sig_high  = float(highs.iloc[-1])

    # ── Dead market filter ───────────────────────────────────────────────────
    atr_pct = atr_now / current * 100 if current > 0 else 0.0
    if atr_pct < MIN_ATR_PCT:
        return {"signal": "WAIT", "reason": f"dead market ATR%={atr_pct:.3f}"}

    # ── Volume confirmation ──────────────────────────────────────────────────
    vol_ok = vol_now > vol_ma_now if vol_ma_now > 0 else False

    # ── Cross detection ──────────────────────────────────────────────────────
    bull_cross = fast_prev <= slow_prev and fast_now > slow_now
    bear_cross = fast_prev >= slow_prev and fast_now < slow_now

    # ── BUY ──────────────────────────────────────────────────────────────────
    if (
        bull_cross
        and m5_trend == "UP"
        and current > trend_now          # price above M1 trend EMA
        and 50 < rsi_now < 78            # bullish momentum, not overbought
        and vol_ok
    ):
        sl   = sig_low - atr_now * SL_BUFFER_ATR_MULT
        risk = current - sl
        if risk <= 0:
            return {"signal": "WAIT", "reason": "BUY SL above entry"}
        tp = current + risk * MIN_RR
        rr = (tp - current) / risk
        return {
            "signal": "BUY",
            "entry":  current,
            "sl":     round(sl, 6),
            "tp":     round(tp, 6),
            "rr":     round(rr, 2),
            "atr":    atr_now,
            "reason": (
                f"EMA{M1_FAST_EMA}x{M1_SLOW_EMA} bull cross | "
                f"M5 UP | RSI={rsi_now:.1f} | ATR={atr_now:.4f}"
            ),
        }

    # ── SELL ─────────────────────────────────────────────────────────────────
    if (
        bear_cross
        and m5_trend == "DOWN"
        and current < trend_now          # price below M1 trend EMA
        and 22 < rsi_now < 50            # bearish momentum, not oversold
        and vol_ok
    ):
        sl   = sig_high + atr_now * SL_BUFFER_ATR_MULT
        risk = sl - current
        if risk <= 0:
            return {"signal": "WAIT", "reason": "SELL SL below entry"}
        tp = current - risk * MIN_RR
        rr = (current - tp) / risk
        return {
            "signal": "SELL",
            "entry":  current,
            "sl":     round(sl, 6),
            "tp":     round(tp, 6),
            "rr":     round(rr, 2),
            "atr":    atr_now,
            "reason": (
                f"EMA{M1_FAST_EMA}x{M1_SLOW_EMA} bear cross | "
                f"M5 DOWN | RSI={rsi_now:.1f} | ATR={atr_now:.4f}"
            ),
        }

    # ── WAIT — explain why ───────────────────────────────────────────────────
    if not bull_cross and not bear_cross:
        reason = (
            f"no cross | fast={fast_now:.4f} slow={slow_now:.4f} | "
            f"M5={m5_trend} RSI={rsi_now:.1f}"
        )
    elif not vol_ok:
        reason = f"low volume {vol_now:.0f} < MA {vol_ma_now:.0f}"
    elif bull_cross and m5_trend != "UP":
        reason = f"bull cross but M5 trend={m5_trend}"
    elif bear_cross and m5_trend != "DOWN":
        reason = f"bear cross but M5 trend={m5_trend}"
    elif rsi_now >= 78 or rsi_now <= 22:
        reason = f"RSI extreme={rsi_now:.1f}"
    else:
        reason = f"conditions not met | M5={m5_trend} RSI={rsi_now:.1f}"

    return {"signal": "WAIT", "reason": reason}


# ─────────────────────────────────────────────────────────────────────────────
# Trade execution
# ─────────────────────────────────────────────────────────────────────────────

def execute_scalp_trade(client: UMFutures, symbol: str, sig: dict,
                        balance: float, dry_run: bool = False) -> bool:
    signal = sig["signal"]
    entry  = sig["entry"]
    sl     = sig["sl"]
    tp     = sig["tp"]
    rr     = sig["rr"]
    reason = sig["reason"]

    if signal not in ("BUY", "SELL"):
        return False

    # Direction sanity
    if signal == "BUY" and not (tp > entry > sl):
        log.warning(f"  [GATE] BUY direction fail: TP({tp}) > entry({entry}) > SL({sl})")
        return False
    if signal == "SELL" and not (sl > entry > tp):
        log.warning(f"  [GATE] SELL direction fail: SL({sl}) > entry({entry}) > TP({tp})")
        return False

    if rr < MIN_RR:
        log.warning(f"  [GATE] R:R {rr:.2f} < {MIN_RR}")
        return False

    min_notional = get_symbol_min_notional(client, symbol)
    qty = calc_quantity(balance, entry, sl, symbol, min_notional)
    if qty <= 0:
        log.warning(f"  [GATE] qty=0 or below min notional {min_notional}")
        return False

    log.info(
        f"  [EXEC] {signal} {symbol} | entry≈{entry} SL={sl} TP={tp} "
        f"R:R={rr:.2f} qty={qty} dry={dry_run}"
    )

    if dry_run:
        log.info(f"  [DRY]  {reason}")
        return True

    sym_pair   = f"{symbol}USDT"
    pending_id = None
    try:
        pending_id = db_create_pending(symbol, signal, "scalp_ema_cross")
        client.change_leverage(symbol=sym_pair, leverage=LEVERAGE)

        order        = client.new_order(symbol=sym_pair, side=signal, type="MARKET", quantity=qty)
        order_id     = str(order.get("orderId", ""))
        actual_price = float(order.get("avgPrice", entry)) or entry

        db_confirm_open(pending_id, actual_price, qty, sl, tp, order_id, reason)
        log.info(f"  [EXEC] Filled @ {actual_price}  orderId={order_id}")
        return True

    except Exception as e:
        log_error(f"[{symbol}] Order failed", e)
        if pending_id:
            db_cancel_pending(pending_id)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Monitor loop  (software SL/TP enforcement — same pattern as engine.py)
# ─────────────────────────────────────────────────────────────────────────────

def _monitor_loop(client: UMFutures):
    log.info("[MONITOR] Scalp monitor started (5s interval)")
    while True:
        try:
            db_cleanup_stale_pending()
            open_trades = db_get_open_trades()

            # Phase 1: enforce SL/TP by mark price
            for trade in list(open_trades):
                sl = trade.get("stop_loss") or 0
                tp = trade.get("take_profit") or 0
                if not sl and not tp:
                    continue
                sym = trade["symbol"]
                try:
                    mark = float(client.ticker_price(symbol=f"{sym}USDT")["price"])
                except Exception:
                    continue

                hit = None
                if trade["side"] == "BUY":
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
                    close_side = "SELL" if trade["side"] == "BUY" else "BUY"
                    try:
                        client.new_order(
                            symbol=f"{sym}USDT", side=close_side,
                            type="MARKET", quantity=trade["quantity"],
                        )
                        db_close_trade(trade["id"], mark, hit)
                        open_trades = [t for t in open_trades if t["id"] != trade["id"]]
                        log.info(f"[MONITOR] {sym} {hit} fired @ {mark}")
                    except Exception as e:
                        log.warning(f"[MONITOR] Close failed for {sym}: {e}")

            # Phase 2: reconcile DB with Binance (catch native closes)
            for trade in open_trades:
                sym = trade["symbol"]
                pos = get_open_position(client, sym)
                if pos is not None:
                    continue
                try:
                    exit_price = float(client.ticker_price(symbol=f"{sym}USDT")["price"])
                except Exception:
                    continue
                sl_t, tp_t, side = trade["stop_loss"], trade["take_profit"], trade["side"]
                if side == "BUY":
                    reason = ("take_profit" if exit_price >= tp_t
                              else "stop_loss" if exit_price <= sl_t
                              else "native_close")
                else:
                    reason = ("take_profit" if exit_price <= tp_t
                              else "stop_loss" if exit_price >= sl_t
                              else "native_close")
                db_close_trade(trade["id"], exit_price, reason)

        except Exception as e:
            log.warning(f"[MONITOR] Error: {e}")

        time.sleep(MONITOR_INTERVAL)


# ─────────────────────────────────────────────────────────────────────────────
# Main cycle
# ─────────────────────────────────────────────────────────────────────────────

def run_symbol_cycle(client: UMFutures, symbol: str,
                     balance: float, dry_run: bool = False):
    log.info(f"  {symbol}  {datetime.now(UTC).strftime('%H:%M:%S UTC')}")

    # Skip if already in a position
    pos = get_open_position(client, symbol)
    if pos:
        upnl_pct = pos["upnl"] / (pos["entry_price"] * pos["amount"]) * 100
        log.info(
            f"  [SKIP] open {pos['side']} @ {pos['entry_price']} "
            f"mark={pos['mark_price']} uPnL={upnl_pct:+.2f}%"
        )
        return

    # Cooldown after recent close
    since_close = time.time() - _last_trade_close.get(symbol, 0)
    if since_close < COOLDOWN_SECONDS:
        log.info(f"  [SKIP] cooldown {int(COOLDOWN_SECONDS - since_close)}s remaining")
        return

    try:
        df_m1 = fetch_ohlcv(client, symbol, "1m", 200)
        df_m5 = fetch_ohlcv(client, symbol, "5m", 100)
    except Exception as e:
        log_error(f"[{symbol}] OHLCV fetch failed", e)
        return

    sig = compute_scalp_signal(df_m1, df_m5)
    log.info(f"  [{sig['signal']}] {sig['reason']}")

    if sig["signal"] == "WAIT":
        return

    execute_scalp_trade(client, symbol, sig, balance, dry_run)


def run_once(dry_run: bool = False):
    log.info("=" * 55)
    log.info(f"SCALP ENGINE  {'DRY' if dry_run else 'LIVE'}  {datetime.now(UTC).isoformat()}")
    log.info("=" * 55)
    client  = get_client()
    balance = get_account_balance(client)
    log.info(f"Balance: {balance:.2f} USDT")
    for symbol in SYMBOLS:
        try:
            run_symbol_cycle(client, symbol, balance, dry_run)
        except Exception as e:
            log_error(f"[{symbol}] Unhandled error", e)


def run_loop(dry_run: bool = False):
    log.info("Scalp engine started — 1-min cycle")
    try:
        db_ensure_trades_table()
    except Exception as e:
        log.warning(f"DB table init failed (will retry): {e}")
    monitor_client = get_client()
    t = threading.Thread(target=_monitor_loop, args=(monitor_client,), daemon=True)
    t.start()

    while True:
        start = time.time()
        try:
            run_once(dry_run=dry_run)
        except Exception as e:
            log_error("Loop error", e)
        sleep_for = max(0.0, CYCLE_INTERVAL - (time.time() - start))
        if sleep_for > 0:
            time.sleep(sleep_for)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scalp Engine — EMA cross momentum")
    parser.add_argument("--loop",    action="store_true", help="Run continuous 1-min loop")
    parser.add_argument("--once",    action="store_true", help="Run single cycle and exit")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run", help="No order execution")
    args = parser.parse_args()

    if args.loop:
        run_loop(dry_run=args.dry_run)
    else:
        run_once(dry_run=args.dry_run)
