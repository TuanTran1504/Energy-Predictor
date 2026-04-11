"""
backtest_scalp.py - Historical replay backtester for engine_scalp.py.

Replays the EMA-cross London/NY-session scalping strategy on Binance Futures public candles.
Uses:
  - M1 entries (EMA9/21 fresh cross, RSI zone, volume filter)
  - M5 trend filter (close vs EMA21)
  - Fixed-% SL (0.4%) and TP (0.8%)
  - BE stop move at +0.4% profit (no partial close)
  - Session: 14:00–17:00 UTC (London/NY overlap)
  - Daily loss limit: 3% halt
  - Max 6 trades per session

Run:
  python backtest_scalp.py --days 365 --symbols BTC,ETH,SOL
"""

import argparse
import logging
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from functools import reduce

import numpy as np
import pandas as pd

from backtest import fetch_binance_klines
from engine_scalp import (
    ACCOUNT_TYPE,
    BE_TRIGGER_PCT,
    BUY_RSI_MAX,
    BUY_RSI_MIN,
    COOLDOWN_SECONDS,
    DAILY_LOSS_LIMIT_PCT,
    DEFAULT_MIN_NOTIONAL,
    LEVERAGE,
    M1_FAST_EMA,
    M1_SLOW_EMA,
    M5_TREND_EMA,
    MAX_POSITION_FRACTION,
    MAX_POSITION_FRACTION_BY_SYMBOL,
    MAX_SESSION_TRADES,
    MIN_RR,
    POSITION_RISK_PCT,
    QTY_PRECISION,
    RSI_LEN,
    SELL_RSI_MAX,
    SELL_RSI_MIN,
    SESSION_END_HOUR_UTC,
    SESSION_START_HOUR_UTC,
    SL_PCT,
    SYMBOLS,
    VOLUME_MIN_RATIO,
    VOL_MA_LEN,
)

UTC = timezone.utc
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_DIR / "backtest_scalp.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("backtest_scalp")

INITIAL_BALANCE = 10_000.0
FEE_RATE_PER_SIDE = 0.0004
SLIPPAGE_RATE_PER_SIDE = 0.0002
LOSS_COOLDOWN_SECONDS = 1200  # 20 min after a stop loss before next entry


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(span=period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _floor_qty(qty: float, symbol: str) -> float:
    factor = 10 ** QTY_PRECISION.get(symbol, 2)
    return math.floor(float(qty) * factor) / factor


def _calc_quantity(balance: float, entry: float, sl: float, symbol: str,
                   min_notional: float) -> float:
    sl_dist = abs(entry - sl)
    if sl_dist <= 0:
        return 0.0
    risk_usdt = balance * POSITION_RISK_PCT
    max_frac = MAX_POSITION_FRACTION_BY_SYMBOL.get(symbol, MAX_POSITION_FRACTION)
    pos_value = min((risk_usdt / sl_dist) * entry, balance * max_frac * LEVERAGE)
    if pos_value < min_notional:
        return 0.0
    qty = pos_value / entry
    factor = 10 ** QTY_PRECISION.get(symbol, 2)
    qty = math.floor(qty * factor) / factor
    if qty <= 0 or qty * entry < min_notional:
        return 0.0
    return qty


def _fetch_symbol_data(symbol: str, days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    end_ms = int(datetime.now(UTC).timestamp() * 1000)
    start_ms = int((datetime.now(UTC) - timedelta(days=days + 3)).timestamp() * 1000)
    log.info(f"Fetching {symbol} scalp history for {days}d...")
    df_m1 = fetch_binance_klines(symbol, "1m", start_ms, end_ms)
    df_m5 = fetch_binance_klines(symbol, "5m", start_ms, end_ms)
    log.info(f"  {symbol} M1={len(df_m1)} M5={len(df_m5)}")
    return df_m1.reset_index(drop=True), df_m5.reset_index(drop=True)


def _is_active_session(ts_ms: int) -> bool:
    hour = datetime.fromtimestamp(ts_ms / 1000, UTC).hour
    if SESSION_START_HOUR_UTC == SESSION_END_HOUR_UTC:
        return True
    if SESSION_START_HOUR_UTC < SESSION_END_HOUR_UTC:
        return SESSION_START_HOUR_UTC <= hour < SESSION_END_HOUR_UTC
    return hour >= SESSION_START_HOUR_UTC or hour < SESSION_END_HOUR_UTC


def _prepare_symbol_frames(symbol: str, days: int) -> pd.DataFrame:
    df_m1, df_m5 = _fetch_symbol_data(symbol, days)

    df_m1["close_time"] = pd.to_numeric(df_m1["close_time"])
    df_m5["close_time"] = pd.to_numeric(df_m5["close_time"])

    m1 = df_m1[["timestamp", "close_time", "open", "high", "low", "close", "volume"]].copy()
    m5 = df_m5[["timestamp", "close_time", "close"]].copy()

    m1["ema_fast"] = _ema(m1["close"], M1_FAST_EMA)
    m1["ema_slow"] = _ema(m1["close"], M1_SLOW_EMA)
    m1["ema_fast_prev"] = m1["ema_fast"].shift(1)
    m1["ema_slow_prev"] = m1["ema_slow"].shift(1)
    m1["rsi"] = _rsi(m1["close"], RSI_LEN)
    m1["vol_ma"] = m1["volume"].rolling(VOL_MA_LEN).mean()

    m5["m5_ema21"] = _ema(m5["close"], M5_TREND_EMA)

    merged = pd.merge_asof(
        m1.sort_values("close_time"),
        m5[["close_time", "m5_ema21"]].sort_values("close_time"),
        on="close_time",
        direction="backward",
    )

    merged["symbol"] = symbol
    return merged.reset_index(drop=True)


def _compute_signal_row(row: pd.Series) -> dict | None:
    ts_ms = int(row["close_time"])
    if not _is_active_session(ts_ms):
        return None

    current = float(row["close"])
    if not np.isfinite(current) or current <= 0:
        return None

    fast_now  = float(row["ema_fast"])
    fast_prev = float(row["ema_fast_prev"]) if np.isfinite(row["ema_fast_prev"]) else np.nan
    slow_now  = float(row["ema_slow"])
    slow_prev = float(row["ema_slow_prev"]) if np.isfinite(row["ema_slow_prev"]) else np.nan

    if not (np.isfinite(fast_prev) and np.isfinite(slow_prev)):
        return None

    rsi_now = float(row["rsi"])
    if not np.isfinite(rsi_now):
        return None

    vol_ma_now = float(row["vol_ma"]) if np.isfinite(row["vol_ma"]) else 0.0
    vol_now = float(row["volume"])
    vol_ok = (vol_now / vol_ma_now) >= VOLUME_MIN_RATIO if vol_ma_now > 0 else False

    m5_ema21 = row["m5_ema21"]
    if not np.isfinite(m5_ema21):
        return None
    m5_trend = "UP" if current > float(m5_ema21) else "DOWN"

    bull_cross = fast_prev <= slow_prev and fast_now > slow_now
    bear_cross = fast_prev >= slow_prev and fast_now < slow_now

    if (
        m5_trend == "UP"
        and bull_cross
        and BUY_RSI_MIN <= rsi_now <= BUY_RSI_MAX
        and vol_ok
    ):
        sl = current * (1 - SL_PCT)
        tp = current + (current - sl) * MIN_RR
        return {
            "signal": "BUY",
            "setup": "scalp_ema_cross",
            "entry": current,
            "sl": round(sl, 6),
            "tp": round(tp, 6),
        }

    if (
        m5_trend == "DOWN"
        and bear_cross
        and SELL_RSI_MIN <= rsi_now <= SELL_RSI_MAX
        and vol_ok
    ):
        sl = current * (1 + SL_PCT)
        tp = current - (sl - current) * MIN_RR
        return {
            "signal": "SELL",
            "setup": "scalp_ema_cross",
            "entry": current,
            "sl": round(sl, 6),
            "tp": round(tp, 6),
        }

    return None


def _trade_pnl(side: str, entry: float, exit_price: float, qty: float) -> float:
    if side == "BUY":
        return qty * (exit_price - entry)
    return qty * (entry - exit_price)


def _fill_cost(notional: float, fee_rate: float, slip_rate: float) -> float:
    return notional * (fee_rate + slip_rate)


def _close_trade(trade: dict, exit_price: float, ts_ms: int, reason: str,
                 balance: float, fee_rate: float, slip_rate: float) -> tuple[float, dict]:
    qty = float(trade["qty"])
    pnl = _trade_pnl(trade["side"], trade["entry"], exit_price, qty)
    exit_cost = _fill_cost(qty * exit_price, fee_rate, slip_rate)
    realized = pnl - exit_cost
    trade["realized_pnl"] += realized
    balance += realized
    record = {
        "symbol": trade["symbol"],
        "side": trade["side"],
        "setup": trade["setup"],
        "entry_ts": trade["entry_ts_iso"],
        "exit_ts": datetime.fromtimestamp(ts_ms / 1000, UTC).isoformat(),
        "entry_price": round(trade["entry"], 6),
        "exit_price": round(exit_price, 6),
        "quantity": round(trade["initial_qty"], 8),
        "outcome": reason,
        "be_moved": bool(trade["be_moved"]),
        "gross_pnl_usdt": round(trade["realized_pnl"] + trade["entry_cost"] + trade["exit_cost_accum"] + exit_cost, 4),
        "net_pnl_usdt": round(trade["realized_pnl"], 4),
        "held_min": round((ts_ms - trade["entry_ts"]) / 60_000, 1),
    }
    return balance, record


def run_backtest_scalp(days: int, symbols: list[str], initial_balance: float,
                       fee_rate: float, slip_rate: float,
                       output_prefix: str | None = None) -> dict:
    log.info("=" * 60)
    log.info(
        f"BACKTEST-SCALP {','.join(symbols)} {days}d "
        f"started {datetime.now(UTC).isoformat()} balance={initial_balance:.2f}"
    )
    log.info("=" * 60)

    data: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        data[symbol] = _prepare_symbol_frames(symbol, days)

    timelines = [df["timestamp"] for df in data.values() if not df.empty]
    if not timelines:
        return {"trades": 0}
    timeline = (
        sorted(reduce(lambda left, right: left.union(right), (pd.Index(x) for x in timelines)).tolist())
        if len(timelines) > 1
        else timelines[0].tolist()
    )
    index_maps = {sym: {int(ts): i for i, ts in enumerate(df["timestamp"].tolist())} for sym, df in data.items()}

    balance = float(initial_balance)
    equity_curve = [balance]
    last_close_ts = {sym: 0 for sym in symbols}
    last_close_reason: dict[str, str] = {sym: "" for sym in symbols}
    open_trades: dict[str, dict] = {}
    trade_records: list[dict] = []
    per_symbol_results = {sym: {"trades": 0, "pnl": 0.0} for sym in symbols}

    # Daily loss limit: (date_str, start_balance)
    daily_start_balance: dict[str, tuple[str, float]] = {}
    # Session trade count: (date_str, count)
    session_trade_count: dict[str, tuple[str, int]] = {}

    total = len(timeline)
    step = max(1, total // 20)

    for idx_t, ts_ms in enumerate(timeline):
        if idx_t % step == 0:
            log.info(f"  Progress {idx_t}/{total} ({idx_t/total*100:.0f}%) balance={balance:.2f} open={len(open_trades)}")

        closed_this_ts: set[str] = set()
        date_str = datetime.fromtimestamp(ts_ms / 1000, UTC).strftime("%Y-%m-%d")

        # ── Manage open trades ────────────────────────────────────────────────
        for symbol in symbols:
            row_idx = index_maps[symbol].get(int(ts_ms))
            if row_idx is None:
                continue
            row = data[symbol].iloc[row_idx]

            trade = open_trades.get(symbol)
            if trade is None:
                continue

            hi = float(row["high"])
            lo = float(row["low"])

            if trade["side"] == "BUY":
                stop_hit = lo <= trade["sl"]
                tp_hit   = hi >= trade["tp"]
            else:
                stop_hit = hi >= trade["sl"]
                tp_hit   = lo <= trade["tp"]

            if stop_hit:
                exit_price = trade["sl"]
                trade["exit_cost_accum"] += _fill_cost(trade["qty"] * exit_price, fee_rate, slip_rate)
                balance, record = _close_trade(trade, exit_price, int(row["close_time"]), "stop_loss", balance, fee_rate, slip_rate)
                trade_records.append(record)
                per_symbol_results[symbol]["trades"] += 1
                per_symbol_results[symbol]["pnl"] += record["net_pnl_usdt"]
                last_close_ts[symbol] = int(row["close_time"])
                last_close_reason[symbol] = "stop_loss"
                del open_trades[symbol]
                closed_this_ts.add(symbol)
                continue

            if tp_hit:
                exit_price = trade["tp"]
                trade["exit_cost_accum"] += _fill_cost(trade["qty"] * exit_price, fee_rate, slip_rate)
                balance, record = _close_trade(trade, exit_price, int(row["close_time"]), "take_profit", balance, fee_rate, slip_rate)
                trade_records.append(record)
                per_symbol_results[symbol]["trades"] += 1
                per_symbol_results[symbol]["pnl"] += record["net_pnl_usdt"]
                last_close_ts[symbol] = int(row["close_time"])
                last_close_reason[symbol] = "take_profit"
                del open_trades[symbol]
                closed_this_ts.add(symbol)
                continue

            # Move SL to breakeven at +BE_TRIGGER_PCT
            if not trade["be_moved"]:
                entry = trade["entry"]
                if trade["side"] == "BUY" and hi >= entry * (1 + BE_TRIGGER_PCT):
                    trade["sl"] = entry
                    trade["be_moved"] = True
                elif trade["side"] == "SELL" and lo <= entry * (1 - BE_TRIGGER_PCT):
                    trade["sl"] = entry
                    trade["be_moved"] = True

        # ── Look for new entries ──────────────────────────────────────────────
        for symbol in symbols:
            if symbol in open_trades or symbol in closed_this_ts:
                continue
            row_idx = index_maps[symbol].get(int(ts_ms))
            if row_idx is None or row_idx < 200:
                continue
            row = data[symbol].iloc[row_idx]

            # Daily loss limit
            prev_date, prev_start = daily_start_balance.get(symbol, (None, None))
            if prev_date != date_str:
                daily_start_balance[symbol] = (date_str, balance)
                prev_start = balance
            if balance <= prev_start * (1 - DAILY_LOSS_LIMIT_PCT):
                continue

            # Session trade count
            prev_sess_date, sess_count = session_trade_count.get(symbol, (None, 0))
            if prev_sess_date != date_str:
                session_trade_count[symbol] = (date_str, 0)
                sess_count = 0
            if sess_count >= MAX_SESSION_TRADES:
                continue

            # Cooldown
            effective_cooldown = (
                LOSS_COOLDOWN_SECONDS
                if last_close_reason.get(symbol) == "stop_loss"
                else COOLDOWN_SECONDS
            )
            if int(row["close_time"]) - last_close_ts[symbol] < effective_cooldown * 1000:
                continue

            sig = _compute_signal_row(row)
            if sig is None:
                continue

            qty = _calc_quantity(balance, float(sig["entry"]), float(sig["sl"]), symbol, DEFAULT_MIN_NOTIONAL)
            if qty <= 0:
                continue

            entry_cost = _fill_cost(qty * float(sig["entry"]), fee_rate, slip_rate)
            balance -= entry_cost
            session_trade_count[symbol] = (date_str, sess_count + 1)

            open_trades[symbol] = {
                "symbol": symbol,
                "side": sig["signal"],
                "setup": sig["setup"],
                "entry": float(sig["entry"]),
                "sl": float(sig["sl"]),
                "tp": float(sig["tp"]),
                "qty": float(qty),
                "initial_qty": float(qty),
                "entry_ts": int(row["close_time"]),
                "entry_ts_iso": datetime.fromtimestamp(int(row["close_time"]) / 1000, UTC).isoformat(),
                "be_moved": False,
                "realized_pnl": -entry_cost,
                "entry_cost": entry_cost,
                "exit_cost_accum": 0.0,
            }

        equity_curve.append(balance)

    # Close any still-open trades at last candle close
    for symbol, trade in list(open_trades.items()):
        df = data[symbol]
        last_row = df.iloc[-1]
        exit_price = float(last_row["close"])
        trade["exit_cost_accum"] += _fill_cost(trade["qty"] * exit_price, fee_rate, slip_rate)
        balance, record = _close_trade(trade, exit_price, int(last_row["close_time"]), "timeout", balance, fee_rate, slip_rate)
        trade_records.append(record)
        per_symbol_results[symbol]["trades"] += 1
        per_symbol_results[symbol]["pnl"] += record["net_pnl_usdt"]
        equity_curve.append(balance)

    if not trade_records:
        log.info("No trades generated.")
        return {"trades": 0}

    df_trades = pd.DataFrame(trade_records)
    wins = df_trades[df_trades["net_pnl_usdt"] > 0]
    losses = df_trades[df_trades["net_pnl_usdt"] <= 0]
    eq = np.array(equity_curve, dtype=float)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak * 100

    results = {
        "account_type": ACCOUNT_TYPE,
        "symbols": ",".join(symbols),
        "days": days,
        "total_trades": int(len(df_trades)),
        "wins": int(len(wins)),
        "losses": int(len(losses)),
        "win_rate_pct": round(len(wins) / len(df_trades) * 100, 2),
        "total_pnl_usdt": round(float(df_trades["net_pnl_usdt"].sum()), 2),
        "profit_factor": round(
            abs(float(wins["net_pnl_usdt"].sum()) / float(losses["net_pnl_usdt"].sum())),
            2,
        ) if len(losses) > 0 and abs(float(losses["net_pnl_usdt"].sum())) > 1e-9 else float("inf"),
        "max_drawdown_pct": round(float(dd.min()), 2),
        "final_balance": round(float(balance), 2),
    }

    log.info("\n" + "=" * 60)
    log.info(f"SCALP BACKTEST RESULTS {days}d {'/'.join(symbols)}")
    log.info("=" * 60)
    for k, v in results.items():
        log.info(f"  {k:<20} {v}")
    log.info("=" * 60)
    log.info("Per-symbol breakdown:")
    for symbol in symbols:
        grp = df_trades[df_trades["symbol"] == symbol]
        if grp.empty:
            log.info(f"  {symbol:<6} no trades")
            continue
        wr = (grp["net_pnl_usdt"] > 0).mean() * 100
        log.info(
            f"  {symbol:<6} trades={len(grp):<4} pnl={grp['net_pnl_usdt'].sum():>9.2f} "
            f"win_rate={wr:>5.1f}%"
        )

    ts_tag = datetime.now(UTC).strftime("%Y%m%d_%H%M")
    default_csv = LOG_DIR / f"backtest_scalp_{'_'.join(symbols)}_{days}d_{ts_tag}.csv"
    df_trades.to_csv(default_csv, index=False)
    log.info(f"Trade log saved to {default_csv}")

    if output_prefix:
        out_path = Path(output_prefix)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_trades.to_csv(out_path, index=False)
        log.info(f"Trade log also saved to {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Backtest engine_scalp on Binance history")
    parser.add_argument("--days", type=int, default=365, help="Number of days to replay")
    parser.add_argument("--symbols", default="BTC,ETH,SOL", help="Comma-separated symbol list")
    parser.add_argument("--initial-balance", type=float, default=INITIAL_BALANCE, help="Starting USDT balance")
    parser.add_argument("--fee-rate", type=float, default=FEE_RATE_PER_SIDE, help="Fee rate per fill side")
    parser.add_argument("--slippage-rate", type=float, default=SLIPPAGE_RATE_PER_SIDE, help="Slippage rate per fill side")
    parser.add_argument("--output", default=None, help="Optional output CSV path")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        symbols = list(SYMBOLS)
    run_backtest_scalp(args.days, symbols, args.initial_balance, args.fee_rate, args.slippage_rate, args.output)


if __name__ == "__main__":
    main()
