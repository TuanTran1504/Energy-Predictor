"""
backtest.py — Historical replay backtester for the 5-min strategy.

Uses rule-based pattern detection (no LLM) for speed.
Fetches historical M5, M15, H1 data from Binance public API.

Run:
  python backtest.py --symbol BTC --days 30
  python backtest.py --symbol ETH --days 60 --output results/eth_60d.csv

Output:
  - Console summary: win rate, P&L, Sharpe, max drawdown
  - CSV trade log (one row per trade)
  - Log file: logs/backtest.log
"""

import argparse
import csv
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
UTC = timezone.utc
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from strategy_core import (
    compute_indicators, compute_score, find_sr_levels,
    check_technical_gates, get_range_bias,
    detect_candle_pattern, detect_bb_mean_reversion, classify_trend,
)

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_DIR / "backtest.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("backtest")

BINANCE_BASE  = "https://fapi.binance.com"
LEVERAGE      = 5
SL_MAX_PCT    = 0.008    # 0.8%
SL_MIN_PCT    = 0.002    # 0.2%
MIN_RR        = 1.5
POSITION_RISK = 0.01     # 1% of balance per trade
INITIAL_BALANCE = 10_000.0
SCORE_THRESHOLD  = 3


def fetch_binance_klines(symbol: str, interval: str,
                         start_ms: int, end_ms: int) -> pd.DataFrame:
    """
    Fetches all klines between start_ms and end_ms from Binance Futures.
    Handles pagination (Binance returns max 1500 per call).
    """
    cols = ["timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_vol", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore"]
    all_rows = []
    cur_start = start_ms

    while cur_start < end_ms:
        url = (
            f"{BINANCE_BASE}/fapi/v1/klines"
            f"?symbol={symbol}USDT&interval={interval}"
            f"&startTime={cur_start}&endTime={end_ms}&limit=1500"
        )
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        all_rows.extend(data)
        cur_start = int(data[-1][0]) + 1
        if len(data) < 1500:
            break
        time.sleep(0.1)   # Binance rate limit

    if not all_rows:
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(all_rows, columns=cols)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col])
    df["timestamp"] = pd.to_numeric(df["timestamp"])
    return df.reset_index(drop=True)


def load_historical_data(symbol: str, days: int) -> dict[str, pd.DataFrame]:
    """
    Loads H1, M15, M5 data for the given symbol and number of days.
    Returns {"1h": df, "15m": df, "5m": df}
    """
    end_ms   = int(datetime.now(UTC).timestamp() * 1000)
    # Extra lookback for indicator warm-up (EMA89 needs 89+ bars)
    start_ms = int((datetime.now(UTC) - timedelta(days=days + 5)).timestamp() * 1000)

    log.info(f"Fetching {symbol} data for {days} days...")
    result = {}
    for interval in ["1h", "15m", "5m"]:
        log.info(f"  {interval}...", )
        df = fetch_binance_klines(symbol, interval, start_ms, end_ms)
        log.info(f"  {interval}: {len(df)} candles")
        result[interval] = df

    return result


def generate_signal(df_m5: pd.DataFrame, idx: int, context: dict) -> dict | None:
    """
    Rule-based pattern check at candle idx in df_m5.
    Returns signal dict {"signal", "entry", "sl", "tp", "setup"} or None.
    """
    if idx < 20:
        return None

    pattern = detect_candle_pattern(df_m5, idx)
    if pattern["direction"] == "NONE":
        return None

    sig    = pattern["direction"]
    setup  = pattern["pattern"]
    h1     = context["h1_trend"]
    sr     = context.get("sr", {})
    is_range = context["is_range"]
    allowed  = context.get("allowed_direction", "BOTH")
    range_bias = context.get("range_bias", "MIDDLE")

    if allowed == "BUY" and sig != "BUY":
        return None
    if allowed == "SELL" and sig != "SELL":
        return None

    if not is_range:
        if h1 == "UPTREND" and sig != "BUY":
            return None
        if h1 == "DOWNTREND" and sig != "SELL":
            return None

    if is_range:
        if range_bias == "NEAR_SUPPORT" and sig != "BUY":
            return None
        if range_bias == "NEAR_RESISTANCE" and sig != "SELL":
            return None
        if range_bias == "MIDDLE":
            return None

    entry = float(df_m5["close"].iloc[idx])
    atr   = context.get("atr_m15", entry * 0.005)

    if sig == "BUY":
        sl = min(float(df_m5["low"].iloc[idx]), entry) - entry * 0.0025
        sl_dist = entry - sl
        if sl_dist / entry > SL_MAX_PCT:
            sl = entry * (1 - SL_MAX_PCT)
        if sl_dist / entry < SL_MIN_PCT:
            return None
        resistance = sr.get("resistance", entry * 1.02)
        tp = min(resistance - entry * 0.001, entry + sl_dist * MIN_RR)
        if tp <= entry:
            return None
    else:  # SELL
        sl = max(float(df_m5["high"].iloc[idx]), entry) + entry * 0.0025
        sl_dist = sl - entry
        if sl_dist / entry > SL_MAX_PCT:
            sl = entry * (1 + SL_MAX_PCT)
        if sl_dist / entry < SL_MIN_PCT:
            return None
        support = sr.get("support", entry * 0.98)
        tp = max(support + entry * 0.001, entry - sl_dist * MIN_RR)  # TP toward support
        if tp >= entry:
            return None

    reward = abs(tp - entry)
    risk   = abs(sl - entry)
    rr     = reward / risk if risk > 0 else 0
    if rr < MIN_RR:
        return None

    return {
        "signal": sig,
        "entry":  round(entry, 4),
        "sl":     round(sl, 4),
        "tp":     round(tp, 4),
        "rr":     round(rr, 2),
        "setup":  setup,
    }


def simulate_exit(df_m5: pd.DataFrame, signal_idx: int, signal: dict,
                  max_candles: int = 200) -> dict:
    """
    Simulates SL/TP exit on candles after signal_idx.
    Returns {"outcome": "TP"|"SL"|"TIMEOUT", "exit_price", "exit_idx", "pnl_pct"}.
    """
    entry  = signal["entry"]
    sl     = signal["sl"]
    tp     = signal["tp"]
    side   = signal["signal"]

    for i in range(signal_idx + 1, min(signal_idx + max_candles, len(df_m5))):
        row = df_m5.iloc[i]
        lo  = float(row["low"])
        hi  = float(row["high"])

        if side == "BUY":
            if lo <= sl:
                pnl = (sl - entry) / entry * LEVERAGE
                return {"outcome": "SL", "exit_price": sl, "exit_idx": i, "pnl_pct": pnl}
            if hi >= tp:
                pnl = (tp - entry) / entry * LEVERAGE
                return {"outcome": "TP", "exit_price": tp, "exit_idx": i, "pnl_pct": pnl}
        else:  # SELL
            if hi >= sl:
                pnl = (entry - sl) / entry * LEVERAGE
                return {"outcome": "SL", "exit_price": sl, "exit_idx": i, "pnl_pct": pnl}
            if lo <= tp:
                pnl = (entry - tp) / entry * LEVERAGE
                return {"outcome": "TP", "exit_price": tp, "exit_idx": i, "pnl_pct": pnl}

    # Timeout — close at last candle close
    close = float(df_m5["close"].iloc[min(signal_idx + max_candles, len(df_m5) - 1)])
    if side == "BUY":
        pnl = (close - entry) / entry * LEVERAGE
    else:
        pnl = (entry - close) / entry * LEVERAGE
    return {"outcome": "TIMEOUT", "exit_price": close,
            "exit_idx": signal_idx + max_candles, "pnl_pct": pnl}


def run_backtest(symbol: str, days: int, output_csv: str | None = None) -> dict:
    log.info("=" * 60)
    log.info(f"BACKTEST  {symbol}  {days}d  started {datetime.now(UTC).isoformat()}")
    log.info("=" * 60)

    data   = load_historical_data(symbol, days)
    df_h1  = data["1h"]
    df_m15 = data["15m"]
    df_m5  = data["5m"]

    warmup_ts = int((datetime.now(UTC) - timedelta(days=days)).timestamp() * 1000)
    df_m5_bt  = df_m5[df_m5["timestamp"] >= warmup_ts].reset_index(drop=True)
    log.info(f"Backtest window: {len(df_m5_bt)} M5 candles")

    trades       = []
    balance      = INITIAL_BALANCE
    equity_curve = [balance]
    open_trade   = None

    total = len(df_m5_bt)
    step  = max(1, total // 20)

    for i, row in df_m5_bt.iterrows():
        if i % step == 0:
            log.info(f"  Progress {i}/{total} ({i/total*100:.0f}%) | "
                     f"balance={balance:.0f} trades={len(trades)}")

        ts_ms = int(row["timestamp"])

        if open_trade is not None:
            sig    = open_trade["signal"]
            entry  = sig["entry"]
            sl     = sig["sl"]
            tp     = sig["tp"]
            side   = sig["signal"]
            lo, hi = float(row["low"]), float(row["high"])

            exit_hit = None
            if side == "BUY":
                if lo <= sl:
                    exit_hit = {"outcome": "SL", "exit_price": sl}
                elif hi >= tp:
                    exit_hit = {"outcome": "TP", "exit_price": tp}
            else:
                if hi >= sl:
                    exit_hit = {"outcome": "SL", "exit_price": sl}
                elif lo <= tp:
                    exit_hit = {"outcome": "TP", "exit_price": tp}

            if exit_hit:
                ep    = exit_hit["exit_price"]
                pnl_pct = ((ep - entry) / entry * LEVERAGE
                           if side == "BUY"
                           else (entry - ep) / entry * LEVERAGE)
                pnl_usdt = balance * POSITION_RISK * pnl_pct
                balance += pnl_usdt

                held_min = (ts_ms - open_trade["entry_ts"]) / 60_000
                trade_record = {
                    **open_trade["signal"],
                    "open_ts":    open_trade["open_ts"],
                    "mode":       open_trade.get("mode", "?"),
                    "score":      open_trade.get("score", 0),
                    "h1_trend":   open_trade.get("h1_trend", "?"),
                    "outcome":    exit_hit["outcome"],
                    "exit_price": round(ep, 4),
                    "exit_ts":    datetime.fromtimestamp(ts_ms / 1000, UTC).isoformat(),
                    "pnl_pct":    round(pnl_pct * 100, 3),
                    "pnl_usdt":   round(pnl_usdt, 2),
                    "balance":    round(balance, 2),
                    "held_min":   round(held_min, 1),
                }
                trades.append(trade_record)
                equity_curve.append(balance)
                open_trade = None
                log.info(
                    f"  CLOSE {exit_hit['outcome']} | {side} {symbol} | "
                    f"entry={entry} exit={ep} pnl={pnl_pct*100:+.2f}% "
                    f"({pnl_usdt:+.2f} USDT) | bal={balance:.0f}"
                )
                continue   # don't look for new entry on the same candle

        if open_trade is not None:
            continue

        h1_slice  = df_h1[df_h1["timestamp"] <= ts_ms].tail(150)
        m15_slice = df_m15[df_m15["timestamp"] <= ts_ms].tail(100)
        m5_slice  = df_m5[df_m5["timestamp"] <= ts_ms].tail(100)

        if len(h1_slice) < 90 or len(m15_slice) < 50 or len(m5_slice) < 50:
            continue

        ctx = compute_indicators(
            h1_slice.reset_index(drop=True),
            m15_slice.reset_index(drop=True),
            m5_slice.reset_index(drop=True),
        )
        ctx["symbol"] = f"{symbol}/USDT"

        ctx["sr"] = find_sr_levels(
            h1_slice.reset_index(drop=True),
            ctx["current_price"],
            m15_slice.reset_index(drop=True),
        )

        score, _ = compute_score(ctx)
        ctx["score"] = score

        if score < SCORE_THRESHOLD:
            continue

        m5_local_idx = len(m5_slice) - 1

        # Setup E disabled — all modes use technical gates
        tech_ok, _ = check_technical_gates(ctx)
        if not tech_ok:
            continue

        if ctx["is_range"]:
            ctx["range_bias"] = get_range_bias(ctx)

        # No ML in backtest — derive allowed direction from H1 trend
        if ctx["h1_trend"] == "UPTREND":
            ctx["allowed_direction"] = "BUY"
        elif ctx["h1_trend"] == "DOWNTREND":
            ctx["allowed_direction"] = "SELL"
        else:
            ctx["allowed_direction"] = "BOTH"

        signal = generate_signal(
            m5_slice.reset_index(drop=True), m5_local_idx, ctx
        )

        if signal is None:
            continue

        open_trade = {
            "signal":      signal,
            "entry_ts":    ts_ms,
            "open_ts":     datetime.fromtimestamp(ts_ms / 1000, UTC).isoformat(),
            "mode":        ctx["market_mode"],
            "score":       score,
            "h1_trend":    ctx["h1_trend"],
        }
        log.info(
            f"  OPEN {signal['signal']} {symbol} | setup={signal['setup']} "
            f"entry={signal['entry']} SL={signal['sl']} TP={signal['tp']} "
            f"R:R={signal['rr']} | mode={ctx['market_mode']} score={score}"
        )

    return _compute_results(trades, equity_curve, symbol, days, output_csv)


def _compute_results(trades: list, equity_curve: list,
                     symbol: str, days: int, output_csv: str | None) -> dict:
    if not trades:
        log.info("No trades generated.")
        return {"trades": 0}

    df = pd.DataFrame(trades)
    wins   = df[df["pnl_usdt"] > 0]
    losses = df[df["pnl_usdt"] <= 0]

    win_rate     = len(wins) / len(df) * 100
    total_pnl    = df["pnl_usdt"].sum()
    avg_win      = wins["pnl_usdt"].mean()    if len(wins) > 0 else 0
    avg_loss     = losses["pnl_usdt"].mean()  if len(losses) > 0 else 0
    profit_factor = abs(wins["pnl_usdt"].sum() / losses["pnl_usdt"].sum()) if len(losses) > 0 else float("inf")

    eq = np.array(equity_curve)
    peak = np.maximum.accumulate(eq)
    dd   = (eq - peak) / peak * 100
    max_dd = float(dd.min())

    daily_pnl = df.groupby(df["open_ts"].str[:10])["pnl_usdt"].sum()
    sharpe    = (daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)
                 if daily_pnl.std() > 0 else 0)

    results = {
        "symbol":         symbol,
        "days":           days,
        "total_trades":   len(df),
        "wins":           len(wins),
        "losses":         len(losses),
        "win_rate_pct":   round(win_rate, 1),
        "total_pnl_usdt": round(total_pnl, 2),
        "avg_win_usdt":   round(avg_win, 2),
        "avg_loss_usdt":  round(avg_loss, 2),
        "profit_factor":  round(profit_factor, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe_ratio":   round(sharpe, 2),
        "final_balance":  round(equity_curve[-1], 2),
    }

    log.info("\n" + "=" * 60)
    log.info(f"BACKTEST RESULTS — {symbol} {days}d")
    log.info("=" * 60)
    for k, v in results.items():
        log.info(f"  {k:<25} {v}")
    log.info("=" * 60)

    # Setup breakdown
    if "setup" in df.columns:
        log.info("\nSetup breakdown:")
        for setup, grp in df.groupby("setup"):
            w = (grp["pnl_usdt"] > 0).sum()
            log.info(f"  {setup:<25} {len(grp)} trades  {w}/{len(grp)} wins  "
                     f"pnl={grp['pnl_usdt'].sum():.2f}")

    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        log.info(f"\nTrade log saved to {output_csv}")

    default_csv = LOG_DIR / f"backtest_{symbol}_{days}d_{datetime.now(UTC).strftime('%Y%m%d_%H%M')}.csv"
    df.to_csv(default_csv, index=False)
    log.info(f"Trade log also saved to {default_csv}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest the 5-min strategy")
    parser.add_argument("--symbol", default="BTC", choices=["BTC", "ETH"],
                        help="Symbol to backtest")
    parser.add_argument("--days",   type=int, default=30,
                        help="Number of days of history to test")
    parser.add_argument("--output", default=None,
                        help="Path to save trade CSV (optional)")
    args = parser.parse_args()

    run_backtest(args.symbol, args.days, args.output)
