"""
LLM-driven trading engine for BTC/ETH futures on Binance Testnet.

Strategy:
  - Gathers all market context (ML predictions, fear/greed, funding rates, price action)
  - Asks GPT-4o-mini to make a structured trading decision for each symbol
  - Executes the LLM's decision: open long/short, close, hold, or adjust size
  - Monitor loop checks SL/TP every 5 minutes
  - Signal loop runs LLM decision cycle every 24h

Run:
  python engine.py --loop    # full engine (monitor + LLM signals)
  python engine.py --once    # one LLM decision cycle and exit
"""

import os
import json
import logging
import argparse
import threading
import time
from datetime import datetime, UTC
from pathlib import Path

import psycopg2
from dotenv import load_dotenv
from binance.um_futures import UMFutures
from openai import OpenAI

# ── Load env ───────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT / ".env")
load_dotenv(ROOT / "backend" / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
API_KEY      = os.getenv("BINANCE_FUTURES_API_KEY", "")
API_SECRET   = os.getenv("BINANCE_FUTURES_SECRET_KEY", "")
DATABASE_URL = os.getenv("DATABASE_URL", "")
REDIS_URL    = os.getenv("REDIS_URL", "")
OPENAI_KEY   = os.getenv("OPENAI_API_KEY", "")

TESTNET_BASE     = "https://testnet.binancefuture.com"
SYMBOLS          = ["BTC", "ETH"]
HORIZONS         = [1, 7]
LEVERAGE         = 5
STOP_LOSS_PCT    = 0.03    # 3%  — hard floor, LLM can tighten but not widen
TAKE_PROFIT_PCT  = 0.06    # 6%  — hard ceiling, LLM can lower but not raise
MAX_POSITION_PCT = 0.10    # never risk more than 10% of account margin
MONITOR_INTERVAL = 5 * 60   # DB sync every 5 minutes
SIGNAL_INTERVAL  = 4 * 3600  # LLM decision every 4 hours


# ── Binance client ─────────────────────────────────────────────────────────────
def get_client() -> UMFutures:
    return UMFutures(key=API_KEY, secret=API_SECRET, base_url=TESTNET_BASE)


# ── Redis ──────────────────────────────────────────────────────────────────────
def get_all_predictions() -> dict:
    """Returns all cached predictions from Redis: {BTC_1d: {...}, BTC_7d: {...}, ...}"""
    try:
        import redis
        url = REDIS_URL
        if url.startswith("redis://"):
            url = "rediss://" + url[8:]
        r = redis.from_url(url, decode_responses=True, socket_timeout=3)
        result = {}
        for sym in SYMBOLS:
            for h in HORIZONS:
                val = r.get(f"prediction:{sym}:{h}d")
                if val:
                    result[f"{sym}_{h}d"] = json.loads(val)
        return result
    except Exception as e:
        log.warning(f"Redis read failed: {e}")
        return {}


# ── DB ─────────────────────────────────────────────────────────────────────────
def get_conn():
    return psycopg2.connect(DATABASE_URL, sslmode="require")


def get_open_trades() -> list[dict]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, symbol, side, entry_price, quantity, stop_loss,
                       take_profit, confidence, horizon, binance_order_id, opened_at
                FROM trades WHERE status = 'OPEN'
            """)
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]


def save_trade(trade: dict) -> int:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO trades
                  (symbol, side, status, entry_price, quantity, leverage,
                   stop_loss, take_profit, confidence, horizon,
                   binance_order_id, notes)
                VALUES (%s,%s,'OPEN',%s,%s,%s,%s,%s,%s,%s,%s,%s)
                RETURNING id
            """, (
                trade["symbol"], trade["side"],
                trade["entry_price"], trade["quantity"], LEVERAGE,
                trade["stop_loss"], trade["take_profit"],
                trade["confidence"], trade["horizon"],
                trade.get("binance_order_id"), trade.get("notes"),
            ))
            tid = cur.fetchone()[0]
        conn.commit()
    return tid


def close_trade(trade_id: int, exit_price: float, reason: str):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT entry_price, quantity, side FROM trades WHERE id=%s",
                (trade_id,)
            )
            row = cur.fetchone()
            if not row:
                return
            entry_price, quantity, side = row
            if side == "BUY":
                pnl_pct = (exit_price - entry_price) / entry_price * LEVERAGE
            else:
                pnl_pct = (entry_price - exit_price) / entry_price * LEVERAGE
            pnl_usdt = pnl_pct * (quantity * entry_price / LEVERAGE)

            cur.execute("""
                UPDATE trades SET
                    status='CLOSED', exit_price=%s, pnl_usdt=%s,
                    pnl_pct=%s, close_reason=%s, closed_at=NOW()
                WHERE id=%s
            """, (exit_price, round(pnl_usdt, 4), round(pnl_pct * 100, 4), reason, trade_id))
        conn.commit()
    log.info(f"Trade {trade_id} closed — {reason} @ {exit_price} pnl={pnl_pct*100:.2f}%")


def get_recent_prices(symbol: str, days: int = 7) -> list[dict]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DATE(fetched_at) as d, AVG(close_usd) as price
                FROM crypto_prices
                WHERE symbol=%s AND interval_minutes=30
                  AND fetched_at >= NOW() - INTERVAL '%s days'
                GROUP BY d ORDER BY d DESC
            """, (symbol, days))
            return [{"date": str(r[0]), "price": round(r[1], 2)} for r in cur.fetchall()]


def get_trade_history(limit: int = 20) -> list[dict]:
    """Recent closed trades for LLM context — wins, losses, reasons, durations."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT symbol, side, entry_price, exit_price, pnl_usdt, pnl_pct,
                       close_reason, opened_at, closed_at,
                       EXTRACT(EPOCH FROM (closed_at - opened_at)) / 3600 AS duration_hours
                FROM trades
                WHERE status = 'CLOSED' AND pnl_usdt IS NOT NULL
                ORDER BY closed_at DESC LIMIT %s
            """, (limit,))
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]


def get_account_stats() -> dict:
    """Aggregate win rate, total P&L, best/worst trade."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    COUNT(*)                                            AS total_trades,
                    SUM(CASE WHEN pnl_usdt > 0 THEN 1 ELSE 0 END)     AS wins,
                    SUM(CASE WHEN pnl_usdt <= 0 THEN 1 ELSE 0 END)    AS losses,
                    COALESCE(SUM(pnl_usdt), 0)                         AS total_pnl,
                    COALESCE(AVG(pnl_usdt), 0)                         AS avg_pnl,
                    COALESCE(MAX(pnl_usdt), 0)                         AS best_trade,
                    COALESCE(MIN(pnl_usdt), 0)                         AS worst_trade
                FROM trades WHERE status = 'CLOSED' AND pnl_usdt IS NOT NULL
            """)
            row = cur.fetchone()
            cols = [d[0] for d in cur.description]
            return dict(zip(cols, row)) if row else {}


def get_last_llm_decisions(limit: int = 3) -> list[dict]:
    """Last N LLM decisions so the model can see its own recent reasoning."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT decided_at, decision FROM llm_decisions
                ORDER BY decided_at DESC LIMIT %s
            """, (limit,))
            return [{"decided_at": str(r[0]), "decision": r[1]} for r in cur.fetchall()]


def get_latest_market_signals() -> dict:
    signals = {}
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT date, value FROM fear_greed_index ORDER BY date DESC LIMIT 1")
            row = cur.fetchone()
            if row:
                v = row[1]
                label = ("Extreme Fear" if v <= 20 else "Fear" if v <= 40 else
                         "Neutral" if v <= 60 else "Greed" if v <= 80 else "Extreme Greed")
                signals["fear_greed"] = {"value": v, "label": label, "date": str(row[0])}

            for sym in SYMBOLS:
                cur.execute(
                    "SELECT date, rate_avg FROM funding_rates WHERE symbol=%s ORDER BY date DESC LIMIT 1",
                    (sym,)
                )
                row = cur.fetchone()
                if row:
                    signals[f"{sym.lower()}_funding"] = {
                        "rate_avg": row[1],
                        "date": str(row[0]),
                        "pct": f"{row[1]*100:.4f}%",
                    }
    return signals


# ── LLM decision ───────────────────────────────────────────────────────────────
DECISION_SCHEMA = """
You must respond with ONLY a valid JSON object — no markdown, no explanation outside the JSON.

Schema:
{
  "decisions": [
    {
      "symbol": "BTC" or "ETH",
      "action": "open_long" | "open_short" | "close" | "hold",
      "position_size_multiplier": 0.5 | 1.0 | 1.5 | 2.0,
      "stop_loss_pct": float (e.g. 0.03 for 3%),
      "take_profit_pct": float (e.g. 0.06 for 6%),
      "confidence_score": float 0-1 (your confidence in this decision),
      "reasoning": "brief explanation"
    }
  ],
  "market_summary": "one sentence overall market assessment"
}

Rules:
- position_size_multiplier: 0.5=half size, 1.0=normal, 1.5=large, 2.0=max
- stop_loss_pct must be between 0.01 and 0.05 (1%-5%)
- take_profit_pct must be between 0.02 and 0.10 (2%-10%)
- If action is "hold" or "close", position_size_multiplier = 0
- Only output "close" if there is currently an open position to close
- Be conservative — it's better to hold than to trade on weak signals
"""


def get_llm_decision(
    predictions: dict,
    market_signals: dict,
    price_data: dict,
    open_trades: list[dict],
    account_balance: float,
    binance_positions: dict,
    trade_history: list[dict],
    account_stats: dict,
    last_decisions: list[dict],
) -> dict | None:
    """Ask GPT-4o-mini to make a structured trading decision given full context."""

    oai = OpenAI(api_key=OPENAI_KEY)

    lines = ["=== ML MODEL PREDICTIONS ==="]
    for key, pred in predictions.items():
        lines.append(
            f"{key}: {pred['direction']} | UP {pred['up_prob']*100:.1f}% / "
            f"DOWN {pred['down_prob']*100:.1f}% | confidence {pred['confidence']*100:.1f}%"
        )

    lines.append("\n=== MARKET SIGNALS ===")
    if "fear_greed" in market_signals:
        fg = market_signals["fear_greed"]
        lines.append(f"Fear & Greed: {fg['value']:.0f}/100 ({fg['label']}) as of {fg['date']}")
    for sym in SYMBOLS:
        k = f"{sym.lower()}_funding"
        if k in market_signals:
            fr = market_signals[k]
            lines.append(f"{sym} funding rate: {fr['pct']} daily avg as of {fr['date']}")

    lines.append("\n=== RECENT PRICE ACTION (newest first) ===")
    for sym, prices in price_data.items():
        if prices:
            price_str = " → ".join([f"${p['price']:,.0f}" for p in prices[:7]])
            lines.append(f"{sym}: {price_str}")

    # Account with performance stats
    stats = account_stats
    win_rate = (stats["wins"] / stats["total_trades"] * 100) if stats.get("total_trades") else 0
    lines.append(f"\n=== ACCOUNT ===")
    lines.append(f"Balance: {account_balance:.2f} USDT | Leverage: {LEVERAGE}x")
    lines.append(
        f"Total P&L: {stats.get('total_pnl', 0):+.2f} USDT | "
        f"Win rate: {win_rate:.0f}% ({stats.get('wins', 0)}W / {stats.get('losses', 0)}L) | "
        f"Avg trade: {stats.get('avg_pnl', 0):+.2f} USDT | "
        f"Best: {stats.get('best_trade', 0):+.2f} | Worst: {stats.get('worst_trade', 0):+.2f}"
    )

    # Open positions enriched with live Binance mark price and unrealized P&L
    lines.append("\n=== OPEN POSITIONS ===")
    if open_trades:
        for t in open_trades:
            sym = t["symbol"]
            bn  = binance_positions.get(sym)
            now = datetime.now(UTC)
            opened = t["opened_at"]
            if opened.tzinfo is None:
                opened = opened.replace(tzinfo=UTC)
            hours_open = (now - opened).total_seconds() / 3600

            if bn:
                mark  = bn["mark_price"]
                upnl  = bn.get("unrealized_pnl", 0)
                entry = bn["entry_price"]
                sl_dist = (mark - t["stop_loss"])   / mark * 100 if t["side"] == "BUY" else (t["stop_loss"]   - mark) / mark * 100
                tp_dist = (t["take_profit"] - mark) / mark * 100 if t["side"] == "BUY" else (mark - t["take_profit"]) / mark * 100
                lines.append(
                    f"{sym} {'LONG' if t['side'] == 'BUY' else 'SHORT'} | "
                    f"Entry: ${entry:,.2f} | Mark: ${mark:,.2f} | "
                    f"Unrealized P&L: {upnl:+.2f} USDT | "
                    f"SL dist: {sl_dist:.1f}% | TP dist: {tp_dist:.1f}% | "
                    f"Open: {hours_open:.1f}h"
                )
            else:
                lines.append(
                    f"{sym} {'LONG' if t['side'] == 'BUY' else 'SHORT'} | "
                    f"Entry: ${t['entry_price']:,.2f} | SL: ${t['stop_loss']:,.2f} | "
                    f"TP: ${t['take_profit']:,.2f} | Open: {hours_open:.1f}h"
                )
    else:
        lines.append("None")

    # Recent trade history
    lines.append("\n=== RECENT TRADE HISTORY (newest first) ===")
    if trade_history:
        for t in trade_history[:10]:
            dur = f"{t['duration_hours']:.1f}h" if t.get("duration_hours") else "?"
            lines.append(
                f"{t['symbol']} {'LONG' if t['side'] == 'BUY' else 'SHORT'} | "
                f"P&L: {t['pnl_usdt']:+.2f} USDT ({t['pnl_pct']:+.2f}%) | "
                f"held {dur} | closed: {t['close_reason']}"
            )
    else:
        lines.append("No closed trades yet")

    # Last LLM decisions so model can see its own recent reasoning
    lines.append("\n=== YOUR LAST DECISIONS ===")
    if last_decisions:
        for d in last_decisions:
            dec = d["decision"]
            summary = []
            for item in dec.get("decisions", []):
                summary.append(f"{item['symbol']}={item['action']} (conf {item.get('confidence_score', 0):.0%})")
            lines.append(
                f"{d['decided_at'][:16]} | {', '.join(summary)} | "
                f"\"{dec.get('market_summary', '')}\""
            )
    else:
        lines.append("No previous decisions")

    context = "\n".join(lines)

    system_prompt = f"""You are an expert crypto futures trader with deep knowledge of technical analysis,
market microstructure, and risk management. You make disciplined, data-driven trading decisions.

You are managing a TESTNET futures account trading BTC and ETH with {LEVERAGE}x leverage.
Positions are protected by native Binance SL/TP orders — you do not need to micro-manage exits.
Your decisions run every 4 hours. Focus on the bigger picture.

Key guidelines:
- Study your recent trade history to learn what is working and what is not
- If win rate is low, be more selective — tighten entry criteria
- Unrealized P&L shows where open positions stand right now — factor this in
- SL dist / TP dist show how close a position is to its exits
- Your previous decisions give context — avoid flip-flopping without strong reason
- Be conservative: a missed opportunity is better than a bad trade
- Only trade when ML model, sentiment, and price action agree

{DECISION_SCHEMA}"""

    user_prompt = f"""Here is the full current context. Make your trading decisions.

{context}

Based on all of the above, what should we do for BTC and ETH right now?
Remember: respond with ONLY the JSON object."""

    log.info("[LLM] Requesting trading decision from GPT-4o-mini...")
    try:
        response = oai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=1000,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
        decision = json.loads(raw)
        log.info(f"[LLM] Decision: {json.dumps(decision, indent=2)}")
        return decision
    except Exception as e:
        log.error(f"[LLM] Decision failed: {e}")
        return None


# ── Position execution ─────────────────────────────────────────────────────────
def calc_quantity(balance: float, size_multiplier: float, entry_price: float, symbol: str) -> float:
    """Calculate position quantity based on LLM size multiplier."""
    base_risk_pct = 0.02 * size_multiplier
    risk_usdt = balance * min(base_risk_pct, MAX_POSITION_PCT)
    position_value = (risk_usdt / STOP_LOSS_PCT) * LEVERAGE
    position_value = min(position_value, balance * MAX_POSITION_PCT * LEVERAGE)
    qty = position_value / entry_price
    return round(qty, 3) if symbol == "BTC" else round(qty, 2)


def execute_decision(client: UMFutures, symbol: str, decision: dict, balance: float, open_trades: list[dict]):
    action     = decision["action"]
    sl_pct     = max(0.01, min(0.05, decision.get("stop_loss_pct", STOP_LOSS_PCT)))
    tp_pct     = max(0.02, min(0.10, decision.get("take_profit_pct", TAKE_PROFIT_PCT)))
    multiplier = decision.get("position_size_multiplier", 1.0)
    reasoning  = decision.get("reasoning", "")
    confidence = decision.get("confidence_score", 0.5)

    log.info(f"[{symbol}] Action={action} | confidence={confidence:.1%} | reason: {reasoning}")

    sym_pair = f"{symbol}USDT"
    open_trade = next((t for t in open_trades if t["symbol"] == symbol), None)

    if action == "hold":
        log.info(f"[{symbol}] Holding — no action taken")
        return

    if action == "close":
        if not open_trade:
            log.info(f"[{symbol}] No open position to close")
            return
        ticker = client.ticker_price(symbol=sym_pair)
        price  = float(ticker["price"])
        _close_position(client, open_trade, price, "llm_decision")
        return

    # Close opposite position before opening new one
    if open_trade:
        opposite = (open_trade["side"] == "BUY" and action == "open_short") or \
                   (open_trade["side"] == "SELL" and action == "open_long")
        if opposite:
            ticker = client.ticker_price(symbol=sym_pair)
            price  = float(ticker["price"])
            log.info(f"[{symbol}] LLM flip — closing existing {open_trade['side']}")
            _close_position(client, open_trade, price, "llm_signal_flip")
        elif not opposite:
            log.info(f"[{symbol}] Already have {open_trade['side']} position — holding")
            return

    # Open new position
    side = "BUY" if action == "open_long" else "SELL"
    try:
        client.change_leverage(symbol=sym_pair, leverage=LEVERAGE)
        ticker      = client.ticker_price(symbol=sym_pair)
        entry_price = float(ticker["price"])
        quantity    = calc_quantity(balance, multiplier, entry_price, symbol)

        if quantity <= 0:
            log.warning(f"[{symbol}] Quantity too small — skipping")
            return

        if side == "BUY":
            stop_loss   = round(entry_price * (1 - sl_pct), 2)
            take_profit = round(entry_price * (1 + tp_pct), 2)
        else:
            stop_loss   = round(entry_price * (1 + sl_pct), 2)
            take_profit = round(entry_price * (1 - tp_pct), 2)

        log.info(f"[{symbol}] Opening {side} {quantity} @ ~{entry_price} | SL={stop_loss} TP={take_profit}")

        order = client.new_order(
            symbol=sym_pair, side=side, type="MARKET", quantity=quantity,
        )
        order_id    = str(order.get("orderId", ""))
        actual_price = float(order.get("avgPrice", entry_price)) or entry_price

        trade_id = save_trade({
            "symbol":           symbol,
            "side":             side,
            "entry_price":      actual_price,
            "quantity":         quantity,
            "stop_loss":        stop_loss,
            "take_profit":      take_profit,
            "confidence":       confidence,
            "horizon":          1,
            "binance_order_id": order_id,
            "notes":            f"LLM: {reasoning[:200]}",
        })
        log.info(f"[{symbol}] Trade {trade_id} opened — {side} {quantity} @ {actual_price}")

        # Place native SL/TP on Binance — these fire even if our engine goes down
        _place_native_sl_tp(client, symbol, side, stop_loss, take_profit)

    except Exception as e:
        log.error(f"[{symbol}] Failed to execute {action}: {e}")


def _place_native_sl_tp(client: UMFutures, symbol: str, side: str,
                        stop_loss: float, take_profit: float):
    """
    Place native STOP_MARKET and TAKE_PROFIT_MARKET orders on Binance.
    These execute on Binance's servers instantly when price is hit,
    even if our engine is offline. closePosition=True closes the full position.
    """
    sym_pair   = f"{symbol}USDT"
    close_side = "SELL" if side == "BUY" else "BUY"
    try:
        client.new_order(
            symbol=sym_pair, side=close_side,
            type="STOP_MARKET", stopPrice=str(stop_loss),
            closePosition=True, timeInForce="GTE_GTC",
        )
        log.info(f"[{symbol}] Native SL order placed @ {stop_loss}")
    except Exception as e:
        log.warning(f"[{symbol}] Could not place native SL: {e}")
    try:
        client.new_order(
            symbol=sym_pair, side=close_side,
            type="TAKE_PROFIT_MARKET", stopPrice=str(take_profit),
            closePosition=True, timeInForce="GTE_GTC",
        )
        log.info(f"[{symbol}] Native TP order placed @ {take_profit}")
    except Exception as e:
        log.warning(f"[{symbol}] Could not place native TP: {e}")


def _get_binance_position_qty(client: UMFutures, symbol: str) -> float:
    """Fetch the actual position size from Binance — source of truth for closing."""
    sym_pair = f"{symbol}USDT"
    try:
        risk = client.get_position_risk(symbol=sym_pair)
        for pos in risk:
            amt = float(pos.get("positionAmt", 0))
            if amt != 0:
                return abs(amt)
    except Exception as e:
        log.warning(f"[{symbol}] Could not fetch Binance position qty: {e}")
    return 0.0


def _close_position(client: UMFutures, trade: dict, price: float, reason: str):
    """Manually close a position (LLM decision or signal flip).
    Cancels native SL/TP orders first to avoid double-close conflicts."""
    sym      = trade["symbol"]
    sym_pair = f"{sym}USDT"
    close_side = "SELL" if trade["side"] == "BUY" else "BUY"
    try:
        # Cancel native SL/TP orders so they don't conflict with our market close
        try:
            client.cancel_open_orders(symbol=sym_pair)
            log.info(f"[{sym}] Cancelled open SL/TP orders before manual close")
        except Exception as e:
            log.warning(f"[{sym}] Could not cancel open orders: {e}")

        # Always use real Binance qty — DB value may differ from actual fill
        real_qty = _get_binance_position_qty(client, sym)
        if real_qty <= 0:
            log.warning(f"[{sym}] No open position on Binance — marking DB trade closed anyway")
            close_trade(trade["id"], price, reason)
            return

        log.info(f"[{sym}] Closing {real_qty} (DB had {trade['quantity']}) via {close_side} reduceOnly")
        client.new_order(
            symbol=sym_pair, side=close_side, type="MARKET",
            quantity=real_qty, reduceOnly=True,
        )
        close_trade(trade["id"], price, reason)
    except Exception as e:
        log.error(f"Failed to close trade {trade['id']}: {e}")


# ── Main loops ─────────────────────────────────────────────────────────────────
def run_llm_cycle():
    log.info("=" * 60)
    log.info("LLM trading cycle starting...")

    client = get_client()

    # Get account balance
    try:
        account  = client.account()
        bal_info = next((a for a in account["assets"] if a["asset"] == "USDT"), None)
        balance  = float(bal_info["walletBalance"]) if bal_info else 1000.0
        log.info(f"Account balance: {balance:.2f} USDT")
    except Exception as e:
        log.error(f"Failed to get balance: {e}")
        return

    # Gather all context
    predictions       = get_all_predictions()
    market_signals    = get_latest_market_signals()
    price_data        = {sym: get_recent_prices(sym, 7) for sym in SYMBOLS}
    open_trades       = get_open_trades()
    binance_positions = _get_binance_positions(client)
    trade_history     = get_trade_history(limit=20)
    account_stats     = get_account_stats()
    last_decisions    = get_last_llm_decisions(limit=3)

    if not predictions:
        log.warning("No predictions in Redis — run POST /predict/precompute first")
        return

    # Ask LLM for decision
    decision = get_llm_decision(
        predictions, market_signals, price_data, open_trades, balance,
        binance_positions, trade_history, account_stats, last_decisions,
    )
    if not decision:
        log.error("LLM returned no decision — skipping cycle")
        return

    log.info(f"Market summary: {decision.get('market_summary', '')}")

    # Execute each symbol's decision
    for d in decision.get("decisions", []):
        symbol = d.get("symbol", "").upper()
        if symbol not in SYMBOLS:
            continue
        execute_decision(client, symbol, d, balance, open_trades)

    # Save LLM reasoning to DB for dashboard display
    _save_llm_decision(decision)


def _save_llm_decision(decision: dict):
    """Save the full LLM decision to DB so the dashboard can display it."""
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS llm_decisions (
                        id          BIGSERIAL PRIMARY KEY,
                        decided_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        decision    JSONB NOT NULL
                    )
                """)
                cur.execute(
                    "INSERT INTO llm_decisions (decision) VALUES (%s)",
                    (json.dumps(decision),)
                )
            conn.commit()
    except Exception as e:
        log.warning(f"Could not save LLM decision to DB: {e}")


def _get_binance_positions(client: UMFutures) -> dict:
    """Returns {symbol: {mark_price, position_amt, entry_price}} for all open positions."""
    result = {}
    try:
        for sym in SYMBOLS:
            sym_pair = f"{sym}USDT"
            risk = client.get_position_risk(symbol=sym_pair)
            for pos in risk:
                amt = float(pos.get("positionAmt", 0))
                if amt != 0:
                    result[sym] = {
                        "mark_price":    float(pos.get("markPrice", 0)),
                        "position_amt":  amt,
                        "entry_price":   float(pos.get("entryPrice", 0)),
                    }
    except Exception as e:
        log.warning(f"[MONITOR] Could not fetch Binance positions: {e}")
    return result


def _get_exit_price(client: UMFutures, symbol: str) -> float | None:
    """Get the fill price of the most recent closing trade for this symbol."""
    try:
        trades = client.get_account_trades(symbol=f"{symbol}USDT", limit=10)
        if trades:
            # Most recent trade — sorted oldest→newest by Binance
            return float(trades[-1]["price"])
    except Exception as e:
        log.warning(f"[{symbol}] Could not fetch exit price from trade history: {e}")
    return None


def monitor_loop():
    """
    Syncs DB state with Binance every 5 minutes.
    SL/TP execution is handled natively by Binance — we just detect when
    a position has closed and record it in our DB with the correct exit price.
    """
    log.info("Monitor loop started — syncing DB state with Binance every 5 minutes")
    client = get_client()
    check_count = 0
    while True:
        try:
            open_trades       = get_open_trades()
            binance_positions = _get_binance_positions(client)
            check_count += 1

            if check_count % 12 == 0:
                log.info(f"[MONITOR] Heartbeat — DB={len(open_trades)} open, Binance={len(binance_positions)} open")

            for trade in open_trades:
                sym = trade["symbol"]
                bn  = binance_positions.get(sym)

                if bn is not None:
                    # Position still open — log for debugging
                    log.debug(f"[MONITOR] {sym} still open | mark={bn['mark_price']} "
                              f"SL={trade['stop_loss']} TP={trade['take_profit']}")
                    continue

                # Position gone from Binance — native SL/TP fired (or manually closed)
                exit_price = _get_exit_price(client, sym)
                if exit_price is None:
                    # Fall back to last mark price from position risk (already 0 now)
                    ticker = client.ticker_price(symbol=f"{sym}USDT")
                    exit_price = float(ticker["price"])

                # Determine close reason from exit price vs SL/TP levels
                if trade["side"] == "BUY":
                    reason = ("take_profit" if exit_price >= trade["take_profit"]
                              else "stop_loss" if exit_price <= trade["stop_loss"]
                              else "native_close")
                else:
                    reason = ("take_profit" if exit_price <= trade["take_profit"]
                              else "stop_loss" if exit_price >= trade["stop_loss"]
                              else "native_close")

                log.info(f"[MONITOR] {sym} position closed on Binance — reason={reason} exit={exit_price}")
                close_trade(trade["id"], exit_price, reason)

        except Exception as e:
            log.error(f"[MONITOR] Error: {e}")
        time.sleep(MONITOR_INTERVAL)


def signal_loop():
    """Runs LLM decision cycle every 4h. Waits 4h before first run
    so it doesn't double-fire when started right after --once."""
    log.info(f"Signal loop started — first LLM decision in 4h")
    time.sleep(SIGNAL_INTERVAL)   # wait before first automatic cycle
    while True:
        try:
            run_llm_cycle()
        except Exception as e:
            log.error(f"[SIGNAL] Error: {e}")
        log.info("Next LLM decision in 4h...")
        time.sleep(SIGNAL_INTERVAL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", action="store_true", help="Run monitor + LLM signal loops")
    parser.add_argument("--once", action="store_true", help="Run one LLM decision cycle and exit")
    args = parser.parse_args()

    if args.once:
        run_llm_cycle()
    elif args.loop:
        t_monitor = threading.Thread(target=monitor_loop, daemon=True, name="monitor")
        t_signal  = threading.Thread(target=signal_loop,  daemon=True, name="signal")
        t_monitor.start()
        t_signal.start()
        log.info("Trading engine running — LLM decisions every 24h, SL/TP monitor every 5m")
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            log.info("Shutting down.")
    else:
        run_llm_cycle()
