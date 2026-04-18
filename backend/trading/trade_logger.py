"""
trade_logger.py — Structured logging for the 5-min trading engine.

Log files (written to backend/trading/logs/):
  trading.log     — every cycle: gates, score, AI decision, execution
  trades.log      — one JSON line per opened/closed trade
  skipped.log     — every cycle that was filtered out by a gate
  errors.log      — exceptions and API failures only

Usage:
  from trade_logger import get_logger, log_trade_open, log_trade_close, log_skip, log_cycle

All public functions are thread-safe (RotatingFileHandler uses file locks).
"""

import json
import logging
import os
import threading
import urllib.request
from datetime import datetime, timezone, timedelta
UTC = timezone.utc
VN_TZ = timezone(timedelta(hours=7))
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

_TG_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")
_TG_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

def _tg_send(text: str):
    """Fire-and-forget Telegram message."""
    if not _TG_TOKEN or not _TG_CHAT_ID:
        return
    def _send():
        try:
            url = f"https://api.telegram.org/bot{_TG_TOKEN}/sendMessage"
            data = json.dumps({"chat_id": _TG_CHAT_ID, "text": text, "parse_mode": "Markdown"}).encode()
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=5)
        except Exception:
            pass
    threading.Thread(target=_send, daemon=True).start()

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)


class _VNFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=VN_TZ)
        return dt.strftime(datefmt or "%Y-%m-%d %H:%M:%S")


_DETAIL_FMT = _VNFormatter(
    "%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_SIMPLE_FMT = _VNFormatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def _make_handler(filename: str, fmt: logging.Formatter,
                  max_mb: int = 20, backup: int = 5) -> RotatingFileHandler:
    h = RotatingFileHandler(
        LOG_DIR / filename,
        maxBytes=max_mb * 1024 * 1024,
        backupCount=backup,
        encoding="utf-8",
    )
    h.setFormatter(fmt)
    return h


def _build_logger(name: str, filename: str, level=logging.INFO,
                  fmt: logging.Formatter = _DETAIL_FMT) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(_make_handler(filename, fmt))
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    logger.propagate = False
    return logger


_cycle_log  = _build_logger("cycle",   "trading.log",  logging.DEBUG)
_trade_log  = _build_logger("trades",  "trades.log",   logging.INFO, _SIMPLE_FMT)
_skip_log   = _build_logger("skipped", "skipped.log",  logging.INFO, _SIMPLE_FMT)
_error_log  = _build_logger("errors",  "errors.log",   logging.ERROR)

_JSONL_PATH = LOG_DIR / "trades.jsonl"

def _append_jsonl(record: dict):
    """Append one JSON line to trades.jsonl — pure JSON, no logging wrapper."""
    with open(_JSONL_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def get_logger() -> logging.Logger:
    """Main cycle logger — use for general INFO/DEBUG messages."""
    return _cycle_log


def log_cycle_start(symbol: str, mode: str, score: int, score_max: int = 5):
    _cycle_log.info(
        f"{'─'*56}\n"
        f"  CYCLE START  {symbol}  mode={mode}  score={score}/{score_max}"
    )


def log_gate_pass(gate: str, detail: str = ""):
    _cycle_log.info(f"  [GATE ✓] {gate}  {detail}")


def log_gate_fail(gate: str, reason: str, symbol: str = "", context: dict = None):
    """Logs gate failure to both trading.log and skipped.log."""
    msg = f"SKIP | {gate} | {reason}"
    if context:
        msg += (
            f" | mode={context.get('market_mode','?')}"
            f" score={context.get('score','?')}"
            f" price={context.get('current_price','?')}"
            f" funding={context.get('funding_rate',0):+.4f}%"
        )
    _cycle_log.warning(f"  [GATE ✗] {gate}: {reason}")
    _skip_log.info(msg)


def log_ai_request(symbol: str, setup_hint: str):
    _cycle_log.info(f"  [AI ▶] Requesting Gemini decision | {symbol} | hint={setup_hint}")


def log_ai_response(decision: dict):
    sig    = decision.get("signal", "?")
    setup  = decision.get("analysis", {}).get("setup_identified", "?")
    reason = decision.get("reason", "")
    entry  = decision.get("entry_price", "?")
    sl     = decision.get("stop_loss", "?")
    tp     = decision.get("take_profit", "?")
    _cycle_log.info(
        f"  [AI ◀] signal={sig}  setup={setup}\n"
        f"          entry={entry}  SL={sl}  TP={tp}\n"
        f"\n"
        f"          reason: {reason}"
    )


def log_trade_open(symbol: str, side: str, entry: float, sl: float, tp: float,
                   rr: float, setup: str, reason: str, trade_id: Optional[str] = None,
                   context: dict = None):
    record = {
        "event":    "OPEN",
        "ts":       datetime.now(VN_TZ).isoformat(),
        "id":       trade_id,
        "symbol":   symbol,
        "side":     side,
        "entry":    entry,
        "sl":       sl,
        "tp":       tp,
        "rr":       round(rr, 2),
        "setup":    setup,
        "reason":   reason,
        "score":    context.get("score") if context else None,
        "mode":     context.get("market_mode") if context else None,
        "funding":  context.get("funding_rate") if context else None,
        "fear_greed": context.get("fear_greed") if context else None,
        "ml_direction": context.get("ml_direction") if context else None,
    }
    _trade_log.info(json.dumps(record))
    _append_jsonl(record)
    side_emoji = "🟢" if side == "BUY" else "🔴"
    _tg_send(
        f"{side_emoji} *TRADE OPEN* — {symbol} {side}\n"
        f"Entry: `{entry}` | SL: `{sl}` | TP: `{tp}`\n"
        f"R:R: `{rr:.2f}` | Setup: {setup}\n"
        f"_{reason}_"
    )
    _cycle_log.info(
        f"  [TRADE OPEN] {side} {symbol} | entry={entry} SL={sl} TP={tp} "
        f"R:R={rr:.2f} | setup={setup}"
    )


def log_trade_close(symbol: str, side: str, entry: float, exit_price: float,
                    pnl_pct: float, pnl_usdt: float, reason: str,
                    trade_id: Optional[str] = None, duration_min: float = 0):
    outcome = "WIN" if pnl_usdt > 0 else "LOSS"
    record = {
        "event":        "CLOSE",
        "ts":           datetime.now(VN_TZ).isoformat(),
        "id":           trade_id,
        "symbol":       symbol,
        "side":         side,
        "entry":        entry,
        "exit":         exit_price,
        "pnl_pct":      round(pnl_pct, 4),
        "pnl_usdt":     round(pnl_usdt, 4),
        "outcome":      outcome,
        "reason":       reason,
        "duration_min": round(duration_min, 1),
    }
    _trade_log.info(json.dumps(record))
    _append_jsonl(record)
    outcome_emoji = "✅" if pnl_usdt > 0 else "❌"
    pnl_str = f"+{pnl_usdt:.2f}" if pnl_usdt >= 0 else f"{pnl_usdt:.2f}"
    pct_str = f"+{pnl_pct*100:.2f}%" if pnl_pct >= 0 else f"{pnl_pct*100:.2f}%"
    _tg_send(
        f"{outcome_emoji} *TRADE CLOSE* — {symbol} {side}\n"
        f"P&L: `{pnl_str} USDT` ({pct_str})\n"
        f"Entry: `{entry}` → Exit: `{exit_price}`\n"
        f"Reason: {reason} | Held: {duration_min:.0f}min"
    )
    _cycle_log.info(
        f"  [TRADE CLOSE] {outcome} | {side} {symbol} | "
        f"entry={entry} exit={exit_price} pnl={pnl_pct*100:+.2f}% ({pnl_usdt:+.2f} USDT) "
        f"reason={reason} held={duration_min:.0f}min"
    )


def log_skip(gate: str, reason: str, context: dict = None, decision: dict = None):
    """Convenience wrapper — mirrors the trader_bot interface."""
    log_gate_fail(gate, reason, context=context)
    if decision:
        sig = decision.get("signal", "?")
        _skip_log.info(
            f"  AI={sig} setup={decision.get('analysis',{}).get('setup_identified','?')} "
            f"entry={decision.get('entry_price','?')} "
            f"SL={decision.get('stop_loss','?')} TP={decision.get('take_profit','?')}"
        )


def log_error(msg: str, exc: Exception = None):
    if exc:
        _error_log.exception(f"{msg}: {exc}")
    else:
        _error_log.error(msg)
    _cycle_log.error(f"  [ERROR] {msg}")


def log_cycle_summary(symbol: str, signal: str, executed: bool, balance: float,
                      context: dict = None, decision: dict = None):
    ctx  = context  or {}
    dec  = decision or {}
    mode  = ctx.get("market_mode", "?")
    score = ctx.get("score", "?")
    price = ctx.get("current_price", ctx.get("close", "?"))
    entry = dec.get("entry_price") or ctx.get("setup_e_entry", "—")
    sl    = dec.get("stop_loss",  "—")
    tp    = dec.get("take_profit","—")
    tp1   = dec.get("take_profit_1")
    tp2   = dec.get("take_profit_2")
    target_mode = dec.get("target_mode")
    reason = dec.get("reason", ctx.get("skip_reason", "—"))
    setup  = dec.get("analysis", {}).get("setup_identified") or ctx.get("setup", "—")
    status = "EXECUTED" if executed else ("SIGNAL" if signal not in ("WAIT", "SKIP") else signal)
    target_line = (
        f"  │  target_mode={target_mode}  TP1={tp1}  TP2={tp2}\n"
        if target_mode or tp1 is not None or tp2 is not None
        else ""
    )
    _cycle_log.info(
        f"\n  ┌─ DECISION  {symbol}  {'─'*40}\n"
        f"  │  mode={mode}  score={score}/5  price={price}\n"
        f"  │  signal={signal}  setup={setup}  status={status}\n"
        f"  │  entry={entry}  SL={sl}  TP={tp}\n"
        f"{target_line}"
        f"  │  reason: {reason}\n"
        f"  │  balance={balance:.2f} USDT\n"
        f"  └{'─'*50}"
    )

