"""
Telegram bot for crypto-agent trading notifications and queries.

Commands:
  /start   — welcome message
  /status  — all open positions
  /live    — live account positions only
  /trades  — last 10 closed trades
  /pnl     — P&L summary (win rate, total)
"""

import os
import logging
from datetime import datetime, timezone
from pathlib import Path

import psycopg2
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT / ".env")

TOKEN        = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID      = int(os.getenv("TELEGRAM_CHAT_ID", "0"))
DATABASE_URL = os.getenv("DATABASE_URL", "")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

UTC = timezone.utc


def get_db():
    return psycopg2.connect(DATABASE_URL)


def fmt_pnl(val):
    if val is None:
        return "—"
    return f"+{val:.2f}" if val >= 0 else f"{val:.2f}"


def fmt_pct(val):
    if val is None:
        return "—"
    return f"+{val:.2f}%" if val >= 0 else f"{val:.2f}%"


# ── Helpers ────────────────────────────────────────────────────────────────

def fetch_open_trades(account_type=None):
    conn = get_db()
    try:
        cur = conn.cursor()
        if account_type:
            cur.execute("""
                SELECT symbol, side, entry_price, stop_loss, take_profit,
                       quantity, leverage, opened_at,
                       COALESCE(account_type, 'testnet') as account_type
                FROM trades
                WHERE status = 'OPEN'
                  AND COALESCE(account_type, 'testnet') = %s
                ORDER BY opened_at DESC
            """, (account_type,))
        else:
            cur.execute("""
                SELECT symbol, side, entry_price, stop_loss, take_profit,
                       quantity, leverage, opened_at,
                       COALESCE(account_type, 'testnet') as account_type
                FROM trades
                WHERE status = 'OPEN'
                ORDER BY opened_at DESC
            """)
        return cur.fetchall()
    finally:
        conn.close()


def fetch_closed_trades(limit=10, account_type=None):
    conn = get_db()
    try:
        cur = conn.cursor()
        if account_type:
            cur.execute("""
                SELECT symbol, side, entry_price, exit_price,
                       pnl_usdt, pnl_pct, close_reason, opened_at, closed_at,
                       COALESCE(account_type, 'testnet') as account_type
                FROM trades
                WHERE status = 'CLOSED'
                  AND COALESCE(account_type, 'testnet') = %s
                ORDER BY closed_at DESC
                LIMIT %s
            """, (account_type, limit))
        else:
            cur.execute("""
                SELECT symbol, side, entry_price, exit_price,
                       pnl_usdt, pnl_pct, close_reason, opened_at, closed_at,
                       COALESCE(account_type, 'testnet') as account_type
                FROM trades
                WHERE status = 'CLOSED'
                ORDER BY closed_at DESC
                LIMIT %s
            """, (limit,))
        return cur.fetchall()
    finally:
        conn.close()


def fetch_pnl_summary(account_type=None):
    conn = get_db()
    try:
        cur = conn.cursor()
        if account_type:
            cur.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN pnl_usdt > 0 THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN pnl_usdt <= 0 THEN 1 ELSE 0 END) as losses,
                    SUM(pnl_usdt) as total_pnl,
                    AVG(pnl_usdt) as avg_pnl
                FROM trades
                WHERE status = 'CLOSED'
                  AND COALESCE(account_type, 'testnet') = %s
            """, (account_type,))
        else:
            cur.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN pnl_usdt > 0 THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN pnl_usdt <= 0 THEN 1 ELSE 0 END) as losses,
                    SUM(pnl_usdt) as total_pnl,
                    AVG(pnl_usdt) as avg_pnl
                FROM trades
                WHERE status = 'CLOSED'
            """)
        return cur.fetchone()
    finally:
        conn.close()


# ── Command handlers ───────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 *Crypto Agent Bot*\n\n"
        "Commands:\n"
        "/status — live open positions\n"
        "/trades — last 10 live closed trades\n"
        "/pnl — live account P&L summary\n"
        "/livepnl — same as /pnl\n"
        "/live — same as /status",
        parse_mode="Markdown"
    )


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    rows = fetch_open_trades(account_type="live")
    if not rows:
        await update.message.reply_text("💰 No open live positions.")
        return

    lines = ["💰 *Live Open Positions*\n"]
    for r in rows:
        symbol, side, entry, sl, tp, qty, lev, opened_at, acc = r
        side_emoji = "🟢" if side == "BUY" else "🔴"
        lines.append(
            f"{side_emoji} *{symbol}* {side}\n"
            f"  Entry: `{entry}` | Qty: `{qty}`\n"
            f"  SL: `{sl}` | TP: `{tp}`\n"
            f"  Opened: {opened_at.strftime('%m/%d %H:%M') if opened_at else '—'}\n"
        )
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def cmd_live(update: Update, context: ContextTypes.DEFAULT_TYPE):
    rows = fetch_open_trades(account_type="live")
    if not rows:
        await update.message.reply_text("No open live positions.")
        return

    lines = ["💰 *Live Account Positions*\n"]
    for r in rows:
        symbol, side, entry, sl, tp, qty, lev, opened_at, acc = r
        side_emoji = "🟢" if side == "BUY" else "🔴"
        lines.append(
            f"{side_emoji} *{symbol}* {side}\n"
            f"  Entry: `{entry}` | Qty: `{qty}`\n"
            f"  SL: `{sl}` | TP: `{tp}`\n"
            f"  Opened: {opened_at.strftime('%m/%d %H:%M') if opened_at else '—'}\n"
        )
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def cmd_trades(update: Update, context: ContextTypes.DEFAULT_TYPE):
    rows = fetch_closed_trades(limit=10, account_type="live")
    if not rows:
        await update.message.reply_text("No closed live trades yet.")
        return

    lines = ["📋 *Last 10 Live Trades*\n"]
    for r in rows:
        symbol, side, entry, exit_p, pnl, pnl_pct, reason, opened_at, closed_at, acc = r
        outcome = "✅" if (pnl or 0) > 0 else "❌"
        lines.append(
            f"{outcome} *{symbol}* {side} → "
            f"`{fmt_pnl(pnl)} USDT` ({fmt_pct(pnl_pct)}) — {reason or '—'}\n"
        )
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def cmd_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    row = fetch_pnl_summary(account_type="live")
    if not row or not row[0]:
        await update.message.reply_text("No closed live trades yet.")
        return

    total, wins, losses, total_pnl, avg_pnl = row
    wins = wins or 0
    losses = losses or 0
    win_rate = (wins / total * 100) if total else 0

    await update.message.reply_text(
        f"💰 *Live Account P&L*\n\n"
        f"Total trades: `{total}`\n"
        f"Wins: `{wins}` | Losses: `{losses}`\n"
        f"Win rate: `{win_rate:.1f}%`\n"
        f"Total P&L: `{fmt_pnl(total_pnl)} USDT`\n"
        f"Avg per trade: `{fmt_pnl(avg_pnl)} USDT`",
        parse_mode="Markdown"
    )


async def cmd_livepnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    row = fetch_pnl_summary(account_type="live")
    if not row or not row[0]:
        await update.message.reply_text("No closed live trades yet.")
        return

    total, wins, losses, total_pnl, avg_pnl = row
    wins = wins or 0
    losses = losses or 0
    win_rate = (wins / total * 100) if total else 0

    await update.message.reply_text(
        f"💰 *Live Account P&L*\n\n"
        f"Total trades: `{total}`\n"
        f"Wins: `{wins}` | Losses: `{losses}`\n"
        f"Win rate: `{win_rate:.1f}%`\n"
        f"Total P&L: `{fmt_pnl(total_pnl)} USDT`\n"
        f"Avg per trade: `{fmt_pnl(avg_pnl)} USDT`",
        parse_mode="Markdown"
    )


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start",    cmd_start))
    app.add_handler(CommandHandler("status",   cmd_status))
    app.add_handler(CommandHandler("live",     cmd_live))
    app.add_handler(CommandHandler("trades",   cmd_trades))
    app.add_handler(CommandHandler("pnl",      cmd_pnl))
    app.add_handler(CommandHandler("livepnl",  cmd_livepnl))

    log.info("Telegram bot started — polling...")
    app.run_polling(allowed_updates=["message"])


if __name__ == "__main__":
    main()
