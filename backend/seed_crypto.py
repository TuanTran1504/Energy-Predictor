"""
Backfills BTC and ETH daily OHLCV data from Binance public API.
No API key needed — Binance public endpoint.

Run once:
    python seed_crypto.py
"""
import httpx
import psycopg2
import os
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent.parent / ".env")
DATABASE_URL = os.getenv("DATABASE_URL")

BINANCE_URL = "https://api.binance.com/api/v3/klines"
SYMBOLS = [
    ("BTCUSDT", "BTC"),
    ("ETHUSDT", "ETH"),
]


def fetch_klines(symbol: str, limit: int = 1000) -> list:
    """
    Fetches daily OHLCV candles from Binance.
    Returns up to 1000 days of history — no API key needed.
    """
    params = {
        "symbol": symbol,
        "interval": "1d",
        "limit": limit,
    }
    resp = httpx.get(BINANCE_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def parse_candles(raw: list, display_symbol: str) -> list:
    """
    Parses Binance kline format into clean records.
    
    Binance kline format:
    [open_time, open, high, low, close, volume, close_time, ...]
    """
    records = []
    for c in raw:
        open_time = datetime.utcfromtimestamp(c[0] / 1000)
        open_p  = float(c[1])
        high_p  = float(c[2])
        low_p   = float(c[3])
        close_p = float(c[4])
        volume  = float(c[5])
        change  = ((close_p - open_p) / open_p) * 100 if open_p > 0 else 0

        records.append({
            "symbol":     display_symbol,
            "fetched_at": open_time,
            "open_usd":   open_p,
            "high_usd":   high_p,
            "low_usd":    low_p,
            "close_usd":  close_p,
            "volume_usd": volume * close_p,  # convert to USD volume
            "change_pct": round(change, 4),
        })
    return records


def save_to_supabase(records: list) -> tuple[int, int]:
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    inserted = skipped = 0

    for r in records:
        cur.execute("""
            INSERT INTO crypto_prices
                (symbol, fetched_at, open_usd, high_usd, low_usd,
                 close_usd, volume_usd, change_pct, source)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'binance')
            ON CONFLICT (symbol, fetched_at) DO NOTHING
        """, (
            r["symbol"],
            r["fetched_at"],
            r["open_usd"],
            r["high_usd"],
            r["low_usd"],
            r["close_usd"],
            r["volume_usd"],
            r["change_pct"],
        ))
        if cur.rowcount > 0:
            inserted += 1
        else:
            skipped += 1

    conn.commit()
    cur.close()
    conn.close()
    return inserted, skipped


def main():
    for binance_symbol, display_symbol in SYMBOLS:
        print(f"\nFetching {display_symbol} history from Binance...")
        raw = fetch_klines(binance_symbol, limit=1000)
        records = parse_candles(raw, display_symbol)

        print(f"  Parsed {len(records)} daily candles")
        print(f"  From: {records[0]['fetched_at'].strftime('%Y-%m-%d')}")
        print(f"  To:   {records[-1]['fetched_at'].strftime('%Y-%m-%d')}")
        print(f"  Latest close: ${records[-1]['close_usd']:,.2f}")

        inserted, skipped = save_to_supabase(records)
        print(f"  Saved {inserted} new rows · Skipped {skipped} duplicates")


if __name__ == "__main__":
    main()