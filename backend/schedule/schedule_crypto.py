"""
Modal scheduled job — saves 30-minute BTC and ETH candles to Supabase.
Runs every 30 minutes to keep data current.

Deploy: modal deploy backend/schedule/schedule_crypto.py
"""
import modal

app = modal.App("crypto-30min-saver")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("httpx==0.27.0", "psycopg2-binary==2.9.9")
)


@app.function(
    image=image,
    schedule=modal.Cron("*/30 * * * *"),  # Every 30 minutes
    secrets=[modal.Secret.from_name("energy-forecaster-secrets")],
    timeout=60,
)
def save_crypto_30min():
    import os
    import httpx
    import psycopg2
    from datetime import datetime, timezone, timedelta

    DATABASE_URL = os.environ["DATABASE_URL"]

    SYMBOLS = [
        ("BTCUSDT", "BTC"),
        ("ETHUSDT", "ETH"),
        ("SOLUSDT", "SOL"),
        ("XRPUSDT", "XRP"),
    ]

    conn = psycopg2.connect(DATABASE_URL, sslmode="require")
    cur = conn.cursor()
    saved = 0

    for binance_symbol, display_symbol in SYMBOLS:
        try:
            # Fetch last 2 days of 30-minute candles to ensure we get new data
            resp = httpx.get(
                "https://api.binance.com/api/v3/klines",
                params={
                    "symbol": binance_symbol,
                    "interval": "30m",
                    "limit": 100,  # 100 * 30min = 50 hours of data
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()

            if not data:
                print(f"{display_symbol}: no candle data")
                continue

            # Process each 30-minute candle
            for candle in data:
                open_time_ms = int(candle[0])
                open_time = datetime.fromtimestamp(open_time_ms / 1000, tz=timezone.utc)

                # Skip if this candle is older than 1 hour ago (to avoid duplicates)
                if open_time < datetime.now(timezone.utc) - timedelta(hours=1):
                    continue

                open_p = float(candle[1])
                high_p = float(candle[2])
                low_p = float(candle[3])
                close_p = float(candle[4])
                volume = float(candle[5])
                taker_buy_vol = float(candle[8])
                trade_count = int(candle[10])  
                change = ((close_p - open_p) / open_p) * 100 if open_p > 0 else 0

                cur.execute("""
                    INSERT INTO crypto_prices
                        (symbol, fetched_at, open_usd, high_usd, low_usd,
                        close_usd, volume_usd, change_pct, source, interval_minutes)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'binance', 30)
                    ON CONFLICT (symbol, fetched_at) DO NOTHING
                """, (
                    display_symbol, open_time,
                    open_p, high_p, low_p, close_p,
                    volume * close_p,  # Convert base volume to USD
                    round(change, 4),
                ))

                if cur.rowcount > 0:
                    saved += 1

            print(f"{display_symbol}: processed {len(data)} candles")

        except Exception as e:
            print(f"{display_symbol} fetch failed: {e}")

    from datetime import datetime, timezone
    now_utc = datetime.now(timezone.utc)

    # Fear & Greed — once a day at 08:00 UTC
    if now_utc.hour == 8 and now_utc.minute < 30:
        _fetch_fear_greed(conn)

    # Funding rates — 3x a day at 00:30, 08:30, 16:30 UTC (after each publish)
    if now_utc.hour in (0, 8, 16) and now_utc.minute >= 30:
        _fetch_funding_rates(conn)

    conn.commit()
    cur.close()
    conn.close()
    print(f"Done. Saved {saved} new 30-minute candles.")


def _fetch_fear_greed(conn):
    """
    Fetches today's Fear & Greed value and upserts into fear_greed_index.
    Called from save_crypto_30min at 08:00 UTC — no extra cron slot needed.
    """
    import httpx
    from datetime import datetime, timezone

    try:
        resp = httpx.get(
            "https://api.alternative.me/fng/",
            params={"limit": 3, "format": "json"},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
    except Exception as e:
        print(f"Fear & Greed fetch failed: {e}")
        return

    cur = conn.cursor()
    saved = 0
    for entry in data:
        ts    = int(entry["timestamp"])
        value = float(entry["value"])
        date  = datetime.fromtimestamp(ts, tz=timezone.utc).date()
        cur.execute("""
            INSERT INTO fear_greed_index (date, value, fetched_at)
            VALUES (%s, %s, NOW())
            ON CONFLICT (date) DO UPDATE
                SET value = EXCLUDED.value, fetched_at = NOW()
        """, (date, value))
        if cur.rowcount > 0:
            saved += 1
    cur.close()
    print(f"Fear & Greed: saved/updated {saved} entries.")


def _fetch_funding_rates(conn):
    """
    Fetches the latest funding rates from Gate.io and upserts daily averages
    into funding_rates. Called at 00:30, 08:30, 16:30 UTC — right after each
    8h funding rate publishes. No extra cron slot needed.
    Gate.io used because Binance/Bybit block Modal's US servers.
    """
    import httpx
    from datetime import datetime, timezone
    from collections import defaultdict

    GATE_SYMBOLS = {"BTC": "BTC_USDT", "ETH": "ETH_USDT"}
    cur = conn.cursor()

    for symbol, gate_sym in GATE_SYMBOLS.items():
        try:
            resp = httpx.get(
                "https://api.gateio.ws/api/v4/futures/usdt/funding_rate",
                params={"contract": gate_sym, "limit": 9},
                headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
                timeout=15,
            )
            resp.raise_for_status()
            entries = resp.json()
        except Exception as e:
            print(f"Funding rate fetch failed ({symbol}): {e}")
            continue

        by_date = defaultdict(list)
        for entry in entries:
            ts   = int(entry["t"])
            rate = float(entry["r"])
            date = datetime.fromtimestamp(ts, tz=timezone.utc).date()
            by_date[date].append(rate)

        saved = 0
        for date, rates in by_date.items():
            cur.execute("""
                INSERT INTO funding_rates (symbol, date, rate_avg, fetched_at)
                VALUES (%s, %s, %s, NOW())
                ON CONFLICT (symbol, date) DO UPDATE
                    SET rate_avg = EXCLUDED.rate_avg, fetched_at = NOW()
            """, (symbol, date, sum(rates) / len(rates)))
            if cur.rowcount > 0:
                saved += 1

        print(f"Funding rates ({symbol}): saved/updated {saved} entries.")

    cur.close()


@app.local_entrypoint()
def main():
    save_crypto_30min.remote()