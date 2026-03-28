"""
Modal scheduled job — saves daily BTC and ETH candles to Supabase.
Runs every day at 00:05 UTC (just after midnight when daily candle closes).

Deploy: modal deploy backend/ml/training/crypto_schedule.py
"""
import modal

app = modal.App("crypto-daily-saver")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("httpx==0.27.0", "psycopg2-binary==2.9.9")
)


@app.function(
    image=image,
    schedule=modal.Cron("5 0 * * *"),
    secrets=[modal.Secret.from_name("energy-forecaster-secrets")],
    timeout=60,
)
def save_daily_crypto():
    import os
    import httpx
    import psycopg2
    from datetime import datetime, timezone

    DATABASE_URL = os.environ["DATABASE_URL"]

    SYMBOLS = [
        ("bitcoin", "BTC"),
        ("ethereum", "ETH"),
    ]

    conn = psycopg2.connect(DATABASE_URL, sslmode="require")
    cur = conn.cursor()
    saved = 0

    for coingecko_id, display_symbol in SYMBOLS:
        try:
            # Use market_chart endpoint — supports any day range
            resp = httpx.get(
                f"https://api.coingecko.com/api/v3/coins/{coingecko_id}/market_chart",
                params={
                    "vs_currency": "usd",
                    "days": "1",
                    "interval": "daily",
                },
                timeout=15,
                headers={"Accept": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()

            prices = data.get("prices", [])
            if not prices:
                print(f"{display_symbol}: no price data")
                continue

            # Latest price point
            timestamp_ms, close_p = prices[-1]
            open_p = float(prices[0][1]) if len(prices) > 1 else close_p
            change = ((close_p - open_p) / open_p) * 100
            open_time = datetime.utcfromtimestamp(timestamp_ms / 1000)

            cur.execute("""
                INSERT INTO crypto_prices
                    (symbol, fetched_at, open_usd, high_usd, low_usd,
                    close_usd, volume_usd, change_pct, source)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'coingecko')
                ON CONFLICT (symbol, fetched_at) DO NOTHING
            """, (
                display_symbol, open_time,
                open_p, close_p, close_p, close_p,
                0,
                round(change, 4),
            ))

            if cur.rowcount > 0:
                saved += 1
                print(f"{display_symbol}: ${close_p:,.2f} ({change:+.2f}%) saved")
            else:
                print(f"{display_symbol}: already exists, skipped")

        except Exception as e:
            print(f"{display_symbol} fetch failed: {e}")

    conn.commit()
    cur.close()
    conn.close()
    print(f"Done. Saved {saved} new candles.")


@app.local_entrypoint()
def main():
    save_daily_crypto.remote()