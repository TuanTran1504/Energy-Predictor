"""
One-time script — run once to populate vnd_rates with 2 years of history.
Run with: python -m scripts.backfill_vnd

After this runs, the Go server saves one new row per day going forward.
"""
import httpx
import os
import psycopg2
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv("../.env")

ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")


def fetch_vnd_history() -> dict:
    """
    Fetches full daily USD/VND history from Alpha Vantage.
    Returns dict of {date_str: rate}.
    """
    url = (
        f"https://www.alphavantage.co/query"
        f"?function=FX_DAILY"
        f"&from_symbol=USD"
        f"&to_symbol=VND"
        f"&outputsize=full"
        f"&apikey={ALPHA_VANTAGE_KEY}"
    )

    print("Fetching USD/VND history from Alpha Vantage...")
    response = httpx.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()

    time_series = data.get("Time Series FX (Daily)", {})
    if not time_series:
        raise ValueError(f"No data returned. Response: {data}")

    print(f"Got {len(time_series)} daily records")

    # Extract closing rates
    rates = {}
    for date_str, values in time_series.items():
        rates[date_str] = float(values["4. close"])

    return rates


def backfill(start_date: str = "2024-01-01"):
    """
    Inserts historical VND rates into Supabase.
    Only inserts dates from start_date to yesterday.
    Skips dates that already exist.
    """
    rates = fetch_vnd_history()

    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()

    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = (datetime.utcnow() - timedelta(days=1)).date()
    current = start

    inserted = 0
    skipped = 0

    print(f"Inserting from {start} to {end}...")

    while current <= end:
        date_str = current.strftime("%Y-%m-%d")

        if date_str not in rates:
            current += timedelta(days=1)
            skipped += 1
            continue  # weekend or holiday — no trading data

        rate = rates[date_str]

        # Use INSERT ... ON CONFLICT DO NOTHING to skip duplicates
        cur.execute("""
            INSERT INTO vnd_rates (fetched_at, usd_to_vnd)
            VALUES (%s, %s)
            ON CONFLICT DO NOTHING
        """, (f"{date_str} 00:00:00+00", rate))

        inserted += 1
        current += timedelta(days=1)

    conn.commit()
    cur.close()
    conn.close()

    print(f"Done. Inserted: {inserted} rows. Skipped: {skipped} days (weekends/holidays).")


if __name__ == "__main__":
    backfill(start_date="2024-01-01")