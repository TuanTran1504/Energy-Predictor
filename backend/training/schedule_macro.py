"""
Modal scheduled job — fetches macro events automatically.
Sources:
  FRED API      → Fed rate decisions (changes only), CPI, NFP
  CoinGecko API → BTC/ETH price impact measurement (no geo-restrictions)

Key fix: Fed rates are only saved on actual decision dates (rate changes).
         HOLD days are NOT saved — they create noise in the training data.
         Only real FOMC meeting dates with rate changes are recorded.

Schedule:
  Daily at 06:00 UTC — checks for new macro events

Deploy: modal deploy backend/ml/training/macro_schedule.py
Test:   modal run backend/ml/training/macro_schedule.py
"""
import modal

app = modal.App("macro-event-fetcher")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "httpx==0.27.0",
        "psycopg2-binary==2.9.9",
    )
)

# Known FOMC meeting dates — Fed publishes these a year in advance
# Only dates where we expect a decision (rate change OR hold announcement)
# Source: federalreserve.gov/monetarypolicy/fomccalendars.htm
FOMC_DATES_2026 = [
    "2026-01-28",
    "2026-03-18",
    "2026-05-06",
    "2026-06-17",
    "2026-07-29",
    "2026-09-16",
    "2026-10-28",
    "2026-12-09",
]


@app.function(
    image=image,
    schedule=modal.Cron("0 6 * * *"),  # daily at 06:00 UTC
    secrets=[modal.Secret.from_name("energy-forecaster-secrets")],
    timeout=120,
)
def fetch_macro_events():
    import os
    import httpx
    import psycopg2
    from datetime import datetime, timezone, timedelta

    DATABASE_URL = os.environ["DATABASE_URL"]
    FRED_API_KEY = os.environ.get("FRED_API_KEY", "")

    if not FRED_API_KEY:
        print("ERROR: FRED_API_KEY not set in Modal secret")
        return

    conn = psycopg2.connect(DATABASE_URL, sslmode="require")
    cur = conn.cursor()
    saved = 0

    # ── 1. Fed Rate Decisions ──────────────────────────────────────────────
    # Strategy: fetch DFEDTARU series, only save rows where rate CHANGED
    # This gives us actual decision dates, not every business day
    print("Fetching Fed rate decisions from FRED...")
    try:
        resp = httpx.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={
                "series_id": "DFEDTARU",
                "api_key": FRED_API_KEY,
                "file_type": "json",
                "sort_order": "asc",
                "limit": 1000,
                "observation_start": "2023-07-01",
            },
            timeout=15,
        )
        resp.raise_for_status()
        observations = resp.json().get("observations", [])
        print(f"  Got {len(observations)} FRED observations")

        prev_rate = None
        decisions_found = 0

        for obs in observations:
            if obs["value"] == ".":
                continue

            date_str = obs["date"]
            rate = float(obs["value"])

            if prev_rate is not None and abs(rate - prev_rate) > 0.001:
                # Rate changed — this IS a real decision date
                change = rate - prev_rate
                change_bps = int(round(change * 100))

                if change_bps > 0:
                    actual = f"+{change_bps}bps"
                    description = f"Fed raises rates to {rate:.2f}%"
                else:
                    actual = f"{change_bps}bps"
                    description = f"Fed cuts rates to {rate:.2f}%"

                event_dt = datetime.strptime(date_str, "%Y-%m-%d").replace(
                    tzinfo=timezone.utc
                )

                cur.execute("""
                    INSERT INTO macro_events
                        (event_date, event_type, description, actual, verified)
                    VALUES (%s, 'FED_RATE', %s, %s, TRUE)
                    ON CONFLICT (event_date, event_type) DO NOTHING
                """, (event_dt, description, actual))

                if cur.rowcount > 0:
                    saved += 1
                    decisions_found += 1
                    print(f"  NEW Fed decision: {date_str} → {actual}")

            prev_rate = rate

        # Also check if today is a known FOMC date with no rate change (HOLD)
        # We only record HOLDs on actual FOMC meeting dates
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today_str in FOMC_DATES_2026:
            event_dt = datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            # Get current rate
            if observations:
                current_rate = float(observations[-1]["value"])
                description = f"Fed holds rates at {current_rate:.2f}%"
                cur.execute("""
                    INSERT INTO macro_events
                        (event_date, event_type, description, actual, verified)
                    VALUES (%s, 'FED_RATE', %s, 'HOLD', TRUE)
                    ON CONFLICT (event_date, event_type) DO NOTHING
                """, (event_dt, description))
                if cur.rowcount > 0:
                    saved += 1
                    print(f"  FOMC meeting today ({today_str}) — HOLD recorded")

        print(f"  Fed: {decisions_found} new rate change decisions found")

    except Exception as e:
        print(f"Fed rate fetch failed: {e}")

    # ── 2. CPI Inflation (FRED series: CPIAUCSL) ──────────────────────────
    # Monthly — released ~2 weeks after the reference month ends
    print("Fetching CPI data from FRED...")
    try:
        resp = httpx.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={
                "series_id": "CPIAUCSL",
                "api_key": FRED_API_KEY,
                "file_type": "json",
                "sort_order": "asc",
                "limit": 36,
                "observation_start": "2023-07-01",
            },
            timeout=15,
        )
        resp.raise_for_status()
        observations = resp.json().get("observations", [])

        prev_cpi = None
        for obs in observations:
            if obs["value"] == ".":
                continue

            date_str = obs["date"]
            cpi = float(obs["value"])

            if prev_cpi is not None:
                # Month-over-month annualized
                mom_pct = ((cpi - prev_cpi) / prev_cpi) * 100
                annualized = mom_pct * 12
                description = (
                    f"US CPI {date_str[:7]}: {cpi:.1f} "
                    f"(MoM {mom_pct:+.2f}%, annualized {annualized:+.1f}%)"
                )

                # CPI data is released ~2 weeks after the reference month
                release_dt = (
                    datetime.strptime(date_str, "%Y-%m-%d").replace(
                        tzinfo=timezone.utc
                    ) + timedelta(days=14)
                )

                cur.execute("""
                    INSERT INTO macro_events
                        (event_date, event_type, description, actual, verified)
                    VALUES (%s, 'CPI', %s, %s, TRUE)
                    ON CONFLICT (event_date, event_type) DO NOTHING
                """, (release_dt, description, f"{mom_pct:+.2f}%"))

                if cur.rowcount > 0:
                    saved += 1
                    print(f"  CPI: {date_str} → MoM {mom_pct:+.2f}%")

            prev_cpi = cpi

    except Exception as e:
        print(f"CPI fetch failed: {e}")

    # ── 3. NFP Jobs Report (FRED series: PAYEMS) ──────────────────────────
    # Released first Friday of each month
    print("Fetching NFP data from FRED...")
    try:
        resp = httpx.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={
                "series_id": "PAYEMS",
                "api_key": FRED_API_KEY,
                "file_type": "json",
                "sort_order": "asc",
                "limit": 36,
                "observation_start": "2023-07-01",
            },
            timeout=15,
        )
        resp.raise_for_status()
        observations = resp.json().get("observations", [])

        prev_jobs = None
        for obs in observations:
            if obs["value"] == ".":
                continue

            date_str = obs["date"]
            jobs_total = float(obs["value"])

            if prev_jobs is not None:
                # Change in thousands of jobs
                change_k = jobs_total - prev_jobs
                description = f"US NFP {date_str[:7]}: {change_k:+.0f}k jobs"

                # NFP released ~1 week after month reference date
                release_dt = (
                    datetime.strptime(date_str, "%Y-%m-%d").replace(
                        tzinfo=timezone.utc
                    ) + timedelta(days=35)  # first Friday of following month
                )

                cur.execute("""
                    INSERT INTO macro_events
                        (event_date, event_type, description, actual, verified)
                    VALUES (%s, 'NFP', %s, %s, TRUE)
                    ON CONFLICT (event_date, event_type) DO NOTHING
                """, (release_dt, description, f"{change_k:+.0f}k"))

                if cur.rowcount > 0:
                    saved += 1
                    print(f"  NFP: {date_str} → {change_k:+.0f}k jobs")

            prev_jobs = jobs_total

    except Exception as e:
        print(f"NFP fetch failed: {e}")

    # ── 4. Measure BTC/ETH impact for yesterday's events ──────────────────
    # Uses CoinGecko — no geo-restrictions unlike Binance
    print("Measuring price impact for recent events...")
    try:
        yesterday = datetime.now(timezone.utc) - timedelta(days=1)
        yesterday_start = yesterday.replace(hour=0, minute=0, second=0)
        yesterday_end = yesterday.replace(hour=23, minute=59, second=59)

        cur.execute("""
            SELECT id, event_date
            FROM macro_events
            WHERE event_date BETWEEN %s AND %s
            AND btc_impact_24h IS NULL
        """, (yesterday_start, yesterday_end))

        events_to_measure = cur.fetchall()

        if events_to_measure:
            print(f"  Measuring impact for {len(events_to_measure)} events...")

            # CoinGecko market chart — works from any cloud provider
            btc_resp = httpx.get(
                "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart",
                params={"vs_currency": "usd", "days": "2", "interval": "daily"},
                timeout=15,
                headers={"Accept": "application/json"},
            )
            eth_resp = httpx.get(
                "https://api.coingecko.com/api/v3/coins/ethereum/market_chart",
                params={"vs_currency": "usd", "days": "2", "interval": "daily"},
                timeout=15,
                headers={"Accept": "application/json"},
            )

            btc_prices = btc_resp.json().get("prices", [])
            eth_prices = eth_resp.json().get("prices", [])

            if len(btc_prices) >= 2 and len(eth_prices) >= 2:
                btc_before = float(btc_prices[-2][1])
                btc_after  = float(btc_prices[-1][1])
                btc_impact = ((btc_after - btc_before) / btc_before) * 100

                eth_before = float(eth_prices[-2][1])
                eth_after  = float(eth_prices[-1][1])
                eth_impact = ((eth_after - eth_before) / eth_before) * 100

                for event_id, event_date in events_to_measure:
                    cur.execute("""
                        UPDATE macro_events
                        SET btc_impact_24h = %s, eth_impact_24h = %s
                        WHERE id = %s
                    """, (round(btc_impact, 2), round(eth_impact, 2), event_id))
                    print(
                        f"  Event {event_id} ({event_date.date()}): "
                        f"BTC {btc_impact:+.2f}% ETH {eth_impact:+.2f}%"
                    )
        else:
            print("  No events from yesterday to measure")

    except Exception as e:
        print(f"Impact measurement failed: {e}")

    conn.commit()
    cur.close()
    conn.close()
    print(f"\nDone. Saved {saved} new macro events.")


@app.local_entrypoint()
def main():
    fetch_macro_events.remote()