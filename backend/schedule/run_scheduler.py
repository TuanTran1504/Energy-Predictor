"""
Scheduler service — runs data collection jobs on a fixed schedule.
Runs as a Docker container (replaces Modal for production).

Jobs:
  Every 24h: Fear & Greed index, FRED macro events (FED/CPI/NFP)
  Every 8h:  Binance funding rates
"""
import os
import time
import logging
import threading
from datetime import datetime, timezone, timedelta

import httpx
import psycopg2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-5s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

DATABASE_URL = os.environ["DATABASE_URL"]
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")


def get_conn():
    dsn = DATABASE_URL
    if "sslmode" not in dsn:
        dsn += ("&" if "?" in dsn else "?") + "sslmode=require"
    return psycopg2.connect(dsn)


def fetch_fear_greed():
    log.info("Fetching Fear & Greed index...")
    try:
        resp = httpx.get(
            "https://api.alternative.me/fng/",
            params={"limit": 7, "format": "json"},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])

        conn = get_conn()
        cur = conn.cursor()
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
        conn.commit()
        cur.close()
        conn.close()
        log.info(f"Fear & Greed: saved {len(data)} entries")
    except Exception as e:
        log.error(f"Fear & Greed failed: {e}")


def fetch_funding_rates():
    log.info("Fetching Binance funding rates...")
    for symbol in ["BTC", "ETH"]:
        try:
            resp = httpx.get(
                "https://fapi.binance.com/fapi/v1/fundingRate",
                params={"symbol": f"{symbol}USDT", "limit": 3},
                timeout=15,
            )
            resp.raise_for_status()
            entries = resp.json()
            if not entries:
                continue

            by_date: dict = {}
            for e in entries:
                ts   = int(e["fundingTime"])
                rate = float(e["fundingRate"])
                date = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).date()
                by_date.setdefault(date, []).append(rate)

            conn = get_conn()
            cur  = conn.cursor()
            for date, rates in by_date.items():
                avg = sum(rates) / len(rates)
                cur.execute("""
                    INSERT INTO funding_rates (symbol, date, rate_avg, fetched_at)
                    VALUES (%s, %s, %s, NOW())
                    ON CONFLICT (symbol, date) DO UPDATE
                        SET rate_avg = EXCLUDED.rate_avg, fetched_at = NOW()
                """, (symbol, date, avg))
            conn.commit()
            cur.close()
            conn.close()
            log.info(f"Funding rates {symbol}: saved {len(by_date)} days")
        except Exception as e:
            log.error(f"Funding rates {symbol} failed: {e}")


def fetch_fred():
    if not FRED_API_KEY:
        log.warning("FRED_API_KEY not set — skipping FRED fetch")
        return
    log.info("Fetching FRED macro data...")

    conn = get_conn()
    cur  = conn.cursor()
    saved = 0
    start = (datetime.now(timezone.utc) - timedelta(days=730)).strftime("%Y-%m-%d")

    # Fed Rate decisions
    try:
        resp = httpx.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={"series_id": "DFEDTARU", "api_key": FRED_API_KEY,
                    "file_type": "json", "sort_order": "asc",
                    "limit": 200, "observation_start": start},
            timeout=20,
        )
        obs  = resp.json().get("observations", [])
        prev = None
        for o in obs:
            if o["value"] == ".":
                continue
            rate = float(o["value"])
            if prev is not None and abs(rate - prev) > 0.001:
                bps  = int(round((rate - prev) * 100))
                desc = f"Fed {'raises' if bps > 0 else 'cuts'} rates to {rate:.2f}%"
                ev_dt = datetime.strptime(o["date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
                cur.execute("""
                    INSERT INTO macro_events
                        (event_date, event_type, description, actual, verified)
                    VALUES (%s, 'FED_RATE', %s, %s, TRUE)
                    ON CONFLICT (event_date, event_type) DO NOTHING
                """, (ev_dt, desc, f"{bps:+d}bps"))
                if cur.rowcount > 0:
                    saved += 1
            prev = rate
        log.info(f"FRED fed rate: processed {len(obs)} observations")
    except Exception as e:
        log.error(f"FRED fed rate failed: {e}")

    # CPI
    try:
        resp = httpx.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={"series_id": "CPIAUCSL", "api_key": FRED_API_KEY,
                    "file_type": "json", "limit": 24,
                    "observation_start": start},
            timeout=20,
        )
        obs  = resp.json().get("observations", [])
        prev = None
        for o in obs:
            if o["value"] == ".":
                continue
            cpi = float(o["value"])
            if prev is not None:
                mom = (cpi - prev) / prev * 100
                desc = f"US CPI {o['date'][:7]}: {cpi:.1f} (MoM {mom:+.2f}%)"
                release_dt = (datetime.strptime(o["date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
                              + timedelta(days=14))
                cur.execute("""
                    INSERT INTO macro_events
                        (event_date, event_type, description, actual, verified)
                    VALUES (%s, 'CPI', %s, %s, TRUE)
                    ON CONFLICT (event_date, event_type) DO NOTHING
                """, (release_dt, desc, f"{mom:+.2f}%"))
                if cur.rowcount > 0:
                    saved += 1
            prev = cpi
    except Exception as e:
        log.error(f"FRED CPI failed: {e}")

    # NFP
    try:
        resp = httpx.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={"series_id": "PAYEMS", "api_key": FRED_API_KEY,
                    "file_type": "json", "limit": 24,
                    "observation_start": start},
            timeout=20,
        )
        obs  = resp.json().get("observations", [])
        prev = None
        for o in obs:
            if o["value"] == ".":
                continue
            jobs = float(o["value"])
            if prev is not None:
                change_k = jobs - prev
                desc = f"US NFP {o['date'][:7]}: {change_k:+.0f}k jobs"
                release_dt = (datetime.strptime(o["date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
                              + timedelta(days=35))
                cur.execute("""
                    INSERT INTO macro_events
                        (event_date, event_type, description, actual, verified)
                    VALUES (%s, 'NFP', %s, %s, TRUE)
                    ON CONFLICT (event_date, event_type) DO NOTHING
                """, (release_dt, desc, f"{change_k:+.0f}k"))
                if cur.rowcount > 0:
                    saved += 1
            prev = jobs
    except Exception as e:
        log.error(f"FRED NFP failed: {e}")

    conn.commit()
    cur.close()
    conn.close()
    log.info(f"FRED: saved {saved} new macro events")


def run_daily():
    """Runs immediately on startup, then every 24h."""
    while True:
        log.info("=== Daily jobs starting ===")
        fetch_fear_greed()
        fetch_fred()
        log.info("=== Daily jobs done — next run in 24h ===")
        time.sleep(24 * 3600)


def run_8h():
    """Runs immediately on startup, then every 8h."""
    while True:
        log.info("=== 8h jobs starting ===")
        fetch_funding_rates()
        log.info("=== 8h jobs done — next run in 8h ===")
        time.sleep(8 * 3600)


if __name__ == "__main__":
    log.info("Scheduler service starting...")
    t1 = threading.Thread(target=run_daily, daemon=True, name="daily")
    t2 = threading.Thread(target=run_8h,   daemon=True, name="8h")
    t1.start()
    t2.start()
    while True:
        time.sleep(3600)
