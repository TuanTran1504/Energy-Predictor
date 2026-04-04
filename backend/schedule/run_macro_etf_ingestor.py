"""
Macro + ETF ingestion worker.

Purpose:
  - Persist macro releases and ETF flow data in dedicated tables
  - Upsert idempotently
  - Track revisions when provider values change over time
  - Poll on short intervals for near-real-time updates

Environment:
  DATABASE_URL                  required
  FRED_API_KEY                  optional (enables FRED ingestion)
  ECON_CALENDAR_URL             optional (generic calendar API endpoint)
  ECON_CALENDAR_API_KEY         optional
  ETF_FLOW_API_URL              optional (generic ETF flow API endpoint)
  ETF_FLOW_API_KEY              optional
  ECON_POLL_SECONDS             default 900 (15 minutes)
  ETF_POLL_SECONDS              default 900 (15 minutes)
  REQUEST_TIMEOUT_SECONDS       default 20
"""
import hashlib
import json
import logging
import os
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import psycopg2
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-5s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


DATABASE_URL = os.environ["DATABASE_URL"]
FRED_API_KEY = os.getenv("FRED_API_KEY", "").strip()
ECON_CALENDAR_URL = os.getenv("ECON_CALENDAR_URL", "").strip()
ECON_CALENDAR_API_KEY = os.getenv("ECON_CALENDAR_API_KEY", "").strip()
ETF_FLOW_API_URL = os.getenv("ETF_FLOW_API_URL", "").strip()
ETF_FLOW_API_KEY = os.getenv("ETF_FLOW_API_KEY", "").strip()
ECON_POLL_SECONDS = int(os.getenv("ECON_POLL_SECONDS", "900"))
ETF_POLL_SECONDS = int(os.getenv("ETF_POLL_SECONDS", "900"))
REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "20"))


def get_conn():
    dsn = DATABASE_URL
    if "sslmode" not in dsn:
        dsn += ("&" if "?" in dsn else "?") + "sslmode=require"
    return psycopg2.connect(dsn)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    txt = str(value).strip()
    if not txt or txt == ".":
        return None
    txt = txt.replace(",", "").replace("%", "")
    try:
        return float(txt)
    except ValueError:
        return None


def _parse_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    txt = str(value).strip()
    if not txt:
        return None

    # Common formats first.
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S%z", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(txt, fmt)
            return dt.astimezone(UTC) if dt.tzinfo else dt.replace(tzinfo=UTC)
        except ValueError:
            pass

    try:
        dt = datetime.fromisoformat(txt.replace("Z", "+00:00"))
        return dt.astimezone(UTC) if dt.tzinfo else dt.replace(tzinfo=UTC)
    except ValueError:
        return None


def _row_hash(payload: dict[str, Any]) -> str:
    stable = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()


def _extract_rows(body: Any) -> list[dict[str, Any]]:
    if isinstance(body, list):
        return [r for r in body if isinstance(r, dict)]
    if isinstance(body, dict):
        for key in ("data", "results", "items", "rows"):
            value = body.get(key)
            if isinstance(value, list):
                return [r for r in value if isinstance(r, dict)]
    return []


def ensure_tables():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS macro_releases (
                    id                        BIGSERIAL PRIMARY KEY,
                    source                    TEXT        NOT NULL,
                    series_code               TEXT        NOT NULL,
                    event_name                TEXT        NOT NULL,
                    country                   TEXT        DEFAULT 'US',
                    release_time_utc          TIMESTAMPTZ NOT NULL,
                    period_date               DATE,
                    expected_value            FLOAT,
                    actual_value              FLOAT,
                    previous_value            FLOAT,
                    surprise_value            FLOAT,
                    unit                      TEXT,
                    importance                TEXT,
                    revision_count            INTEGER     NOT NULL DEFAULT 0,
                    row_hash                  TEXT        NOT NULL,
                    raw_payload               JSONB       NOT NULL DEFAULT '{}'::jsonb,
                    actual_first_available_at TIMESTAMPTZ,
                    first_seen_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    last_seen_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    CONSTRAINT macro_releases_unique
                        UNIQUE (source, series_code, release_time_utc, period_date)
                )
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_macro_releases_time
                ON macro_releases (release_time_utc DESC)
                """
            )
            # Safe migration for existing deployments: add column if absent.
            cur.execute(
                """
                ALTER TABLE macro_releases
                ADD COLUMN IF NOT EXISTS actual_first_available_at TIMESTAMPTZ
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS etf_flows (
                    id                   BIGSERIAL PRIMARY KEY,
                    source               TEXT        NOT NULL,
                    flow_date            DATE        NOT NULL,
                    ticker               TEXT        NOT NULL,
                    net_flow_usd         FLOAT,
                    net_flow_btc         FLOAT,
                    aum_usd              FLOAT,
                    volume_usd           FLOAT,
                    premium_discount_pct FLOAT,
                    revision_count       INTEGER     NOT NULL DEFAULT 0,
                    row_hash             TEXT        NOT NULL,
                    raw_payload          JSONB       NOT NULL DEFAULT '{}'::jsonb,
                    first_seen_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    last_seen_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    CONSTRAINT etf_flows_unique
                        UNIQUE (source, flow_date, ticker)
                )
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_etf_flows_date
                ON etf_flows (flow_date DESC)
                """
            )
        conn.commit()
    log.info("Ensured macro_releases and etf_flows tables exist.")


def _upsert_macro(cur, row: dict[str, Any]) -> str:
    cur.execute(
        """
        SELECT id, row_hash, actual_value, actual_first_available_at
        FROM macro_releases
        WHERE source = %s
          AND series_code = %s
          AND release_time_utc = %s
          AND period_date IS NOT DISTINCT FROM %s
        """,
        (row["source"], row["series_code"], row["release_time_utc"], row["period_date"]),
    )
    existing = cur.fetchone()

    if not existing:
        # actual_first_available_at logic on first insert:
        #   - If the row carries an explicit value (FRED rows), use it.
        #   - If actual_value is already present (historical calendar rows),
        #     fall back to release_time_utc as the best available proxy.
        #   - Otherwise leave NULL; it will be stamped on the first revision
        #     where actual transitions from NULL to a real number.
        actual_faa = row.get("actual_first_available_at")
        if actual_faa is None and row["actual_value"] is not None:
            actual_faa = row["release_time_utc"]

        cur.execute(
            """
            INSERT INTO macro_releases (
                source, series_code, event_name, country, release_time_utc,
                period_date, expected_value, actual_value, previous_value,
                surprise_value, unit, importance, row_hash, raw_payload,
                actual_first_available_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)
            """,
            (
                row["source"],
                row["series_code"],
                row["event_name"],
                row["country"],
                row["release_time_utc"],
                row["period_date"],
                row["expected_value"],
                row["actual_value"],
                row["previous_value"],
                row["surprise_value"],
                row["unit"],
                row["importance"],
                row["row_hash"],
                json.dumps(row["raw_payload"]),
                actual_faa,
            ),
        )
        return "inserted"

    existing_id, existing_hash, existing_actual, existing_faa = existing
    if existing_hash == row["row_hash"]:
        cur.execute(
            """
            UPDATE macro_releases
            SET last_seen_at = NOW()
            WHERE id = %s
            """,
            (existing_id,),
        )
        return "unchanged"

    # Stamp actual_first_available_at the moment actual goes NULL → non-null.
    # Once set it is never overwritten (preserves the first-observation time).
    set_faa = (
        existing_faa is None
        and existing_actual is None
        and row["actual_value"] is not None
    )

    cur.execute(
        f"""
        UPDATE macro_releases
        SET event_name = %s,
            country = %s,
            expected_value = %s,
            actual_value = %s,
            previous_value = %s,
            surprise_value = %s,
            unit = %s,
            importance = %s,
            row_hash = %s,
            raw_payload = %s::jsonb,
            revision_count = revision_count + 1,
            {"actual_first_available_at = NOW()," if set_faa else ""}
            last_seen_at = NOW()
        WHERE id = %s
        """,
        (
            row["event_name"],
            row["country"],
            row["expected_value"],
            row["actual_value"],
            row["previous_value"],
            row["surprise_value"],
            row["unit"],
            row["importance"],
            row["row_hash"],
            json.dumps(row["raw_payload"]),
            existing_id,
        ),
    )
    return "revised"


def _upsert_etf(cur, row: dict[str, Any]) -> str:
    cur.execute(
        """
        SELECT id, row_hash
        FROM etf_flows
        WHERE source = %s
          AND flow_date = %s
          AND ticker = %s
        """,
        (row["source"], row["flow_date"], row["ticker"]),
    )
    existing = cur.fetchone()

    if not existing:
        cur.execute(
            """
            INSERT INTO etf_flows (
                source, flow_date, ticker, net_flow_usd, net_flow_btc,
                aum_usd, volume_usd, premium_discount_pct, row_hash, raw_payload
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
            """,
            (
                row["source"],
                row["flow_date"],
                row["ticker"],
                row["net_flow_usd"],
                row["net_flow_btc"],
                row["aum_usd"],
                row["volume_usd"],
                row["premium_discount_pct"],
                row["row_hash"],
                json.dumps(row["raw_payload"]),
            ),
        )
        return "inserted"

    existing_hash = existing[1]
    if existing_hash == row["row_hash"]:
        cur.execute(
            """
            UPDATE etf_flows
            SET last_seen_at = NOW()
            WHERE id = %s
            """,
            (existing[0],),
        )
        return "unchanged"

    cur.execute(
        """
        UPDATE etf_flows
        SET net_flow_usd = %s,
            net_flow_btc = %s,
            aum_usd = %s,
            volume_usd = %s,
            premium_discount_pct = %s,
            row_hash = %s,
            raw_payload = %s::jsonb,
            revision_count = revision_count + 1,
            last_seen_at = NOW()
        WHERE id = %s
        """,
        (
            row["net_flow_usd"],
            row["net_flow_btc"],
            row["aum_usd"],
            row["volume_usd"],
            row["premium_discount_pct"],
            row["row_hash"],
            json.dumps(row["raw_payload"]),
            existing[0],
        ),
    )
    return "revised"


def _fetch_fred_rows() -> list[dict[str, Any]]:
    if not FRED_API_KEY:
        return []

    # release_lag_days: how many days after the period end date the data is
    #   typically published (approximate; avoids lookahead leakage in training).
    # release_hour_utc: the canonical announcement hour so the stored timestamp
    #   is a realistic proxy for when markets first see the data.
    series_map = {
        "DFEDTARU": {
            "event_name": "FED_RATE",
            "unit": "percent",
            "country": "US",
            "release_lag_days": 0,    # FOMC decision effective same day
            "release_hour_utc": 19,   # ~2 pm ET = 19:00 UTC
        },
        "CPIAUCSL": {
            "event_name": "CPI",
            "unit": "index",
            "country": "US",
            "release_lag_days": 45,   # released ~mid following month
            "release_hour_utc": 13,   # 8:30 am ET = 13:30 UTC
        },
        "PAYEMS": {
            "event_name": "NFP",
            "unit": "thousand_persons",
            "country": "US",
            "release_lag_days": 35,   # released first Friday of following month
            "release_hour_utc": 13,
        },
    }
    now = datetime.now(UTC)
    start = (now.replace(year=now.year - 10)).strftime("%Y-%m-%d")

    rows: list[dict[str, Any]] = []
    with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        for series_id, meta in series_map.items():
            resp = client.get(
                "https://api.stlouisfed.org/fred/series/observations",
                params={
                    "series_id": series_id,
                    "api_key": FRED_API_KEY,
                    "file_type": "json",
                    "sort_order": "asc",
                    "observation_start": start,
                    "limit": 1000,
                },
            )
            resp.raise_for_status()
            observations = resp.json().get("observations", [])

            prev_val: float | None = None
            for obs in observations:
                actual = _safe_float(obs.get("value"))
                if actual is None:
                    continue

                obs_dt = _parse_dt(obs.get("date"))
                if obs_dt is None:
                    continue

                # FED series is daily; keep only true changes to avoid noise.
                if series_id == "DFEDTARU" and prev_val is not None and abs(actual - prev_val) < 1e-9:
                    continue

                # Build the publication timestamp:
                #   obs_dt is the *period* date (e.g. Jan 1 for January CPI).
                #   Adding release_lag_days approximates the actual publication
                #   date, preventing lookahead leakage during training.
                period_date = obs_dt.date()
                release_dt = (obs_dt + timedelta(days=meta["release_lag_days"])).replace(
                    hour=meta["release_hour_utc"], minute=30, second=0, microsecond=0
                )

                surprise = None
                if prev_val is not None:
                    surprise = actual - prev_val

                payload = {
                    "source": "fred",
                    "series_code": series_id,
                    "event_name": meta["event_name"],
                    "country": meta["country"],
                    "release_time_utc": release_dt.isoformat(),
                    "period_date": period_date.isoformat(),
                    "expected_value": None,
                    "actual_value": actual,
                    "previous_value": prev_val,
                    "surprise_value": surprise,
                    "unit": meta["unit"],
                    "importance": "high",
                }
                rows.append(
                    {
                        "source": "fred",
                        "series_code": series_id,
                        "event_name": meta["event_name"],
                        "country": meta["country"],
                        "release_time_utc": release_dt,
                        "period_date": period_date,
                        "expected_value": None,
                        "actual_value": actual,
                        "previous_value": prev_val,
                        "surprise_value": surprise,
                        "unit": meta["unit"],
                        "importance": "high",
                        # FRED always has the actual; mark it available at publication.
                        "actual_first_available_at": release_dt,
                        "raw_payload": payload,
                        "row_hash": _row_hash(payload),
                    }
                )
                prev_val = actual

    return rows


def _fetch_calendar_rows() -> list[dict[str, Any]]:
    if not ECON_CALENDAR_URL:
        return []

    headers: dict[str, str] = {"Accept": "application/json"}
    if ECON_CALENDAR_API_KEY:
        headers["Authorization"] = f"Bearer {ECON_CALENDAR_API_KEY}"
        headers["X-API-Key"] = ECON_CALENDAR_API_KEY

    with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        resp = client.get(ECON_CALENDAR_URL, headers=headers)
        resp.raise_for_status()
        rows = _extract_rows(resp.json())

    mapped: list[dict[str, Any]] = []
    for r in rows:
        release_dt = _parse_dt(
            r.get("release_time")
            or r.get("releaseTime")
            or r.get("datetime")
            or r.get("date")
            or r.get("time")
        )
        if release_dt is None:
            continue

        event_name = (
            str(r.get("event_name") or r.get("event") or r.get("indicator") or "").strip()
            or "MACRO_EVENT"
        )
        series_code = (
            str(r.get("series_code") or r.get("symbol") or r.get("event_code") or event_name).strip()
        )

        expected = _safe_float(r.get("expected") or r.get("forecast"))
        actual = _safe_float(r.get("actual"))
        previous = _safe_float(r.get("previous"))
        surprise = _safe_float(r.get("surprise"))
        if surprise is None and expected is not None and actual is not None:
            surprise = actual - expected

        period = _parse_dt(r.get("period") or r.get("period_date") or r.get("reference_date"))
        period_date = period.date() if period else release_dt.date()

        source = str(r.get("source") or "calendar_api").strip().lower()
        payload = {
            "source": source,
            "series_code": series_code,
            "event_name": event_name,
            "country": str(r.get("country") or "US"),
            "release_time_utc": release_dt.isoformat(),
            "period_date": period_date.isoformat(),
            "expected_value": expected,
            "actual_value": actual,
            "previous_value": previous,
            "surprise_value": surprise,
            "unit": r.get("unit"),
            "importance": str(r.get("importance") or "").lower() or None,
        }
        # For calendar rows: actual_first_available_at is set to the scheduled
        # release time when actual is already present (historical backfill).
        # For future/pending events (actual=None) it stays None and will be
        # stamped at NOW() on the first revision that delivers the actual value.
        actual_faa = release_dt if actual is not None else None
        mapped.append(
            {
                "source": source,
                "series_code": series_code,
                "event_name": event_name,
                "country": str(r.get("country") or "US"),
                "release_time_utc": release_dt,
                "period_date": period_date,
                "expected_value": expected,
                "actual_value": actual,
                "previous_value": previous,
                "surprise_value": surprise,
                "unit": str(r.get("unit") or "") or None,
                "importance": str(r.get("importance") or "").lower() or None,
                "actual_first_available_at": actual_faa,
                "raw_payload": payload,
                "row_hash": _row_hash(payload),
            }
        )
    return mapped


def ingest_macro_releases():
    rows: list[dict[str, Any]] = []
    try:
        rows.extend(_fetch_fred_rows())
    except Exception as exc:
        log.error(f"FRED ingestion failed: {exc}")

    try:
        rows.extend(_fetch_calendar_rows())
    except Exception as exc:
        log.error(f"Calendar API ingestion failed: {exc}")

    if not rows:
        log.info("Macro ingestion: no rows fetched.")
        return

    inserted = 0
    revised = 0
    unchanged = 0
    with get_conn() as conn:
        with conn.cursor() as cur:
            for row in rows:
                status = _upsert_macro(cur, row)
                if status == "inserted":
                    inserted += 1
                elif status == "revised":
                    revised += 1
                else:
                    unchanged += 1
        conn.commit()
    log.info(
        f"Macro ingestion complete: fetched={len(rows)} inserted={inserted} "
        f"revised={revised} unchanged={unchanged}"
    )


def _fetch_etf_rows() -> list[dict[str, Any]]:
    if not ETF_FLOW_API_URL:
        return []

    headers: dict[str, str] = {"Accept": "application/json"}
    if ETF_FLOW_API_KEY:
        headers["Authorization"] = f"Bearer {ETF_FLOW_API_KEY}"
        headers["X-API-Key"] = ETF_FLOW_API_KEY

    with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        resp = client.get(ETF_FLOW_API_URL, headers=headers)
        resp.raise_for_status()
        records = _extract_rows(resp.json())

    mapped: list[dict[str, Any]] = []
    for r in records:
        flow_dt = _parse_dt(r.get("flow_date") or r.get("date") or r.get("timestamp"))
        if flow_dt is None:
            continue

        ticker = str(r.get("ticker") or r.get("symbol") or r.get("fund") or "").upper().strip()
        if not ticker:
            continue

        source = str(r.get("source") or "etf_api").strip().lower()
        payload = {
            "source": source,
            "flow_date": flow_dt.date().isoformat(),
            "ticker": ticker,
            "net_flow_usd": _safe_float(r.get("net_flow_usd") or r.get("netflow_usd") or r.get("net_flow") or r.get("flow_usd")),
            "net_flow_btc": _safe_float(r.get("net_flow_btc") or r.get("flow_btc")),
            "aum_usd": _safe_float(r.get("aum_usd") or r.get("aum")),
            "volume_usd": _safe_float(r.get("volume_usd") or r.get("volume")),
            "premium_discount_pct": _safe_float(r.get("premium_discount_pct") or r.get("premium_pct") or r.get("discount_pct")),
        }
        mapped.append(
            {
                "source": source,
                "flow_date": flow_dt.date(),
                "ticker": ticker,
                "net_flow_usd": payload["net_flow_usd"],
                "net_flow_btc": payload["net_flow_btc"],
                "aum_usd": payload["aum_usd"],
                "volume_usd": payload["volume_usd"],
                "premium_discount_pct": payload["premium_discount_pct"],
                "raw_payload": payload,
                "row_hash": _row_hash(payload),
            }
        )
    return mapped


def ingest_etf_flows():
    try:
        rows = _fetch_etf_rows()
    except Exception as exc:
        log.error(f"ETF ingestion failed: {exc}")
        return

    if not rows:
        log.info("ETF ingestion: no rows fetched.")
        return

    inserted = 0
    revised = 0
    unchanged = 0
    with get_conn() as conn:
        with conn.cursor() as cur:
            for row in rows:
                status = _upsert_etf(cur, row)
                if status == "inserted":
                    inserted += 1
                elif status == "revised":
                    revised += 1
                else:
                    unchanged += 1
        conn.commit()

    log.info(
        f"ETF ingestion complete: fetched={len(rows)} inserted={inserted} "
        f"revised={revised} unchanged={unchanged}"
    )


def _macro_loop():
    while True:
        try:
            ingest_macro_releases()
        except Exception as exc:
            log.error(f"Macro loop error: {exc}")
        time.sleep(ECON_POLL_SECONDS)


def _etf_loop():
    while True:
        try:
            ingest_etf_flows()
        except Exception as exc:
            log.error(f"ETF loop error: {exc}")
        time.sleep(ETF_POLL_SECONDS)


if __name__ == "__main__":
    log.info("Starting macro + ETF ingestor...")
    ensure_tables()

    if not FRED_API_KEY and not ECON_CALENDAR_URL:
        log.warning("No macro source configured. Set FRED_API_KEY or ECON_CALENDAR_URL.")
    if not ETF_FLOW_API_URL:
        log.warning("ETF_FLOW_API_URL not set. ETF ingestion will idle.")

    t_macro = threading.Thread(target=_macro_loop, daemon=True, name="macro-loop")
    t_etf = threading.Thread(target=_etf_loop, daemon=True, name="etf-loop")
    t_macro.start()
    t_etf.start()

    while True:
        time.sleep(3600)
