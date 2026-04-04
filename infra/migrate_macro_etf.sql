-- Migration: macro_releases + etf_flows tables
-- Safe to run on a fresh DB or against an existing one.
-- All statements are idempotent (CREATE IF NOT EXISTS / ADD COLUMN IF NOT EXISTS).

-- ── macro_releases ────────────────────────────────────────────────────────────
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
    -- Set once when actual_value is first observed; NULL until then.
    -- Used for point-in-time (as-of) joins in ML training.
    actual_first_available_at TIMESTAMPTZ,
    first_seen_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT macro_releases_unique
        UNIQUE (source, series_code, release_time_utc, period_date)
);

-- If the table already existed without actual_first_available_at, add it now.
ALTER TABLE macro_releases
    ADD COLUMN IF NOT EXISTS actual_first_available_at TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS idx_macro_releases_time
    ON macro_releases (release_time_utc DESC);

CREATE INDEX IF NOT EXISTS idx_macro_releases_event
    ON macro_releases (event_name, release_time_utc DESC);

-- ── etf_flows ─────────────────────────────────────────────────────────────────
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
);

CREATE INDEX IF NOT EXISTS idx_etf_flows_date
    ON etf_flows (flow_date DESC);

CREATE INDEX IF NOT EXISTS idx_etf_flows_ticker_date
    ON etf_flows (ticker, flow_date DESC);
