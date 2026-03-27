CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS grid_observations (
    id                    BIGSERIAL,
    timestamp             TIMESTAMPTZ     NOT NULL,
    settlement_period     INTEGER,
    actual_load_mw        FLOAT,
    generation_wind_mw    FLOAT,
    generation_solar_mw   FLOAT,
    generation_gas_mw     FLOAT,
    generation_nuclear_mw FLOAT,
    grid_imbalance_price  FLOAT,
    temperature_c         FLOAT,
    wind_speed_ms         FLOAT,
    cloud_cover_pct       FLOAT,
    hour                  INTEGER,
    weekday               INTEGER,
    month                 INTEGER,
    is_holiday            BOOLEAN DEFAULT FALSE,
    load_lag_1h           FLOAT,
    load_lag_24h          FLOAT,
    load_lag_168h         FLOAT,
    oil_price_brent       FLOAT,
    oil_delta_pct_30m     FLOAT,
    PRIMARY KEY (id, timestamp)
);

CREATE TABLE IF NOT EXISTS forecasts (
    id              BIGSERIAL,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    forecast_for    TIMESTAMPTZ     NOT NULL,
    forecast_mw     FLOAT           NOT NULL,
    lower_bound_mw  FLOAT           NOT NULL,
    upper_bound_mw  FLOAT           NOT NULL,
    confidence_pct  INTEGER         DEFAULT 90,
    shock_score     FLOAT           NOT NULL,
    shock_active    BOOLEAN         NOT NULL,
    oil_delta_pct   FLOAT,
    headline_score  FLOAT,
    actual_mw       FLOAT,
    mape_pct        FLOAT,
    model_version   TEXT,
    PRIMARY KEY (id, created_at)
);

CREATE TABLE IF NOT EXISTS shock_events (
    id               BIGSERIAL       PRIMARY KEY,
    started_at       TIMESTAMPTZ     NOT NULL,
    ended_at         TIMESTAMPTZ,
    peak_score       FLOAT,
    peak_oil_delta   FLOAT,
    trigger_headline TEXT,
    notes            TEXT
);

CREATE TABLE IF NOT EXISTS model_runs (
    id              BIGSERIAL       PRIMARY KEY,
    run_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    mlflow_run_id   TEXT            UNIQUE,
    mape_overall    FLOAT,
    mape_normal     FLOAT,
    mape_shock      FLOAT,
    train_samples   INTEGER,
    val_samples     INTEGER,
    shock_samples   INTEGER,
    promoted        BOOLEAN         DEFAULT FALSE,
    notes           TEXT
);

INSERT INTO shock_events (started_at, peak_score, peak_oil_delta, trigger_headline, notes)
VALUES (
    '2026-02-28 06:00:00+00',
    0.94,
    10.2,
    'US and Israel launch strikes against Iran nuclear facilities',
    'Start of 2026 Iran-US conflict'
)
ON CONFLICT DO NOTHING;