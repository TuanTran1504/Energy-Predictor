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

CREATE TABLE IF NOT EXISTS vnd_rates (
    id          BIGSERIAL PRIMARY KEY,
    fetched_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    usd_to_vnd  FLOAT NOT NULL
);

CREATE INDEX idx_vnd_rates_fetched_at 
ON vnd_rates (fetched_at DESC);

CREATE TABLE IF NOT EXISTS macro_events (
    id           BIGSERIAL PRIMARY KEY,
    event_date   TIMESTAMPTZ NOT NULL,
    event_type   TEXT NOT NULL,  -- 'FED_RATE', 'CPI', 'NFP', 'HALVING'
    description  TEXT,
    expected     TEXT,           -- market consensus before event
    actual       TEXT,           -- filled in after event
    impact_pct   FLOAT,          -- actual BTC move within 24h
    CONSTRAINT macro_events_date_type_unique UNIQUE (event_date, event_type)
);

DROP TABLE shock_events;

CREATE TABLE shock_events (
    id              BIGSERIAL PRIMARY KEY,
    event_date      TIMESTAMPTZ NOT NULL,
    description     TEXT NOT NULL,
    severity        TEXT NOT NULL CHECK (severity IN ('LOW', 'ELEVATED', 'HIGH')),
    oil_impact      FLOAT DEFAULT 0,
    trigger_headline TEXT,
    peak_score      FLOAT,
    notes           TEXT,
    CONSTRAINT shock_events_date_unique UNIQUE (event_date)
);

CREATE INDEX idx_shock_events_date ON shock_events (event_date DESC);