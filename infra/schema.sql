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

CREATE TABLE IF NOT EXISTS crypto_prices (
    id              BIGSERIAL,
    symbol          TEXT            NOT NULL,
    fetched_at      TIMESTAMPTZ     NOT NULL,
    open_usd        FLOAT           NOT NULL,
    high_usd        FLOAT           NOT NULL,
    low_usd         FLOAT           NOT NULL,
    close_usd       FLOAT           NOT NULL,
    volume_usd      FLOAT           NOT NULL,
    change_pct      FLOAT,
    source          TEXT            DEFAULT 'binance',
    interval_minutes INTEGER        DEFAULT 1440,  -- 1440 minutes = 1 day
    PRIMARY KEY (id, fetched_at)
);
CREATE TABLE crypto_prices (
    id BIGSERIAL PRIMARY KEY,
    symbol TEXT,
    fetched_at TIMESTAMPTZ,
    open_usd FLOAT,
    high_usd FLOAT,
    low_usd FLOAT,
    close_usd FLOAT,
    volume_usd FLOAT,
    change_pct FLOAT,
    source TEXT DEFAULT 'binance',
    interval_minutes INTEGER DEFAULT 1440,  -- 30, 60, 240, 1440, etc.
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_crypto_prices_symbol_time ON crypto_prices (symbol, fetched_at DESC);
CREATE INDEX idx_crypto_prices_interval ON crypto_prices (interval_minutes);

CREATE TABLE IF NOT EXISTS fear_greed_index (
    date        DATE        PRIMARY KEY,
    value       FLOAT       NOT NULL,  -- 0-100 raw value
    fetched_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS funding_rates (
    symbol      TEXT        NOT NULL,
    date        DATE        NOT NULL,
    rate_avg    FLOAT       NOT NULL,  -- daily average of 3 readings
    fetched_at  TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (symbol, date)
);