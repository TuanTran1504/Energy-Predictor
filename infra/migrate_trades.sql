CREATE TABLE IF NOT EXISTS trades (
    id              BIGSERIAL PRIMARY KEY,
    symbol          TEXT NOT NULL,              -- 'BTC', 'ETH'
    side            TEXT NOT NULL,              -- 'BUY' (long) or 'SELL' (short)
    status          TEXT NOT NULL DEFAULT 'OPEN', -- 'OPEN', 'CLOSED', 'CANCELLED'
    entry_price     FLOAT NOT NULL,
    exit_price      FLOAT,
    quantity        FLOAT NOT NULL,             -- in contracts (USDT-margined)
    leverage        INTEGER NOT NULL DEFAULT 5,
    stop_loss       FLOAT NOT NULL,
    take_profit     FLOAT NOT NULL,
    pnl_usdt        FLOAT,                      -- realised PnL in USDT
    pnl_pct         FLOAT,                      -- % return on margin
    confidence      FLOAT NOT NULL,             -- model confidence at entry
    horizon         INTEGER NOT NULL,           -- lookahead days (1 or 7)
    binance_order_id TEXT,                      -- Binance order ID
    close_reason    TEXT,                       -- 'take_profit', 'stop_loss', 'signal_flip', 'manual'
    opened_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    closed_at       TIMESTAMPTZ,
    notes           TEXT
);

CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades (symbol, opened_at DESC);
CREATE INDEX IF NOT EXISTS idx_trades_status  ON trades (status);
