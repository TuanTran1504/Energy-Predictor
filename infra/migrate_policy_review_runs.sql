-- Migration: policy review run history
-- Stores every scheduled policy-review decision (HOLD/ALLOW/PROPOSE).

CREATE TABLE IF NOT EXISTS policy_review_runs (
    id                         BIGSERIAL PRIMARY KEY,
    run_at                     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    engine_name                TEXT        NOT NULL,
    account_type               TEXT        NOT NULL,
    reviewer_model             TEXT,
    guard_decision             TEXT        NOT NULL CHECK (guard_decision IN ('HOLD', 'ALLOW_REVIEW')),
    guard_reason               TEXT        NOT NULL,
    llm_called                 BOOLEAN     NOT NULL DEFAULT FALSE,
    llm_decision               TEXT,
    llm_reason                 TEXT,
    proposed_patch             JSONB       NOT NULL DEFAULT '{}'::jsonb,
    guard_payload              JSONB       NOT NULL DEFAULT '{}'::jsonb,
    llm_payload                JSONB       NOT NULL DEFAULT '{}'::jsonb,
    active_policy_id           BIGINT,
    active_policy_version      INTEGER,
    hours_since_policy_update  FLOAT,
    closed_trades_since_update INTEGER,
    created_at                 TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_policy_review_runs_scope_time
    ON policy_review_runs (engine_name, account_type, run_at DESC);

CREATE INDEX IF NOT EXISTS idx_policy_review_runs_guard_decision
    ON policy_review_runs (guard_decision, run_at DESC);

CREATE INDEX IF NOT EXISTS idx_policy_review_runs_llm_decision
    ON policy_review_runs (llm_decision, run_at DESC);

CREATE INDEX IF NOT EXISTS idx_policy_review_runs_guard_payload
    ON policy_review_runs USING GIN (guard_payload);

CREATE INDEX IF NOT EXISTS idx_policy_review_runs_llm_payload
    ON policy_review_runs USING GIN (llm_payload);
