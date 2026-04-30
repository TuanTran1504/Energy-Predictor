-- Migration: runtime strategy policies
-- Safe for existing Postgres / Supabase databases.

CREATE TABLE IF NOT EXISTS strategy_policies (
    id                BIGSERIAL PRIMARY KEY,
    policy_name       TEXT        NOT NULL DEFAULT 'default',
    engine_name       TEXT        NOT NULL,
    account_type      TEXT        NOT NULL,
    version           INTEGER     NOT NULL,
    status            TEXT        NOT NULL DEFAULT 'draft'
                                  CHECK (status IN ('draft', 'validated', 'active', 'rejected', 'retired')),
    policy_json       JSONB       NOT NULL DEFAULT '{}'::jsonb,
    validation_report JSONB       NOT NULL DEFAULT '{}'::jsonb,
    reason            TEXT,
    source            TEXT        NOT NULL DEFAULT 'manual',
    created_by        TEXT,
    base_policy_id    BIGINT REFERENCES strategy_policies(id) ON DELETE SET NULL,
    effective_from    TIMESTAMPTZ,
    activated_at      TIMESTAMPTZ,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT strategy_policies_version_unique
        UNIQUE (policy_name, engine_name, account_type, version)
);

CREATE INDEX IF NOT EXISTS idx_strategy_policies_lookup
    ON strategy_policies (engine_name, account_type, status, effective_from DESC, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_strategy_policies_created_at
    ON strategy_policies (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_strategy_policies_policy_json
    ON strategy_policies USING GIN (policy_json);

CREATE UNIQUE INDEX IF NOT EXISTS idx_strategy_policies_one_active_per_scope
    ON strategy_policies (engine_name, account_type)
    WHERE status = 'active';
