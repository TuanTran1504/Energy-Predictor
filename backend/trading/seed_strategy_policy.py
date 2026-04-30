"""
Seed a baseline runtime strategy policy from the current environment.

Examples:
  python seed_strategy_policy.py
  python seed_strategy_policy.py --engine-name llm_live --account-type live --reason "baseline from env"
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import psycopg2
from psycopg2.extras import Json
from dotenv import load_dotenv

from strategy_policy import snapshot_policy_from_environment


ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT / ".env")


def main():
    parser = argparse.ArgumentParser(description="Seed strategy_policies from current environment")
    parser.add_argument("--engine-name", default=os.getenv("STRATEGY_POLICY_ENGINE_NAME", "llm_live"))
    parser.add_argument("--account-type", default=os.getenv("STRATEGY_POLICY_ACCOUNT_TYPE", "live"))
    parser.add_argument("--policy-name", default="default")
    parser.add_argument("--source", default="manual")
    parser.add_argument("--created-by", default="seed_strategy_policy.py")
    parser.add_argument("--reason", default="Baseline policy seeded from current environment")
    parser.add_argument("--allow-risk-increase", action="store_true")
    parser.add_argument("--clear-signal", action="store_true")
    parser.add_argument("--evidence-trades", type=int, default=0)
    args = parser.parse_args()

    database_url = os.getenv("DATABASE_URL", "")
    if not database_url:
        raise SystemExit("DATABASE_URL is not configured")

    policy_json = snapshot_policy_from_environment()
    if not policy_json:
        raise SystemExit("No supported env-backed trading settings found to seed")

    validation_report = {
        "allow_risk_increase": bool(args.allow_risk_increase),
        "clear_signal": bool(args.clear_signal),
        "evidence_trades": int(args.evidence_trades),
        "seeded_from_env": True,
    }

    conn = psycopg2.connect(database_url, sslmode="require")
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COALESCE(MAX(version), 0)
                FROM strategy_policies
                WHERE policy_name = %s
                  AND engine_name = %s
                  AND account_type = %s
                """,
                (args.policy_name, args.engine_name, args.account_type),
            )
            version = int(cur.fetchone()[0]) + 1

            cur.execute(
                """
                UPDATE strategy_policies
                SET status = 'retired', updated_at = NOW()
                WHERE engine_name = %s
                  AND account_type = %s
                  AND status = 'active'
                """,
                (args.engine_name, args.account_type),
            )

            cur.execute(
                """
                INSERT INTO strategy_policies
                  (policy_name, engine_name, account_type, version, status,
                   policy_json, validation_report, reason, source, created_by,
                   effective_from, activated_at)
                VALUES
                  (%s, %s, %s, %s, 'active',
                   %s::jsonb, %s::jsonb, %s, %s, %s,
                   NOW(), NOW())
                RETURNING id
                """,
                (
                    args.policy_name,
                    args.engine_name,
                    args.account_type,
                    version,
                    Json(policy_json),
                    Json(validation_report),
                    args.reason,
                    args.source,
                    args.created_by,
                ),
            )
            row_id = int(cur.fetchone()[0])
        conn.commit()
    finally:
        conn.close()

    print(
        f"Seeded strategy policy id={row_id} "
        f"name={args.policy_name} engine={args.engine_name} "
        f"account={args.account_type} version={version}"
    )


if __name__ == "__main__":
    main()
