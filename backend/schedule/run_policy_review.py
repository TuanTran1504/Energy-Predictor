"""
run_policy_review.py — guarded strategy policy reviewer.

Flow:
  1) Evaluate deterministic maturity gate (hours + sample size + cooldown).
  2) If HOLD -> persist run and exit (no LLM call).
  3) If ALLOW_REVIEW -> summarize recent trade performance and ask LLM for a JSON patch.
  4) Validate patch with strict bounds and conservative risk rules.
  5) Optionally auto-activate validated policy (hot-reload picked up by engine on next cycle).
  6) Persist review run details to policy_review_runs.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg2
from dotenv import load_dotenv
from openai import OpenAI
from psycopg2.extras import Json, RealDictCursor

from policy_review_guard import ReviewGuardConfig, evaluate_policy_review_guard


UTC = timezone.utc
ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT / ".env")

log = logging.getLogger(__name__)


def _env_bool(value: str | bool | int | None, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _json_default(value: Any):
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC).isoformat()
    return str(value)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, default=_json_default)


def _safe_table_name(name: str, fallback: str = "trades_live") -> str:
    text = str(name or "").strip()
    if not text:
        return fallback
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", text):
        return text
    return fallback


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _get_path(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _set_path(data: dict[str, Any], path: tuple[str, ...], value: Any):
    cur = data
    for key in path[:-1]:
        child = cur.get(key)
        if not isinstance(child, dict):
            child = {}
            cur[key] = child
        cur = child
    cur[path[-1]] = value


def _del_path(data: dict[str, Any], path: tuple[str, ...]):
    if not path:
        return
    cur: Any = data
    for key in path[:-1]:
        if not isinstance(cur, dict) or key not in cur:
            return
        cur = cur[key]
    if isinstance(cur, dict):
        cur.pop(path[-1], None)


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(_json_dumps(base))
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _flatten_patch(data: Any, prefix: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], Any]]:
    rows: list[tuple[tuple[str, ...], Any]] = []
    if isinstance(data, dict):
        for k, v in data.items():
            rows.extend(_flatten_patch(v, prefix + (str(k),)))
        return rows
    rows.append((prefix, data))
    return rows


@dataclass(frozen=True)
class PolicyKeySpec:
    kind: str
    min_value: float | int | None = None
    max_value: float | int | None = None


ALLOWED_POLICY_KEYS: dict[tuple[str, ...], PolicyKeySpec] = {
    ("global", "fg_extreme_block"): PolicyKeySpec("bool"),
    ("global", "fg_extreme_fear_threshold"): PolicyKeySpec("int", 0, 100),
    ("global", "fg_extreme_greed_threshold"): PolicyKeySpec("int", 0, 100),
    ("global", "tech_allow_sideway"): PolicyKeySpec("bool"),
    ("global", "tech_score_threshold"): PolicyKeySpec("int", 1, 5),
    ("global", "tech_score_threshold_range"): PolicyKeySpec("int", 1, 5),
    ("global", "tech_rollover_score_threshold"): PolicyKeySpec("int", 0, 5),
    ("global", "sl_atr_multiplier"): PolicyKeySpec("float", 0.5, 4.0),
    ("global", "sl_sr_buffer_atr_mult"): PolicyKeySpec("float", 0.0, 1.0),
    ("global", "sl_atr_dynamic_mult"): PolicyKeySpec("float", 0.5, 6.0),
    ("global", "sl_max_pct_ceiling"): PolicyKeySpec("float", 0.003, 0.08),
    ("global", "trade_min_rr"): PolicyKeySpec("float", 0.8, 4.0),
    ("global", "trade_min_rr_range"): PolicyKeySpec("float", 0.8, 4.0),
    ("global", "tp_extension_atr_mult"): PolicyKeySpec("float", 0.0, 1.5),
    ("global", "setup_e_min_rr"): PolicyKeySpec("float", 0.8, 3.0),
    ("symbols", "BTC", "enabled"): PolicyKeySpec("bool"),
    ("symbols", "BTC", "position_risk_pct"): PolicyKeySpec("float", 0.001, 0.03),
    ("symbols", "BTC", "max_position_fraction"): PolicyKeySpec("float", 0.02, 0.40),
    ("symbols", "BTC", "break_even_trigger_r_mult"): PolicyKeySpec("float", 0.1, 3.0),
    ("symbols", "BTC", "high_atr_pct"): PolicyKeySpec("float", 0.005, 0.20),
    ("symbols", "BTC", "high_atr_min_adx"): PolicyKeySpec("float", 5.0, 80.0),
    ("symbols", "BTC", "range_disable_atr_pct"): PolicyKeySpec("float", 0.002, 0.15),
    ("symbols", "BTC", "trail_lookback_bars"): PolicyKeySpec("int", 2, 8),
    ("symbols", "BTC", "trail_stop_atr_mult"): PolicyKeySpec("float", 0.05, 2.0),
    ("symbols", "BTC", "trail_mark_buffer_pct"): PolicyKeySpec("float", 0.0, 0.02),
    ("symbols", "ETH", "enabled"): PolicyKeySpec("bool"),
    ("symbols", "SOL", "enabled"): PolicyKeySpec("bool"),
}

RISK_UP_DIRECTION_HIGHER = "higher_is_riskier"
RISK_UP_DIRECTION_LOWER = "lower_is_riskier"

RISK_RULES: dict[tuple[str, ...], str] = {
    ("global", "tech_allow_sideway"): RISK_UP_DIRECTION_HIGHER,
    ("global", "fg_extreme_block"): RISK_UP_DIRECTION_LOWER,
    ("global", "tech_score_threshold"): RISK_UP_DIRECTION_LOWER,
    ("global", "tech_score_threshold_range"): RISK_UP_DIRECTION_LOWER,
    ("global", "tech_rollover_score_threshold"): RISK_UP_DIRECTION_LOWER,
    ("global", "trade_min_rr"): RISK_UP_DIRECTION_LOWER,
    ("global", "trade_min_rr_range"): RISK_UP_DIRECTION_LOWER,
    ("global", "setup_e_min_rr"): RISK_UP_DIRECTION_LOWER,
    ("symbols", "BTC", "position_risk_pct"): RISK_UP_DIRECTION_HIGHER,
    ("symbols", "BTC", "max_position_fraction"): RISK_UP_DIRECTION_HIGHER,
    ("symbols", "BTC", "break_even_trigger_r_mult"): RISK_UP_DIRECTION_HIGHER,
    ("symbols", "BTC", "trail_stop_atr_mult"): RISK_UP_DIRECTION_HIGHER,
}


def _coerce_by_spec(value: Any, spec: PolicyKeySpec) -> Any:
    if spec.kind == "bool":
        return _env_bool(value)
    if spec.kind == "int":
        parsed = int(value)
        if spec.min_value is not None and parsed < int(spec.min_value):
            raise ValueError(f"value {parsed} < min {spec.min_value}")
        if spec.max_value is not None and parsed > int(spec.max_value):
            raise ValueError(f"value {parsed} > max {spec.max_value}")
        return parsed
    if spec.kind == "float":
        parsed = float(value)
        if spec.min_value is not None and parsed < float(spec.min_value):
            raise ValueError(f"value {parsed} < min {spec.min_value}")
        if spec.max_value is not None and parsed > float(spec.max_value):
            raise ValueError(f"value {parsed} > max {spec.max_value}")
        return round(parsed, 8)
    return value


def _is_risk_increase(base_value: Any, new_value: Any, direction: str) -> bool:
    if base_value is None or new_value is None:
        return False
    if direction == RISK_UP_DIRECTION_HIGHER:
        return new_value > base_value
    return new_value < base_value


def _is_allowed_dynamic_key(path: tuple[str, ...]) -> bool:
    if len(path) != 3:
        return False
    if path[0] == "setups" and path[2] == "enabled":
        return bool(re.match(r"^[A-Z]$", path[1]))
    return False


def _normalize_dynamic_key(path: tuple[str, ...], value: Any) -> tuple[tuple[str, ...], Any] | None:
    if len(path) != 3:
        return None
    head, mid, tail = path
    if head == "setups" and tail == "enabled":
        key = str(mid).upper().strip()[:1]
        if not key or not re.match(r"^[A-Z]$", key):
            return None
        return ("setups", key, "enabled"), _env_bool(value)
    return None


def validate_policy_patch(
    patch: dict[str, Any],
    active_policy_json: dict[str, Any],
) -> tuple[dict[str, Any], list[str], list[str]]:
    errors: list[str] = []
    sanitized: dict[str, Any] = {}
    risk_increase_changes: list[str] = []

    for path, raw_value in _flatten_patch(patch):
        if not path:
            continue
        spec = ALLOWED_POLICY_KEYS.get(path)
        norm_path = path
        value = raw_value

        if spec is None:
            if _is_allowed_dynamic_key(path):
                dynamic = _normalize_dynamic_key(path, raw_value)
                if dynamic is None:
                    errors.append(f"Invalid dynamic key path={'.'.join(path)}")
                    continue
                norm_path, value = dynamic
                _set_path(sanitized, norm_path, value)
                continue
            errors.append(f"Unknown policy key path={'.'.join(path)}")
            continue

        try:
            value = _coerce_by_spec(raw_value, spec)
        except Exception as exc:
            errors.append(f"Invalid value for {'.'.join(path)}: {exc}")
            continue

        _set_path(sanitized, norm_path, value)

    merged = _deep_merge(active_policy_json or {}, sanitized)
    for path, direction in RISK_RULES.items():
        old_value = _get_path(active_policy_json or {}, path)
        new_value = _get_path(merged, path)
        if _is_risk_increase(old_value, new_value, direction):
            risk_increase_changes.append(f"{'.'.join(path)}: {old_value!r} -> {new_value!r}")

    return sanitized, errors, risk_increase_changes


def strip_risk_increase_changes(
    sanitized_patch: dict[str, Any],
    active_policy_json: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    out = json.loads(_json_dumps(sanitized_patch))
    stripped: list[str] = []
    merged = _deep_merge(active_policy_json or {}, out)
    for path, direction in RISK_RULES.items():
        old_value = _get_path(active_policy_json or {}, path)
        new_value = _get_path(merged, path)
        if not _is_risk_increase(old_value, new_value, direction):
            continue
        _del_path(out, path)
        stripped.append(f"{'.'.join(path)}: {old_value!r} -> {new_value!r}")
    return out, stripped


def _is_patch_empty(patch: dict[str, Any]) -> bool:
    return len(_flatten_patch(patch)) == 0


def _fetch_active_policy_row(conn, engine_name: str, account_type: str) -> dict[str, Any] | None:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT id, policy_name, engine_name, account_type, version, status,
                   policy_json, validation_report, source, reason,
                   created_at, updated_at, effective_from, activated_at
            FROM strategy_policies
            WHERE engine_name = %s
              AND account_type = %s
              AND status = 'active'
              AND (effective_from IS NULL OR effective_from <= NOW())
            ORDER BY COALESCE(effective_from, created_at) DESC, version DESC
            LIMIT 1
            """,
            (engine_name, account_type),
        )
        row = cur.fetchone()
        return dict(row) if row else None


def _fetch_recent_setup_symbol_stats(
    conn,
    trades_table: str,
    account_type: str,
    limit_rows: int = 50,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "last_closed_trades_by_symbol_setup": [],
        "last_losing_trades_patterns": [],
    }
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            f"""
            WITH last_closed AS (
                SELECT symbol, COALESCE(setup, 'UNKNOWN') AS setup, pnl_usdt, closed_at
                FROM {trades_table}
                WHERE account_type = %s
                  AND status = 'CLOSED'
                  AND pnl_usdt IS NOT NULL
                ORDER BY closed_at DESC
                LIMIT %s
            )
            SELECT
                symbol,
                setup,
                COUNT(*)::INT AS closed_trades,
                SUM(CASE WHEN pnl_usdt > 0 THEN 1 ELSE 0 END)::INT AS wins,
                ROUND((SUM(CASE WHEN pnl_usdt > 0 THEN 1 ELSE 0 END)::numeric
                     / NULLIF(COUNT(*), 0)) * 100, 2) AS win_rate_pct,
                ROUND(SUM(pnl_usdt)::numeric, 4) AS total_pnl_usdt
            FROM last_closed
            GROUP BY symbol, setup
            ORDER BY total_pnl_usdt ASC, closed_trades DESC, symbol ASC, setup ASC
            """,
            (account_type, int(limit_rows)),
        )
        out["last_closed_trades_by_symbol_setup"] = [dict(r) for r in cur.fetchall()]

        cur.execute(
            f"""
            WITH last_losses AS (
                SELECT symbol, COALESCE(setup, 'UNKNOWN') AS setup, pnl_usdt, closed_at
                FROM {trades_table}
                WHERE account_type = %s
                  AND status = 'CLOSED'
                  AND pnl_usdt < 0
                ORDER BY closed_at DESC
                LIMIT %s
            )
            SELECT
                symbol,
                setup,
                COUNT(*)::INT AS losing_trades,
                ROUND(SUM(pnl_usdt)::numeric, 4) AS losing_pnl_usdt
            FROM last_losses
            GROUP BY symbol, setup
            ORDER BY losing_trades DESC, losing_pnl_usdt ASC, symbol ASC, setup ASC
            """,
            (account_type, int(limit_rows)),
        )
        out["last_losing_trades_patterns"] = [dict(r) for r in cur.fetchall()]
    return out


def _get_llm_client_and_model() -> tuple[OpenAI, str, str]:
    provider = os.getenv("POLICY_REVIEW_LLM_PROVIDER", "openai").strip().lower()

    if provider in {"google", "gemini"}:
        api_key = os.getenv("GOOGLE_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is required for POLICY_REVIEW_LLM_PROVIDER=google")
        base_url = os.getenv(
            "POLICY_REVIEW_LLM_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/openai/",
        ).strip()
        model = os.getenv("POLICY_REVIEW_MODEL", "gemini-2.5-pro").strip()
        return OpenAI(api_key=api_key, base_url=base_url), provider, model

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for policy review")
    base_url = os.getenv("POLICY_REVIEW_LLM_BASE_URL", "").strip() or None
    model = os.getenv("POLICY_REVIEW_MODEL", "gpt-5-mini").strip()
    return OpenAI(api_key=api_key, base_url=base_url), provider, model


def _build_reviewer_prompts(review_input: dict[str, Any]) -> tuple[str, str]:
    system_prompt = (
        "You are a conservative crypto strategy reviewer. "
        "Return strict JSON only. Prefer NO_CHANGE unless evidence is strong. "
        "Do not optimize for trade frequency. Avoid overfitting short windows. "
        "Never suggest risk increases unless clear positive evidence exists after enough samples."
    )

    user_prompt = f"""Review this strategy context and propose either NO_CHANGE or a minimal PATCH.

Scope:
- Engine/account: {review_input.get("engine_name")} / {review_input.get("account_type")}
- Policy cadence: 6h reviewer, but policy changes should be rare and evidence-based.
- Hard behavior:
  1) If sample since last policy update is too small, return NO_CHANGE.
  2) Prefer risk reduction over risk increase when performance degrades.
  3) If proposing risk increase, keep it very small and justify with strong evidence.
  4) Minimize number of changed keys (1-3 keys max unless clearly needed).

Allowed patch paths:
- global.<key> from current runtime policy config
- symbols.BTC.<key> for BTC risk controls
- symbols.<BTC|ETH|SOL>.enabled
- setups.<A..Z>.enabled

Return JSON exactly in this schema:
{{
  "decision": "NO_CHANGE or PROPOSE_CHANGE",
  "reason": "short explanation",
  "confidence": 0.0,
  "patch": {{}},
  "notes": {{
    "risk_increase_requested": false,
    "review_focus": ["optional short bullets"]
  }}
}}

Context JSON:
{_json_dumps(review_input)}
"""
    return system_prompt, user_prompt


def _call_llm_reviewer(review_input: dict[str, Any]) -> dict[str, Any]:
    client, provider, model = _get_llm_client_and_model()
    system_prompt, user_prompt = _build_reviewer_prompts(review_input)

    response = client.chat.completions.create(
        model=model,
        temperature=0.1,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    raw = (response.choices[0].message.content or "").strip()
    parsed = json.loads(raw) if raw else {}
    parsed["_meta"] = {
        "provider": provider,
        "model": model,
    }
    return parsed


def _compute_clear_signal_for_risk_increase(
    guard_context: dict[str, Any],
    min_evidence_trades: int,
) -> tuple[bool, dict[str, Any]]:
    post_update = dict(guard_context.get("post_update_sample") or {})
    windows = dict(guard_context.get("performance_windows") or {})
    last_50 = dict(windows.get("last_50_trades") or {})
    last_7d = dict(windows.get("last_7_days") or {})

    closed_since_update = _to_int(post_update.get("closed_trades_since_update"), 0)
    closed_50 = _to_int(last_50.get("closed_trades"), 0)
    win_rate_50 = _to_float(last_50.get("win_rate_pct"), 0.0)
    pnl_50 = _to_float(last_50.get("total_pnl_usdt"), 0.0)
    pnl_7d = _to_float(last_7d.get("total_pnl_usdt"), 0.0)

    clear_signal = (
        closed_since_update >= min_evidence_trades
        and closed_50 >= min(50, min_evidence_trades)
        and win_rate_50 >= 57.0
        and pnl_50 > 0.0
        and pnl_7d >= 0.0
    )
    evidence = {
        "closed_trades_since_update": closed_since_update,
        "last_50_closed_trades": closed_50,
        "last_50_win_rate_pct": round(win_rate_50, 2),
        "last_50_total_pnl_usdt": round(pnl_50, 4),
        "last_7d_total_pnl_usdt": round(pnl_7d, 4),
        "min_evidence_trades_required": int(min_evidence_trades),
    }
    return clear_signal, evidence


def _insert_policy_review_run(conn, payload: dict[str, Any]) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO policy_review_runs
              (engine_name, account_type, reviewer_model,
               guard_decision, guard_reason,
               llm_called, llm_decision, llm_reason,
               proposed_patch, guard_payload, llm_payload,
               active_policy_id, active_policy_version,
               hours_since_policy_update, closed_trades_since_update)
            VALUES
              (%s, %s, %s,
               %s, %s,
               %s, %s, %s,
               %s::jsonb, %s::jsonb, %s::jsonb,
               %s, %s,
               %s, %s)
            RETURNING id
            """,
            (
                payload["engine_name"],
                payload["account_type"],
                payload.get("reviewer_model"),
                payload["guard_decision"],
                payload["guard_reason"],
                bool(payload.get("llm_called", False)),
                payload.get("llm_decision"),
                payload.get("llm_reason"),
                Json(payload.get("proposed_patch") or {}, dumps=_json_dumps),
                Json(payload.get("guard_payload") or {}, dumps=_json_dumps),
                Json(payload.get("llm_payload") or {}, dumps=_json_dumps),
                payload.get("active_policy_id"),
                payload.get("active_policy_version"),
                payload.get("hours_since_policy_update"),
                payload.get("closed_trades_since_update"),
            ),
        )
        row = cur.fetchone()
    return int(row[0])


def _create_policy_version(
    conn,
    *,
    active_policy: dict[str, Any],
    merged_policy_json: dict[str, Any],
    validation_report: dict[str, Any],
    reason: str,
    reviewer_model: str,
    auto_activate: bool,
) -> tuple[int, int, str]:
    policy_name = str(active_policy.get("policy_name") or "default")
    engine_name = str(active_policy["engine_name"])
    account_type = str(active_policy["account_type"])

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COALESCE(MAX(version), 0)
            FROM strategy_policies
            WHERE policy_name = %s
              AND engine_name = %s
              AND account_type = %s
            """,
            (policy_name, engine_name, account_type),
        )
        next_version = int(cur.fetchone()[0]) + 1

        status = "active" if auto_activate else "validated"
        if auto_activate:
            cur.execute(
                """
                UPDATE strategy_policies
                SET status = 'retired', updated_at = NOW()
                WHERE engine_name = %s
                  AND account_type = %s
                  AND status = 'active'
                """,
                (engine_name, account_type),
            )

        cur.execute(
            """
            INSERT INTO strategy_policies
              (policy_name, engine_name, account_type, version, status,
               policy_json, validation_report, reason, source, created_by,
               base_policy_id, effective_from, activated_at)
            VALUES
              (%s, %s, %s, %s, %s,
               %s::jsonb, %s::jsonb, %s, %s, %s,
               %s, NOW(), %s)
            RETURNING id
            """,
            (
                policy_name,
                engine_name,
                account_type,
                next_version,
                status,
                Json(merged_policy_json, dumps=_json_dumps),
                Json(validation_report, dumps=_json_dumps),
                reason[:500],
                "llm_reviewer",
                reviewer_model[:120],
                int(active_policy["id"]),
                datetime.now(UTC) if auto_activate else None,
            ),
        )
        new_policy_id = int(cur.fetchone()[0])
    return new_policy_id, next_version, status


def run_policy_review_once() -> dict[str, Any]:
    if not _env_bool(os.getenv("POLICY_REVIEW_ENABLED", "1"), default=True):
        return {
            "status": "disabled",
            "message": "POLICY_REVIEW_ENABLED=0",
        }

    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        raise RuntimeError("DATABASE_URL is required")

    engine_name = os.getenv("STRATEGY_POLICY_ENGINE_NAME", "llm_live").strip()
    account_type = os.getenv("STRATEGY_POLICY_ACCOUNT_TYPE", "live").strip()
    trades_table = _safe_table_name(os.getenv("POLICY_REVIEW_TRADES_TABLE", "trades_live"))

    cfg = ReviewGuardConfig(
        min_hours_since_update=_to_float(
            os.getenv("POLICY_REVIEW_MIN_HOURS_SINCE_UPDATE", "24"),
            24.0,
        ),
        min_closed_trades_since_update=_to_int(
            os.getenv("POLICY_REVIEW_MIN_CLOSED_TRADES_SINCE_UPDATE", "20"),
            20,
        ),
        cooldown_hours_since_last_change=_to_float(
            os.getenv("POLICY_REVIEW_COOLDOWN_HOURS", "12"),
            12.0,
        ),
        trades_table=trades_table,
    )
    guard = evaluate_policy_review_guard(
        database_url=database_url,
        engine_name=engine_name,
        account_type=account_type,
        config=cfg,
    )
    guard_payload = asdict(guard)
    guard_ctx = dict(guard_payload.get("context") or {})
    policy_timing = dict(guard_ctx.get("policy_timing") or {})
    post_update = dict(guard_ctx.get("post_update_sample") or {})
    active_policy_ctx = dict(guard_ctx.get("active_policy") or {})

    base_run_payload = {
        "engine_name": engine_name,
        "account_type": account_type,
        "reviewer_model": None,
        "guard_decision": guard.decision,
        "guard_reason": guard.reason,
        "llm_called": False,
        "llm_decision": "NO_CALL",
        "llm_reason": "",
        "proposed_patch": {},
        "guard_payload": guard_payload,
        "llm_payload": {},
        "active_policy_id": active_policy_ctx.get("id"),
        "active_policy_version": active_policy_ctx.get("version"),
        "hours_since_policy_update": policy_timing.get("hours_since_update"),
        "closed_trades_since_update": post_update.get("closed_trades_since_update"),
    }

    conn = psycopg2.connect(database_url, sslmode="require")
    try:
        active_row = _fetch_active_policy_row(conn, engine_name, account_type)
        if not active_row:
            payload = dict(base_run_payload)
            payload["llm_reason"] = "No active policy found in DB scope"
            run_id = _insert_policy_review_run(conn, payload)
            conn.commit()
            return {"status": "hold", "run_id": run_id, "reason": payload["llm_reason"]}

        if guard.decision != "ALLOW_REVIEW":
            payload = dict(base_run_payload)
            payload["llm_reason"] = guard.reason
            run_id = _insert_policy_review_run(conn, payload)
            conn.commit()
            return {"status": "hold", "run_id": run_id, "reason": guard.reason}

        try:
            extra_stats = _fetch_recent_setup_symbol_stats(conn, trades_table, account_type, limit_rows=50)
        except Exception as exc:
            log.warning(f"policy review trade pattern query failed: {exc}")
            extra_stats = {
                "last_closed_trades_by_symbol_setup": [],
                "last_losing_trades_patterns": [],
                "error": str(exc),
            }
        review_input = {
            "engine_name": engine_name,
            "account_type": account_type,
            "guard": guard_payload,
            "active_policy": {
                "id": int(active_row["id"]),
                "policy_name": active_row.get("policy_name"),
                "version": int(active_row.get("version") or 0),
                "policy_json": dict(active_row.get("policy_json") or {}),
                "validation_report": dict(active_row.get("validation_report") or {}),
                "activated_at": active_row.get("activated_at"),
                "effective_from": active_row.get("effective_from"),
            },
            "trade_patterns": extra_stats,
        }

        try:
            reviewer_output = _call_llm_reviewer(review_input)
        except Exception as exc:
            payload = dict(base_run_payload)
            payload["reviewer_model"] = os.getenv("POLICY_REVIEW_MODEL", "unknown")
            payload["llm_called"] = False
            payload["llm_decision"] = "ERROR"
            payload["llm_reason"] = str(exc)[:500]
            payload["llm_payload"] = {
                "error": str(exc),
                "review_input": review_input,
            }
            run_id = _insert_policy_review_run(conn, payload)
            conn.commit()
            return {
                "status": "error",
                "run_id": run_id,
                "reason": payload["llm_reason"],
            }
        meta = dict(reviewer_output.get("_meta") or {})
        reviewer_model = str(meta.get("model") or os.getenv("POLICY_REVIEW_MODEL", "unknown"))
        decision = str(reviewer_output.get("decision") or "NO_CHANGE").strip().upper()
        llm_reason = str(reviewer_output.get("reason") or "").strip()[:500]
        patch = reviewer_output.get("patch")
        patch = patch if isinstance(patch, dict) else {}

        payload = dict(base_run_payload)
        payload["reviewer_model"] = reviewer_model
        payload["llm_called"] = True
        payload["llm_decision"] = decision
        payload["llm_reason"] = llm_reason
        payload["proposed_patch"] = patch
        payload["llm_payload"] = reviewer_output

        if decision != "PROPOSE_CHANGE":
            run_id = _insert_policy_review_run(conn, payload)
            conn.commit()
            return {"status": "no_change", "run_id": run_id, "reason": llm_reason or "NO_CHANGE"}

        active_policy_json = dict(active_row.get("policy_json") or {})
        sanitized_patch, validation_errors, risk_increase_changes = validate_policy_patch(
            patch=patch,
            active_policy_json=active_policy_json,
        )

        clear_signal, evidence = _compute_clear_signal_for_risk_increase(
            guard_ctx,
            min_evidence_trades=max(cfg.min_closed_trades_since_update, 30),
        )

        stripped_risk_changes: list[str] = []
        if risk_increase_changes and not clear_signal:
            sanitized_patch, stripped_risk_changes = strip_risk_increase_changes(
                sanitized_patch=sanitized_patch,
                active_policy_json=active_policy_json,
            )

        if validation_errors or _is_patch_empty(sanitized_patch):
            payload["llm_decision"] = "REJECTED_VALIDATION"
            notes = {
                "validation_errors": validation_errors,
                "risk_increase_changes": risk_increase_changes,
                "risk_increase_stripped": stripped_risk_changes,
                "clear_signal_for_risk_increase": clear_signal,
                "clear_signal_evidence": evidence,
            }
            payload["llm_reason"] = (
                "No effective validated patch"
                if not validation_errors
                else ("; ".join(validation_errors))[:500]
            )
            payload["proposed_patch"] = sanitized_patch
            payload["llm_payload"] = {
                "reviewer_output": reviewer_output,
                "validation": notes,
            }
            run_id = _insert_policy_review_run(conn, payload)
            conn.commit()
            return {
                "status": "rejected",
                "run_id": run_id,
                "reason": payload["llm_reason"],
                "validation": notes,
            }

        merged_policy = _deep_merge(active_policy_json, sanitized_patch)
        auto_apply = _env_bool(os.getenv("POLICY_REVIEW_AUTO_APPLY", "0"), default=False)
        allow_risk_increase = bool(risk_increase_changes) and clear_signal
        validation_report = {
            "review_type": "llm_policy_review",
            "allow_risk_increase": bool(allow_risk_increase),
            "clear_signal": bool(clear_signal),
            "evidence_trades": int(evidence.get("closed_trades_since_update", 0)),
            "risk_increase_changes": risk_increase_changes,
            "risk_increase_stripped": stripped_risk_changes,
            "validation_errors": validation_errors,
            "reviewer_model": reviewer_model,
            "reviewed_at": datetime.now(UTC),
            "guard_decision": guard.decision,
            "guard_reason": guard.reason,
            "guard_config": guard.guard_config,
        }
        new_policy_id, new_policy_version, new_status = _create_policy_version(
            conn,
            active_policy=active_row,
            merged_policy_json=merged_policy,
            validation_report=validation_report,
            reason=llm_reason or "LLM policy review proposal",
            reviewer_model=reviewer_model,
            auto_activate=auto_apply,
        )

        payload["llm_decision"] = "APPLIED" if auto_apply else "VALIDATED"
        payload["llm_reason"] = (
            f"policy_id={new_policy_id} version={new_policy_version} status={new_status}"
        )
        payload["proposed_patch"] = sanitized_patch
        payload["llm_payload"] = {
            "reviewer_output": reviewer_output,
            "validation": {
                "validation_errors": validation_errors,
                "risk_increase_changes": risk_increase_changes,
                "risk_increase_stripped": stripped_risk_changes,
                "clear_signal_for_risk_increase": clear_signal,
                "clear_signal_evidence": evidence,
            },
            "new_policy": {
                "id": new_policy_id,
                "version": new_policy_version,
                "status": new_status,
                "auto_applied": auto_apply,
            },
        }

        run_id = _insert_policy_review_run(conn, payload)
        conn.commit()
        return {
            "status": "applied" if auto_apply else "validated",
            "run_id": run_id,
            "policy_id": new_policy_id,
            "policy_version": new_policy_version,
            "policy_status": new_status,
            "reason": payload["llm_reason"],
        }

    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Run one guarded strategy policy review")
    parser.add_argument("--json-only", action="store_true")
    args = parser.parse_args()

    result = run_policy_review_once()
    output = _json_dumps(result)
    if not args.json_only:
        log.info(f"policy review result: {output}")
    print(output)


if __name__ == "__main__":
    main()
