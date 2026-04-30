from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import psycopg2
from psycopg2.extras import RealDictCursor


def _env_bool(value: str | bool | int | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _get_path(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = data
    for part in path:
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _set_path(data: dict[str, Any], path: tuple[str, ...], value: Any):
    current = data
    for part in path[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[path[-1]] = value


def _format_env_value(kind: str, value: Any) -> str:
    if kind == "bool":
        return "1" if _env_bool(value) else "0"
    if kind == "int":
        return str(int(value))
    if kind == "float":
        return str(float(value))
    return str(value)


def _parse_env_value(kind: str, raw: str) -> Any:
    if kind == "bool":
        return _env_bool(raw)
    if kind == "int":
        return int(raw)
    if kind == "float":
        return float(raw)
    return raw


POLICY_BINDINGS: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (("global", "fg_extreme_block"), "FG_EXTREME_BLOCK", "bool"),
    (("global", "fg_extreme_fear_threshold"), "FG_EXTREME_FEAR_THRESHOLD", "int"),
    (("global", "fg_extreme_greed_threshold"), "FG_EXTREME_GREED_THRESHOLD", "int"),
    (("global", "tech_allow_sideway"), "TECH_ALLOW_SIDEWAY", "bool"),
    (("global", "tech_score_threshold"), "TECH_SCORE_THRESHOLD", "int"),
    (("global", "tech_score_threshold_range"), "TECH_SCORE_THRESHOLD_RANGE", "int"),
    (("global", "tech_rollover_score_threshold"), "TECH_ROLLOVER_SCORE_THRESHOLD", "int"),
    (("global", "sl_atr_multiplier"), "SL_ATR_MULTIPLIER", "float"),
    (("global", "sl_sr_buffer_atr_mult"), "SL_SR_BUFFER_ATR_MULT", "float"),
    (("global", "sl_atr_dynamic_mult"), "SL_ATR_DYNAMIC_MULT", "float"),
    (("global", "sl_max_pct_ceiling"), "SL_MAX_PCT_CEILING", "float"),
    (("global", "trade_min_rr"), "TRADE_MIN_RR", "float"),
    (("global", "trade_min_rr_range"), "TRADE_MIN_RR_RANGE", "float"),
    (("global", "tp_extension_atr_mult"), "TP_EXTENSION_ATR_MULT", "float"),
    (("global", "setup_e_min_rr"), "SETUP_E_MIN_RR", "float"),
    (("symbols", "BTC", "enabled"), "BTC_ENABLED", "bool"),
    (("symbols", "BTC", "position_risk_pct"), "BTC_POSITION_RISK_PCT", "float"),
    (("symbols", "BTC", "max_position_fraction"), "BTC_MAX_POSITION_FRACTION", "float"),
    (("symbols", "BTC", "break_even_trigger_r_mult"), "BTC_BREAK_EVEN_TRIGGER_R_MULT", "float"),
    (("symbols", "BTC", "high_atr_pct"), "BTC_HIGH_ATR_PCT", "float"),
    (("symbols", "BTC", "high_atr_min_adx"), "BTC_HIGH_ATR_MIN_ADX", "float"),
    (("symbols", "BTC", "range_disable_atr_pct"), "BTC_RANGE_DISABLE_ATR_PCT", "float"),
    (("symbols", "BTC", "trail_lookback_bars"), "BTC_TRAIL_LOOKBACK_BARS", "int"),
    (("symbols", "BTC", "trail_stop_atr_mult"), "BTC_TRAIL_STOP_ATR_MULT", "float"),
    (("symbols", "BTC", "trail_mark_buffer_pct"), "BTC_TRAIL_MARK_BUFFER_PCT", "float"),
)

MANAGED_ENV_KEYS: tuple[str, ...] = tuple(sorted({env_key for _, env_key, _ in POLICY_BINDINGS}))


@dataclass
class ActiveStrategyPolicy:
    id: int
    policy_name: str
    engine_name: str
    account_type: str
    version: int
    status: str
    policy_json: dict[str, Any]
    validation_report: dict[str, Any]
    source: str
    effective_from: datetime | None


def capture_managed_environment() -> dict[str, str | None]:
    return {key: os.getenv(key) for key in MANAGED_ENV_KEYS}


def reset_managed_environment(base_env: dict[str, str | None]):
    for key, value in base_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def apply_policy_to_environment(policy_json: dict[str, Any]) -> dict[str, str]:
    applied: dict[str, str] = {}
    if not isinstance(policy_json, dict):
        return applied
    for path, env_key, kind in POLICY_BINDINGS:
        value = _get_path(policy_json, path)
        if value is None:
            continue
        env_value = _format_env_value(kind, value)
        os.environ[env_key] = env_value
        applied[env_key] = env_value
    return applied


_RISK_DIRECTION_HIGHER = "higher_is_riskier"
_RISK_DIRECTION_LOWER = "lower_is_riskier"

RISK_INCREASE_RULES: tuple[tuple[str, str, str], ...] = (
    ("POSITION_RISK_PCT", "float", _RISK_DIRECTION_HIGHER),
    ("BTC_POSITION_RISK_PCT", "float", _RISK_DIRECTION_HIGHER),
    ("MAX_POSITION_FRACTION", "float", _RISK_DIRECTION_HIGHER),
    ("BTC_MAX_POSITION_FRACTION", "float", _RISK_DIRECTION_HIGHER),
    ("BREAK_EVEN_TRIGGER_R_MULT", "float", _RISK_DIRECTION_HIGHER),
    ("BTC_BREAK_EVEN_TRIGGER_R_MULT", "float", _RISK_DIRECTION_HIGHER),
    ("BTC_TRAIL_STOP_ATR_MULT", "float", _RISK_DIRECTION_HIGHER),
    ("TECH_ALLOW_SIDEWAY", "bool", _RISK_DIRECTION_HIGHER),
    ("FG_EXTREME_BLOCK", "bool", _RISK_DIRECTION_LOWER),
    ("TECH_SCORE_THRESHOLD", "int", _RISK_DIRECTION_LOWER),
    ("TECH_SCORE_THRESHOLD_RANGE", "int", _RISK_DIRECTION_LOWER),
    ("TECH_ROLLOVER_SCORE_THRESHOLD", "int", _RISK_DIRECTION_LOWER),
    ("TRADE_MIN_RR", "float", _RISK_DIRECTION_LOWER),
    ("TRADE_MIN_RR_RANGE", "float", _RISK_DIRECTION_LOWER),
    ("SETUP_E_MIN_RR", "float", _RISK_DIRECTION_LOWER),
)


def _parse_value(raw: str | None, kind: str) -> Any:
    if raw is None:
        return None
    if kind == "bool":
        return _env_bool(raw)
    if kind == "int":
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None
    if kind == "float":
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None
    return raw


def _is_risk_increase(base_value: Any, new_value: Any, direction: str) -> bool:
    if base_value is None or new_value is None:
        return False
    if direction == _RISK_DIRECTION_HIGHER:
        return new_value > base_value
    return new_value < base_value


def policy_allows_risk_increase(
    policy: ActiveStrategyPolicy | None,
    min_evidence_trades: int = 30,
) -> tuple[bool, str]:
    if policy is None:
        return False, "No active policy context"
    report = policy.validation_report or {}
    allow_flag = _env_bool(report.get("allow_risk_increase"))
    clear_signal = _env_bool(report.get("clear_signal"))
    try:
        evidence_trades = int(report.get("evidence_trades", 0) or 0)
    except (TypeError, ValueError):
        evidence_trades = 0
    ok = allow_flag and clear_signal and evidence_trades >= min_evidence_trades
    reason = (
        f"allow_risk_increase={allow_flag} clear_signal={clear_signal} "
        f"evidence_trades={evidence_trades} min_required={min_evidence_trades}"
    )
    return ok, reason


def guard_risk_increase_overrides(
    applied_env: dict[str, str],
    base_env: dict[str, str | None],
    allow_risk_increase: bool,
) -> tuple[dict[str, str], list[str]]:
    blocked: list[str] = []
    if allow_risk_increase:
        return applied_env, blocked

    for env_key, kind, direction in RISK_INCREASE_RULES:
        if env_key not in applied_env:
            continue
        base_value = _parse_value(base_env.get(env_key), kind)
        new_value = _parse_value(applied_env.get(env_key), kind)
        if not _is_risk_increase(base_value, new_value, direction):
            continue
        blocked.append(
            f"{env_key}: {base_env.get(env_key)!r} -> {applied_env.get(env_key)!r}"
        )
        if base_env.get(env_key) is None:
            os.environ.pop(env_key, None)
        else:
            os.environ[env_key] = str(base_env[env_key])
        applied_env.pop(env_key, None)
    return applied_env, blocked


def snapshot_policy_from_environment() -> dict[str, Any]:
    policy: dict[str, Any] = {}
    for path, env_key, kind in POLICY_BINDINGS:
        raw = os.getenv(env_key)
        if raw is None or str(raw).strip() == "":
            continue
        try:
            value = _parse_env_value(kind, raw)
        except (TypeError, ValueError):
            continue
        _set_path(policy, path, value)
    return policy


def symbol_enabled_in_policy(policy_json: dict[str, Any], symbol: str) -> bool:
    value = _get_path(policy_json or {}, ("symbols", symbol.upper(), "enabled"))
    return True if value is None else _env_bool(value)


def setup_enabled_in_policy(policy_json: dict[str, Any], setup_name: str) -> bool:
    text = str(setup_name or "").strip().lower()
    code = "?"
    if "setup " in text:
        code = text.split("setup ", 1)[1][:1].upper()
    elif "setup_" in text:
        code = text.split("setup_", 1)[1][:1].upper()
    elif text:
        code = text[:1].upper()
    value = _get_path(policy_json or {}, ("setups", code, "enabled"))
    return True if value is None else _env_bool(value)


def load_active_strategy_policy(engine_name: str, account_type: str) -> ActiveStrategyPolicy | None:
    database_url = os.getenv("DATABASE_URL", "")
    if not database_url:
        return None

    conn = None
    try:
        conn = psycopg2.connect(database_url, sslmode="require", cursor_factory=RealDictCursor)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, policy_name, engine_name, account_type, version, status,
                       policy_json, validation_report, source, effective_from
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
            if not row:
                return None
            return ActiveStrategyPolicy(
                id=int(row["id"]),
                policy_name=str(row["policy_name"]),
                engine_name=str(row["engine_name"]),
                account_type=str(row["account_type"]),
                version=int(row["version"]),
                status=str(row["status"]),
                policy_json=dict(row["policy_json"] or {}),
                validation_report=dict(row.get("validation_report") or {}),
                source=str(row.get("source") or "manual"),
                effective_from=row.get("effective_from"),
            )
    except Exception as e:
        print(
            f"[strategy_policy] load failed for engine={engine_name} account={account_type}: {e}"
        )
        return None
    finally:
        if conn is not None:
            conn.close()
