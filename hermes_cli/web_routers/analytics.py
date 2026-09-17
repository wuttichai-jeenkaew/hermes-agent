"""Raw-YAML config and token/cost analytics dashboard routes.

Extracted from ``hermes_cli.web_server``; app state and helpers are late-bound through
:mod:`hermes_cli.web_deps` (cycle-safe, monkeypatch-friendly).
"""

import asyncio
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from fastapi import APIRouter, HTTPException, Query

from hermes_cli.config import get_config_path, read_raw_config
from hermes_cli.web_deps import late
from hermes_cli.web_routers._common import corrupt_store_as_status
from hermes_cli.web_server_profiles import (
    _approval_mode_of, _aux_task_summary, _aux_usage_rows, _broadcast_gateway_session_info, _is_other_profile, _merge_aux_into_by_model,
)
from hermes_cli.web_models import RawConfigUpdate

router = APIRouter()

# Late-bound so a test's monkeypatch on the owning module wins at call time.
_open_session_db_for_profile = late("_open_session_db_for_profile", "hermes_cli.web_server_sessions")
_session_db_path_for_profile = late("_session_db_path_for_profile", "hermes_cli.web_server_sessions")
_profile_scope = late("_profile_scope", "hermes_cli.web_server_profiles")
save_config = late("save_config", "hermes_cli.config")

# ── Raw YAML config ──────────────────────────────────────────────────────────


@router.get("/api/config/raw")
async def get_config_raw(profile: Optional[str] = None):
    """Raw config.yaml text plus its resolved path.

    ``path`` is resolved inside ``_profile_scope`` so the Config page header
    shows the file the switched profile actually reads/writes — /api/status's
    ``config_path`` is machine-global and always reports the dashboard
    process's own profile, which is wrong under the global profile switcher.
    """
    def _run():
        with _profile_scope(profile):
            path = get_config_path()
        if not path.exists():
            return {"yaml": "", "path": str(path)}
        return {"yaml": path.read_text(encoding="utf-8"), "path": str(path)}

    return await asyncio.to_thread(_run)


@router.put("/api/config/raw")
async def update_config_raw(body: RawConfigUpdate, profile: Optional[str] = None):
    def _run():
        parsed = yaml.safe_load(body.yaml_text)
        if not isinstance(parsed, dict):
            raise HTTPException(status_code=400, detail="YAML must be a mapping")
        with _profile_scope(body.profile or profile):
            # Full-document replacement: the editor owns the whole file; never
            # merge omitted sections back from disk.
            # See #62723.
            approvals_mode_changed = _approval_mode_of(parsed) != _approval_mode_of(read_raw_config())
            save_config(parsed, merge_existing=False)
        # Same indicator refresh as the schema-driven save.
        if approvals_mode_changed and not _is_other_profile(body.profile or profile):
            _broadcast_gateway_session_info()
        return {"ok": True}

    try:
        return await asyncio.to_thread(_run)
    except yaml.YAMLError as e:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {e}")


def _rows(db, sql: str, cutoff: float) -> List[Dict[str, Any]]:
    return [dict(r) for r in db._conn.execute(sql, (cutoff,)).fetchall()]


def _get_usage_analytics(days: int = 30, profile: Optional[str] = None):
    from agent.insights import InsightsEngine

    db = _open_session_db_for_profile(profile, read_only=True)
    try:
        cutoff = time.time() - (days * 86400)
        daily = _rows(db, """
            SELECT date(started_at, 'unixepoch') as day,
                   SUM(input_tokens) as input_tokens,
                   SUM(output_tokens) as output_tokens,
                   SUM(cache_read_tokens) as cache_read_tokens,
                   SUM(reasoning_tokens) as reasoning_tokens,
                   COALESCE(SUM(estimated_cost_usd), 0) as estimated_cost,
                   COALESCE(SUM(actual_cost_usd), 0) as actual_cost,
                   COUNT(*) as sessions,
                   SUM(COALESCE(api_call_count, 0)) as api_calls
            FROM sessions WHERE started_at > ?
            GROUP BY day ORDER BY day
        """, cutoff)

        by_model = _rows(db, """
            SELECT model,
                   SUM(input_tokens) as input_tokens,
                   SUM(output_tokens) as output_tokens,
                   COALESCE(SUM(estimated_cost_usd), 0) as estimated_cost,
                   COUNT(*) as sessions,
                   SUM(COALESCE(api_call_count, 0)) as api_calls
            FROM sessions WHERE started_at > ? AND model IS NOT NULL
            GROUP BY model ORDER BY SUM(input_tokens) + SUM(output_tokens) DESC
        """, cutoff)

        # Fold in auxiliary usage (vision, compression, ...) from session_model_usage.
        # Aux calls never touch the sessions counters, so this is add-only — no double count.
        # Without it the models list shows only the main agent model even when aux models are actively
        # burning tokens (issue #23270).
        aux_rows = _aux_usage_rows(db, cutoff)
        by_model = _merge_aux_into_by_model(by_model, aux_rows)

        totals = _rows(db, """
            SELECT SUM(input_tokens) as total_input,
                   SUM(output_tokens) as total_output,
                   SUM(cache_read_tokens) as total_cache_read,
                   SUM(reasoning_tokens) as total_reasoning,
                   COALESCE(SUM(estimated_cost_usd), 0) as total_estimated_cost,
                   COALESCE(SUM(actual_cost_usd), 0) as total_actual_cost,
                   COUNT(*) as total_sessions,
                   SUM(COALESCE(api_call_count, 0)) as total_api_calls
            FROM sessions WHERE started_at > ?
        """, cutoff)[0]
        usage = InsightsEngine(db).get_usage_breakdown(days=days)

        return {
            "daily": daily,
            "by_model": by_model,
            "by_task": _aux_task_summary(aux_rows),  # "what is compression costing me"
            "totals": totals,
            "period_days": days,
            "skills": usage["skills"],
            "tools": usage["tools"],  # per-tool-name counts; desktop aggregates per toolset
        }
    finally:
        db.close()


@router.get("/api/analytics/usage")
async def get_usage_analytics(
    days: int = Query(30, ge=1, le=365),
    profile: Optional[str] = None,
):
    """``days`` is clamped to 1-365 (idea from #74778): huge or non-positive
    values would force expensive full-history SQL and InsightsEngine work, or
    produce empty/inverted time windows. The UI only offers 7/30/90-day
    presets."""
    with corrupt_store_as_status(_session_db_path_for_profile(profile)):
        return await asyncio.to_thread(_get_usage_analytics, days, profile)


_USAGE_KEYS = (
    "input_tokens", "output_tokens", "cache_read_tokens", "reasoning_tokens",
    "estimated_cost", "actual_cost", "api_calls", "tool_calls",
)


def _has_usage(row: Dict[str, Any]) -> bool:
    return any((row.get(key) or 0) != 0 for key in _USAGE_KEYS)


def _fold_session_only_rows(raw_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fold model rows that carry no billing_provider and no usage into the single
    accounted provider row for that model.

    Session rows can be created before the first billable call finishes; if that early row
    records only the model name while a later row has real accounting, the Models page used
    to show a duplicate "0 tokens / — API calls" card. Only folds when ownership is
    unambiguous (exactly one provider row).
    """
    rows_by_model: Dict[str, List[Dict[str, Any]]] = {}
    for row in raw_rows:
        rows_by_model.setdefault(row.get("model") or "", []).append(row)

    rows: List[Dict[str, Any]] = []
    for model_rows in rows_by_model.values():
        provider_rows = [r for r in model_rows if r.get("billing_provider")]
        if len(provider_rows) != 1:
            rows.extend(model_rows)
            continue
        target = provider_rows[0]
        for row in model_rows:
            if row is target or row.get("billing_provider") or _has_usage(row):
                continue
            target["sessions"] = (target.get("sessions") or 0) + (row.get("sessions") or 0)
            target["last_used_at"] = max(target.get("last_used_at") or 0, row.get("last_used_at") or 0)
            total_tokens = (target.get("input_tokens") or 0) + (target.get("output_tokens") or 0)
            sessions = target.get("sessions") or 0
            target["avg_tokens_per_session"] = total_tokens / sessions if sessions else 0
        rows.append(target)
        rows.extend(
            r for r in model_rows
            if r is not target and (r.get("billing_provider") or _has_usage(r))
        )
    return rows


def _model_capabilities(provider: str, model_name: str) -> dict:
    """models.dev capability metadata for the card; {} when unknown or lookup fails."""
    try:
        from agent.models_dev import get_model_capabilities
        mc = get_model_capabilities(provider=provider, model=model_name)
    except Exception:
        return {}
    if mc is None:
        return {}
    return {
        "supports_tools": mc.supports_tools,
        "supports_vision": mc.supports_vision,
        "supports_reasoning": mc.supports_reasoning,
        "context_window": mc.context_window,
        "max_output_tokens": mc.max_output_tokens,
        "model_family": mc.model_family,
    }


_AUX_SUMMED_KEYS = (
    "input_tokens", "output_tokens", "cache_read_tokens", "reasoning_tokens", "estimated_cost", "sessions", "api_calls",
)
_MODEL_CARD_KEYS = (
    "input_tokens", "output_tokens", "cache_read_tokens", "reasoning_tokens",
    "estimated_cost", "actual_cost", "sessions", "api_calls", "tool_calls",
    "last_used_at", "avg_tokens_per_session",
)


def _get_models_analytics(days: int = 30, profile: Optional[str] = None):
    """Per-model token/cost/session breakdown plus models.dev capability metadata."""
    db = _open_session_db_for_profile(profile, read_only=True)
    try:
        cutoff = time.time() - (days * 86400)

        raw_rows = _rows(db, """
            SELECT model,
                   billing_provider,
                   SUM(input_tokens) as input_tokens,
                   SUM(output_tokens) as output_tokens,
                   SUM(cache_read_tokens) as cache_read_tokens,
                   SUM(reasoning_tokens) as reasoning_tokens,
                   COALESCE(SUM(estimated_cost_usd), 0) as estimated_cost,
                   COALESCE(SUM(actual_cost_usd), 0) as actual_cost,
                   COUNT(*) as sessions,
                   SUM(COALESCE(api_call_count, 0)) as api_calls,
                   SUM(tool_call_count) as tool_calls,
                   MAX(started_at) as last_used_at,
                   AVG(input_tokens + output_tokens) as avg_tokens_per_session
            FROM sessions WHERE started_at > ? AND model IS NOT NULL AND model != ''
            GROUP BY model, billing_provider
            ORDER BY SUM(input_tokens) + SUM(output_tokens) DESC
        """, cutoff)

        # Aux-only models (dedicated vision/compression) as (model, provider) rows,
        # keyed like the GROUP BY above, so they appear on the Models page.
        # See #23270.
        for aux in _aux_usage_rows(db, cutoff):
            raw_rows.append({
                "model": aux.get("model") or "unknown",
                "billing_provider": aux.get("billing_provider") or "",
                **{key: aux.get(key) or 0 for key in _AUX_SUMMED_KEYS},
                "actual_cost": 0,
                "tool_calls": 0,
                "last_used_at": aux.get("last_used_at"),
                "avg_tokens_per_session": 0,
                "aux_task": aux.get("task") or "",
            })

        rows = _fold_session_only_rows(raw_rows)
        rows.sort(
            key=lambda r: (r.get("input_tokens") or 0) + (r.get("output_tokens") or 0),
            reverse=True,
        )

        models = [
            {
                "model": row["model"],
                "provider": row.get("billing_provider") or "",
                **{key: row[key] for key in _MODEL_CARD_KEYS},
                "capabilities": _model_capabilities(row.get("billing_provider") or "", row["model"]),
            }
            for row in rows
        ]

        totals = _rows(db, """
            SELECT COUNT(DISTINCT model) as distinct_models,
                   SUM(input_tokens) as total_input,
                   SUM(output_tokens) as total_output,
                   SUM(cache_read_tokens) as total_cache_read,
                   SUM(reasoning_tokens) as total_reasoning,
                   COALESCE(SUM(estimated_cost_usd), 0) as total_estimated_cost,
                   COALESCE(SUM(actual_cost_usd), 0) as total_actual_cost,
                   COUNT(*) as total_sessions,
                   SUM(COALESCE(api_call_count, 0)) as total_api_calls
            FROM sessions WHERE started_at > ? AND model IS NOT NULL AND model != ''
        """, cutoff)[0]

        return {"models": models, "totals": totals, "period_days": days}
    finally:
        db.close()


@router.get("/api/analytics/models")
async def get_models_analytics(
    days: int = Query(30, ge=1, le=365),
    profile: Optional[str] = None,
):
    """Return model analytics without blocking the serving event loop."""
    with corrupt_store_as_status(_session_db_path_for_profile(profile)):
        return await asyncio.to_thread(_get_models_analytics, days, profile)


def _ninerouter_scope(profile: Optional[str]) -> str:
    from agent.ninerouter_usage import sanitize_scope

    return sanitize_scope(f"profile:{profile or 'current'}")


def _ninerouter_profile_settings(
    profile: Optional[str],
    *,
    profile_home: Optional[Path] = None,
) -> dict[str, Optional[str] | bool]:
    from agent.secret_scope import build_profile_secret_scope
    from hermes_constants import get_hermes_home
    from hermes_cli import web_server as _compat_web_server

    get_process_hermes_home_fn = getattr(_compat_web_server, "get_process_hermes_home")
    normalized_profile = (profile or "current").strip().lower()
    allow_default_data_dir = normalized_profile == "current"
    if not allow_default_data_dir and profile_home is not None:
        try:
            allow_default_data_dir = (
                profile_home.resolve() == get_process_hermes_home_fn().resolve()
            )
        except (OSError, RuntimeError, ValueError):
            allow_default_data_dir = False
    if not allow_default_data_dir and profile_home is None:
        return {
            "management_base_url": None,
            "auth_cookie": None,
            "data_dir": None,
            "allow_default_data_dir": False,
        }

    selected_home = profile_home or get_hermes_home()
    try:
        values = build_profile_secret_scope(selected_home)
    except Exception:
        values = {}

    def value(name: str) -> Optional[str]:
        raw = values.get(name) if isinstance(values, dict) else None
        if not isinstance(raw, str):
            return None
        text = raw.strip()
        return text or None

    return {
        "management_base_url": value("HERMES_9ROUTER_MANAGEMENT_URL"),
        "auth_cookie": value("HERMES_9ROUTER_AUTH_COOKIE"),
        "data_dir": value("HERMES_9ROUTER_DATA_DIR"),
        "allow_default_data_dir": allow_default_data_dir,
    }


def _get_usage_quota(profile: Optional[str] = None) -> dict:
    from datetime import datetime, timezone
    from pathlib import Path
    from agent.account_usage import AccountUsageSnapshot, fetch_account_usage
    from agent.ninerouter_usage import (
        fetch_ninerouter_account_usage,
        resolve_ninerouter_cli_token,
        sanitize_scope,
    )
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from hermes_cli import web_server as _compat_web_server

    config_profile_scope = getattr(_compat_web_server, "_config_profile_scope")
    load_config_fn = getattr(_compat_web_server, "load_config")
    get_process_hermes_home_fn = getattr(_compat_web_server, "get_process_hermes_home")
    with config_profile_scope(profile) as scoped_profile_home:
        from hermes_constants import get_hermes_home

        profile_name = (profile or "").strip().lower()
        is_current_profile = not profile_name or profile_name == "current"
        if not is_current_profile and scoped_profile_home is not None:
            try:
                is_current_profile = (
                    scoped_profile_home.resolve() == get_process_hermes_home_fn().resolve()
                )
            except (OSError, RuntimeError, ValueError):
                is_current_profile = False
        request_scope = getattr(_compat_web_server, "_ninerouter_scope")(profile)
        profile_home = (
            scoped_profile_home
            if scoped_profile_home is not None
            else (get_hermes_home() if is_current_profile else None)
        )
        ninerouter_settings = getattr(_compat_web_server, "_ninerouter_profile_settings")(
            profile,
            profile_home=profile_home,
        )
        cfg = load_config_fn() or {}
        model_cfg = cfg.get("model") or {}
        configured: list[str] = []
        if isinstance(model_cfg, dict):
            provider = str(model_cfg.get("provider") or "").strip().lower()
            if provider and provider not in {"auto", "custom"}:
                configured.append(provider)
        providers_cfg = cfg.get("providers") or {}
        if isinstance(providers_cfg, dict):
            configured.extend(str(name).strip().lower() for name in providers_cfg if str(name).strip())
        providers = list(dict.fromkeys(configured))
        snapshots: list[dict] = []
        for provider in providers:
            snapshot: Optional[AccountUsageSnapshot] = None
            response_provider = provider
            if provider == "openai-api":
                response_provider = "9router"
                profile_binding_missing = (
                    not is_current_profile and scoped_profile_home is None
                )
                if profile_binding_missing:
                    snapshot = AccountUsageSnapshot(
                        provider=response_provider,
                        source="unavailable",
                        fetched_at=datetime.now(timezone.utc),
                        title="9Router usage & quota",
                        unavailable_reason=(
                            "The selected profile has no bound quota-management home."
                        ),
                        scope=request_scope,
                    )
                else:
                    try:
                        runtime = resolve_runtime_provider(requested=provider)
                        management_base_url = ninerouter_settings["management_base_url"]
                        data_dir = ninerouter_settings["data_dir"]
                        auth_cookie = ninerouter_settings["auth_cookie"]
                        cli_token = resolve_ninerouter_cli_token(
                            runtime.get("base_url"),
                            management_base_url,
                            data_dir=data_dir,
                            use_process_env=False,
                            allow_default_data_dir=bool(
                                ninerouter_settings["allow_default_data_dir"]
                            ),
                        )
                        has_profile_cookie = (
                            isinstance(auth_cookie, str)
                            and bool(auth_cookie.strip())
                            and len(auth_cookie) <= 4096
                            and not any(
                                ord(char) < 32 or 127 <= ord(char) <= 159
                                for char in auth_cookie
                            )
                        )
                        if not is_current_profile and not (has_profile_cookie or cli_token):
                            snapshot = AccountUsageSnapshot(
                                provider=response_provider,
                                source="unavailable",
                                fetched_at=datetime.now(timezone.utc),
                                title="9Router usage & quota",
                                unavailable_reason=(
                                    "The selected profile has no bound quota-management credential."
                                ),
                                scope=request_scope,
                            )
                        else:
                            snapshot = fetch_ninerouter_account_usage(
                                base_url=runtime.get("base_url"),
                                management_base_url=management_base_url,
                                auth_cookie=auth_cookie,
                                cli_token=cli_token,
                                scope=request_scope,
                            )
                    except Exception:
                        snapshot = None
            if provider in {"openai-codex", "anthropic", "openrouter"}:
                try:
                    runtime = resolve_runtime_provider(requested=provider)
                    snapshot = fetch_account_usage(
                        provider,
                        base_url=runtime.get("base_url"),
                        api_key=runtime.get("api_key"),
                    )
                except Exception:
                    snapshot = None
            if snapshot is None:
                snapshots.append({
                    "provider": response_provider,
                    "source": "unavailable",
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                    "title": "9Router usage & quota" if response_provider == "9router" else "Account limits",
                    "plan": None,
                    "windows": [],
                    "details": [],
                    "routes": [],
                    "scope": request_scope if response_provider == "9router" else None,
                    "stale": False,
                    "partial": False,
                    "unavailable_reason": "No quota data was returned. The provider may be unsupported, credentials may be unavailable, or the request may have failed.",
                    "available": False,
                })
                continue
            snapshots.append({
                "provider": snapshot.provider,
                "source": snapshot.source,
                "fetched_at": snapshot.fetched_at.isoformat(),
                "title": snapshot.title,
                "plan": snapshot.plan,
                "windows": [
                    {"label": w.label, "used_percent": w.used_percent, "reset_at": w.reset_at.isoformat() if w.reset_at else None, "detail": w.detail}
                    for w in snapshot.windows
                ],
                "details": list(snapshot.details),
                "routes": [
                    {
                        "route": route.route,
                        "provider": route.provider,
                        "account": route.account,
                        "usage": route.usage,
                        "limit": route.limit,
                        "remaining": route.remaining,
                        "remaining_percent": route.remaining_percent,
                        "unit": route.unit,
                        "reset_at": route.reset_at.isoformat() if route.reset_at else None,
                        "status": route.status,
                        "source": route.source,
                        "detail": route.detail,
                    }
                    for route in snapshot.routes
                ],
                "scope": sanitize_scope(snapshot.scope) if response_provider == "9router" else None,
                "stale": snapshot.stale,
                "partial": snapshot.partial,
                "unavailable_reason": snapshot.unavailable_reason,
                "available": snapshot.available,
            })
        return {"providers": snapshots}


@router.get("/api/usage/quota")
async def get_usage_quota(profile: Optional[str] = None):
    """Return provider-reported account limits without exposing credentials."""
    return await asyncio.to_thread(_get_usage_quota, profile)
