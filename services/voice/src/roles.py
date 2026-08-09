"""Rôles spécialistes — outils dynamiques ask_<key> pour le LLM de chat."""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any

import asyncpg

from .db import get_pool

logger = logging.getLogger(__name__)

ROLES_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS llm_roles (
  id SERIAL PRIMARY KEY,
  key TEXT NOT NULL UNIQUE,
  name TEXT NOT NULL,
  description TEXT NOT NULL DEFAULT '',
  system_prompt TEXT NOT NULL,
  ollama_model TEXT NOT NULL,
  function_keys JSONB NOT NULL DEFAULT '[]'::jsonb,
  enabled BOOLEAN NOT NULL DEFAULT true,
  sort_order INT NOT NULL DEFAULT 0,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_llm_roles_enabled ON llm_roles (enabled, sort_order);
"""

ROLES_MIGRATE_SQL = """
ALTER TABLE llm_roles
  ADD COLUMN IF NOT EXISTS function_keys JSONB NOT NULL DEFAULT '[]'::jsonb;
"""

_KEY_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")


def slugify_key(raw: str) -> str:
    s = (raw or "").strip().lower()
    s = s.replace("-", "_").replace(" ", "_")
    s = re.sub(r"[^a-z0-9_]", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = "role"
    if s[0].isdigit():
        s = f"r_{s}"
    return s[:32]


def tool_name_for_key(key: str) -> str:
    return f"ask_{key}"


def key_from_tool_name(name: str) -> str | None:
    if not name.startswith("ask_"):
        return None
    key = name[4:]
    if not _KEY_RE.match(key):
        return None
    # Réserver les outils non-rôles
    if key in ("heavy_model",):
        return None
    return key


def _row_to_dict(row: asyncpg.Record) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in row.keys():
        val = row[k]
        if isinstance(val, datetime):
            out[k] = val.isoformat()
        else:
            out[k] = val
    # JSONB peut arriver en str / list
    raw_keys = out.get("function_keys") or []
    if isinstance(raw_keys, str):
        try:
            import json

            raw_keys = json.loads(raw_keys)
        except json.JSONDecodeError:
            raw_keys = []
    if not isinstance(raw_keys, list):
        raw_keys = []
    from .functions import normalize_function_keys

    out["function_keys"] = normalize_function_keys(
        [str(x) for x in raw_keys], scope="specialist"
    )
    out["tool_name"] = tool_name_for_key(str(out["key"]))
    return out


async def ensure_roles_schema() -> None:
    pool = get_pool()
    async with pool.acquire() as conn:
        await conn.execute(ROLES_SCHEMA_SQL)
        await conn.execute(ROLES_MIGRATE_SQL)


async def list_roles(*, enabled_only: bool = False) -> list[dict[str, Any]]:
    pool = get_pool()
    async with pool.acquire() as conn:
        if enabled_only:
            rows = await conn.fetch(
                """
                SELECT * FROM llm_roles
                WHERE enabled = true
                ORDER BY sort_order ASC, id ASC
                """
            )
        else:
            rows = await conn.fetch(
                "SELECT * FROM llm_roles ORDER BY sort_order ASC, id ASC"
            )
    return [_row_to_dict(r) for r in rows]


async def get_role_by_key(key: str) -> dict[str, Any] | None:
    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM llm_roles WHERE key = $1", key.strip().lower()
        )
    return _row_to_dict(row) if row else None


async def get_role_by_id(role_id: int) -> dict[str, Any] | None:
    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM llm_roles WHERE id = $1", role_id)
    return _row_to_dict(row) if row else None


async def create_role(data: dict[str, Any]) -> dict[str, Any]:
    import json

    from .functions import normalize_function_keys

    key = slugify_key(str(data.get("key") or data.get("name") or ""))
    if not _KEY_RE.match(key):
        raise ValueError(f"Clé invalide: {key} (a-z, 0-9, _, max 32)")
    name = (data.get("name") or key).strip()
    description = (data.get("description") or "").strip()
    system_prompt = (data.get("system_prompt") or "").strip()
    if not system_prompt:
        raise ValueError("system_prompt requis")
    model = (data.get("ollama_model") or "").strip()
    if not model:
        raise ValueError("ollama_model requis")
    enabled = bool(data.get("enabled", True))
    sort_order = int(data.get("sort_order") or 0)
    function_keys = normalize_function_keys(
        list(data.get("function_keys") or []), scope="specialist"
    )

    pool = get_pool()
    async with pool.acquire() as conn:
        try:
            row = await conn.fetchrow(
                """
                INSERT INTO llm_roles (
                  key, name, description, system_prompt, ollama_model,
                  function_keys, enabled, sort_order
                ) VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, $8)
                RETURNING *
                """,
                key,
                name,
                description,
                system_prompt,
                model,
                json.dumps(function_keys),
                enabled,
                sort_order,
            )
        except asyncpg.UniqueViolationError as exc:
            raise ValueError(f"Clé déjà utilisée: {key}") from exc
    assert row is not None
    logger.info(
        "Rôle créé key=%s model=%s functions=%s", key, model, function_keys
    )
    return _row_to_dict(row)


async def update_role(role_id: int, patch: dict[str, Any]) -> dict[str, Any]:
    import json

    from .functions import normalize_function_keys

    existing = await get_role_by_id(role_id)
    if not existing:
        raise ValueError("Rôle introuvable")

    fields: dict[str, Any] = {}
    if "name" in patch and patch["name"] is not None:
        fields["name"] = str(patch["name"]).strip()
    if "description" in patch and patch["description"] is not None:
        fields["description"] = str(patch["description"]).strip()
    if "system_prompt" in patch and patch["system_prompt"] is not None:
        sp = str(patch["system_prompt"]).strip()
        if not sp:
            raise ValueError("system_prompt vide")
        fields["system_prompt"] = sp
    if "ollama_model" in patch and patch["ollama_model"] is not None:
        model = str(patch["ollama_model"]).strip()
        if not model:
            raise ValueError("ollama_model vide")
        fields["ollama_model"] = model
    if "enabled" in patch and patch["enabled"] is not None:
        fields["enabled"] = bool(patch["enabled"])
    if "sort_order" in patch and patch["sort_order"] is not None:
        fields["sort_order"] = int(patch["sort_order"])
    if "key" in patch and patch["key"] is not None:
        key = slugify_key(str(patch["key"]))
        if not _KEY_RE.match(key):
            raise ValueError(f"Clé invalide: {key}")
        fields["key"] = key
    if "function_keys" in patch and patch["function_keys"] is not None:
        fields["function_keys"] = normalize_function_keys(
            list(patch["function_keys"] or []), scope="specialist"
        )

    if not fields:
        return existing

    sets = []
    args: list[Any] = []
    for i, (k, v) in enumerate(fields.items(), start=1):
        if k == "function_keys":
            sets.append(f"{k} = ${i}::jsonb")
            args.append(json.dumps(v))
        else:
            sets.append(f"{k} = ${i}")
            args.append(v)
    sets.append("updated_at = now()")
    args.append(role_id)
    sql = (
        f"UPDATE llm_roles SET {', '.join(sets)} "
        f"WHERE id = ${len(args)} RETURNING *"
    )

    pool = get_pool()
    async with pool.acquire() as conn:
        try:
            row = await conn.fetchrow(sql, *args)
        except asyncpg.UniqueViolationError as exc:
            raise ValueError("Clé déjà utilisée") from exc
    if row is None:
        raise ValueError("Rôle introuvable")
    return _row_to_dict(row)


async def delete_role(role_id: int) -> bool:
    pool = get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM llm_roles WHERE id = $1", role_id
        )
    return result.endswith("1")


def format_history_block(
    history: list[dict[str, str]],
    *,
    limit: int = 8,
    pending_user: str | None = None,
) -> str:
    """Formate les N derniers messages pour le spécialiste."""
    msgs = list(history[-max(0, limit) :]) if limit else []
    lines: list[str] = []
    for m in msgs:
        role = m.get("role") or "?"
        content = (m.get("content") or "").strip()
        if not content:
            continue
        label = {
            "user": "Utilisateur",
            "assistant": "Compagnon (TARS)",
            "system": "Système",
        }.get(role, role)
        # Tronquer un peu pour éviter les prompts monstrueux
        if len(content) > 1200:
            content = content[:1200] + "…"
        lines.append(f"[{label}] {content}")
    if pending_user and pending_user.strip():
        lines.append(f"[Utilisateur] {pending_user.strip()}")
    if not lines:
        return "(aucun historique récent)"
    return "\n".join(lines)


def role_tool_schema(role: dict[str, Any]) -> dict[str, Any]:
    key = role["key"]
    name = role.get("name") or key
    desc = (role.get("description") or "").strip()
    tool = tool_name_for_key(key)
    fn_keys = role.get("function_keys") or []
    fn_hint = (
        f" Fonctions du spécialiste : {', '.join(fn_keys)}."
        if fn_keys
        else " Ce spécialiste n'a pas de fonctions outils assignées."
    )
    description = (
        f"Consulte le spécialiste « {name} » (rôle {key}). "
        + (f"{desc} " if desc else "")
        + "Annonce d'abord oralement que tu vas l'appeler, "
        "puis appelle cet outil. "
        + fn_hint
        + " La citation / résumé sera gérée ensuite."
    )
    return {
        "type": "function",
        "function": {
            "name": tool,
            "description": description.strip(),
            "parameters": {
                "type": "object",
                "required": ["task"],
                "properties": {
                    "task": {
                        "type": "string",
                        "description": (
                            "Consigne courte pour le spécialiste "
                            "(ce qu'il doit faire / répondre)"
                        ),
                    },
                },
            },
        },
    }


def roles_system_hint(roles: list[dict[str, Any]]) -> str:
    enabled = [r for r in roles if r.get("enabled")]
    if not enabled:
        return ""
    bits = []
    for r in enabled:
        tool = tool_name_for_key(r["key"])
        label = r.get("name") or r["key"]
        blurb = (r.get("description") or "").strip()
        if blurb:
            bits.append(f"- {tool} : {label} — {blurb}")
        else:
            bits.append(f"- {tool} : {label}")
    return (
        "Specialistes (ne choisis jamais un modele Ollama toi-meme) :\n"
        + "\n".join(bits)
        + "\nAvant d'appeler un specialiste, annonce-le a l'oral, puis appelle l'outil. "
        "Apres : resume brièvement et demande s'il faut lire la reponse complete "
        "(outil read_specialist_reply si oui)."
    )
