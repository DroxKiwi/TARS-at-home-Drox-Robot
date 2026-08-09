"""Fleet — inventaire nœuds/hosts + audit commandes (cerveau)."""

from __future__ import annotations

import json
import logging
import secrets
from datetime import datetime
from typing import Any

import asyncpg

from .db import get_pool

logger = logging.getLogger(__name__)

FLEET_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS fleet_nodes (
  id SERIAL PRIMARY KEY,
  node_key TEXT NOT NULL UNIQUE,
  name TEXT NOT NULL,
  kind TEXT NOT NULL DEFAULT 'edge',
  token_hash TEXT NOT NULL,
  last_seen_at TIMESTAMPTZ,
  online BOOLEAN NOT NULL DEFAULT false,
  meta JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS fleet_hosts (
  id SERIAL PRIMARY KEY,
  host_key TEXT NOT NULL UNIQUE,
  label TEXT NOT NULL,
  hostname TEXT,
  ip TEXT,
  ssh_user TEXT NOT NULL DEFAULT 'tars',
  ssh_port INT NOT NULL DEFAULT 22,
  ssh_key_path TEXT,
  tags JSONB NOT NULL DEFAULT '[]'::jsonb,
  enabled BOOLEAN NOT NULL DEFAULT true,
  notes TEXT NOT NULL DEFAULT '',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS fleet_audit (
  id BIGSERIAL PRIMARY KEY,
  actor TEXT NOT NULL DEFAULT 'llm',
  tool_name TEXT NOT NULL,
  target TEXT,
  request JSONB NOT NULL DEFAULT '{}'::jsonb,
  response JSONB NOT NULL DEFAULT '{}'::jsonb,
  ok BOOLEAN NOT NULL DEFAULT false,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


def _row(row: asyncpg.Record) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in row.keys():
        val = row[k]
        if isinstance(val, datetime):
            out[k] = val.isoformat()
        else:
            out[k] = val
    return out


def _parse_json(val: Any, default: Any) -> Any:
    if val is None:
        return default
    if isinstance(val, (dict, list)):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except json.JSONDecodeError:
            return default
    return default


async def ensure_fleet_schema() -> None:
    pool = get_pool()
    async with pool.acquire() as conn:
        await conn.execute(FLEET_SCHEMA_SQL)
    logger.info("Schéma fleet OK")


def hash_token(token: str) -> str:
    import hashlib

    return hashlib.sha256(token.encode("utf-8")).hexdigest()


async def register_or_rotate_node(
    *,
    node_key: str,
    name: str,
    kind: str = "edge",
) -> dict[str, Any]:
    """Crée un nœud et retourne le token en clair une seule fois."""
    token = secrets.token_urlsafe(24)
    th = hash_token(token)
    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO fleet_nodes (node_key, name, kind, token_hash, online)
            VALUES ($1, $2, $3, $4, false)
            ON CONFLICT (node_key) DO UPDATE SET
              name = EXCLUDED.name,
              kind = EXCLUDED.kind,
              token_hash = EXCLUDED.token_hash,
              updated_at = now()
            RETURNING *
            """,
            node_key.strip().lower(),
            name.strip() or node_key,
            kind,
            th,
        )
    assert row is not None
    data = _row(row)
    data["token"] = token  # clair une fois
    data.pop("token_hash", None)
    return data


async def get_node_by_key(node_key: str) -> dict[str, Any] | None:
    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM fleet_nodes WHERE node_key = $1",
            node_key.strip().lower(),
        )
    if not row:
        return None
    data = _row(row)
    data["meta"] = _parse_json(data.get("meta"), {})
    return data


async def verify_node_token(node_key: str, token: str) -> dict[str, Any] | None:
    node = await get_node_by_key(node_key)
    if not node:
        return None
    if node.get("token_hash") != hash_token(token):
        return None
    return node


async def set_node_online(node_key: str, online: bool, meta: dict | None = None) -> None:
    pool = get_pool()
    async with pool.acquire() as conn:
        if meta is not None:
            await conn.execute(
                """
                UPDATE fleet_nodes
                SET online = $2, last_seen_at = now(), meta = $3::jsonb, updated_at = now()
                WHERE node_key = $1
                """,
                node_key.strip().lower(),
                online,
                json.dumps(meta),
            )
        else:
            await conn.execute(
                """
                UPDATE fleet_nodes
                SET online = $2, last_seen_at = CASE WHEN $2 THEN now() ELSE last_seen_at END,
                    updated_at = now()
                WHERE node_key = $1
                """,
                node_key.strip().lower(),
                online,
            )


async def list_nodes() -> list[dict[str, Any]]:
    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, node_key, name, kind, last_seen_at, online, meta, created_at "
            "FROM fleet_nodes ORDER BY name ASC"
        )
    out = []
    for r in rows:
        d = _row(r)
        d["meta"] = _parse_json(d.get("meta"), {})
        out.append(d)
    return out


async def upsert_host(data: dict[str, Any]) -> dict[str, Any]:
    host_key = str(data.get("host_key") or "").strip().lower()
    if not host_key:
        raise ValueError("host_key requis")
    label = str(data.get("label") or host_key).strip()
    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO fleet_hosts (
              host_key, label, hostname, ip, ssh_user, ssh_port, ssh_key_path,
              tags, enabled, notes
            ) VALUES (
              $1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9, $10
            )
            ON CONFLICT (host_key) DO UPDATE SET
              label = EXCLUDED.label,
              hostname = EXCLUDED.hostname,
              ip = EXCLUDED.ip,
              ssh_user = EXCLUDED.ssh_user,
              ssh_port = EXCLUDED.ssh_port,
              ssh_key_path = COALESCE(EXCLUDED.ssh_key_path, fleet_hosts.ssh_key_path),
              tags = EXCLUDED.tags,
              enabled = EXCLUDED.enabled,
              notes = EXCLUDED.notes,
              updated_at = now()
            RETURNING *
            """,
            host_key,
            label,
            (data.get("hostname") or None),
            (data.get("ip") or None),
            str(data.get("ssh_user") or "tars"),
            int(data.get("ssh_port") or 22),
            (data.get("ssh_key_path") or None),
            json.dumps(data.get("tags") or []),
            bool(data.get("enabled", True)),
            str(data.get("notes") or ""),
        )
    assert row is not None
    d = _row(row)
    d["tags"] = _parse_json(d.get("tags"), [])
    return d


async def list_hosts(*, enabled_only: bool = True) -> list[dict[str, Any]]:
    pool = get_pool()
    async with pool.acquire() as conn:
        if enabled_only:
            rows = await conn.fetch(
                "SELECT * FROM fleet_hosts WHERE enabled = true ORDER BY label ASC"
            )
        else:
            rows = await conn.fetch("SELECT * FROM fleet_hosts ORDER BY label ASC")
    out = []
    for r in rows:
        d = _row(r)
        d["tags"] = _parse_json(d.get("tags"), [])
        out.append(d)
    return out


async def get_host(host_key: str) -> dict[str, Any] | None:
    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM fleet_hosts WHERE host_key = $1",
            host_key.strip().lower(),
        )
    if not row:
        return None
    d = _row(row)
    d["tags"] = _parse_json(d.get("tags"), [])
    return d


async def audit_log(
    *,
    tool_name: str,
    target: str | None,
    request: dict[str, Any],
    response: dict[str, Any],
    ok: bool,
    actor: str = "llm",
) -> None:
    pool = get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO fleet_audit (actor, tool_name, target, request, response, ok)
            VALUES ($1, $2, $3, $4::jsonb, $5::jsonb, $6)
            """,
            actor,
            tool_name,
            target,
            json.dumps(request),
            json.dumps(response),
            ok,
        )
