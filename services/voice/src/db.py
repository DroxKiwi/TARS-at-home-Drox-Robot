"""PostgreSQL — réglages UI / system prompt (ligne unique app_settings)."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import asyncpg

from .config import DEFAULT_SYSTEM_PROMPT, Settings

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS app_settings (
  id SMALLINT PRIMARY KEY DEFAULT 1 CHECK (id = 1),
  system_prompt TEXT NOT NULL,
  ollama_base_url TEXT NOT NULL,
  ollama_model TEXT NOT NULL,
  heavy_ollama_model TEXT NOT NULL DEFAULT '',
  stt_model TEXT NOT NULL,
  tts_backend TEXT NOT NULL DEFAULT 'kokoro',
  tts_voice TEXT NOT NULL,
  enable_thinking BOOLEAN NOT NULL DEFAULT false,
  kokoro_speed REAL NOT NULL DEFAULT 1.0,
  silence_ms INT NOT NULL DEFAULT 700,
  vad_rms_threshold REAL NOT NULL DEFAULT 0.015,
  barge_in_enabled BOOLEAN NOT NULL DEFAULT true,
  barge_in_min_speech_ms INT NOT NULL DEFAULT 180,
  role_history_messages INT NOT NULL DEFAULT 8,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

MIGRATE_SQL = """
ALTER TABLE app_settings
  ADD COLUMN IF NOT EXISTS heavy_ollama_model TEXT NOT NULL DEFAULT '';
ALTER TABLE app_settings
  ADD COLUMN IF NOT EXISTS role_history_messages INT NOT NULL DEFAULT 8;
"""

SETTINGS_COLUMNS = (
    "system_prompt",
    "ollama_base_url",
    "ollama_model",
    "heavy_ollama_model",
    "stt_model",
    "tts_backend",
    "tts_voice",
    "enable_thinking",
    "kokoro_speed",
    "silence_ms",
    "vad_rms_threshold",
    "barge_in_enabled",
    "barge_in_min_speech_ms",
    "role_history_messages",
    "updated_at",
)

_pool: asyncpg.Pool | None = None


def _row_to_dict(row: asyncpg.Record) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for col in SETTINGS_COLUMNS:
        val = row[col]
        if isinstance(val, datetime):
            out[col] = val.isoformat()
        else:
            out[col] = val
    return out


def seed_dict_from_settings(settings: Settings) -> dict[str, Any]:
    """Valeurs initiales depuis env / defaults config."""
    return {
        "system_prompt": settings.system_prompt or DEFAULT_SYSTEM_PROMPT,
        "ollama_base_url": settings.ollama_base_url,
        "ollama_model": settings.ollama_model,
        "heavy_ollama_model": getattr(settings, "heavy_ollama_model", "") or "",
        "stt_model": settings.whisper_model,
        "tts_backend": settings.tts_backend or "kokoro",
        "tts_voice": settings.kokoro_voice,
        "enable_thinking": False,
        "kokoro_speed": float(settings.kokoro_speed),
        "silence_ms": int(settings.silence_ms),
        "vad_rms_threshold": float(settings.vad_rms_threshold),
        "barge_in_enabled": bool(settings.barge_in_enabled),
        "barge_in_min_speech_ms": int(settings.barge_in_min_speech_ms),
        "role_history_messages": int(
            getattr(settings, "role_history_messages", 8) or 8
        ),
    }


def apply_row_to_settings(settings: Settings, row: dict[str, Any]) -> None:
    """Applique une ligne DB sur l'objet Settings runtime (mutable)."""
    if "system_prompt" in row and row["system_prompt"] is not None:
        settings.system_prompt = str(row["system_prompt"])
    if "ollama_base_url" in row and row["ollama_base_url"] is not None:
        settings.ollama_base_url = str(row["ollama_base_url"])
    if "ollama_model" in row and row["ollama_model"] is not None:
        settings.ollama_model = str(row["ollama_model"])
    if "heavy_ollama_model" in row and row["heavy_ollama_model"] is not None:
        settings.heavy_ollama_model = str(row["heavy_ollama_model"])
    if "stt_model" in row and row["stt_model"] is not None:
        settings.whisper_model = str(row["stt_model"])
    if "tts_backend" in row and row["tts_backend"] is not None:
        settings.tts_backend = str(row["tts_backend"])
    if "tts_voice" in row and row["tts_voice"] is not None:
        settings.kokoro_voice = str(row["tts_voice"])
    if "kokoro_speed" in row and row["kokoro_speed"] is not None:
        settings.kokoro_speed = float(row["kokoro_speed"])
    if "silence_ms" in row and row["silence_ms"] is not None:
        settings.silence_ms = int(row["silence_ms"])
    if "vad_rms_threshold" in row and row["vad_rms_threshold"] is not None:
        settings.vad_rms_threshold = float(row["vad_rms_threshold"])
    if "barge_in_enabled" in row and row["barge_in_enabled"] is not None:
        settings.barge_in_enabled = bool(row["barge_in_enabled"])
    if "barge_in_min_speech_ms" in row and row["barge_in_min_speech_ms"] is not None:
        settings.barge_in_min_speech_ms = int(row["barge_in_min_speech_ms"])
    if "role_history_messages" in row and row["role_history_messages"] is not None:
        settings.role_history_messages = max(0, int(row["role_history_messages"]))


async def init_db(settings: Settings) -> asyncpg.Pool:
    global _pool
    if _pool is not None:
        return _pool

    url = (settings.database_url or "").strip()
    if not url:
        raise RuntimeError("DATABASE_URL manquant")

    logger.info("Connexion Postgres…")
    _pool = await asyncpg.create_pool(url, min_size=1, max_size=5)
    assert _pool is not None
    async with _pool.acquire() as conn:
        await conn.execute(SCHEMA_SQL)
        await conn.execute(MIGRATE_SQL)
        existing = await conn.fetchrow("SELECT id FROM app_settings WHERE id = 1")
        if existing is None:
            seed = seed_dict_from_settings(settings)
            await conn.execute(
                """
                INSERT INTO app_settings (
                  id, system_prompt, ollama_base_url, ollama_model,
                  heavy_ollama_model, stt_model,
                  tts_backend, tts_voice, enable_thinking, kokoro_speed,
                  silence_ms, vad_rms_threshold, barge_in_enabled,
                  barge_in_min_speech_ms, role_history_messages
                ) VALUES (
                  1, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14
                )
                """,
                seed["system_prompt"],
                seed["ollama_base_url"],
                seed["ollama_model"],
                seed["heavy_ollama_model"],
                seed["stt_model"],
                seed["tts_backend"],
                seed["tts_voice"],
                seed["enable_thinking"],
                seed["kokoro_speed"],
                seed["silence_ms"],
                seed["vad_rms_threshold"],
                seed["barge_in_enabled"],
                seed["barge_in_min_speech_ms"],
                seed["role_history_messages"],
            )
            logger.info("app_settings seedé depuis env/defaults")
        else:
            logger.info("app_settings déjà présent")

    from .roles import ensure_roles_schema
    from .fleet_db import ensure_fleet_schema

    await ensure_roles_schema()
    await ensure_fleet_schema()
    return _pool


async def close_db() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def get_pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("DB non initialisée")
    return _pool


async def get_app_settings() -> dict[str, Any]:
    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM app_settings WHERE id = 1"
        )
        if row is None:
            raise RuntimeError("app_settings vide")
        return _row_to_dict(row)


async def update_app_settings(patch: dict[str, Any]) -> dict[str, Any]:
    """Mise à jour partielle ; ignore clés inconnues / None."""
    allowed = {c for c in SETTINGS_COLUMNS if c != "updated_at"}
    fields: dict[str, Any] = {}
    for key, val in patch.items():
        if key not in allowed or val is None:
            continue
        fields[key] = val

    if not fields:
        return await get_app_settings()

    sets = []
    args: list[Any] = []
    for i, (key, val) in enumerate(fields.items(), start=1):
        sets.append(f"{key} = ${i}")
        args.append(val)
    sets.append("updated_at = now()")
    sql = f"UPDATE app_settings SET {', '.join(sets)} WHERE id = 1 RETURNING *"

    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(sql, *args)
        if row is None:
            raise RuntimeError("Échec update app_settings")
        return _row_to_dict(row)
