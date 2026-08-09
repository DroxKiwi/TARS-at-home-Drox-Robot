"""Outils LLM (Ollama tool calling) — panneau, web, délégation modèle lourd."""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import httpx

from .config import get_settings
from .functions import (
    CHAT_DEFAULT_FUNCTION_KEYS,
    CHAT_ONLY_TOOLS,
    format_action,
    schemas_for_keys,
)

logger = logging.getLogger(__name__)

ProgressCb = Callable[[dict[str, Any]], Awaitable[None]]


@dataclass
class ToolContext:
    """Contexte runtime pour outils qui touchent le pipeline (VRAM, etc.)."""

    orchestrator: Any | None = None
    on_progress: ProgressCb | None = None
    roles: list[dict[str, Any]] | None = None
    history: list[dict[str, str]] | None = None
    pending_user: str | None = None
    history_limit: int = 8
    allowed_function_keys: set[str] | None = None
    caller: str = "chat"  # chat | specialist


BASE_TOOLS: list[dict[str, Any]] = [
    *schemas_for_keys(CHAT_DEFAULT_FUNCTION_KEYS),
    *CHAT_ONLY_TOOLS,
]

# Alias rétrocompat (tests / imports)
TOOLS = BASE_TOOLS

SYSTEM_TOOLS_HINT = (
    "Tu disposes d'outils : show_shape(shape, color), clear_canvas(), "
    "web_search(query), ask_<role> pour les specialistes, "
    "et read_specialist_reply pour lire la reponse complete d'un specialiste. "
    "Avant chaque outil, annonce a l'oral ce que tu vas faire, puis appelle l'outil. "
    "Apres un specialiste : resume en 1-3 phrases, puis demande si l'utilisateur "
    "veut ecouter la reponse complete. S'il accepte, appelle read_specialist_reply. "
    "Ne lis jamais du markdown a l'oral."
)


def build_tools(roles: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    from .roles import role_tool_schema

    tools = list(BASE_TOOLS)
    for role in roles or []:
        if not role.get("enabled", True):
            continue
        tools.append(role_tool_schema(role))
    return tools

_COLOR_FR = {
    "bleu": "blue",
    "rouge": "red",
    "vert": "green",
    "jaune": "yellow",
    "orange": "orange",
    "violet": "purple",
    "rose": "pink",
    "noir": "black",
    "blanc": "white",
    "gris": "gray",
    "cyan": "cyan",
    "marron": "brown",
}

_SHAPE_FR = {
    "cercle": "circle",
    "rond": "circle",
    "carré": "square",
    "carre": "square",
    "triangle": "triangle",
}


def normalize_color(color: str) -> str:
    c = (color or "").strip().lower()
    return _COLOR_FR.get(c, c or "blue")


def normalize_shape(shape: str) -> str:
    s = (shape or "").strip().lower()
    s = _SHAPE_FR.get(s, s)
    if s not in ("circle", "square", "triangle"):
        return "circle"
    return s


def _format_search_for_llm(query: str, results: list[dict[str, str]]) -> str:
    if not results:
        return f"Aucun résultat pour: {query}"
    lines = [f"Résultats pour « {query} » ({len(results)}) :"]
    for i, r in enumerate(results, 1):
        title = r.get("title") or "(sans titre)"
        url = r.get("url") or ""
        snippet = (r.get("content") or "").strip()
        if len(snippet) > 280:
            snippet = snippet[:277] + "..."
        lines.append(f"{i}. {title}\n   URL: {url}\n   {snippet}")
    return "\n".join(lines)


async def _web_search(query: str, max_results: int) -> dict[str, Any]:
    q = (query or "").strip()
    if not q:
        return {
            "ok": False,
            "content": "Requête vide.",
            "ui": None,
        }

    n = max(1, min(int(max_results or 5), 8))
    settings = get_settings()
    base = (settings.searxng_url or "").rstrip("/")
    if not base:
        return {
            "ok": False,
            "content": "SEARXNG_URL non configuré.",
            "ui": None,
        }

    url = f"{base}/search"
    params = {
        "q": q,
        "format": "json",
        "language": "fr-FR",
        # Évite Brave/Startpage (captcha / 429) — DuckDuckGo + wiki + Google
        "engines": "duckduckgo,wikipedia,wikidata,google",
    }
    headers = {
        # Satisfait botdetection quand on appelle sans reverse-proxy
        "X-Forwarded-For": "127.0.0.1",
        "X-Real-IP": "127.0.0.1",
    }
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(25.0, connect=8.0)) as client:
            r = await client.get(url, params=params, headers=headers)
            r.raise_for_status()
            data = r.json()
    except httpx.HTTPStatusError as exc:
        logger.warning("SearXNG HTTP %s: %s", exc.response.status_code, exc)
        return {
            "ok": False,
            "content": (
                f"SearXNG erreur HTTP {exc.response.status_code}. "
                "Vérifie que le service tourne et que format=json est autorisé."
            ),
            "ui": None,
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("SearXNG indisponible: %s", exc)
        return {
            "ok": False,
            "content": f"Recherche web indisponible: {exc}",
            "ui": None,
        }

    unresponsive = data.get("unresponsive_engines") or []
    if unresponsive:
        logger.info("SearXNG moteurs en échec (partiel): %s", unresponsive)

    raw = data.get("results") or []
    results: list[dict[str, str]] = []
    for item in raw[:n]:
        results.append(
            {
                "title": str(item.get("title") or "").strip(),
                "url": str(item.get("url") or "").strip(),
                "content": str(item.get("content") or "").strip(),
            }
        )

    if not results and unresponsive:
        return {
            "ok": False,
            "content": (
                f"Aucun résultat pour « {q} » "
                f"(moteurs indisponibles: {unresponsive}). Réessaie plus tard."
            ),
            "ui": {"action": "web_search", "query": q, "count": 0, "results": []},
        }

    content = _format_search_for_llm(q, results)
    ui = {
        "action": "web_search",
        "query": q,
        "count": len(results),
        "results": results,
    }
    return {"ok": True, "content": content, "ui": ui}


async def execute_tool(
    name: str,
    arguments: dict[str, Any] | None,
    ctx: ToolContext | None = None,
) -> dict[str, Any]:
    """Exécute un outil. Retourne {ok, content, ui, ...} pour LLM + WebSocket."""
    if ctx and ctx.allowed_function_keys is not None:
        if name not in ctx.allowed_function_keys:
            return {
                "ok": False,
                "content": f"Fonction non autorisée pour ce spécialiste: {name}",
                "ui": None,
            }
        if name.startswith("ask_") or name == "read_specialist_reply":
            return {
                "ok": False,
                "content": f"Outil réservé au chat: {name}",
                "ui": None,
            }

    args = arguments or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}

    if name == "clear_canvas":
        ui = {"action": "clear"}
        return {
            "ok": True,
            "content": "Panneau effacé.",
            "ui": ui,
        }

    if name == "show_shape":
        shape = normalize_shape(str(args.get("shape", "circle")))
        color = normalize_color(str(args.get("color", "blue")))
        clear = bool(args.get("clear", False))
        ui = {"action": "show_shape", "shape": shape, "color": color, "clear": clear}
        return {
            "ok": True,
            "content": f"Forme affichée: {shape} {color}"
            + (" (panneau effacé avant)." if clear else "."),
            "ui": ui,
        }

    if name == "web_search":
        max_results = args.get("max_results", 5)
        try:
            max_results = int(max_results)
        except (TypeError, ValueError):
            max_results = 5
        return await _web_search(str(args.get("query", "")), max_results)

    # --- Flotte / edge ---
    if name in (
        "lan_list_hosts",
        "ssh_run",
        "edge_wifi_scan",
        "edge_wifi_status",
        "edge_bt_scan",
        "edge_bt_connect",
        "edge_exec",
    ):
        from . import fleet, fleet_db

        if name == "lan_list_hosts":
            data = await fleet.lan_list_hosts()
            await fleet_db.audit_log(
                tool_name=name,
                target=None,
                request={},
                response=data,
                ok=bool(data.get("ok")),
            )
            hosts_n = len(data.get("hosts") or [])
            nodes_n = len(data.get("nodes") or [])
            return {
                "ok": True,
                "content": (
                    f"Flotte: {hosts_n} hôte(s) SSH, {nodes_n} nœud(s) edge. "
                    f"Détail JSON: {json.dumps(data, ensure_ascii=False)[:2500]}"
                ),
                "ui": {"action": "fleet_list", **data},
            }

        if name == "ssh_run":
            host_key = str(args.get("host_key") or "").strip()
            command = str(args.get("command") or "").strip()
            data = await fleet.ssh_run(host_key, command)
            await fleet_db.audit_log(
                tool_name=name,
                target=host_key,
                request={"command": command},
                response=data,
                ok=bool(data.get("ok")),
            )
            if not data.get("ok"):
                return {
                    "ok": False,
                    "content": data.get("error") or "SSH échoué",
                    "ui": {"action": "ssh_run", **data},
                }
            out = data.get("stdout") or "(vide)"
            err = data.get("stderr") or ""
            content = f"SSH {host_key} OK:\n{out}"
            if err:
                content += f"\nstderr: {err}"
            return {
                "ok": True,
                "content": content[:3500],
                "ui": {"action": "ssh_run", **data},
            }

        node_key = str(args.get("node_key") or "").strip().lower()
        method_map = {
            "edge_wifi_scan": ("wifi_scan", {}),
            "edge_wifi_status": ("wifi_status", {}),
            "edge_bt_scan": ("bt_scan", {}),
            "edge_bt_connect": (
                "bt_connect",
                {
                    "address": str(args.get("address") or "").strip(),
                    "name": str(args.get("name") or "").strip(),
                },
            ),
            "edge_exec": (
                "exec",
                {"command": str(args.get("command") or "").strip()},
            ),
        }
        method, params = method_map[name]
        data = await fleet.edge_call(node_key, method, params)
        await fleet_db.audit_log(
            tool_name=name,
            target=node_key,
            request={"method": method, "params": params},
            response=data,
            ok=bool(data.get("ok")),
        )
        if not data.get("ok"):
            return {
                "ok": False,
                "content": data.get("error") or f"Échec {name}",
                "ui": {"action": name, **data},
            }
        return {
            "ok": True,
            "content": (
                f"{name}@{node_key}: "
                f"{json.dumps(data.get('result'), ensure_ascii=False)[:2500]}"
            ),
            "ui": {"action": name, **data},
        }

    if name == "read_specialist_reply":
        orch = ctx.orchestrator if ctx else None
        if orch is None:
            return {
                "ok": False,
                "content": "Pipeline indisponible.",
                "ui": None,
            }
        pending = getattr(orch.pipeline.session, "pending_specialist", None)
        if not pending or not (pending.get("reply") or "").strip():
            return {
                "ok": False,
                "content": (
                    "Aucune réponse spécialiste en attente. "
                    "Il faut d'abord consulter un spécialiste."
                ),
                "ui": None,
            }
        from .speech_text import markdown_to_speech

        role_name = str(pending.get("role_name") or pending.get("role_key") or "spécialiste")
        raw = str(pending.get("reply") or "")
        speak = markdown_to_speech(raw)
        return {
            "ok": True,
            "content": (
                f"Lecture de la réponse complète du spécialiste « {role_name} ». "
                "Le texte oral est diffusé automatiquement — "
                "réponds seulement par une très courte confirmation "
                "(ex. « Voilà, c'est tout. ») sans répéter le contenu."
            ),
            "ui": {
                "action": "read_specialist",
                "role_name": role_name,
                "reply": raw,
                "speak": speak,
            },
            "speak_text": speak,
        }

    from .roles import (
        format_history_block,
        get_role_by_key,
        key_from_tool_name,
    )

    role_key = key_from_tool_name(name)
    if role_key:
        orch = ctx.orchestrator if ctx else None
        if orch is None:
            return {
                "ok": False,
                "content": "Orchestrateur VRAM indisponible.",
                "ui": None,
            }
        role = None
        if ctx and ctx.roles:
            role = next(
                (r for r in ctx.roles if r.get("key") == role_key and r.get("enabled")),
                None,
            )
        if role is None:
            role = await get_role_by_key(role_key)
        if role is None or not role.get("enabled", True):
            return {
                "ok": False,
                "content": f"Rôle inconnu ou désactivé: {role_key}",
                "ui": None,
            }
        task = str(args.get("task") or "").strip()
        if not task:
            return {
                "ok": False,
                "content": f"{name}: argument task manquant.",
                "ui": None,
            }
        limit = int(ctx.history_limit if ctx else 8)
        history_block = format_history_block(
            ctx.history or [] if ctx else [],
            limit=limit,
            pending_user=ctx.pending_user if ctx else None,
        )
        result = await orch.ask_role(
            role=role,
            task=task,
            history_block=history_block,
            on_progress=ctx.on_progress if ctx else None,
        )
        pending = result.pop("pending_specialist", None)
        orch.pipeline.session.pending_specialist = pending
        return result

    # Legacy alias
    if name == "ask_heavy_model":
        return {
            "ok": False,
            "content": (
                "ask_heavy_model est remplacé par les rôles ask_<clé>. "
                "Crée un rôle dans la config."
            ),
            "ui": None,
        }

    logger.warning("Outil inconnu: %s", name)
    return {
        "ok": False,
        "content": f"Outil inconnu: {name}",
        "ui": None,
    }


def parse_tool_call(tc: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Normalise un tool_call Ollama → (name, arguments)."""
    fn = tc.get("function") or tc
    name = fn.get("name") or ""
    raw_args = fn.get("arguments")
    if raw_args is None:
        args: dict[str, Any] = {}
    elif isinstance(raw_args, dict):
        args = raw_args
    elif isinstance(raw_args, str):
        try:
            args = json.loads(raw_args) if raw_args.strip() else {}
        except json.JSONDecodeError:
            args = {}
    else:
        args = {}
    return name, args
