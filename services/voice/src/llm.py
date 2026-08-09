"""Client LLM Ollama — streaming, historique multi-tours, tools, cancel."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urlparse

import httpx

from .config import Settings
from .roles import key_from_tool_name, list_roles, roles_system_hint
from .tools import ToolContext, build_tools, execute_tool, parse_tool_call

logger = logging.getLogger(__name__)

THINK_TAG_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)


@dataclass
class LlmDelta:
    kind: Literal[
        "content",
        "thinking",
        "tool_call",
        "tool_result",
        "stage",
        "resource",
        "flush",
    ]
    text: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


def normalize_ollama_url(url: str) -> str:
    u = (url or "").strip().rstrip("/")
    if not u:
        raise ValueError("URL Ollama vide")
    parsed = urlparse(u)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("URL invalide (ex. http://127.0.0.1:11434)")
    return u


def strip_think_tags(text: str) -> str:
    return THINK_TAG_RE.sub("", text or "").strip()


def _ensure_spoken_sentence(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    if t[-1] not in ".!?…":
        t += "."
    return t


def _default_tool_announce(
    name: str,
    args: dict[str, Any],
    roles: list[dict[str, Any]],
) -> str:
    """Annonce orale par défaut si le modèle n'en fournit pas."""
    role_key = key_from_tool_name(name)
    if role_key:
        role = next((r for r in roles if r.get("key") == role_key), None)
        label = (role or {}).get("name") or role_key
        return f"Je vais appeler le spécialiste {label}."
    if name == "web_search":
        q = str(args.get("query") or "").strip()
        if q:
            return f"Je vais lancer une recherche web sur « {q} »."
        return "Je vais lancer une recherche web."
    if name == "show_shape":
        return "Je vais afficher une forme sur le panneau."
    if name == "clear_canvas":
        return "Je vais effacer le panneau."
    if name == "read_specialist_reply":
        return "Je vais te lire la réponse complète du spécialiste."
    return f"Je vais utiliser l'outil {name}."


def _specialist_citation_speech(
    role_name: str,
    reply: str,
    actions: list[str] | None = None,
    *,
    max_chars: int = 900,
) -> str:
    """Ancien helper — conservé pour import éventuel ; préférer résumé LLM."""
    from .speech_text import markdown_to_speech

    body = markdown_to_speech(reply or "")
    if len(body) > max_chars:
        body = body[: max_chars - 1].rstrip() + "…"
    text = f"Le spécialiste {role_name} a dit ceci : {body}"
    acts = [a.strip() for a in (actions or []) if a and str(a).strip()]
    if acts:
        text += " Et il a effectué les actions : " + ", ".join(acts)
    return _ensure_spoken_sentence(text)


async def probe_ollama(base_url: str) -> dict[str, Any]:
    url = normalize_ollama_url(base_url)
    try:
        async with httpx.AsyncClient(
            base_url=url,
            timeout=httpx.Timeout(8.0, connect=4.0),
        ) as client:
            r = await client.get("/api/tags")
            r.raise_for_status()
            models = sorted(
                m.get("name")
                for m in r.json().get("models", [])
                if m.get("name")
            )
            return {"ok": True, "base_url": url, "models": models}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "base_url": url, "models": [], "error": str(exc)}


class OllamaLLM:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._base_url = settings.ollama_base_url.rstrip("/")
        self._model = settings.ollama_model
        self._enable_thinking = False
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(180.0, connect=10.0),
        )
        self._history: list[dict[str, str]] = []
        self._pending_user: str | None = None
        self._stream_resp: httpx.Response | None = None
        self._tool_context: ToolContext | None = None
        self._roles_cache: list[dict[str, Any]] = []

    def set_tool_context(self, ctx: ToolContext | None) -> None:
        self._tool_context = ctx

    def set_roles_cache(self, roles: list[dict[str, Any]]) -> None:
        self._roles_cache = list(roles)

    async def refresh_roles(self) -> list[dict[str, Any]]:
        try:
            self._roles_cache = await list_roles(enabled_only=False)
        except Exception:  # noqa: BLE001
            logger.warning("Impossible de charger les rôles", exc_info=True)
            self._roles_cache = []
        return self._roles_cache

    @property
    def pending_user(self) -> str | None:
        return self._pending_user

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def model(self) -> str:
        return self._model

    @property
    def enable_thinking(self) -> bool:
        return self._enable_thinking

    @property
    def history(self) -> list[dict[str, str]]:
        return list(self._history)

    def clear_history(self) -> None:
        self._history.clear()
        self._pending_user = None
        logger.info("Historique LLM effacé")

    def append_user(self, text: str) -> None:
        cleaned = (text or "").strip()
        if not cleaned:
            return
        if self._pending_user is not None:
            self._pending_user = cleaned
            return
        self._pending_user = cleaned

    def commit_assistant(self, spoken_text: str) -> None:
        spoken = (spoken_text or "").strip()
        if self._pending_user is None:
            if spoken:
                self._history.append({"role": "assistant", "content": spoken})
            return
        user = self._pending_user
        self._pending_user = None
        self._history.append({"role": "user", "content": user})
        if spoken:
            self._history.append({"role": "assistant", "content": spoken})
        logger.info(
            "Commit historique user=%d chars assistant=%d chars (hist=%d msgs)",
            len(user),
            len(spoken),
            len(self._history),
        )

    def discard_pending_user(self) -> None:
        self._pending_user = None

    def _build_messages(self, user_text: str) -> list[dict[str, Any]]:
        system = self.settings.system_prompt
        hint = roles_system_hint(
            [r for r in self._roles_cache if r.get("enabled")]
        )
        if hint:
            system = f"{system}\n\n{hint}"
        pending = None
        if self._tool_context and getattr(self._tool_context, "orchestrator", None):
            pending = getattr(
                self._tool_context.orchestrator.pipeline.session,
                "pending_specialist",
                None,
            )
        if pending and (pending.get("reply") or "").strip():
            rn = pending.get("role_name") or pending.get("role_key") or "spécialiste"
            system = (
                f"{system}\n\n"
                f"Note: une reponse complete du specialiste « {rn} » est disponible. "
                "Si l'utilisateur demande de l'ecouter / dit oui, "
                "appelle read_specialist_reply."
            )
        return [
            {"role": "system", "content": system},
            *self._history,
            {"role": "user", "content": user_text},
        ]

    async def configure(
        self,
        base_url: str | None = None,
        model: str | None = None,
        enable_thinking: bool | None = None,
    ) -> None:
        if base_url is not None:
            url = normalize_ollama_url(base_url)
            if url != self._base_url:
                await self._client.aclose()
                self._base_url = url
                self._client = httpx.AsyncClient(
                    base_url=self._base_url,
                    timeout=httpx.Timeout(180.0, connect=10.0),
                )
        if model is not None and model.strip():
            self._model = model.strip()
        if enable_thinking is not None:
            self._enable_thinking = bool(enable_thinking)

    async def close(self) -> None:
        await self.cancel_stream()
        await self._client.aclose()

    async def cancel_stream(self) -> None:
        resp = self._stream_resp
        self._stream_resp = None
        if resp is not None:
            try:
                await resp.aclose()
            except Exception:  # noqa: BLE001
                logger.debug("Fermeture stream LLM déjà close", exc_info=True)

    async def health(self) -> dict[str, Any]:
        result = await probe_ollama(self._base_url)
        result["selected"] = self._model
        result["enable_thinking"] = self._enable_thinking
        result["history_messages"] = len(self._history)
        result["tools_enabled"] = self.settings.tools_enabled
        return result

    async def warm(self, keep_alive: str = "30m") -> None:
        payload = {
            "model": self._model,
            "prompt": "",
            "stream": False,
            "keep_alive": keep_alive,
        }
        logger.info("Warm Ollama model=%s keep_alive=%s", self._model, keep_alive)
        r = await self._client.post("/api/generate", json=payload, timeout=300.0)
        r.raise_for_status()

    async def _run_tool_stream(
        self, name: str, args: dict[str, Any]
    ) -> AsyncIterator[LlmDelta]:
        """Exécute un outil ; yield stage/resource puis tool_result."""
        q: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        base = self._tool_context or ToolContext()
        limit = int(getattr(self.settings, "role_history_messages", 8) or 8)

        async def on_progress(event: dict[str, Any]) -> None:
            await q.put(event)

        ctx = ToolContext(
            orchestrator=base.orchestrator,
            on_progress=on_progress,
            roles=self._roles_cache,
            history=list(self._history),
            pending_user=self._pending_user,
            history_limit=limit,
        )

        async def _exec() -> dict[str, Any]:
            try:
                return await execute_tool(name, args, ctx=ctx)
            finally:
                await q.put(None)

        task = asyncio.create_task(_exec())
        while True:
            item = await q.get()
            if item is None:
                break
            if item.get("kind") == "specialist_tool_ui":
                yield LlmDelta(
                    kind="tool_result",
                    text="",
                    meta={
                        "name": item.get("name") or "specialist_tool",
                        "ok": item.get("ok"),
                        "ui": item.get("ui"),
                    },
                )
                if item.get("resource"):
                    yield LlmDelta(
                        kind="resource",
                        text="",
                        meta=item,
                    )
                continue
            yield LlmDelta(
                kind="stage",
                text=str(item.get("stage") or ""),
                meta=item,
            )
            if item.get("resource"):
                yield LlmDelta(
                    kind="resource",
                    text=str(item.get("stage") or ""),
                    meta=item,
                )
        result = await task
        yield LlmDelta(
            kind="tool_result",
            text=result.get("content") or "",
            meta={
                "name": name,
                "ok": result.get("ok"),
                "ui": result.get("ui"),
                "speak_text": result.get("speak_text"),
            },
        )

    async def _chat_once(
        self,
        messages: list[dict[str, Any]],
        *,
        use_tools: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model,
            "stream": False,
            "messages": messages,
            "think": self._enable_thinking,
        }
        if use_tools and self.settings.tools_enabled:
            enabled = [r for r in self._roles_cache if r.get("enabled")]
            payload["tools"] = build_tools(enabled)

        try:
            r = await self._client.post("/api/chat", json=payload, timeout=180.0)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in (400, 422) and "think" in payload:
                logger.warning("Retry chat sans champ think (%s)", exc)
                payload.pop("think", None)
                r = await self._client.post("/api/chat", json=payload, timeout=180.0)
                r.raise_for_status()
                return r.json()
            raise

    async def chat_stream(
        self,
        user_text: str,
        cancel: asyncio.Event | None = None,
    ) -> AsyncIterator[LlmDelta]:
        """Tools (si besoin) puis stream de la réponse vocale."""
        if not self._roles_cache:
            await self.refresh_roles()
        self.append_user(user_text)
        messages: list[dict[str, Any]] = self._build_messages(user_text)
        use_tools = bool(self.settings.tools_enabled)
        max_rounds = max(1, int(self.settings.tools_max_rounds))
        already_yielded_final = False

        if use_tools:
            try:
                for round_i in range(max_rounds):
                    if cancel is not None and cancel.is_set():
                        return
                    data = await self._chat_once(messages, use_tools=True)
                    msg = data.get("message") or {}
                    tool_calls = msg.get("tool_calls") or []
                    if not tool_calls:
                        content = msg.get("content") or ""
                        thinking = msg.get("thinking") or ""
                        if thinking:
                            yield LlmDelta(kind="thinking", text=thinking)
                        if content:
                            yield LlmDelta(kind="content", text=content)
                            already_yielded_final = True
                        break

                    # Annonce orale AVANT les outils (TTS encore chargé)
                    preamble = strip_think_tags(msg.get("content") or "").strip()
                    if thinking := (msg.get("thinking") or ""):
                        yield LlmDelta(kind="thinking", text=thinking)
                    if preamble:
                        yield LlmDelta(
                            kind="content",
                            text=_ensure_spoken_sentence(preamble),
                        )
                    else:
                        # Une annonce par outil si le modèle reste muet
                        for tc in tool_calls:
                            n, a = parse_tool_call(tc)
                            announce = _default_tool_announce(
                                n, a, self._roles_cache
                            )
                            yield LlmDelta(
                                kind="content",
                                text=_ensure_spoken_sentence(announce),
                            )
                    # Force la synthèse TTS avant déchargement éventuel
                    yield LlmDelta(kind="flush", text="")

                    messages.append(
                        {
                            "role": "assistant",
                            "content": msg.get("content") or "",
                            "tool_calls": tool_calls,
                        }
                    )
                    specialist_pending_round = False
                    for tc in tool_calls:
                        name, args = parse_tool_call(tc)
                        logger.info(
                            "Tool call round=%s name=%s args=%s",
                            round_i,
                            name,
                            args,
                        )
                        yield LlmDelta(
                            kind="tool_call",
                            text=name,
                            meta={"name": name, "arguments": args},
                        )
                        tool_content = ""
                        tool_meta: dict[str, Any] = {}
                        async for delta in self._run_tool_stream(name, args):
                            if delta.kind == "tool_result":
                                tool_content = delta.text
                                tool_meta = delta.meta or {}
                            yield delta
                        messages.append(
                            {
                                "role": "tool",
                                "tool_name": name,
                                "content": tool_content,
                            }
                        )
                        # Lecture complète : le serveur dicte le texte nettoyé
                        if (
                            name == "read_specialist_reply"
                            and tool_meta.get("ok")
                            and (tool_meta.get("speak_text") or "").strip()
                        ):
                            from .speech_text import markdown_to_speech

                            spoken = markdown_to_speech(
                                str(tool_meta.get("speak_text") or "")
                            )
                            # Découper grossièrement pour le TTS phrase-par-phrase
                            yield LlmDelta(kind="content", text=_ensure_spoken_sentence(spoken))
                            yield LlmDelta(kind="flush", text="")
                            already_yielded_final = True
                            break
                        if key_from_tool_name(name) and tool_meta.get("ok"):
                            specialist_pending_round = True

                    if already_yielded_final:
                        break
                    # Après spécialiste : laisse le LLM résumer + demander lecture
                    if specialist_pending_round:
                        continue
                else:
                    logger.warning("Tools: max rounds (%s) atteint", max_rounds)
            except httpx.HTTPStatusError as exc:
                logger.warning(
                    "Tools indisponibles (%s) — fallback sans tools",
                    exc.response.status_code if exc.response else exc,
                )

        if already_yielded_final:
            return

        base_payload: dict[str, Any] = {
            "model": self._model,
            "stream": True,
            "messages": messages,
            "think": self._enable_thinking,
        }
        try:
            async for delta in self._stream_chat(base_payload, cancel):
                yield delta
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in (400, 422):
                logger.warning("Retry chat stream sans champ think (%s)", exc)
                payload = {k: v for k, v in base_payload.items() if k != "think"}
                async for delta in self._stream_chat(payload, cancel):
                    yield delta
            else:
                raise

    async def _stream_chat(
        self,
        payload: dict[str, Any],
        cancel: asyncio.Event | None,
    ) -> AsyncIterator[LlmDelta]:
        async with self._client.stream("POST", "/api/chat", json=payload) as resp:
            self._stream_resp = resp
            try:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if cancel is not None and cancel.is_set():
                        logger.info("Stream LLM annulé (barge-in / cancel)")
                        break
                    if not line:
                        continue
                    data = json.loads(line)
                    if data.get("done"):
                        break
                    msg = data.get("message") or {}
                    thinking = msg.get("thinking") or ""
                    content = msg.get("content") or ""
                    if thinking:
                        yield LlmDelta(kind="thinking", text=thinking)
                    if content:
                        if "<think>" in content.lower() and not self._enable_thinking:
                            content = strip_think_tags(content)
                            if not content:
                                continue
                        yield LlmDelta(kind="content", text=content)
            finally:
                self._stream_resp = None
