"""Orchestration VRAM : chat léger ↔ spécialiste (rôle / modèle)."""

from __future__ import annotations

import asyncio
import gc
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

ProgressCb = Callable[[dict[str, Any]], Awaitable[None]]


def _format_specialist_tool_result(
    *,
    role_name: str,
    reply: str,
    actions: list[str] | None,
    ok: bool,
    total_ms: float,
    model: str,
) -> str:
    """Texte renvoyé au LLM chat : résumé + proposition de lecture complète."""
    acts = [a.strip() for a in (actions or []) if a and str(a).strip()]
    if not ok:
        return (
            f"Echec du specialiste « {role_name} » ({model}).\n"
            f"Detail: {reply.strip() or 'erreur inconnue'}\n\n"
            "Consigne pour toi (TARS): dis a l'oral que le specialiste "
            "n'a pas pu repondre, en une phrase courte."
        )
    actions_block = (
        "\n".join(f"- {a}" for a in acts) if acts else "(aucune)"
    )
    return (
        f"## Reponse complete du specialiste « {role_name} » ({model}, {total_ms} ms)\n"
        f"(visible aussi dans l'interface — ne la lis PAS en entier maintenant)\n\n"
        f"{reply.strip()}\n\n"
        f"## Actions effectuees par le specialiste\n"
        f"{actions_block}\n\n"
        "## Consigne pour toi (TARS) — reponse orale obligatoire\n"
        "1. Interprete et resume la reponse du specialiste en 1 a 3 phrases orales, "
        "claires, sans markdown ni listes a puces.\n"
        "2. Puis demande explicitement a l'utilisateur s'il veut ecouter "
        "la reponse complete "
        "(ex. « Tu veux que je te lise la reponse complete ? »).\n"
        "3. Si plus tard l'utilisateur accepte, appelle l'outil "
        "read_specialist_reply — ne reinvente pas le texte.\n"
        "N'attribue pas le travail a toi-meme : c'est le specialiste."
    )


@dataclass
class ResourceSnapshot:
    actor: str = "idle"  # idle | chat | speech | heavy | swapping
    chat_model: str | None = None
    heavy_model: str | None = None
    active_role: str | None = None
    active_role_name: str | None = None
    stt_loaded: bool = False
    stt_id: str | None = None
    tts_loaded: bool = False
    tts_backend: str | None = None
    tts_id: str | None = None
    stage: str | None = None
    stages: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "actor": self.actor,
            "chat_model": self.chat_model,
            "heavy_model": self.heavy_model,
            "active_role": self.active_role,
            "active_role_name": self.active_role_name,
            "stt_loaded": self.stt_loaded,
            "stt_id": self.stt_id,
            "tts_loaded": self.tts_loaded,
            "tts_backend": self.tts_backend,
            "tts_id": self.tts_id,
            "stage": self.stage,
            "stages": list(self.stages),
        }


class VramOrchestrator:
    """Swap speech + chat LLM pour laisser la place à un spécialiste."""

    def __init__(self, pipeline: Any) -> None:
        self.pipeline = pipeline
        self.snap = ResourceSnapshot()
        self._lock = asyncio.Lock()
        self._refresh_snap()

    def _refresh_snap(self) -> None:
        p = self.pipeline
        self.snap.chat_model = p.llm.model
        self.snap.stt_loaded = p.stt.loaded
        self.snap.stt_id = p.stt.model_id
        self.snap.tts_loaded = bool(getattr(p.tts, "loaded", False))
        self.snap.tts_backend = getattr(p.tts, "backend", None)
        self.snap.tts_id = p.tts.voice_id

    def status(self) -> dict[str, Any]:
        self._refresh_snap()
        return self.snap.to_dict()

    async def _emit(self, on_progress: ProgressCb | None, event: dict[str, Any]) -> None:
        self.snap.stage = event.get("stage")
        if event.get("actor"):
            self.snap.actor = event["actor"]
        if "active_role" in event:
            self.snap.active_role = event.get("active_role")
        if "active_role_name" in event:
            self.snap.active_role_name = event.get("active_role_name")
        if "heavy_model" in event:
            self.snap.heavy_model = event.get("heavy_model")
        if event.get("stage") and event.get("ms") is not None:
            self.snap.stages.append(
                {
                    "stage": event["stage"],
                    "ms": event.get("ms"),
                    "ok": event.get("ok", True),
                    "detail": event.get("detail"),
                }
            )
        if on_progress:
            await on_progress({**event, "resource": self.status()})

    async def _ollama_unload(self, model: str) -> None:
        client = self.pipeline.llm._client
        await client.post(
            "/api/generate",
            json={"model": model, "keep_alive": 0},
            timeout=120.0,
        )

    async def _ollama_warm(self, model: str, keep_alive: str = "30m") -> None:
        client = self.pipeline.llm._client
        await client.post(
            "/api/generate",
            json={
                "model": model,
                "prompt": "",
                "stream": False,
                "keep_alive": keep_alive,
            },
            timeout=600.0,
        )

    async def _ollama_chat_role(
        self,
        *,
        model: str,
        system_prompt: str,
        role_name: str,
        task: str,
        history_block: str,
        function_keys: list[str] | None = None,
        on_progress: ProgressCb | None = None,
    ) -> tuple[str, list[str]]:
        from .functions import format_action, schemas_for_keys
        from .tools import ToolContext, execute_tool, parse_tool_call

        client = self.pipeline.llm._client
        tools = schemas_for_keys(function_keys or [])
        allow = set(function_keys or [])
        tools_hint = ""
        if tools:
            tools_hint = (
                " Tu disposes d'outils assignes a ton role ; "
                "utilise-les si utile puis reponds clairement."
            )
        framing = (
            f"Tu es le specialiste « {role_name} », appele en renfort "
            "par un compagnon vocal (TARS). "
            "Tu recois l'historique recent de la discussion et une consigne. "
            "Reponds de facon precise et utile pour que TARS puisse resumer a l'oral. "
            "Pas d'emojis. Francais par defaut sauf si la tache est en autre langue."
            + tools_hint
        )
        system = f"{system_prompt.strip()}\n\n---\n{framing}".strip()
        user = (
            "## Ou tu te trouves\n"
            "Conversation vocale en cours entre l'utilisateur et TARS "
            "(compagnon leger). On t'appelle pour une expertise ponctuelle.\n\n"
            "## Historique recent\n"
            f"{history_block.strip()}\n\n"
            "## Consigne de TARS\n"
            f"{task.strip()}"
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        actions: list[str] = []
        max_rounds = max(1, int(getattr(self.pipeline.settings, "tools_max_rounds", 4)))
        ctx = ToolContext(
            orchestrator=self,
            allowed_function_keys=allow if allow else set(),
            caller="specialist",
        )
        # Si aucune fonction : allow vide bloque tout — passer None pour n'envoyer aucun tool
        if not allow:
            ctx.allowed_function_keys = set()

        answer = ""
        for round_i in range(max_rounds if tools else 1):
            payload: dict[str, Any] = {
                "model": model,
                "stream": False,
                "keep_alive": "5m",
                "messages": messages,
            }
            if tools:
                payload["tools"] = tools
            r = await client.post("/api/chat", json=payload, timeout=600.0)
            r.raise_for_status()
            msg = r.json().get("message") or {}
            tool_calls = msg.get("tool_calls") or []
            content = (msg.get("content") or "").strip()
            if not tool_calls:
                answer = content
                break

            messages.append(
                {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tool_calls,
                }
            )
            for tc in tool_calls:
                name, args = parse_tool_call(tc)
                await self._emit(
                    on_progress,
                    {
                        "kind": "stage",
                        "stage": "specialist_tool",
                        "actor": "heavy",
                        "detail": format_action(name, args),
                        "ok": True,
                        "ms": 0,
                    },
                )
                result = await execute_tool(name, args, ctx=ctx)
                actions.append(format_action(name, args))
                if result.get("ui") and on_progress:
                    await on_progress(
                        {
                            "kind": "specialist_tool_ui",
                            "ui": result.get("ui"),
                            "name": name,
                            "ok": result.get("ok"),
                            "resource": self.status(),
                        }
                    )
                messages.append(
                    {
                        "role": "tool",
                        "tool_name": name,
                        "content": result.get("content") or "",
                    }
                )
        else:
            if not answer:
                answer = content

        return answer.strip(), actions

    async def ask_role(
        self,
        *,
        role: dict[str, Any],
        task: str,
        history_block: str = "",
        on_progress: ProgressCb | None = None,
    ) -> dict[str, Any]:
        settings = self.pipeline.settings
        heavy = (role.get("ollama_model") or "").strip()
        role_key = str(role.get("key") or "")
        role_name = str(role.get("name") or role_key)
        system_prompt = str(role.get("system_prompt") or "").strip()
        if not heavy:
            return {
                "ok": False,
                "content": f"Rôle {role_key}: aucun modèle Ollama lié.",
            }
        if not system_prompt:
            return {
                "ok": False,
                "content": f"Rôle {role_key}: system_prompt vide.",
            }

        chat_model = self.pipeline.llm.model
        stt_id = self.pipeline.stt.model_id or settings.whisper_model
        tts_voice = self.pipeline.tts.voice_id or settings.kokoro_voice
        tts_backend = getattr(self.pipeline.tts, "backend", "kokoro") or "kokoro"
        session_cfg = self.pipeline.session.config

        async with self._lock:
            self.snap.stages = []
            t0 = time.perf_counter()
            await self._emit(
                on_progress,
                {
                    "kind": "stage_start",
                    "stage": "delegate_start",
                    "actor": "swapping",
                    "active_role": role_key,
                    "active_role_name": role_name,
                    "heavy_model": heavy,
                    "detail": f"{role_name} → {heavy}",
                },
            )

            t = time.perf_counter()
            await self._emit(
                on_progress,
                {
                    "kind": "stage_start",
                    "stage": "unload_speech",
                    "actor": "swapping",
                    "active_role": role_key,
                    "heavy_model": heavy,
                },
            )
            try:
                await asyncio.to_thread(self.pipeline.stt.unload)
                await asyncio.to_thread(self.pipeline.tts.unload)
                gc.collect()
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:  # noqa: BLE001
                    pass
                await self._emit(
                    on_progress,
                    {
                        "kind": "stage",
                        "stage": "unload_speech",
                        "actor": "swapping",
                        "ok": True,
                        "ms": round((time.perf_counter() - t) * 1000, 1),
                    },
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("unload speech")
                await self._emit(
                    on_progress,
                    {
                        "kind": "stage",
                        "stage": "unload_speech",
                        "ok": False,
                        "detail": str(exc),
                        "ms": round((time.perf_counter() - t) * 1000, 1),
                    },
                )

            t = time.perf_counter()
            await self._emit(
                on_progress,
                {
                    "kind": "stage_start",
                    "stage": "unload_chat_llm",
                    "actor": "swapping",
                    "detail": chat_model,
                },
            )
            try:
                await self._ollama_unload(chat_model)
                await self._emit(
                    on_progress,
                    {
                        "kind": "stage",
                        "stage": "unload_chat_llm",
                        "ok": True,
                        "ms": round((time.perf_counter() - t) * 1000, 1),
                    },
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("unload chat llm: %s", exc)
                await self._emit(
                    on_progress,
                    {
                        "kind": "stage",
                        "stage": "unload_chat_llm",
                        "ok": False,
                        "detail": str(exc),
                        "ms": round((time.perf_counter() - t) * 1000, 1),
                    },
                )

            answer = ""
            actions: list[str] = []
            heavy_ok = False
            t = time.perf_counter()
            await self._emit(
                on_progress,
                {
                    "kind": "stage_start",
                    "stage": "load_heavy",
                    "actor": "heavy",
                    "active_role": role_key,
                    "active_role_name": role_name,
                    "heavy_model": heavy,
                    "detail": heavy,
                },
            )
            try:
                await self._ollama_warm(heavy, keep_alive="10m")
                await self._emit(
                    on_progress,
                    {
                        "kind": "stage",
                        "stage": "load_heavy",
                        "ok": True,
                        "ms": round((time.perf_counter() - t) * 1000, 1),
                    },
                )
                t = time.perf_counter()
                await self._emit(
                    on_progress,
                    {
                        "kind": "stage_start",
                        "stage": "heavy_run",
                        "actor": "heavy",
                        "active_role": role_key,
                        "heavy_model": heavy,
                    },
                )
                answer, actions = await self._ollama_chat_role(
                    model=heavy,
                    system_prompt=system_prompt,
                    role_name=role_name,
                    task=task,
                    history_block=history_block,
                    function_keys=list(role.get("function_keys") or []),
                    on_progress=on_progress,
                )
                heavy_ok = bool(answer) or bool(actions)
                await self._emit(
                    on_progress,
                    {
                        "kind": "stage",
                        "stage": "heavy_run",
                        "ok": heavy_ok,
                        "ms": round((time.perf_counter() - t) * 1000, 1),
                        "detail": f"{len(answer)} chars, {len(actions)} actions",
                    },
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("role model")
                actions = []
                await self._emit(
                    on_progress,
                    {
                        "kind": "stage",
                        "stage": "heavy_run",
                        "ok": False,
                        "detail": str(exc),
                        "ms": round((time.perf_counter() - t) * 1000, 1),
                    },
                )
                answer = f"Echec specialiste {role_name}: {exc}"

            t = time.perf_counter()
            await self._emit(
                on_progress,
                {
                    "kind": "stage_start",
                    "stage": "unload_heavy",
                    "actor": "swapping",
                },
            )
            try:
                await self._ollama_unload(heavy)
                await self._emit(
                    on_progress,
                    {
                        "kind": "stage",
                        "stage": "unload_heavy",
                        "ok": True,
                        "ms": round((time.perf_counter() - t) * 1000, 1),
                    },
                )
            except Exception as exc:  # noqa: BLE001
                await self._emit(
                    on_progress,
                    {
                        "kind": "stage",
                        "stage": "unload_heavy",
                        "ok": False,
                        "detail": str(exc),
                        "ms": round((time.perf_counter() - t) * 1000, 1),
                    },
                )

            t = time.perf_counter()
            await self._emit(
                on_progress,
                {
                    "kind": "stage_start",
                    "stage": "reload_chat_llm",
                    "actor": "swapping",
                    "detail": chat_model,
                },
            )
            try:
                await self.pipeline.llm.configure(model=chat_model)
                await self.pipeline.llm.warm(keep_alive="30m")
                await self._emit(
                    on_progress,
                    {
                        "kind": "stage",
                        "stage": "reload_chat_llm",
                        "ok": True,
                        "ms": round((time.perf_counter() - t) * 1000, 1),
                    },
                )
            except Exception as exc:  # noqa: BLE001
                await self._emit(
                    on_progress,
                    {
                        "kind": "stage",
                        "stage": "reload_chat_llm",
                        "ok": False,
                        "detail": str(exc),
                        "ms": round((time.perf_counter() - t) * 1000, 1),
                    },
                )

            t = time.perf_counter()
            await self._emit(
                on_progress,
                {
                    "kind": "stage_start",
                    "stage": "reload_speech",
                    "actor": "swapping",
                },
            )
            try:
                await asyncio.to_thread(self.pipeline.stt.load, stt_id)
                load_kwargs: dict[str, Any] = {"backend": tts_backend}
                if session_cfg and tts_backend == "cosyvoice":
                    load_kwargs["prompt_text"] = session_cfg.cosy_prompt_text
                    load_kwargs["prompt_wav_path"] = session_cfg.cosy_prompt_wav_path
                    load_kwargs["use_default_prompt"] = (
                        session_cfg.cosy_use_default_prompt
                    )
                await asyncio.to_thread(
                    self.pipeline.tts.load, tts_voice, **load_kwargs
                )
                await self._emit(
                    on_progress,
                    {
                        "kind": "stage",
                        "stage": "reload_speech",
                        "ok": True,
                        "ms": round((time.perf_counter() - t) * 1000, 1),
                    },
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("reload speech")
                await self._emit(
                    on_progress,
                    {
                        "kind": "stage",
                        "stage": "reload_speech",
                        "ok": False,
                        "detail": str(exc),
                        "ms": round((time.perf_counter() - t) * 1000, 1),
                    },
                )

            total_ms = round((time.perf_counter() - t0) * 1000, 1)
            await self._emit(
                on_progress,
                {
                    "kind": "stage",
                    "stage": "delegate_done",
                    "actor": "chat",
                    "ok": heavy_ok,
                    "ms": total_ms,
                    "detail": f"total {total_ms} ms",
                    "active_role": None,
                    "active_role_name": None,
                    "heavy_model": None,
                },
            )
            self.snap.actor = "chat"
            self.snap.active_role = None
            self.snap.active_role_name = None
            self.snap.heavy_model = None
            self._refresh_snap()

            content = _format_specialist_tool_result(
                role_name=role_name,
                reply=answer,
                actions=actions,
                ok=heavy_ok,
                total_ms=total_ms,
                model=heavy,
            )
            return {
                "ok": heavy_ok,
                "content": content,
                "ui": {
                    "action": "role_delegate",
                    "role_key": role_key,
                    "role_name": role_name,
                    "heavy_model": heavy,
                    "total_ms": total_ms,
                    "reply": answer if heavy_ok else "",
                    "actions": list(actions),
                    "function_keys": list(role.get("function_keys") or []),
                    "stages": list(self.snap.stages),
                },
                "pending_specialist": (
                    {
                        "role_key": role_key,
                        "role_name": role_name,
                        "reply": answer,
                        "actions": list(actions),
                        "model": heavy,
                    }
                    if heavy_ok and answer
                    else None
                ),
            }
