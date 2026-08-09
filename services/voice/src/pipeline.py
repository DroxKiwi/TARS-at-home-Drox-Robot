"""Pipeline STT → LLM stream → TTS phrase-par-phrase (interruptible)."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from .catalog import catalog
from .config import Settings
from .llm import OllamaLLM, probe_ollama
from .metrics import TurnMetrics
from .orchestrator import VramOrchestrator
from .speech_text import markdown_to_speech
from .stt import SpeechToText
from .tools import ToolContext
from .tts import TextToSpeech

logger = logging.getLogger(__name__)

SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")


@dataclass
class SessionConfig:
    ollama_base_url: str
    ollama_model: str
    stt_model: str
    tts_voice: str
    enable_thinking: bool = False
    tts_backend: str = "kokoro"
    cosy_prompt_text: str = ""
    cosy_prompt_wav_path: str | None = None
    cosy_use_default_prompt: bool = False


@dataclass
class SessionState:
    ready: bool = False
    loading: bool = False
    error: str | None = None
    config: SessionConfig | None = None
    steps: dict[str, str] = field(default_factory=dict)
    # Dernière réponse spécialiste (lecture complète à la demande)
    pending_specialist: dict[str, Any] | None = None


@dataclass
class PipelineEvent:
    kind: Literal[
        "llm_token",
        "thinking_token",
        "tool_call",
        "tool_result",
        "stage",
        "resource",
        "sentence_audio",
        "done",
        "interrupted",
        "metrics_partial",
    ]
    text: str = ""
    audio: np.ndarray | None = None
    sample_rate: int = 0
    meta: dict[str, Any] | None = None
    synthesize_ms: float = 0.0


class VoicePipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.stt = SpeechToText(settings)
        self.llm = OllamaLLM(settings)
        self.tts = TextToSpeech(settings)
        self.session = SessionState()
        self.orch = VramOrchestrator(self)
        self.llm.set_tool_context(ToolContext(orchestrator=self.orch))

    async def close(self) -> None:
        await self.llm.close()

    def set_actor(self, actor: str) -> None:
        self.orch.snap.actor = actor
        self.orch._refresh_snap()

    def status(self) -> dict:
        return {
            "ready": self.session.ready,
            "loading": self.session.loading,
            "error": self.session.error,
            "steps": self.session.steps,
            "resource": self.orch.status(),
            "roles": [
                {
                    "key": r.get("key"),
                    "name": r.get("name"),
                    "tool_name": r.get("tool_name"),
                    "ollama_model": r.get("ollama_model"),
                    "enabled": r.get("enabled"),
                }
                for r in getattr(self.llm, "_roles_cache", [])
                if r.get("enabled")
            ],
            "active": {
                "stt": self.stt.model_id,
                "llm": self.llm.model if self.session.ready else None,
                "tts": self.tts.voice_id,
                "tts_backend": self.tts.backend if self.session.ready else None,
                "ollama_base_url": self.llm.base_url if self.session.ready else None,
                "enable_thinking": self.llm.enable_thinking if self.session.ready else None,
                "history_messages": len(self.llm.history),
                "role_history_messages": int(
                    getattr(self.settings, "role_history_messages", 8) or 8
                ),
                "pending_specialist": bool(self.session.pending_specialist),
                "pending_specialist_role": (
                    (self.session.pending_specialist or {}).get("role_name")
                ),
            },
        }

    async def health(self) -> dict:
        return {
            "service": "tars-voice",
            "session": self.status(),
            "catalog": catalog(),
            "defaults": {
                "ollama_base_url": self.settings.ollama_base_url,
                "ollama_model": self.settings.ollama_model,
                "role_history_messages": int(
                    getattr(self.settings, "role_history_messages", 8) or 8
                ),
                "stt_model": self.settings.whisper_model,
                "tts_voice": self.settings.kokoro_voice,
                "tts_backend": self.settings.tts_backend,
                "enable_thinking": False,
                "silence_ms": self.settings.silence_ms,
                "vad_rms_threshold": self.settings.vad_rms_threshold,
                "barge_in_enabled": self.settings.barge_in_enabled,
                "barge_in_min_speech_ms": self.settings.barge_in_min_speech_ms,
                "cosyvoice_url": self.settings.cosyvoice_url,
            },
        }

    async def load_session(self, cfg: SessionConfig) -> dict:
        if self.session.loading:
            return {"ok": False, "error": "Chargement déjà en cours", **self.status()}

        self.session.loading = True
        self.session.ready = False
        self.session.error = None
        self.session.steps = {
            "ollama": "pending",
            "stt": "pending",
            "tts": "pending",
            "llm_warm": "pending",
        }
        self.session.config = cfg

        try:
            self.session.steps["ollama"] = "testing"
            probe = await probe_ollama(cfg.ollama_base_url)
            if not probe.get("ok"):
                raise RuntimeError(probe.get("error") or "Ollama injoignable")
            if cfg.ollama_model not in probe.get("models", []):
                raise RuntimeError(
                    f"Modèle LLM inconnu sur Ollama: {cfg.ollama_model}"
                )
            await self.llm.configure(
                cfg.ollama_base_url,
                cfg.ollama_model,
                enable_thinking=cfg.enable_thinking,
            )
            self.llm.clear_history()
            self.session.steps["ollama"] = "ok"

            self.session.steps["stt"] = "loading"
            await asyncio.to_thread(self.stt.load, cfg.stt_model)
            self.session.steps["stt"] = "ok"

            prompt_text = cfg.cosy_prompt_text
            if (
                cfg.tts_backend == "cosyvoice"
                and not (prompt_text or "").strip()
                and cfg.cosy_prompt_wav_path
                and not cfg.cosy_use_default_prompt
            ):
                logger.info(
                    "Auto-transcription du sample CosyVoice: %s",
                    cfg.cosy_prompt_wav_path,
                )
                prompt_text = await asyncio.to_thread(
                    self.stt.transcribe_wav_file, cfg.cosy_prompt_wav_path
                )
                if not prompt_text.strip():
                    raise RuntimeError(
                        "Impossible de transcrire le sample — "
                        "vérifie l'audio ou saisis la transcription à la main."
                    )
                cfg.cosy_prompt_text = prompt_text
                logger.info("Prompt CosyVoice STT: %s", prompt_text[:160])

            self.session.steps["tts"] = "loading"
            await asyncio.to_thread(
                self.tts.load,
                cfg.tts_voice,
                backend=cfg.tts_backend,
                prompt_text=prompt_text,
                prompt_wav_path=cfg.cosy_prompt_wav_path,
                use_default_prompt=cfg.cosy_use_default_prompt,
            )
            self.session.steps["tts"] = "ok"

            self.session.steps["llm_warm"] = "loading"
            await self.llm.warm(keep_alive="30m")
            self.session.steps["llm_warm"] = "ok"

            self.session.ready = True
            self.set_actor("chat")
            logger.info(
                "Session prête STT=%s LLM=%s TTS=%s/%s think=%s",
                cfg.stt_model,
                cfg.ollama_model,
                cfg.tts_backend,
                cfg.tts_voice,
                cfg.enable_thinking,
            )
            return {"ok": True, **self.status()}
        except Exception as exc:  # noqa: BLE001
            logger.exception("Échec chargement session")
            self.session.ready = False
            self.session.error = str(exc)
            for k, v in list(self.session.steps.items()):
                if v in ("pending", "testing", "loading"):
                    self.session.steps[k] = "error"
            return {"ok": False, "error": str(exc), **self.status()}
        finally:
            self.session.loading = False

    async def unload_session(self) -> dict:
        """Décharge STT/TTS + modèles Ollama (chat + rôles) pour libérer la VRAM."""
        if self.session.loading:
            return {"ok": False, "error": "Chargement en cours", **self.status()}

        unloaded: list[str] = []
        errors: list[str] = []
        self.set_actor("swapping")

        # Speech
        try:
            await asyncio.to_thread(self.stt.unload)
            unloaded.append("stt")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"stt: {exc}")
        try:
            await asyncio.to_thread(self.tts.unload)
            unloaded.append("tts")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"tts: {exc}")

        # Ollama — chat + modèles liés aux rôles
        models: set[str] = set()
        if self.llm.model:
            models.add(self.llm.model)
        for role in getattr(self.llm, "_roles_cache", []) or []:
            m = (role.get("ollama_model") or "").strip()
            if m:
                models.add(m)

        async def _ollama_unload(model: str) -> None:
            await self.llm._client.post(
                "/api/generate",
                json={"model": model, "keep_alive": 0},
                timeout=120.0,
            )

        for model in sorted(models):
            try:
                await _ollama_unload(model)
                unloaded.append(f"ollama:{model}")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Unload Ollama %s: %s", model, exc)
                errors.append(f"ollama:{model}: {exc}")

        try:
            import gc

            gc.collect()
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

        self.session.ready = False
        self.session.error = None
        self.session.steps = {
            "stt": "unloaded",
            "tts": "unloaded",
            "ollama": "unloaded",
            "llm_warm": "unloaded",
        }
        self.llm.clear_history()
        self.set_actor("idle")
        self.orch.snap.active_role = None
        self.orch.snap.active_role_name = None
        self.orch.snap.heavy_model = None
        self.orch.snap.stage = "unloaded"
        logger.info("Session déchargée unloaded=%s errors=%s", unloaded, errors)
        return {
            "ok": len(errors) == 0,
            "unloaded": unloaded,
            "errors": errors,
            **self.status(),
        }

    def require_ready(self) -> None:
        if not self.session.ready:
            raise RuntimeError("Session non prête — charge la configuration d'abord")

    def transcribe(self, pcm_f32: np.ndarray, sample_rate: int) -> str:
        self.require_ready()
        return self.stt.transcribe(pcm_f32, sample_rate)

    async def respond_stream(
        self,
        user_text: str,
        metrics: TurnMetrics,
        cancel: asyncio.Event | None = None,
    ) -> AsyncIterator[PipelineEvent]:
        """Stream LLM + TTS. cancel → stop ; text de done/interrupted = phrases audio émises."""
        self.require_ready()
        buffer = ""
        full_content: list[str] = []
        spoken_emitted: list[str] = []
        sentence_idx = 0
        cancelled = False

        metrics.start_llm()

        async def _emit_sentence(sentence: str) -> PipelineEvent | None:
            nonlocal sentence_idx, cancelled
            sentence = markdown_to_speech(sentence or "").strip()
            if not sentence:
                return None
            t_tts = time.perf_counter()
            audio, sr = await asyncio.to_thread(self.tts.synthesize, sentence)
            if cancel is not None and cancel.is_set():
                cancelled = True
                return None
            synth_ms = (time.perf_counter() - t_tts) * 1000.0
            metrics.add_tts_sentence(
                index=sentence_idx,
                text=sentence,
                synthesize_ms=synth_ms,
                audio=audio,
                sample_rate=sr,
            )
            spoken_emitted.append(sentence)
            sentence_idx += 1
            return PipelineEvent(
                kind="sentence_audio",
                text=sentence,
                audio=audio,
                sample_rate=sr,
                synthesize_ms=round(synth_ms, 2),
                meta={
                    "sentence_index": sentence_idx - 1,
                    "duration_ms": round(
                        (int(getattr(audio, "size", len(audio))) / max(1, sr))
                        * 1000.0,
                        2,
                    ),
                },
            )

        async for delta in self.llm.chat_stream(user_text, cancel=cancel):
            if cancel is not None and cancel.is_set():
                cancelled = True
                break

            if delta.kind == "thinking":
                metrics.on_thinking_delta(delta.text)
                yield PipelineEvent(kind="thinking_token", text=delta.text)
                continue

            if delta.kind == "flush":
                # Synthèse immédiate (annonce avant unload VRAM / citation)
                tail = buffer.strip()
                buffer = ""
                if tail:
                    pev = await _emit_sentence(tail)
                    if pev is not None:
                        yield pev
                if cancelled:
                    break
                continue

            if delta.kind == "tool_call":
                yield PipelineEvent(
                    kind="tool_call",
                    text=delta.text,
                    meta=delta.meta or None,
                )
                continue

            if delta.kind == "stage":
                yield PipelineEvent(
                    kind="stage",
                    text=delta.text,
                    meta=delta.meta or None,
                )
                continue

            if delta.kind == "resource":
                yield PipelineEvent(
                    kind="resource",
                    text=delta.text,
                    meta=delta.meta or None,
                )
                continue

            if delta.kind == "tool_result":
                yield PipelineEvent(
                    kind="tool_result",
                    text=delta.text,
                    meta=delta.meta or None,
                )
                continue

            metrics.on_content_delta(delta.text)
            full_content.append(delta.text)
            yield PipelineEvent(kind="llm_token", text=delta.text)
            buffer += delta.text

            while True:
                if cancel is not None and cancel.is_set():
                    cancelled = True
                    break
                m = SENTENCE_END.search(buffer)
                if not m:
                    break
                sentence = buffer[: m.end()].strip()
                buffer = buffer[m.end() :]
                pev = await _emit_sentence(sentence)
                if pev is not None:
                    yield pev
                if cancelled:
                    break
            if cancelled:
                break

        metrics.end_llm_stream()

        if cancel is not None and cancel.is_set():
            cancelled = True

        if not cancelled:
            tail = buffer.strip()
            if tail:
                pev = await _emit_sentence(tail)
                if pev is not None:
                    yield pev
                if cancelled:
                    pass

        emitted = " ".join(spoken_emitted).strip()
        full = "".join(full_content).strip()
        if cancelled:
            metrics.interrupted = True
            logger.info(
                "Réponse interrompue spoken_emitted=%d chars llm=%d chars",
                len(emitted),
                len(full),
            )
            yield PipelineEvent(kind="interrupted", text=emitted)
        else:
            logger.info("Réponse LLM: %s", full[:200])
            yield PipelineEvent(kind="done", text=emitted or full)
