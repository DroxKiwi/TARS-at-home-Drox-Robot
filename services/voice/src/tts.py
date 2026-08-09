"""Facade TTS — Kokoro (défaut) ou CosyVoice (clonage optionnel)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .config import Settings
from .tts_cosyvoice import CosyVoiceTTS
from .tts_kokoro import KokoroTTS

logger = logging.getLogger(__name__)


class TextToSpeech:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.backend = "kokoro"
        self._kokoro = KokoroTTS(settings)
        self._cosy = CosyVoiceTTS(settings)
        self._active: KokoroTTS | CosyVoiceTTS = self._kokoro

    @property
    def loaded(self) -> bool:
        return self._active.loaded

    @property
    def voice_id(self) -> str | None:
        return self._active.voice_id

    @property
    def sample_rate(self) -> int:
        return int(getattr(self._active, "sample_rate", 24000))

    def unload(self) -> None:
        self._kokoro.unload()
        self._cosy.unload()

    def load(self, voice_id: str | None = None, **kwargs: Any) -> None:
        backend = (kwargs.pop("backend", None) or self.settings.tts_backend or "kokoro")
        backend = str(backend).strip().lower()
        if backend not in ("kokoro", "cosyvoice"):
            raise RuntimeError(f"Backend TTS inconnu: {backend}")

        # Un seul moteur en mémoire GPU/CPU à la fois
        if backend == "kokoro":
            self._cosy.unload()
            self._kokoro.load(voice_id, **kwargs)
            self._active = self._kokoro
        else:
            self._kokoro.unload()
            self._cosy.load(voice_id, **kwargs)
            self._active = self._cosy
        self.backend = backend
        self.settings.tts_backend = backend
        logger.info("TTS backend actif=%s voice=%s", backend, self.voice_id)

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        return self._active.synthesize(text)

    def cosy_health(self) -> dict:
        return self._cosy.health()
