"""TTS Kokoro ONNX (Apache-2.0) — chargement à la demande."""

from __future__ import annotations

import logging
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np

from .catalog import TTS_VOICES
from .config import Settings

logger = logging.getLogger(__name__)

KOKORO_MODEL_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
    "kokoro-v1.0.onnx"
)
KOKORO_VOICES_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
    "voices-v1.0.bin"
)

_VOICE_LANG = {v["id"]: v["lang"] for v in TTS_VOICES}


def _ensure_file(path: str, url: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and p.stat().st_size > 0:
        return
    logger.info("Téléchargement %s → %s", url, path)
    urlretrieve(url, path)


class KokoroTTS:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._kokoro = None
        self._voice: str = settings.kokoro_voice
        self.sample_rate = 24000

    @property
    def loaded(self) -> bool:
        return self._kokoro is not None

    @property
    def voice_id(self) -> str | None:
        return self._voice if self._kokoro is not None else None

    def unload(self) -> None:
        self._kokoro = None

    def load(self, voice_id: str | None = None, **_kwargs) -> None:
        voice = (voice_id or self.settings.kokoro_voice).strip()
        if self._kokoro is not None and self._voice == voice:
            logger.info("Kokoro déjà chargé (voix=%s)", voice)
            return

        _ensure_file(self.settings.kokoro_model_path, KOKORO_MODEL_URL)
        _ensure_file(self.settings.kokoro_voices_path, KOKORO_VOICES_URL)

        from kokoro_onnx import Kokoro

        self._kokoro = Kokoro(
            self.settings.kokoro_model_path,
            self.settings.kokoro_voices_path,
        )
        self._voice = voice
        self.settings.kokoro_voice = voice
        logger.info("Kokoro prêt (voix=%s)", voice)

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        if self._kokoro is None:
            raise RuntimeError("Kokoro non chargé — lance la configuration d'abord")
        text = (text or "").strip()
        if not text:
            return np.zeros(0, dtype=np.float32), self.sample_rate

        preferred = _VOICE_LANG.get(self._voice, "en-us")
        langs = [preferred]
        for fallback in ("fr-fr", "en-us", "en-gb"):
            if fallback not in langs:
                langs.append(fallback)

        last_err: Exception | None = None
        for lang in langs:
            try:
                samples, sample_rate = self._kokoro.create(
                    text,
                    voice=self._voice,
                    speed=self.settings.kokoro_speed,
                    lang=lang,
                )
                audio = np.asarray(samples, dtype=np.float32)
                self.sample_rate = int(sample_rate)
                return audio, self.sample_rate
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                logger.warning("Kokoro lang=%s échoué: %s", lang, exc)
        raise RuntimeError(f"TTS impossible: {last_err}")
