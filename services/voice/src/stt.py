"""STT via faster-whisper (MIT) — chargement à la demande."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from faster_whisper import WhisperModel

from .config import Settings

logger = logging.getLogger(__name__)


class SpeechToText:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model: Optional[WhisperModel] = None
        self._loaded_id: str | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def model_id(self) -> str | None:
        return self._loaded_id

    def unload(self) -> None:
        self._model = None
        self._loaded_id = None

    def load(self, model_id: str | None = None) -> None:
        mid = (model_id or self.settings.whisper_model).strip()
        if self._model is not None and self._loaded_id == mid:
            logger.info("Whisper déjà chargé: %s", mid)
            return

        self.unload()
        logger.info(
            "Chargement Whisper %s (%s / %s)",
            mid,
            self.settings.whisper_device,
            self.settings.whisper_compute_type,
        )
        try:
            self._model = WhisperModel(
                mid,
                device=self.settings.whisper_device,
                compute_type=self.settings.whisper_compute_type,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("CUDA Whisper indisponible (%s) — fallback CPU int8", exc)
            self._model = WhisperModel(mid, device="cpu", compute_type="int8")
        self._loaded_id = mid
        self.settings.whisper_model = mid

    def transcribe(self, audio_f32: np.ndarray, sample_rate: int) -> str:
        if self._model is None:
            raise RuntimeError("Whisper non chargé — lance la configuration d'abord")
        if sample_rate != 16000:
            duration = len(audio_f32) / sample_rate
            target = int(duration * 16000)
            x_old = np.linspace(0.0, 1.0, num=len(audio_f32), endpoint=False)
            x_new = np.linspace(0.0, 1.0, num=target, endpoint=False)
            audio_f32 = np.interp(x_new, x_old, audio_f32).astype(np.float32)

        segments, _info = self._model.transcribe(
            audio_f32,
            language=self.settings.whisper_language or None,
            vad_filter=True,
            beam_size=1,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()

    def transcribe_wav_file(self, path: str) -> str:
        """Transcrit un fichier audio (WAV/etc.) pour prompt CosyVoice."""
        import soundfile as sf

        audio, sr = sf.read(path, dtype="float32", always_2d=False)
        if getattr(audio, "ndim", 1) > 1:
            audio = np.mean(audio, axis=1).astype(np.float32)
        text = self.transcribe(np.asarray(audio, dtype=np.float32), int(sr))
        logger.info("STT fichier %s → %d chars", path, len(text))
        return text
