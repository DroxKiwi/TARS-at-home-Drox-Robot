"""Client HTTP vers le service CosyVoice (clonage zero-shot)."""

from __future__ import annotations

import base64
import logging
import time
from pathlib import Path

import httpx
import numpy as np

from .config import Settings

logger = logging.getLogger(__name__)


class CosyVoiceTTS:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._loaded = False
        self._voice_label = "cosyvoice-clone"
        self.sample_rate = 24000
        self._prompt_text = ""
        self._prompt_wav_b64: str | None = None

    @property
    def base_url(self) -> str:
        return self.settings.cosyvoice_url.rstrip("/")

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def voice_id(self) -> str | None:
        return self._voice_label if self._loaded else None

    def unload(self) -> None:
        try:
            httpx.post(f"{self.base_url}/unload", timeout=30.0)
        except Exception:  # noqa: BLE001
            logger.debug("unload CosyVoice remote failed", exc_info=True)
        self._loaded = False

    def health(self) -> dict:
        try:
            r = httpx.get(f"{self.base_url}/health", timeout=5.0)
            r.raise_for_status()
            return {"ok": True, **r.json()}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}

    def load(
        self,
        voice_id: str | None = None,
        *,
        prompt_text: str = "",
        prompt_wav_path: str | None = None,
        prompt_wav_b64: str | None = None,
        use_default_prompt: bool = False,
        **_kwargs,
    ) -> None:
        transcript = (prompt_text or "").strip()
        b64 = prompt_wav_b64
        if not b64 and prompt_wav_path:
            p = Path(prompt_wav_path)
            if not p.exists():
                raise RuntimeError(f"Sample introuvable: {p}")
            b64 = base64.b64encode(p.read_bytes()).decode("ascii")

        if not transcript and not use_default_prompt and not b64:
            raise RuntimeError(
                "CosyVoice: fournis un sample WAV et/ou sa transcription"
            )

        if not transcript:
            # Démo CosyVoice (mandarin dans le sample officiel)
            transcript = "希望你以后能够做的比我还好呦。"
            use_default_prompt = True

        # Attend le prefetch HF (plusieurs Go) avant /load — évite un timeout opaque
        deadline = time.time() + 3600.0
        while time.time() < deadline:
            h = self.health()
            if h.get("ok") and h.get("model_ready"):
                break
            if h.get("download_state") == "error":
                raise RuntimeError(
                    h.get("download_error")
                    or h.get("error")
                    or "Téléchargement CosyVoice échoué"
                )
            pct = h.get("progress_pct")
            missing = h.get("missing") or []
            logger.info(
                "CosyVoice prefetch… %s%% (%s)",
                pct if pct is not None else "?",
                ", ".join(missing[:3]) or h.get("download_state") or "…",
            )
            time.sleep(3.0)
        else:
            raise RuntimeError(
                "Timeout: modèle CosyVoice toujours incomplet après 1 h. "
                "Laisse tars-tts-cosy tourner (sans docker compose up --build)."
            )

        payload = {
            "prompt_text": transcript,
            "prompt_wav_b64": b64,
            "use_default_prompt": bool(use_default_prompt and not b64),
        }
        logger.info("Chargement CosyVoice remote %s …", self.base_url)
        r = httpx.post(
            f"{self.base_url}/load",
            json=payload,
            timeout=httpx.Timeout(900.0, connect=30.0),
        )
        if r.status_code >= 400:
            detail = r.text
            try:
                detail = r.json().get("detail") or detail
            except Exception:  # noqa: BLE001
                pass
            raise RuntimeError(f"CosyVoice load échoué: {detail}")
        data = r.json()
        self.sample_rate = int(data.get("sample_rate") or 24000)
        self._prompt_text = transcript
        self._prompt_wav_b64 = b64
        self._voice_label = voice_id or "cosyvoice-clone"
        self._loaded = True
        logger.info("CosyVoice remote prêt")

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        if not self._loaded:
            raise RuntimeError("CosyVoice non chargé")
        text = (text or "").strip()
        if not text:
            return np.zeros(0, dtype=np.float32), self.sample_rate

        r = httpx.post(
            f"{self.base_url}/synthesize",
            json={"text": text},
            timeout=httpx.Timeout(180.0, connect=10.0),
        )
        if r.status_code >= 400:
            detail = r.text
            try:
                detail = r.json().get("detail") or detail
            except Exception:  # noqa: BLE001
                pass
            raise RuntimeError(f"CosyVoice synth échoué: {detail}")
        data = r.json()
        raw = base64.b64decode(data["audio_b64"])
        audio = np.frombuffer(raw, dtype=np.float32).copy()
        sr = int(data.get("sample_rate") or self.sample_rate)
        self.sample_rate = sr
        return audio, sr
