"""TTS CosyVoice 3 — thin HTTP wrapper autour de AutoModel (API officielle)."""

from __future__ import annotations

import base64
import logging
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("tars.cosy")

COSY_ROOT = Path(os.environ.get("COSYVOICE_ROOT", "/opt/CosyVoice"))
MODEL_DIR = Path(
    os.environ.get(
        "COSYVOICE_MODEL_DIR",
        "/root/.cache/cosyvoice/Fun-CosyVoice3-0.5B",
    )
)
HF_REPO = os.environ.get(
    "COSYVOICE_HF_REPO", "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
)
DEFAULT_PROMPT = Path(
    os.environ.get(
        "COSYVOICE_DEFAULT_PROMPT",
        str(COSY_ROOT / "asset" / "zero_shot_prompt.wav"),
    )
)
_EXPECTED_MODEL_BYTES = int(
    os.environ.get("COSYVOICE_EXPECTED_BYTES", str(3_200_000_000))
)

_lock = threading.Lock()
_model = None
_prompt_wav: Path | None = None
_prompt_text: str = ""
_spk_id: str = ""
_sample_rate = 24000
_USE_FP16 = os.environ.get("COSYVOICE_FP16", "1").strip() not in ("0", "false", "False")

_prefetch_lock = threading.Lock()
_prefetch_state = "idle"
_prefetch_error: str | None = None
_prefetch_started_at: float | None = None
_prefetch_thread: threading.Thread | None = None


def _ensure_sys_path() -> None:
    root = str(COSY_ROOT)
    matcha = str(COSY_ROOT / "third_party" / "Matcha-TTS")
    for p in (root, matcha):
        if p not in sys.path:
            sys.path.insert(0, p)


def _dir_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def _missing_parts() -> list[str]:
    missing: list[str] = []
    if not (MODEL_DIR / "cosyvoice3.yaml").exists():
        missing.append("cosyvoice3.yaml")
    blank = MODEL_DIR / "CosyVoice-BlankEN"
    weight_ok = any(
        (blank / name).exists() and (blank / name).stat().st_size > 50_000_000
        for name in ("model.safetensors", "pytorch_model.bin")
    )
    if not weight_ok:
        missing.append("CosyVoice-BlankEN/model.safetensors")
    for name in ("llm.pt", "flow.pt", "hift.pt"):
        p = MODEL_DIR / name
        if not p.exists() or p.stat().st_size < 1_000:
            missing.append(name)
    return missing


def _model_complete() -> bool:
    return not _missing_parts()


def _download_model() -> None:
    global _prefetch_state, _prefetch_error, _prefetch_started_at
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if _model_complete():
        _prefetch_state = "ready"
        _prefetch_error = None
        logger.info("Modèle CosyVoice complet: %s", MODEL_DIR)
        return

    with _prefetch_lock:
        if _prefetch_state == "downloading":
            return
        _prefetch_state = "downloading"
        _prefetch_error = None
        _prefetch_started_at = time.time()

    logger.warning(
        "Téléchargement HF %s → %s (ne redémarre pas Docker)",
        HF_REPO,
        MODEL_DIR,
    )
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=HF_REPO, local_dir=str(MODEL_DIR), max_workers=4)
        if not _model_complete():
            raise RuntimeError(
                f"Téléchargement incomplet (manque: {_missing_parts()})"
            )
        _prefetch_state = "ready"
        _prefetch_error = None
        logger.info("Téléchargement CosyVoice OK (%.1f Go)", _dir_size(MODEL_DIR) / 1e9)
    except Exception as exc:  # noqa: BLE001
        _prefetch_state = "error"
        _prefetch_error = str(exc)
        logger.exception("Échec téléchargement CosyVoice")
        raise


def _start_prefetch() -> None:
    global _prefetch_thread
    if _model_complete():
        _prefetch_state = "ready"
        return
    if _prefetch_thread and _prefetch_thread.is_alive():
        return

    def worker() -> None:
        try:
            _download_model()
        except Exception:  # noqa: BLE001
            pass

    _prefetch_thread = threading.Thread(target=worker, name="cosy-prefetch", daemon=True)
    _prefetch_thread.start()


def _wait_model_ready(timeout_s: float = 3600.0) -> None:
    if _model_complete():
        _prefetch_state = "ready"
        return
    if _prefetch_state != "downloading":
        _start_prefetch()
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _model_complete() and _prefetch_state == "ready":
            return
        if _prefetch_state == "error":
            raise RuntimeError(_prefetch_error or "Téléchargement CosyVoice échoué")
        time.sleep(2.0)
    raise RuntimeError(
        f"Timeout téléchargement CosyVoice ({timeout_s:.0f}s, "
        f"{_dir_size(MODEL_DIR) / 1e9:.1f} Go)"
    )


def _load_model() -> None:
    global _model, _sample_rate
    if _model is not None:
        return
    _ensure_sys_path()
    _wait_model_ready()
    from cosyvoice.cli.cosyvoice import AutoModel

    logger.info("Chargement AutoModel %s (fp16=%s)", MODEL_DIR, _USE_FP16)
    _model = AutoModel(model_dir=str(MODEL_DIR), fp16=_USE_FP16)
    _sample_rate = int(getattr(_model, "sample_rate", 24000))
    logger.info("CosyVoice prêt (sr=%s fp16=%s)", _sample_rate, _USE_FP16)


def _write_wav_b64(b64: str, dest: Path) -> Path:
    raw = base64.b64decode(b64)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".upload.bin")
    tmp.write_bytes(raw)
    try:
        audio, sr = sf.read(tmp, dtype="float32", always_2d=False)
        if getattr(audio, "ndim", 1) > 1:
            audio = np.mean(audio, axis=1)
        sf.write(dest, audio, int(sr))
        tmp.unlink(missing_ok=True)
    except Exception:
        tmp.replace(dest)
    return dest


def _status_payload() -> dict:
    size = _dir_size(MODEL_DIR)
    missing = _missing_parts()
    elapsed = None
    if _prefetch_started_at and _prefetch_state == "downloading":
        elapsed = round(time.time() - _prefetch_started_at, 1)
    return {
        "service": "tars-tts-cosy",
        "loaded": _model is not None,
        "model_ready": _model_complete(),
        "download_state": "ready" if _model_complete() else _prefetch_state,
        "download_error": _prefetch_error,
        "bytes_on_disk": size,
        "bytes_expected": _EXPECTED_MODEL_BYTES,
        "progress_pct": (
            100
            if _model_complete()
            else min(99, int(100 * size / max(1, _EXPECTED_MODEL_BYTES)))
        ),
        "missing": missing,
        "elapsed_s": elapsed,
        "prompt_ready": _prompt_wav is not None and bool(_prompt_text),
        "sample_rate": _sample_rate,
        "model_dir": str(MODEL_DIR),
        "hf_repo": HF_REPO,
        "license": "Apache-2.0",
    }


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info(
        "CosyVoice TTS — model_dir=%s cosy_root=%s", MODEL_DIR, COSY_ROOT
    )
    _start_prefetch()
    yield
    global _model
    _model = None


app = FastAPI(title="TARS CosyVoice TTS", lifespan=lifespan)


class LoadRequest(BaseModel):
    prompt_text: str = Field(..., description="Transcript du sample de référence")
    prompt_wav_b64: str | None = None
    use_default_prompt: bool = False


class SynthRequest(BaseModel):
    text: str


@app.get("/health")
def health():
    return _status_payload()


@app.post("/load")
def load(body: LoadRequest):
    global _prompt_wav, _prompt_text, _spk_id
    with _lock:
        try:
            _load_model()
            prompt_dir = Path("/data/voices")
            prompt_dir.mkdir(parents=True, exist_ok=True)
            dest = prompt_dir / "active_prompt.wav"

            if body.prompt_wav_b64:
                _write_wav_b64(body.prompt_wav_b64, dest)
            elif body.use_default_prompt or not dest.exists():
                if not DEFAULT_PROMPT.exists():
                    raise RuntimeError(f"Sample démo introuvable: {DEFAULT_PROMPT}")
                dest.write_bytes(DEFAULT_PROMPT.read_bytes())

            try:
                info = sf.info(str(dest))
                dur = float(info.duration)
                if dur > 30.0:
                    raise RuntimeError(
                        f"Sample trop long ({dur:.1f}s) — CosyVoice limite à 30s"
                    )
            except RuntimeError:
                raise
            except Exception:  # noqa: BLE001
                logger.warning("Impossible de lire la durée du sample", exc_info=True)

            transcript = (body.prompt_text or "").strip()
            if not transcript:
                raise RuntimeError("prompt_text requis")

            if "<|endofprompt|>" not in transcript:
                prompt_text = (
                    f"You are a helpful assistant.<|endofprompt|>{transcript}"
                )
            else:
                prompt_text = transcript

            _prompt_wav = dest
            _prompt_text = prompt_text
            # Cache speaker une fois — sinon CosyVoice ré-extrait le WAV à chaque phrase
            _spk_id = "tars_clone"
            norm_prompt = _model.frontend.text_normalize(
                prompt_text, split=False, text_frontend=True
            )
            t0 = time.perf_counter()
            _model.add_zero_shot_spk(norm_prompt, str(dest), _spk_id)
            logger.info(
                "Speaker cache OK id=%s (%.0f ms) — extractions prompt évitées ensuite",
                _spk_id,
                (time.perf_counter() - t0) * 1000,
            )
            # Warm-up court avec speaker déjà en cache
            warm = (
                "Systems nominal. Confirmed. Standing by for further instructions "
                "and awaiting your next command regarding mission parameters."
            )
            t1 = time.perf_counter()
            _synthesize_locked(warm)
            logger.info("Warm-up synth %.0f ms", (time.perf_counter() - t1) * 1000)
            return {
                "ok": True,
                "prompt_wav": str(dest),
                "sample_rate": _sample_rate,
                "spk_id": _spk_id,
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("Échec load CosyVoice")
            raise HTTPException(status_code=500, detail=str(exc)) from exc


def _synthesize_locked(text: str) -> tuple[np.ndarray, int]:
    global _model, _prompt_wav, _prompt_text, _spk_id, _sample_rate
    if _model is None or _prompt_wav is None or not _prompt_text:
        raise RuntimeError("CosyVoice non chargé — POST /load d'abord")
    text = (text or "").strip()
    if not text:
        return np.zeros(0, dtype=np.float32), _sample_rate

    t0 = time.perf_counter()
    chunks: list[np.ndarray] = []
    # zero_shot_spk_id : réutilise embeddings/tokens du prompt (gros gain latence)
    for _i, item in enumerate(
        _model.inference_zero_shot(
            text,
            _prompt_text,
            str(_prompt_wav),
            zero_shot_spk_id=_spk_id or "",
            stream=False,
        )
    ):
        speech = item["tts_speech"]
        arr = speech.squeeze().detach().cpu().numpy().astype(np.float32)
        chunks.append(arr)
    if not chunks:
        return np.zeros(0, dtype=np.float32), _sample_rate
    audio = np.concatenate(chunks)
    dur_s = float(audio.size) / float(_sample_rate)
    elapsed = time.perf_counter() - t0
    logger.info(
        "synth %.0f ms · audio %.1f s · RTF %.2f · spk=%s",
        elapsed * 1000,
        dur_s,
        elapsed / max(dur_s, 1e-3),
        _spk_id or "(raw)",
    )
    return audio, _sample_rate


@app.post("/synthesize")
def synthesize(body: SynthRequest):
    with _lock:
        try:
            audio, sr = _synthesize_locked(body.text)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Synth CosyVoice")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "sample_rate": sr,
        "audio_b64": base64.b64encode(audio.astype(np.float32).tobytes()).decode("ascii"),
        "samples": int(audio.size),
        "dtype": "float32",
    }


@app.post("/unload")
def unload():
    global _model
    with _lock:
        _model = None
    return {"ok": True}
