"""Catalogue des modèles disponibles (licences permissives)."""

from __future__ import annotations

# faster-whisper / CTranslate2 — poids Whisper MIT
STT_MODELS: list[dict[str, str]] = [
    {
        "id": "large-v3-turbo",
        "label": "Whisper large-v3-turbo (recommandé · ~1.5 Go)",
        "vram": "~1.5 Go",
    },
    {
        "id": "distil-large-v3",
        "label": "Distil-Whisper large-v3 (rapide · ~1.5 Go)",
        "vram": "~1.5 Go",
    },
    {
        "id": "medium",
        "label": "Whisper medium (~1.5 Go)",
        "vram": "~1.5 Go",
    },
    {
        "id": "small",
        "label": "Whisper small (~0.5 Go)",
        "vram": "~0.5 Go",
    },
    {
        "id": "base",
        "label": "Whisper base (léger)",
        "vram": "~0.3 Go",
    },
    {
        "id": "tiny",
        "label": "Whisper tiny (minimal)",
        "vram": "~0.1 Go",
    },
]

# Kokoro voices (Apache-2.0) — ids officiels kokoro-v1.0
TTS_VOICES: list[dict[str, str]] = [
    {"id": "af_heart", "label": "AF Heart (EN · défaut)", "lang": "en-us"},
    {"id": "af_bella", "label": "AF Bella (EN)", "lang": "en-us"},
    {"id": "af_nicole", "label": "AF Nicole (EN)", "lang": "en-us"},
    {"id": "af_sarah", "label": "AF Sarah (EN)", "lang": "en-us"},
    {"id": "am_adam", "label": "AM Adam (EN)", "lang": "en-us"},
    {"id": "am_michael", "label": "AM Michael (EN)", "lang": "en-us"},
    {"id": "bf_emma", "label": "BF Emma (EN-GB)", "lang": "en-gb"},
    {"id": "bf_isabella", "label": "BF Isabella (EN-GB)", "lang": "en-gb"},
    {"id": "bm_george", "label": "BM George (EN-GB)", "lang": "en-gb"},
    {"id": "bm_lewis", "label": "BM Lewis (EN-GB)", "lang": "en-gb"},
    {"id": "ff_siwis", "label": "FF Siwis (FR)", "lang": "fr-fr"},
    {"id": "jf_alpha", "label": "JF Alpha (JA)", "lang": "ja"},
    {"id": "jf_gongitsune", "label": "JF Gongitsune (JA)", "lang": "ja"},
    {"id": "jm_kumo", "label": "JM Kumo (JA)", "lang": "ja"},
    {"id": "zf_xiaobei", "label": "ZF Xiaobei (ZH)", "lang": "zh"},
    {"id": "zm_yunxi", "label": "ZM Yunxi (ZH)", "lang": "zh"},
]

TTS_BACKENDS: list[dict[str, str]] = [
    {
        "id": "kokoro",
        "label": "Kokoro (rapide · voix prédéfinies · Apache-2.0)",
        "clone": "false",
    },
    {
        "id": "cosyvoice",
        "label": "CosyVoice 3 (clonage · Apache-2.0 · +VRAM)",
        "clone": "true",
    },
]


def catalog() -> dict:
    return {
        "stt": STT_MODELS,
        "tts": TTS_VOICES,
        "tts_backends": TTS_BACKENDS,
    }
