"""Configuration TARS Voice — licences permissives uniquement (voir docs/LICENSES.md).

Les réglages UI / system prompt sont persistés en Postgres (voir db.py).
Les champs ci-dessous seedent la DB au premier démarrage et couvrent l'infra.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_SYSTEM_PROMPT = (
    "Tu es TARS, un robot compagnon francophone. Reponses concises, "
    "humour sec, honnete. Reponds toujours dans la langue de l'utilisateur "
    "(francais par defaut). Pas d'emojis, pas de markdown a l'oral. "
    "Outils : show_shape, clear_canvas, web_search, ask_<role>, "
    "read_specialist_reply. "
    "Avant chaque outil, annonce a l'oral ce que tu vas faire, puis appelle-le. "
    "Apres un specialiste : resume en 1 a 3 phrases, puis demande si "
    "l'utilisateur veut ecouter la reponse complete. "
    "S'il accepte, appelle read_specialist_reply. "
    "Ne choisis jamais un modele Ollama toi-meme."
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    tars_host: str = "0.0.0.0"
    tars_port: int = 9743

    database_url: str = "postgresql://tars:tars@tars-postgres:5432/tars"

    ollama_base_url: str = "http://host.docker.internal:11434"
    ollama_model: str = "qwen2.5:7b-instruct-q4_K_M"
    # Modèle Ollama lourd legacy (préférer llm_roles) — seed seulement
    heavy_ollama_model: str = ""
    # Nb de messages d'historique transmis au spécialiste
    role_history_messages: int = 8

    whisper_model: str = "large-v3-turbo"
    whisper_device: str = "cuda"
    whisper_compute_type: str = "float16"
    whisper_language: str = "fr"

    kokoro_voice: str = "af_heart"
    kokoro_speed: float = 1.0
    # Modèles Kokoro téléchargés au premier run (Apache-2.0)
    kokoro_model_path: str = "/root/.cache/kokoro/kokoro-v1.0.onnx"
    kokoro_voices_path: str = "/root/.cache/kokoro/voices-v1.0.bin"

    # TTS backend: kokoro | cosyvoice
    tts_backend: str = "kokoro"
    cosyvoice_url: str = "http://tars-tts-cosy:9750"
    cosyvoice_voices_dir: str = "/data/voices"

    # Seed DB seulement — ensuite éditable via UI /api/settings
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    # Tool calling Ollama (panneau géométrique, recherche web, etc.)
    tools_enabled: bool = True
    tools_max_rounds: int = 4

    # SearXNG (métamoteur local) pour web_search
    searxng_url: str = "http://tars-searxng:8080"

    sample_rate: int = 16000
    silence_ms: int = 700
    max_utterance_ms: int = 15000
    # Seuil énergie simple (RMS) pour VAD basique — pas de modèle propriétaire
    vad_rms_threshold: float = 0.015

    # Barge-in (couper TTS quand l'utilisateur reprend la parole)
    barge_in_enabled: bool = True
    barge_in_min_speech_ms: int = 180

    # Export métriques / runs (monté en volume Docker → lisible sur l'hôte)
    metrics_dir: str = "/data/metrics"
    metrics_export_enabled: bool = True

    # Flotte : allowlist SSH (regex séparées par des virgules). Vide = défaut sûr.
    fleet_ssh_allowlist: str = ""

    # CORS pour UI Next (LAN) — * en dev local trusté
    cors_origins: str = "*"
    web_ui_url: str = "http://localhost:3000"


@lru_cache
def get_settings() -> Settings:
    return Settings()
