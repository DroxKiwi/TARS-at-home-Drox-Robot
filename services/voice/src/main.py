"""TARS Voice — serveur Web UI + WebSocket (port 9743)."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import struct
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import fleet, fleet_db
from .catalog import catalog
from .config import get_settings
from .db import (
    apply_row_to_settings,
    close_db,
    get_app_settings,
    init_db,
    update_app_settings,
)
from .export import MetricsExporter
from .llm import probe_ollama
from .metrics import TurnMetrics
from .pipeline import SessionConfig, VoicePipeline
from .roles import create_role, delete_role, list_roles, update_role
from .functions import list_catalog as list_function_catalog

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("tars")

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
pipeline: VoicePipeline | None = None
exporter: MetricsExporter | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global pipeline, exporter
    settings = get_settings()
    await init_db(settings)
    row = await get_app_settings()
    apply_row_to_settings(settings, row)
    pipeline = VoicePipeline(settings)
    await pipeline.llm.refresh_roles()
    if settings.metrics_export_enabled:
        exporter = MetricsExporter(settings.metrics_dir)
        logger.info("Export métriques activé → %s", settings.metrics_dir)
    else:
        exporter = None
    logger.info(
        "API prête sur %s:%s (UI Next → %s)",
        settings.tars_host,
        settings.tars_port,
        settings.web_ui_url,
    )
    yield
    if pipeline:
        await pipeline.close()
    await close_db()


app = FastAPI(title="TARS Voice API", lifespan=lifespan)

_settings0 = get_settings()
_cors = [o.strip() for o in (_settings0.cors_origins or "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors if _cors != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    settings = get_settings()
    ui = settings.web_ui_url.rstrip("/")
    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>TARS API</title>
  <style>
    body {{ font-family: system-ui, sans-serif; background:#0c1118; color:#e8eef7;
           max-width:36rem; margin:3rem auto; padding:0 1.25rem; line-height:1.5; }}
    a {{ color:#c4a574; }}
    code {{ background:#151d29; padding:0.15rem 0.4rem; }}
  </style>
</head>
<body>
  <h1>TARS Voice API</h1>
  <p>L’interface web est sur <a href="{ui}">{ui}</a> (Next.js, port 3000).</p>
  <p>Endpoints : <code>/health</code>, <code>/api/*</code>, WebSocket <code>/ws</code>.</p>
  <p>Legacy UI statique encore sous <a href="/static/index.html">/static/index.html</a>.</p>
</body>
</html>"""


@app.get("/health")
async def health():
    assert pipeline is not None
    return await pipeline.health()


class OllamaTestRequest(BaseModel):
    base_url: str = Field(..., description="URL Ollama")


class AppSettingsUpdate(BaseModel):
    system_prompt: str | None = None
    ollama_base_url: str | None = None
    ollama_model: str | None = None
    heavy_ollama_model: str | None = None
    stt_model: str | None = None
    tts_backend: str | None = None
    tts_voice: str | None = None
    enable_thinking: bool | None = None
    kokoro_speed: float | None = None
    silence_ms: int | None = None
    vad_rms_threshold: float | None = None
    barge_in_enabled: bool | None = None
    barge_in_min_speech_ms: int | None = None
    role_history_messages: int | None = None


class RoleCreate(BaseModel):
    key: str | None = None
    name: str
    description: str = ""
    system_prompt: str
    ollama_model: str
    function_keys: list[str] = []
    enabled: bool = True
    sort_order: int = 0


class RoleUpdate(BaseModel):
    key: str | None = None
    name: str | None = None
    description: str | None = None
    system_prompt: str | None = None
    ollama_model: str | None = None
    function_keys: list[str] | None = None
    enabled: bool | None = None
    sort_order: int | None = None


@app.post("/api/ollama/test")
async def ollama_test(body: OllamaTestRequest):
    return await probe_ollama(body.base_url)


@app.get("/api/settings")
async def api_get_settings():
    return await get_app_settings()


@app.put("/api/settings")
async def api_put_settings(body: AppSettingsUpdate):
    """Persiste les réglages UI ; applique immédiatement prompt / VAD / barge-in / thinking."""
    assert pipeline is not None
    settings = get_settings()
    patch = body.model_dump(exclude_none=True)
    row = await update_app_settings(patch)
    apply_row_to_settings(settings, row)
    if "enable_thinking" in patch:
        await pipeline.llm.configure(enable_thinking=bool(patch["enable_thinking"]))
    return row


@app.get("/api/roles")
async def api_list_roles():
    return {"roles": await list_roles(enabled_only=False)}


@app.get("/api/functions")
async def api_list_functions():
    """Catalogue des fonctions assignables (chat / spécialistes)."""
    return {
        "functions": list_function_catalog(),
        "specialist": list_function_catalog(scope="specialist"),
        "chat": list_function_catalog(scope="chat"),
    }


@app.post("/api/roles")
async def api_create_role(body: RoleCreate):
    assert pipeline is not None
    try:
        role = await create_role(body.model_dump())
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    await pipeline.llm.refresh_roles()
    return {"ok": True, "role": role}


@app.put("/api/roles/{role_id}")
async def api_update_role(role_id: int, body: RoleUpdate):
    assert pipeline is not None
    try:
        role = await update_role(role_id, body.model_dump(exclude_none=True))
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    await pipeline.llm.refresh_roles()
    return {"ok": True, "role": role}


@app.delete("/api/roles/{role_id}")
async def api_delete_role(role_id: int):
    assert pipeline is not None
    ok = await delete_role(role_id)
    await pipeline.llm.refresh_roles()
    return {"ok": ok}


# --- Flotte (cerveau : inventaire + SSH ; edge : uplink WS) ---


class FleetNodeCreate(BaseModel):
    node_key: str = Field(..., min_length=1, max_length=64)
    name: str = Field("", max_length=120)
    kind: str = Field("edge", max_length=32)


class FleetHostUpsert(BaseModel):
    host_key: str = Field(..., min_length=1, max_length=64)
    label: str = Field("", max_length=120)
    hostname: str | None = None
    ip: str | None = None
    ssh_user: str = "tars"
    ssh_port: int = 22
    ssh_key_path: str | None = None
    tags: list[str] = Field(default_factory=list)
    enabled: bool = True
    notes: str = ""


@app.get("/api/fleet")
async def api_fleet_overview():
    return await fleet.lan_list_hosts()


@app.get("/api/fleet/nodes")
async def api_fleet_nodes():
    nodes = await fleet_db.list_nodes()
    online = set(fleet.edge_hub.online_keys())
    for n in nodes:
        n["uplink_online"] = n["node_key"] in online
        n.pop("token_hash", None)
    return {"nodes": nodes}


@app.post("/api/fleet/nodes")
async def api_fleet_register_node(body: FleetNodeCreate):
    """Enregistre un nœud edge et renvoie le token (une seule fois)."""
    try:
        node = await fleet_db.register_or_rotate_node(
            node_key=body.node_key,
            name=body.name or body.node_key,
            kind=body.kind or "edge",
        )
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "node": node}


@app.get("/api/fleet/hosts")
async def api_fleet_hosts():
    return {"hosts": await fleet_db.list_hosts(enabled_only=False)}


@app.post("/api/fleet/hosts")
async def api_fleet_upsert_host(body: FleetHostUpsert):
    try:
        host = await fleet_db.upsert_host(body.model_dump())
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "host": host}


@app.get("/api/catalog")
async def api_catalog():
    settings = get_settings()
    try:
        db_row = await get_app_settings()
    except Exception:  # noqa: BLE001
        db_row = None
    defaults = {
        "ollama_base_url": settings.ollama_base_url,
        "ollama_model": settings.ollama_model,
        "role_history_messages": int(
            getattr(settings, "role_history_messages", 8) or 8
        ),
        "stt_model": settings.whisper_model,
        "tts_voice": settings.kokoro_voice,
        "tts_backend": settings.tts_backend,
        "cosyvoice_url": settings.cosyvoice_url,
        "system_prompt": settings.system_prompt,
        "enable_thinking": False,
        "kokoro_speed": settings.kokoro_speed,
        "silence_ms": settings.silence_ms,
        "vad_rms_threshold": settings.vad_rms_threshold,
        "barge_in_enabled": settings.barge_in_enabled,
        "barge_in_min_speech_ms": settings.barge_in_min_speech_ms,
    }
    if db_row:
        defaults.update(
            {
                "ollama_base_url": db_row["ollama_base_url"],
                "ollama_model": db_row["ollama_model"],
                "role_history_messages": db_row.get("role_history_messages") or 8,
                "stt_model": db_row["stt_model"],
                "tts_voice": db_row["tts_voice"],
                "tts_backend": db_row["tts_backend"],
                "system_prompt": db_row["system_prompt"],
                "enable_thinking": db_row["enable_thinking"],
                "kokoro_speed": db_row["kokoro_speed"],
                "silence_ms": db_row["silence_ms"],
                "vad_rms_threshold": db_row["vad_rms_threshold"],
                "barge_in_enabled": db_row["barge_in_enabled"],
                "barge_in_min_speech_ms": db_row["barge_in_min_speech_ms"],
            }
        )
    return {**catalog(), "defaults": defaults}


@app.get("/api/session")
async def api_session():
    assert pipeline is not None
    return pipeline.status()


@app.get("/api/tts/cosy/health")
async def cosy_health():
    assert pipeline is not None
    return pipeline.tts.cosy_health()


class SessionLoadRequest(BaseModel):
    ollama_base_url: str
    ollama_model: str
    stt_model: str
    tts_voice: str
    enable_thinking: bool = False
    tts_backend: str = "kokoro"
    cosy_prompt_text: str = ""
    cosy_use_default_prompt: bool = False


@app.post("/api/tts/prompt")
async def upload_tts_prompt(file: UploadFile = File(...)):
    """Upload d'un sample WAV pour clonage CosyVoice."""
    settings = get_settings()
    dest_dir = Path(settings.cosyvoice_voices_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "active_prompt.wav"
    data = await file.read()
    if len(data) < 1000:
        return {"ok": False, "error": "Fichier trop court"}
    dest.write_bytes(data)
    return {"ok": True, "path": str(dest), "bytes": len(data)}


@app.post("/api/session/load")
async def api_session_load(body: SessionLoadRequest):
    """Charge STT + TTS + warm LLM en VRAM, puis autorise la discussion."""
    assert pipeline is not None
    settings = get_settings()
    # Persiste les choix UI (hors system_prompt — géré via PUT /api/settings)
    row = await update_app_settings(
        {
            "ollama_base_url": body.ollama_base_url,
            "ollama_model": body.ollama_model,
            "stt_model": body.stt_model,
            "tts_voice": body.tts_voice,
            "tts_backend": body.tts_backend or "kokoro",
            "enable_thinking": body.enable_thinking,
        }
    )
    apply_row_to_settings(settings, row)
    prompt_path = None
    if body.tts_backend == "cosyvoice":
        candidate = Path(settings.cosyvoice_voices_dir) / "active_prompt.wav"
        if candidate.exists():
            prompt_path = str(candidate)
    cfg = SessionConfig(
        ollama_base_url=body.ollama_base_url,
        ollama_model=body.ollama_model,
        stt_model=body.stt_model,
        tts_voice=body.tts_voice,
        enable_thinking=body.enable_thinking,
        tts_backend=body.tts_backend or "kokoro",
        cosy_prompt_text=body.cosy_prompt_text or "",
        cosy_prompt_wav_path=prompt_path,
        cosy_use_default_prompt=body.cosy_use_default_prompt,
    )
    result = await pipeline.load_session(cfg)
    if exporter is not None:
        exporter.write_session(
            {
                "ready": bool(result.get("ready")),
                "error": result.get("error"),
                "steps": result.get("steps") or {},
                "active": result.get("active") or {
                    "stt": cfg.stt_model,
                    "llm": cfg.ollama_model,
                    "tts": cfg.tts_voice,
                    "tts_backend": cfg.tts_backend,
                    "ollama_base_url": cfg.ollama_base_url,
                    "enable_thinking": cfg.enable_thinking,
                },
            }
        )
    return result


@app.post("/api/session/unload")
async def api_session_unload():
    """Décharge STT/TTS + Ollama pour libérer la VRAM avant de quitter."""
    assert pipeline is not None
    result = await pipeline.unload_session()
    if exporter is not None:
        exporter.write_session(
            {
                "ready": False,
                "unloaded": result.get("unloaded") or [],
                "errors": result.get("errors") or [],
                "active": result.get("active") or {},
            }
        )
    return result


def _pcm16_b64_to_f32(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    count = len(raw) // 2
    samples = struct.unpack("<" + "h" * count, raw[: count * 2])
    return (np.asarray(samples, dtype=np.float32) / 32768.0).clip(-1.0, 1.0)


def _f32_to_pcm16_b64(audio: np.ndarray) -> str:
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16).tobytes()
    return base64.b64encode(pcm).decode("ascii")


def _resample_f32(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr or audio.size == 0:
        return audio
    duration = len(audio) / src_sr
    target = max(1, int(duration * dst_sr))
    x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=target, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def _session_info() -> dict:
    assert pipeline is not None
    if not pipeline.session.config:
        return {}
    c = pipeline.session.config
    return {
        "stt": c.stt_model,
        "llm": c.ollama_model,
        "tts": c.tts_voice,
        "tts_backend": c.tts_backend,
        "ollama_base_url": c.ollama_base_url,
        "enable_thinking": c.enable_thinking,
    }


@app.websocket("/ws/edge")
async def ws_edge(ws: WebSocket) -> None:
    """Uplink sortant des nœuds Pi → cerveau (auth node_key + token)."""
    node_key = (ws.query_params.get("node_key") or "").strip().lower()
    token = (ws.query_params.get("token") or "").strip()
    if not node_key or not token:
        await ws.close(code=4401)
        return
    node = await fleet_db.verify_node_token(node_key, token)
    if node is None:
        await ws.close(code=4403)
        return
    await ws.accept()
    await fleet.edge_hub.attach(node_key, ws)
    await ws.send_json({"type": "welcome", "node_key": node_key})
    logger.info("Edge uplink connecté: %s", node_key)
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(msg, dict):
                await fleet.edge_hub.handle_edge_message(node_key, msg)
    except WebSocketDisconnect:
        logger.info("Edge uplink déconnecté: %s", node_key)
    finally:
        await fleet.edge_hub.detach(node_key, ws)


@app.websocket("/ws")
async def ws_voice(ws: WebSocket) -> None:
    assert pipeline is not None
    settings = get_settings()
    await ws.accept()
    await ws.send_json(
        {
            "type": "ready",
            "sample_rate": settings.sample_rate,
            "session_ready": pipeline.session.ready,
            "resource": pipeline.orch.status(),
            "vad": {
                "silence_ms": settings.silence_ms,
                "rms_threshold": settings.vad_rms_threshold,
            },
            "barge_in": {
                "enabled": settings.barge_in_enabled,
                "min_speech_ms": settings.barge_in_min_speech_ms,
            },
        }
    )

    chunks: list[np.ndarray] = []
    busy = False
    chat_active = False
    incoming: asyncio.Queue[dict | None] = asyncio.Queue()
    deferred: list[dict] = []
    # Contrôle du tour en cours
    turn_cancel: asyncio.Event | None = None
    spoken_from_client: str | None = None
    spoken_commit_event = asyncio.Event()

    async def reader() -> None:
        try:
            while True:
                message = await ws.receive()
                if message.get("type") == "websocket.disconnect":
                    await incoming.put(None)
                    break
                data = message.get("text")
                if not data:
                    continue
                await incoming.put(json.loads(data))
        except WebSocketDisconnect:
            await incoming.put(None)
        except Exception:  # noqa: BLE001
            logger.exception("WS reader error")
            await incoming.put(None)

    reader_task = asyncio.create_task(reader())

    async def handle_control_during_turn(event: dict) -> bool:
        """True si l'event est consommé comme contrôle du tour courant."""
        nonlocal chat_active, spoken_from_client, chunks
        etype = event.get("type")
        if etype == "barge_in" and settings.barge_in_enabled:
            spoken_from_client = (event.get("spoken_text") or "").strip()
            spoken_commit_event.set()
            if turn_cancel is not None:
                turn_cancel.set()
            await pipeline.llm.cancel_stream()
            logger.info(
                "Barge-in reçu spoken=%d chars",
                len(spoken_from_client or ""),
            )
            return True
        if etype == "spoken_commit":
            spoken_from_client = (event.get("spoken_text") or "").strip()
            spoken_commit_event.set()
            return True
        if etype == "audio_chunk" and chat_active:
            chunks.append(_pcm16_b64_to_f32(event["audio"]))
            return True
        if etype == "chat_stop":
            chat_active = False
            chunks = []
            if turn_cancel is not None:
                turn_cancel.set()
            spoken_commit_event.set()
            await pipeline.llm.cancel_stream()
            await ws.send_json({"type": "chat_stopped"})
            await ws.send_json({"type": "status", "state": "idle"})
            return True
        if etype == "ping":
            await ws.send_json(
                {
                    "type": "pong",
                    "session_ready": pipeline.session.ready,
                    "chat_active": chat_active,
                }
            )
            return True
        return False

    async def run_turn(event: dict) -> None:
        nonlocal busy, turn_cancel, spoken_from_client, chunks, chat_active, deferred
        if not chat_active:
            return
        if busy or not chunks:
            if chat_active:
                await ws.send_json({"type": "status", "state": "listening"})
            return
        if not pipeline.session.ready:
            await ws.send_json(
                {
                    "type": "error",
                    "message": "Charge d'abord la configuration (STT / LLM / TTS).",
                }
            )
            return

        busy = True
        turn_cancel = asyncio.Event()
        spoken_from_client = None
        spoken_commit_event.clear()
        deferred = []
        metrics = TurnMetrics()
        client_vad_ms = event.get("client_vad_ms")
        n_chunks = len(chunks)
        await ws.send_json({"type": "status", "state": "stt"})
        pipeline.set_actor("speech")
        await ws.send_json(
            {"type": "resource", "resource": pipeline.orch.status()}
        )
        audio = np.concatenate(chunks)
        chunks = []
        metrics.set_input_audio(
            samples=int(audio.size),
            sample_rate=settings.sample_rate,
            chunks=n_chunks,
        )
        turn_error: str | None = None
        user_text = ""
        assistant_parts: list[str] = []
        thinking_parts: list[str] = []
        assistant_final = ""
        interrupted = False
        control_task: asyncio.Task | None = None

        async def control_loop() -> None:
            while True:
                try:
                    ev = await asyncio.wait_for(incoming.get(), timeout=0.25)
                except asyncio.TimeoutError:
                    if not busy:
                        return
                    continue
                if ev is None:
                    await incoming.put(None)
                    if turn_cancel is not None:
                        turn_cancel.set()
                    spoken_commit_event.set()
                    return
                handled = await handle_control_during_turn(ev)
                if not handled:
                    deferred.append(ev)
                if not busy:
                    return

        try:
            control_task = asyncio.create_task(control_loop())
            metrics.start_stt()
            user_text = await asyncio.to_thread(
                pipeline.transcribe, audio, settings.sample_rate
            )
            metrics.end_stt(user_text)
            await ws.send_json(
                {
                    "type": "user_transcript",
                    "text": user_text,
                    "metrics": {
                        "stt_ms": metrics.stt_ms,
                        "input_duration_ms": metrics.input_duration_ms,
                        "stt_realtime_factor": metrics.stt_realtime_factor,
                    },
                }
            )
            if not user_text:
                spoken_commit_event.set()
                return

            if turn_cancel.is_set():
                interrupted = True
                spoken_commit_event.set()
                pipeline.llm.discard_pending_user()
                return

            await ws.send_json({"type": "status", "state": "streaming"})
            pipeline.set_actor("chat")
            await ws.send_json(
                {"type": "resource", "resource": pipeline.orch.status()}
            )
            await ws.send_json({"type": "assistant_reset"})

            async for pev in pipeline.respond_stream(
                user_text, metrics, cancel=turn_cancel
            ):
                if pev.kind == "thinking_token":
                    thinking_parts.append(pev.text)
                    await ws.send_json({"type": "thinking_token", "text": pev.text})
                elif pev.kind == "tool_call":
                    await ws.send_json(
                        {
                            "type": "tool_call",
                            "name": (pev.meta or {}).get("name") or pev.text,
                            "arguments": (pev.meta or {}).get("arguments") or {},
                        }
                    )
                elif pev.kind == "stage":
                    meta = pev.meta or {}
                    await ws.send_json(
                        {
                            "type": "stage",
                            "stage": meta.get("stage") or pev.text,
                            "kind": meta.get("kind"),
                            "actor": meta.get("actor"),
                            "ok": meta.get("ok"),
                            "ms": meta.get("ms"),
                            "detail": meta.get("detail"),
                            "resource": meta.get("resource"),
                        }
                    )
                elif pev.kind == "resource":
                    meta = pev.meta or {}
                    await ws.send_json(
                        {
                            "type": "resource",
                            "resource": meta.get("resource")
                            or pipeline.orch.status(),
                        }
                    )
                elif pev.kind == "tool_result":
                    await ws.send_json(
                        {
                            "type": "tool_result",
                            "name": (pev.meta or {}).get("name"),
                            "ok": (pev.meta or {}).get("ok"),
                            "content": pev.text,
                            "ui": (pev.meta or {}).get("ui"),
                        }
                    )
                elif pev.kind == "llm_token":
                    await ws.send_json(
                        {
                            "type": "assistant_token",
                            "text": pev.text,
                            "ttft_ms": metrics.llm_ttft_ms,
                            "display": False,
                        }
                    )
                elif pev.kind == "sentence_audio":
                    if turn_cancel.is_set():
                        interrupted = True
                        break
                    assistant_parts.append(pev.text)
                    assert pev.audio is not None
                    t_enc = time.perf_counter()
                    play = _resample_f32(
                        pev.audio, pev.sample_rate, settings.sample_rate
                    )
                    b64 = _f32_to_pcm16_b64(play)
                    enc_ms = (time.perf_counter() - t_enc) * 1000.0
                    if metrics.tts_sentences:
                        metrics.tts_sentences[-1].resample_encode_ms = round(
                            enc_ms, 2
                        )
                    duration_ms = (pev.meta or {}).get("duration_ms")
                    if duration_ms is None:
                        duration_ms = round(
                            (play.size / max(1, settings.sample_rate)) * 1000.0, 2
                        )
                    await ws.send_json({"type": "status", "state": "speaking"})
                    t_ws = time.perf_counter()
                    await ws.send_json(
                        {
                            "type": "assistant_audio",
                            "text": pev.text,
                            "sample_rate": settings.sample_rate,
                            "audio": b64,
                            "sentence_index": (pev.meta or {}).get("sentence_index"),
                            "duration_ms": duration_ms,
                            "metrics": {
                                "sentence_index": (pev.meta or {}).get(
                                    "sentence_index"
                                ),
                                "synthesize_ms": pev.synthesize_ms,
                                "resample_encode_ms": round(enc_ms, 2),
                                "e2e_to_first_audio_ms": metrics.e2e_to_first_audio_ms,
                                "duration_ms": duration_ms,
                            },
                        }
                    )
                    if metrics.tts_sentences:
                        metrics.tts_sentences[-1].ws_send_ms = round(
                            (time.perf_counter() - t_ws) * 1000.0, 2
                        )
                elif pev.kind == "interrupted":
                    interrupted = True
                    assistant_final = pev.text or " ".join(assistant_parts).strip()
                elif pev.kind == "done":
                    assistant_final = (
                        pev.text or " ".join(assistant_parts).strip()
                    )

            # Pas d'audio joué → commit immédiat (évite timeout 120s)
            if not assistant_parts and not spoken_commit_event.is_set():
                spoken_from_client = spoken_from_client or ""
                spoken_commit_event.set()

            if not spoken_commit_event.is_set():
                try:
                    await asyncio.wait_for(spoken_commit_event.wait(), timeout=120.0)
                except asyncio.TimeoutError:
                    logger.warning("Timeout spoken_commit — fallback serveur")

            spoken = (
                spoken_from_client
                if spoken_from_client is not None
                else assistant_final
            )
            spoken = (spoken or "").strip()
            metrics.spoken_chars = len(spoken)
            metrics.interrupted = interrupted or metrics.interrupted
            pipeline.llm.commit_assistant(spoken)
            assistant_final = spoken

            if interrupted or metrics.interrupted:
                await ws.send_json(
                    {"type": "assistant_interrupted", "text": spoken}
                )
            else:
                await ws.send_json({"type": "assistant_done", "text": spoken})

        except Exception as exc:  # noqa: BLE001
            turn_error = str(exc)
            logger.exception("Erreur tour de parole")
            await ws.send_json({"type": "error", "message": str(exc)})
            pipeline.llm.discard_pending_user()
        finally:
            busy = False
            turn_cancel = None
            if control_task is not None:
                control_task.cancel()
                try:
                    await control_task
                except asyncio.CancelledError:
                    pass
            for ev in deferred:
                await incoming.put(ev)
            deferred = []
            payload = metrics.finalize(error=turn_error)
            if client_vad_ms is not None:
                payload["client_vad_ms"] = client_vad_ms
            payload["interrupted"] = bool(
                payload.get("interrupted") or interrupted
            )
            payload["spoken_chars"] = metrics.spoken_chars
            await ws.send_json({"type": "turn_metrics", "metrics": payload})
            if exporter is not None:
                exporter.write_turn(
                    payload,
                    user_text=user_text,
                    assistant_text=assistant_final,
                    thinking_text="".join(thinking_parts),
                    session=_session_info(),
                    extra={
                        "client_vad_ms": client_vad_ms,
                        "interrupted": metrics.interrupted,
                        "spoken_chars": metrics.spoken_chars,
                    },
                )
            if chat_active:
                await ws.send_json({"type": "status", "state": "listening"})
            else:
                await ws.send_json({"type": "status", "state": "idle"})

    try:
        while True:
            event = await incoming.get()
            if event is None:
                break

            etype = event.get("type")

            if etype == "ping":
                await ws.send_json(
                    {
                        "type": "pong",
                        "session_ready": pipeline.session.ready,
                        "chat_active": chat_active,
                    }
                )
                continue

            if etype == "chat_start":
                if not pipeline.session.ready:
                    await ws.send_json(
                        {
                            "type": "error",
                            "message": "Charge d'abord la configuration.",
                        }
                    )
                    continue
                chat_active = True
                chunks = []
                pipeline.llm.clear_history()
                await ws.send_json({"type": "chat_started"})
                await ws.send_json({"type": "status", "state": "listening"})
                continue

            if etype == "chat_stop":
                chat_active = False
                chunks = []
                if turn_cancel is not None:
                    turn_cancel.set()
                await pipeline.llm.cancel_stream()
                await ws.send_json({"type": "chat_stopped"})
                await ws.send_json({"type": "status", "state": "idle"})
                continue

            if etype == "set_thinking":
                enabled = bool(event.get("enable_thinking", False))
                await pipeline.llm.configure(enable_thinking=enabled)
                await ws.send_json(
                    {"type": "thinking_set", "enable_thinking": enabled}
                )
                continue

            if etype == "audio_chunk":
                if not pipeline.session.ready or not chat_active:
                    continue
                if busy:
                    # Pendant STT/LLM avant barge : ignorer pour éviter feedback
                    # (le client ne devrait envoyer qu'en listening / après barge)
                    continue
                chunks.append(_pcm16_b64_to_f32(event["audio"]))
                continue

            if etype == "end_utterance":
                await run_turn(event)
                continue

            if etype == "barge_in":
                # Hors tour : ignore
                continue

            if etype == "spoken_commit":
                spoken_from_client = (event.get("spoken_text") or "").strip()
                spoken_commit_event.set()
                continue

            if etype == "cancel":
                chunks = []
                if turn_cancel is not None:
                    turn_cancel.set()
                await pipeline.llm.cancel_stream()
                if chat_active:
                    await ws.send_json({"type": "status", "state": "listening"})
                else:
                    await ws.send_json({"type": "status", "state": "idle"})

    except WebSocketDisconnect:
        logger.info("Client déconnecté")
    except Exception:  # noqa: BLE001
        logger.exception("WebSocket error")
    finally:
        reader_task.cancel()
        try:
            await reader_task
        except asyncio.CancelledError:
            pass


def main() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "src.main:app",
        host=settings.tars_host,
        port=settings.tars_port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
