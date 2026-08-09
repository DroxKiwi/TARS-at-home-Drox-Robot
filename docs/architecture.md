# Architecture TARS Voice

## Objectif

Conversation vocale **100 % serveur** :

- le client (PC / téléphone) n’exécute que le **navigateur** (micro + lecture audio)
- STT, LLM, TTS tournent sur la machine GPU (dev ou VM)
- UI Next.js sur le port **3000** · API FastAPI sur **9743**

## Flux

```text
Navigateur (LAN)
    │  HTTP :3000 (UI Next)
    │  HTTP + WebSocket :9743 (API)
    ▼
┌─────────────────────────────────────┐
│  tars-web (Next.js)                 │
└─────────────────────────────────────┘
┌─────────────────────────────────────┐
│  tars-voice (FastAPI)               │
│  /api/* · /ws · CORS                │
│         │                           │
│         ▼                           │
│  faster-whisper (STT, GPU)          │
│         │                           │
│         ▼                           │
│  Ollama HTTP (hôte)                 │
│         │                           │
│         ▼                           │
│  Kokoro TTS (CPU) / Cosy optionnel  │
└─────────────────────────────────────┘
         │
         ▼
   PostgreSQL (settings)
```

Latence perçue : synthèse **phrase par phrase** dès que le LLM produit un `.` / `!` / `?`.

## Profils

| Fichier | Usage |
|---------|--------|
| `docker-compose.dev.yml` | PC Windows + RTX, Ollama via `host.docker.internal`, Next en `Dockerfile.dev` |
| `docker-compose.yml` | Prod Linux VM, Next image standalone |

## Séparation Ollama

Ollama **n’est pas** dans l’image voix (cycle de vie / VRAM gérés à part).

## Évolutions prévues

1. HTTPS LAN (mkcert / Caddy) pour micro mobile
2. Streaming STT partiel
3. Historique multi-tours en DB
4. Auth si exposition hors LAN trusté

## Flotte / uplink

Inventaire + SSH côté cerveau ; Wi‑Fi/BT côté Pi via agent uplink (`/ws/edge`).
Détail : [fleet-uplink.md](./fleet-uplink.md).
