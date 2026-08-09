# TARS-at-home-Drox-Robot

Agent vocal **STT → Ollama → TTS**, tout sur le serveur.  
UI **Next.js** (mobile-first) sur le port **3000** · API FastAPI sur **9743**.

Le navigateur (PC / téléphone) ne fait que micro + lecture audio.

## Stack (licences permissives)

| Couche | Techno | Licence |
|--------|--------|---------|
| UI | Next.js + Tailwind + shadcn-style | MIT |
| API + WS | FastAPI | MIT |
| STT | faster-whisper | MIT |
| LLM | Ollama local (ex. Qwen2.5) | MIT + Apache-2.0 modèle |
| TTS | Kokoro ONNX | Apache-2.0 |
| Settings | PostgreSQL | — |

Détail : [docs/LICENSES.md](docs/LICENSES.md) · Archi : [docs/architecture.md](docs/architecture.md)

## Prérequis

1. **Docker** + NVIDIA Container Toolkit (GPU)
2. **Ollama** installé **sur l’hôte** (pas dans le compose voix)
3. Un modèle instruct **compatible tools**, ex. :

```bash
ollama pull qwen2.5:7b-instruct-q4_K_M
```

## Démarrage — PC de dev (RTX 12 Go)

```powershell
copy .env.example .env
# Éditer .env si besoin (DATABASE_URL, OLLAMA_*, device Whisper…)

# Premier lancement (build images, dont tars-web) :
docker compose -f docker-compose.dev.yml up --build -d

# Ensuite au quotidien — SANS --build :
docker compose -f docker-compose.dev.yml up -d
```

- UI : [https://localhost:3000](https://localhost:3000) (HTTPS auto-signé — accepter l’avertissement)
- API health : [http://localhost:9743/health](http://localhost:9743/health)

### UI Next — HTTPS (micro LAN)

La page est en **HTTPS** ; `/api` et `/ws` sont **proxifiés** vers `tars-voice` (même origin → pas de mixed-content, micro OK sur téléphone).

**Sur l’hôte** (si tu préfères Node local) :

```powershell
cd apps\web
npm install
# optionnel : IP LAN dans le certificat
$env:HTTPS_SANS="192.168.x.x"
$env:VOICE_INTERNAL_URL="http://127.0.0.1:9743"
npm run dev
```

**Via Docker** : inclus dans `docker compose -f docker-compose.dev.yml up` (service `tars-web`).

Téléphone (même Wi‑Fi) : `https://<IP_PC>:3000` → accepter le certificat une fois, puis le micro fonctionne.

Pare-feu : TCP **3000** (+ **9743** si tu appelles l’API en direct).

### Exposition LAN (autre appareil)

1. IP du PC : `ipconfig` → IPv4
2. Pare-feu : TCP **3000** (UI HTTPS)
3. Optionnel dans `.env` : `HTTPS_SANS=192.168.1.42` (SAN du certificat)
4. Téléphone : `https://<IP>:3000` — **accepter le certificat** auto-signé, puis autoriser le micro

### Réglages / Postgres

System prompt + choix UI persistés dans **PostgreSQL** (`tars-postgres`, port hôte **5433**).

- Édition dans l’UI → `GET/PUT /api/settings`
- `.env` = infra seulement (`DATABASE_URL`, ports, chemins, GPU, URLs services)

## Démarrage — VM prod (Linux)

```bash
cp .env.example .env
# OLLAMA_BASE_URL=http://172.17.0.1:11434
docker compose up --build -d
```

Depuis le LAN : `http://<IP_VM>:3000` (UI) · API `:9743`

## Utilisation

1. Configurer Ollama / STT / TTS / prompt, puis **Charger en VRAM**
2. **Démarrer le chat** (VAD client)
3. Métriques latence (panneau collapsible)

### TTS optionnel — CosyVoice

Moteur **CosyVoice 3** dans la config (service `tars-tts-cosy` :9750).  
Roadmap Kokoro → RVC : [docs/roadmap-kokoro-rvc.md](docs/roadmap-kokoro-rvc.md).

### Tools

- « affiche un cercle bleu » → `show_shape`
- recherche web → `web_search` (SearXNG :8088)

Préférer un modèle **tools** (`qwen2.5`) — Gemma liste souvent les outils sans les appeler.

## Structure

```text
docker-compose.yml / docker-compose.dev.yml
apps/web/                # Next.js UI (tars-web :3000)
services/voice/          # FastAPI API + WS (tars-voice :9743)
services/tts-cosy/
services/searxng/
data/metrics/
docs/
```

## Note PersonaPlex

Documenté dans [docs/personaplex-setup.md](docs/personaplex-setup.md) — pas le chemin produit.
