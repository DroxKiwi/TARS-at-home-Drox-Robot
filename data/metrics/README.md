# TARS metrics (hors container)

Bind mount Docker → hôte :

- Dev : `./data/metrics` ← `/data/metrics` (`docker-compose.dev.yml`)
- Prod : idem (`docker-compose.yml`)

Fichiers générés (non versionnés) :

- `runs.jsonl` — un JSON par tour (transcripts + timings)
- `sessions.jsonl` — chargements de session
- `turns/<turn_id>.json` — détail par tour

Lisible directement depuis le repo Windows, sans `docker exec`.
