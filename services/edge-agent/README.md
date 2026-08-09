# Agent edge (Raspberry Pi)

Uplink sortant vers le cerveau TARS (`/ws/edge`). Gère Wi‑Fi / Bluetooth / exec
allowlistés côté Pi ; le cerveau orchestre SSH/LAN et les outils LLM.

## Enregistrement

Sur le cerveau :

```bash
curl -X POST http://localhost:9743/api/fleet/nodes \
  -H 'Content-Type: application/json' \
  -d '{"node_key":"pi-salon","name":"Pi salon"}'
```

Conserver le `token` renvoyé (une seule fois).

## Lancer sur le Pi

```bash
cd services/edge-agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export TARS_EDGE_URL=ws://IP_CERVEAU:9743/ws/edge
export TARS_NODE_KEY=pi-salon
export TARS_NODE_TOKEN=...
python agent.py
```

Sous Windows / sans `nmcli`, les handlers Wi‑Fi/BT renvoient des données mock.

Voir aussi `docs/fleet-uplink.md`.
