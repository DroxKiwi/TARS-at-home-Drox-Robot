# Flotte & uplink

## Principes

| Couche | Rôle |
|--------|------|
| **Cerveau** (`tars-voice`) | Inventaire hosts/nodes, SSH LAN allowlisté, hub WebSocket uplink, outils LLM |
| **Edge** (Pi) | Agent sortant Wi‑Fi / Bluetooth / exec local allowlisté |

Les modèles (léger ou spécialistes) n’ont jamais de shell libre : uniquement les
fonctions catalogue (`lan_list_hosts`, `ssh_run`, `edge_wifi_*`, `edge_bt_*`,
`edge_exec`), assignables aux rôles via `function_keys`.

```text
  Pi ──(WS uplink sortant)──► cerveau :9743/ws/edge
                                    │
  LLM tools ──► fleet.edge_call / ssh_run
```

L’uplink est initié par le Pi (passe NAT). Une fois connecté, le cerveau (donc
chat léger **et** modèles lourds via outils) peut pousser des commandes RPC.

## API cerveau

- `GET /api/fleet` — hosts + nodes + état uplink
- `POST /api/fleet/nodes` — enregistre un nœud, renvoie `token` une fois
- `POST /api/fleet/hosts` — upsert hôte SSH (ip, user, clé montée dans le conteneur)
- `WS /ws/edge?node_key=&token=` — uplink

## Sécurité

- SSH : regex allowlist (`FLEET_SSH_ALLOWLIST` ou défaut sûr)
- Edge exec : allowlist locale dans `services/edge-agent/handlers.py`
- Token nœud stocké hashé (SHA-256) en Postgres
- Audit : table `fleet_audit`

## Outils LLM

Chat : `lan_list_hosts` par défaut. Assigner `ssh_run` / `edge_*` aux rôles
experts (ou au chat) dans la config UI.
