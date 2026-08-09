#!/usr/bin/env python3
"""Agent edge Pi — uplink WebSocket sortant vers le cerveau TARS.

Usage:
  export TARS_EDGE_URL=ws://192.168.1.10:9743/ws/edge
  export TARS_NODE_KEY=pi-salon
  export TARS_NODE_TOKEN=<token from POST /api/fleet/nodes>
  python agent.py

Le Pi initie la connexion (NAT / CGNAT friendly). Le cerveau pousse des
commandes RPC (wifi_*, bt_*, exec) ; le modèle léger ou un spécialiste
les déclenche via les outils catalogue.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import signal
import sys

try:
    import websockets
    from websockets.exceptions import ConnectionClosed
except ImportError:
    print("Installer: pip install -r requirements.txt", file=sys.stderr)
    raise

from handlers import dispatch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("edge-agent")


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


async def run_once(url: str, node_key: str, token: str) -> None:
    qs = f"?node_key={node_key}&token={token}"
    full = url.rstrip("/") + qs
    logger.info("Connexion uplink %s …", url)
    async with websockets.connect(full, ping_interval=20, ping_timeout=20) as ws:
        await ws.send(
            json.dumps(
                {
                    "type": "hello",
                    "meta": {
                        "platform": platform.system(),
                        "machine": platform.machine(),
                        "python": platform.python_version(),
                    },
                }
            )
        )
        logger.info("Uplink OK (%s)", node_key)
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(msg, dict) or msg.get("type") != "command":
                continue
            req_id = msg.get("id")
            method = str(msg.get("method") or "")
            params = msg.get("params") if isinstance(msg.get("params"), dict) else {}
            logger.info("RPC %s %s", method, req_id)
            outcome = await dispatch(method, params)
            reply = {
                "type": "result",
                "id": req_id,
                "ok": bool(outcome.get("ok")),
                "result": outcome.get("result"),
                "error": outcome.get("error"),
            }
            await ws.send(json.dumps(reply))


async def main() -> None:
    base = _env("TARS_EDGE_URL", "ws://127.0.0.1:9743/ws/edge")
    node_key = _env("TARS_NODE_KEY", "pi-dev")
    token = _env("TARS_NODE_TOKEN")
    if not token:
        logger.error("TARS_NODE_TOKEN requis (POST /api/fleet/nodes sur le cerveau)")
        sys.exit(1)

    stop = asyncio.Event()

    def _stop(*_a: object) -> None:
        stop.set()

    try:
        signal.signal(signal.SIGINT, _stop)
        signal.signal(signal.SIGTERM, _stop)
    except Exception:  # noqa: BLE001
        pass

    backoff = 2.0
    while not stop.is_set():
        try:
            await run_once(base, node_key, token)
            backoff = 2.0
        except ConnectionClosed as exc:
            logger.warning("Uplink fermé: %s", exc)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Uplink erreur: %s", exc)
        if stop.is_set():
            break
        logger.info("Reconnexion dans %.0fs…", backoff)
        try:
            await asyncio.wait_for(stop.wait(), timeout=backoff)
        except asyncio.TimeoutError:
            pass
        backoff = min(backoff * 1.5, 60.0)


if __name__ == "__main__":
    asyncio.run(main())
