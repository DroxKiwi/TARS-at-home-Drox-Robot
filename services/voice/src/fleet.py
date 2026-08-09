"""Gateway flotte — SSH/LAN côté cerveau + RPC uplink vers nœuds edge (Pi)."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from fastapi import WebSocket

from . import fleet_db
from .config import get_settings

logger = logging.getLogger(__name__)

# Commandes SSH autorisées (préfixe exact ou regex simple)
_DEFAULT_SSH_ALLOW = (
    r"^uname( -a)?$",
    r"^hostname$",
    r"^uptime$",
    r"^df -h$",
    r"^free -h$",
    r"^cat /etc/os-release$",
    r"^whoami$",
    r"^pwd$",
    r"^ls( -[alA]+)?( /\S*)?$",
    r"^ping -c [1-3] [0-9.a-zA-Z._-]+$",
)


class EdgeHub:
    """Nœuds edge connectés en WebSocket (uplink sortant du Pi)."""

    def __init__(self) -> None:
        self._sockets: dict[str, WebSocket] = {}
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._lock = asyncio.Lock()
        self._req_i = 0

    def online_keys(self) -> list[str]:
        return sorted(self._sockets.keys())

    async def attach(self, node_key: str, ws: WebSocket) -> None:
        async with self._lock:
            old = self._sockets.get(node_key)
            self._sockets[node_key] = ws
        if old is not None and old is not ws:
            try:
                await old.close()
            except Exception:  # noqa: BLE001
                pass
        await fleet_db.set_node_online(node_key, True)

    async def detach(self, node_key: str, ws: WebSocket | None = None) -> None:
        async with self._lock:
            cur = self._sockets.get(node_key)
            if ws is not None and cur is not ws:
                return
            self._sockets.pop(node_key, None)
        await fleet_db.set_node_online(node_key, False)

    async def handle_edge_message(self, node_key: str, msg: dict[str, Any]) -> None:
        if msg.get("type") == "hello":
            await fleet_db.set_node_online(
                node_key, True, meta=msg.get("meta") if isinstance(msg.get("meta"), dict) else {}
            )
            return
        if msg.get("type") == "result":
            req_id = str(msg.get("id") or "")
            fut = self._pending.pop(req_id, None)
            if fut and not fut.done():
                fut.set_result(msg)
            return

    async def call(
        self,
        node_key: str,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float = 45.0,
    ) -> dict[str, Any]:
        ws = self._sockets.get(node_key)
        if ws is None:
            return {
                "ok": False,
                "error": f"Nœud « {node_key} » hors ligne (pas d'uplink).",
            }
        self._req_i += 1
        req_id = f"{node_key}-{self._req_i}"
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending[req_id] = fut
        payload = {
            "type": "command",
            "id": req_id,
            "method": method,
            "params": params or {},
        }
        try:
            await ws.send_json(payload)
            msg = await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            self._pending.pop(req_id, None)
            return {"ok": False, "error": f"Timeout RPC edge ({method})"}
        except Exception as exc:  # noqa: BLE001
            self._pending.pop(req_id, None)
            return {"ok": False, "error": str(exc)}
        return {
            "ok": bool(msg.get("ok")),
            "result": msg.get("result"),
            "error": msg.get("error"),
            "method": method,
            "node_key": node_key,
        }


edge_hub = EdgeHub()


def _ssh_allow_patterns() -> list[re.Pattern[str]]:
    settings = get_settings()
    raw = getattr(settings, "fleet_ssh_allowlist", "") or ""
    lines = [ln.strip() for ln in raw.split(",") if ln.strip()]
    if not lines:
        lines = list(_DEFAULT_SSH_ALLOW)
    return [re.compile(p) for p in lines]


def is_ssh_command_allowed(command: str) -> bool:
    cmd = " ".join((command or "").strip().split())
    if not cmd or len(cmd) > 240:
        return False
    # Interdit les injections grossières
    if any(x in cmd for x in (";", "&&", "||", "|", "`", "$(", "\n", "\r")):
        return False
    return any(p.match(cmd) for p in _ssh_allow_patterns())


async def lan_list_hosts() -> dict[str, Any]:
    hosts = await fleet_db.list_hosts(enabled_only=True)
    nodes = await fleet_db.list_nodes()
    online = set(edge_hub.online_keys())
    for n in nodes:
        n["uplink_online"] = n.get("node_key") in online
    return {
        "ok": True,
        "hosts": [
            {
                "host_key": h["host_key"],
                "label": h["label"],
                "ip": h.get("ip"),
                "hostname": h.get("hostname"),
                "ssh_user": h.get("ssh_user"),
                "tags": h.get("tags") or [],
            }
            for h in hosts
        ],
        "nodes": [
            {
                "node_key": n["node_key"],
                "name": n["name"],
                "kind": n["kind"],
                "online": bool(n.get("online")) or n["node_key"] in online,
                "uplink_online": n["node_key"] in online,
                "last_seen_at": n.get("last_seen_at"),
            }
            for n in nodes
        ],
    }


async def ssh_run(host_key: str, command: str) -> dict[str, Any]:
    host = await fleet_db.get_host(host_key)
    if not host or not host.get("enabled"):
        return {"ok": False, "error": f"Hôte inconnu ou désactivé: {host_key}"}
    cmd = " ".join((command or "").strip().split())
    if not is_ssh_command_allowed(cmd):
        return {
            "ok": False,
            "error": (
                "Commande SSH refusée (allowlist). "
                "Exemples: uname -a, hostname, uptime, df -h."
            ),
        }
    target = host.get("ip") or host.get("hostname")
    if not target:
        return {"ok": False, "error": "Hôte sans IP/hostname"}

    ssh_user = host.get("ssh_user") or "tars"
    ssh_port = int(host.get("ssh_port") or 22)
    key_path = host.get("ssh_key_path")
    args = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ConnectTimeout=8",
        "-p",
        str(ssh_port),
    ]
    if key_path:
        args.extend(["-i", str(key_path)])
    args.append(f"{ssh_user}@{target}")
    args.append(cmd)

    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=30.0)
    except FileNotFoundError:
        return {
            "ok": False,
            "error": "Client ssh absent dans le conteneur (openssh-client).",
        }
    except asyncio.TimeoutError:
        return {"ok": False, "error": "Timeout SSH"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}

    stdout = (stdout_b or b"").decode("utf-8", errors="replace").strip()
    stderr = (stderr_b or b"").decode("utf-8", errors="replace").strip()
    ok = proc.returncode == 0
    return {
        "ok": ok,
        "host_key": host_key,
        "command": cmd,
        "exit_code": proc.returncode,
        "stdout": stdout[:4000],
        "stderr": stderr[:2000],
    }


async def edge_call(node_key: str, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    return await edge_hub.call(node_key.strip().lower(), method, params or {})
