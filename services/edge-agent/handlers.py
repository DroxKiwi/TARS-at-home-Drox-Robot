"""Handlers locaux Wi‑Fi / Bluetooth / exec (allowlist) pour le nœud Pi."""

from __future__ import annotations

import asyncio
import logging
import platform
import re
import shutil
from typing import Any

logger = logging.getLogger("edge-handlers")

_EXEC_ALLOW = (
    r"^uname( -a)?$",
    r"^hostname$",
    r"^uptime$",
    r"^whoami$",
    r"^pwd$",
    r"^cat /sys/class/net/.+/operstate$",
)

_MOCK = platform.system() == "Windows" or not shutil.which("nmcli")


async def _run(cmd: list[str], timeout: float = 20.0) -> tuple[int, str, str]:
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out_b, err_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except FileNotFoundError:
        return 127, "", f"commande introuvable: {cmd[0]}"
    except asyncio.TimeoutError:
        return 124, "", "timeout"
    return (
        proc.returncode or 0,
        (out_b or b"").decode("utf-8", errors="replace"),
        (err_b or b"").decode("utf-8", errors="replace"),
    )


def _exec_allowed(command: str) -> bool:
    cmd = " ".join((command or "").strip().split())
    if not cmd or len(cmd) > 200:
        return False
    if any(x in cmd for x in (";", "&&", "||", "|", "`", "$(", "\n")):
        return False
    return any(re.match(p, cmd) for p in _EXEC_ALLOW)


async def wifi_scan(_params: dict[str, Any]) -> dict[str, Any]:
    if _MOCK:
        return {
            "mock": True,
            "networks": [
                {"ssid": "Maison-Demo", "signal": 78, "security": "WPA2"},
                {"ssid": "Invites", "signal": 42, "security": "WPA2"},
            ],
        }
    code, out, err = await _run(["nmcli", "-t", "-f", "SSID,SIGNAL,SECURITY", "dev", "wifi", "list"])
    if code != 0:
        return {"ok": False, "error": err or out or f"exit {code}"}
    networks = []
    for line in out.splitlines():
        parts = line.split(":")
        if len(parts) >= 3 and parts[0]:
            networks.append(
                {"ssid": parts[0], "signal": parts[1], "security": parts[2]}
            )
    return {"networks": networks[:40]}


async def wifi_status(_params: dict[str, Any]) -> dict[str, Any]:
    if _MOCK:
        return {
            "mock": True,
            "connected": True,
            "ssid": "Maison-Demo",
            "ip": "192.168.1.50",
        }
    code, out, err = await _run(["nmcli", "-t", "-f", "DEVICE,TYPE,STATE,CONNECTION", "dev", "status"])
    if code != 0:
        return {"error": err or out}
    return {"devices": [ln for ln in out.splitlines() if ln][:20]}


async def bt_scan(_params: dict[str, Any]) -> dict[str, Any]:
    if _MOCK or not shutil.which("bluetoothctl"):
        return {
            "mock": True,
            "devices": [
                {"address": "AA:BB:CC:DD:EE:01", "name": "Speaker-Demo"},
                {"address": "AA:BB:CC:DD:EE:02", "name": "Phone-Demo"},
            ],
        }
    # Scan court non-interactif
    await _run(["bluetoothctl", "--timeout", "5", "scan", "on"], timeout=8.0)
    code, out, err = await _run(["bluetoothctl", "devices"])
    devices = []
    for line in out.splitlines():
        # Device AA:BB:… Name
        m = re.match(r"Device\s+([0-9A-Fa-f:]+)\s+(.*)$", line.strip())
        if m:
            devices.append({"address": m.group(1), "name": m.group(2).strip()})
    return {"devices": devices[:40], "stderr": err[:200] if err else ""}


async def bt_connect(params: dict[str, Any]) -> dict[str, Any]:
    address = str(params.get("address") or "").strip()
    if not re.match(r"^([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}$", address):
        return {"ok": False, "error": "Adresse MAC invalide"}
    if _MOCK or not shutil.which("bluetoothctl"):
        return {
            "mock": True,
            "connected": True,
            "address": address,
            "name": params.get("name") or "",
        }
    code, out, err = await _run(["bluetoothctl", "connect", address], timeout=25.0)
    return {
        "ok": code == 0,
        "address": address,
        "stdout": out[:1000],
        "stderr": err[:500],
    }


async def local_exec(params: dict[str, Any]) -> dict[str, Any]:
    command = str(params.get("command") or "").strip()
    if not _exec_allowed(command):
        return {"ok": False, "error": "Commande refusée (allowlist edge)"}
    if _MOCK and platform.system() == "Windows":
        return {"mock": True, "command": command, "stdout": f"(mock) ok: {command}"}
    # shell=False : on passe via sh -c uniquement si allowlist déjà validée
    code, out, err = await _run(["/bin/sh", "-c", command], timeout=15.0)
    return {
        "ok": code == 0,
        "exit_code": code,
        "stdout": out[:3000],
        "stderr": err[:1000],
    }


HANDLERS = {
    "wifi_scan": wifi_scan,
    "wifi_status": wifi_status,
    "bt_scan": bt_scan,
    "bt_connect": bt_connect,
    "exec": local_exec,
}


async def dispatch(method: str, params: dict[str, Any] | None) -> dict[str, Any]:
    fn = HANDLERS.get(method)
    if fn is None:
        return {"ok": False, "error": f"Méthode inconnue: {method}"}
    try:
        result = await fn(params or {})
        if isinstance(result, dict) and "ok" in result and result.get("ok") is False:
            return result
        return {"ok": True, "result": result}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Handler %s", method)
        return {"ok": False, "error": str(exc)}
