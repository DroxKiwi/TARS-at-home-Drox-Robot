"""Catalogue des fonctions appelables — assignables aux experts (rôles)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

Scope = Literal["chat", "specialist"]


@dataclass(frozen=True)
class FunctionDef:
    key: str
    name: str
    description: str
    """Description courte pour l'UI."""
    scopes: tuple[Scope, ...]
    """Qui peut recevoir cette fonction (chat léger et/ou spécialiste)."""
    schema: dict[str, Any]
    """Schéma Ollama / OpenAI tool."""


def _fn(
    key: str,
    name: str,
    description: str,
    scopes: tuple[Scope, ...],
    parameters: dict[str, Any],
    *,
    tool_description: str | None = None,
) -> FunctionDef:
    return FunctionDef(
        key=key,
        name=name,
        description=description,
        scopes=scopes,
        schema={
            "type": "function",
            "function": {
                "name": key,
                "description": tool_description or description,
                "parameters": parameters,
            },
        },
    )


# Catalogue unique — ajouter ici les nouvelles capacités
FUNCTION_CATALOG: dict[str, FunctionDef] = {
    "show_shape": _fn(
        "show_shape",
        "Afficher une forme",
        "Dessine une forme colorée sur le panneau de l'interface.",
        ("chat", "specialist"),
        {
            "type": "object",
            "required": ["shape", "color"],
            "properties": {
                "shape": {
                    "type": "string",
                    "enum": ["circle", "square", "triangle"],
                    "description": "Forme à dessiner",
                },
                "color": {
                    "type": "string",
                    "description": "Couleur CSS (blue, red, #0066ff…)",
                },
                "clear": {
                    "type": "boolean",
                    "description": "Effacer le panneau avant de dessiner",
                },
            },
        },
        tool_description=(
            "Affiche une forme géométrique colorée sur le panneau de l'interface. "
            "À utiliser pour un cercle, carré, triangle, etc."
        ),
    ),
    "clear_canvas": _fn(
        "clear_canvas",
        "Effacer le panneau",
        "Efface toutes les formes du panneau géométrique.",
        ("chat", "specialist"),
        {"type": "object", "properties": {}},
    ),
    "web_search": _fn(
        "web_search",
        "Recherche web",
        "Recherche des infos à jour via SearXNG (actualité, faits, dates).",
        ("chat", "specialist"),
        {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Requête de recherche (mots-clés clairs)",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Nombre de résultats (1–8, défaut 5)",
                },
            },
        },
        tool_description=(
            "Recherche des informations à jour sur le web via SearXNG. "
            "Pour l'actualité, faits récents, dates, définitions. "
            "Ne pas inventer : cherche puis résume."
        ),
    ),
    # --- Flotte / réseau (cerveau) ---
    "lan_list_hosts": _fn(
        "lan_list_hosts",
        "Lister le réseau / la flotte",
        "Liste les hôtes SSH enregistrés et les nœuds edge (Pi) + état uplink.",
        ("chat", "specialist"),
        {"type": "object", "properties": {}},
        tool_description=(
            "Liste les machines connues du cerveau (hosts SSH) et les nœuds edge "
            "connectés via uplink. Ne scanne pas le réseau librement."
        ),
    ),
    "ssh_run": _fn(
        "ssh_run",
        "Commande SSH",
        "Exécute une commande allowlistée sur un hôte SSH enregistré (côté cerveau).",
        ("chat", "specialist"),
        {
            "type": "object",
            "required": ["host_key", "command"],
            "properties": {
                "host_key": {
                    "type": "string",
                    "description": "Clé de l'hôte (ex. pi-salon, nas)",
                },
                "command": {
                    "type": "string",
                    "description": "Commande allowlistée (uname -a, hostname, uptime…)",
                },
            },
        },
    ),
    # --- Edge Pi (via uplink) ---
    "edge_wifi_scan": _fn(
        "edge_wifi_scan",
        "Scan Wi‑Fi (nœud)",
        "Demande à un nœud edge (Pi) de scanner les réseaux Wi‑Fi locaux.",
        ("chat", "specialist"),
        {
            "type": "object",
            "required": ["node_key"],
            "properties": {
                "node_key": {
                    "type": "string",
                    "description": "Clé du nœud edge (ex. pi-salon)",
                },
            },
        },
    ),
    "edge_wifi_status": _fn(
        "edge_wifi_status",
        "État Wi‑Fi (nœud)",
        "État des interfaces Wi‑Fi / uplink du nœud edge.",
        ("chat", "specialist"),
        {
            "type": "object",
            "required": ["node_key"],
            "properties": {
                "node_key": {"type": "string", "description": "Clé du nœud edge"},
            },
        },
    ),
    "edge_bt_scan": _fn(
        "edge_bt_scan",
        "Scan Bluetooth (nœud)",
        "Demande à un nœud edge de scanner les périphériques Bluetooth.",
        ("chat", "specialist"),
        {
            "type": "object",
            "required": ["node_key"],
            "properties": {
                "node_key": {"type": "string", "description": "Clé du nœud edge"},
            },
        },
    ),
    "edge_bt_connect": _fn(
        "edge_bt_connect",
        "Connexion Bluetooth (nœud)",
        "Demande au nœud edge de se connecter à un appareil Bluetooth connu.",
        ("chat", "specialist"),
        {
            "type": "object",
            "required": ["node_key", "address"],
            "properties": {
                "node_key": {"type": "string", "description": "Clé du nœud edge"},
                "address": {
                    "type": "string",
                    "description": "Adresse MAC Bluetooth (AA:BB:…)",
                },
                "name": {
                    "type": "string",
                    "description": "Nom optionnel de l'appareil",
                },
            },
        },
    ),
    "edge_exec": _fn(
        "edge_exec",
        "Commande sur nœud edge",
        "Exécute une commande allowlistée sur le Pi via l'uplink (pas SSH LAN).",
        ("chat", "specialist"),
        {
            "type": "object",
            "required": ["node_key", "command"],
            "properties": {
                "node_key": {"type": "string"},
                "command": {
                    "type": "string",
                    "description": "Commande allowlistée locale au nœud",
                },
            },
        },
    ),
}

# Outils réservés au chat léger (pas assignables aux experts)
CHAT_ONLY_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_specialist_reply",
            "description": (
                "Lit à voix haute la dernière réponse complète d'un spécialiste "
                "(après confirmation de l'utilisateur)."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

# Fonctions toujours disponibles pour le chat (en plus des ask_<role>)
CHAT_DEFAULT_FUNCTION_KEYS: tuple[str, ...] = (
    "show_shape",
    "clear_canvas",
    "web_search",
    "lan_list_hosts",
)


def list_catalog(*, scope: Scope | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for fn in FUNCTION_CATALOG.values():
        if scope and scope not in fn.scopes:
            continue
        out.append(
            {
                "key": fn.key,
                "name": fn.name,
                "description": fn.description,
                "scopes": list(fn.scopes),
            }
        )
    return sorted(out, key=lambda x: x["key"])


def schemas_for_keys(keys: list[str] | tuple[str, ...] | None) -> list[dict[str, Any]]:
    """Schémas Ollama pour une liste de clés catalogue."""
    tools: list[dict[str, Any]] = []
    seen: set[str] = set()
    for key in keys or []:
        k = str(key).strip()
        if not k or k in seen:
            continue
        fn = FUNCTION_CATALOG.get(k)
        if fn is None:
            continue
        seen.add(k)
        tools.append(fn.schema)
    return tools


def normalize_function_keys(
    keys: list[str] | None,
    *,
    scope: Scope = "specialist",
) -> list[str]:
    """Filtre / déduplique les clés valides pour un scope."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in keys or []:
        k = str(raw).strip()
        fn = FUNCTION_CATALOG.get(k)
        if fn is None or scope not in fn.scopes:
            continue
        if k in seen:
            continue
        seen.add(k)
        out.append(k)
    return out


def format_action(name: str, args: dict[str, Any] | None) -> str:
    args = args or {}
    if not args:
        return f"{name}()"
    parts = []
    for k, v in list(args.items())[:6]:
        parts.append(f"{k}={v!r}")
    return f"{name}({', '.join(parts)})"
