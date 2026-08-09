"""Export persistant des runs (métriques + transcripts) vers un volume monté."""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("tars.export")


class MetricsExporter:
    """Écrit JSONL + JSON par tour sous METRICS_DIR (volume Docker → host)."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.turns_dir = self.root / "turns"
        self.runs_path = self.root / "runs.jsonl"
        self.sessions_path = self.root / "sessions.jsonl"
        self._lock = threading.Lock()
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.turns_dir.mkdir(parents=True, exist_ok=True)
        marker = self.root / "README.txt"
        if not marker.exists():
            marker.write_text(
                "Exports TARS Voice — un enregistrement JSONL par tour (runs.jsonl),\n"
                "détail par tour dans turns/<turn_id>.json, sessions dans sessions.jsonl.\n",
                encoding="utf-8",
            )

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _append_jsonl(self, path: Path, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        with self._lock:
            with path.open("a", encoding="utf-8") as f:
                f.write(line)

    def write_session(self, session: dict[str, Any]) -> Path:
        record = {
            "ts": self._now_iso(),
            "kind": "session_load",
            **session,
        }
        self._append_jsonl(self.sessions_path, record)
        logger.info("Session exportée → %s", self.sessions_path)
        return self.sessions_path

    def write_turn(
        self,
        metrics: dict[str, Any],
        *,
        user_text: str = "",
        assistant_text: str = "",
        thinking_text: str = "",
        session: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        turn_id = str(metrics.get("turn_id") or "unknown")
        record: dict[str, Any] = {
            "ts": self._now_iso(),
            "kind": "turn",
            "turn_id": turn_id,
            "transcript": {
                "user": user_text,
                "assistant": assistant_text,
                "thinking": thinking_text,
            },
            "session": session or {},
            "metrics": metrics,
        }
        if extra:
            record["extra"] = extra

        turn_path = self.turns_dir / f"{turn_id}.json"
        with self._lock:
            turn_path.write_text(
                json.dumps(record, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            with self.runs_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

        logger.info(
            "Turn exporté → %s (aussi append %s)",
            turn_path,
            self.runs_path.name,
        )
        return turn_path
