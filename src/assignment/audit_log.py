"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import time
import uuid


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, dict] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Store an input event until its corresponding output is available."""
        correlation_id = request_id or f"req-{uuid.uuid4().hex[:12]}"
        self._open[correlation_id] = {
            "request_id": correlation_id,
            "user_id": user_id,
            "input": text,
            "started_at": utc_now_iso(),
            "started_monotonic": time.perf_counter(),
        }
        return correlation_id

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Complete an audit event with its decision and measured latency."""
        correlation_id = request_id or user_id
        pending = self._open.pop(correlation_id, None)

        # Support the simple record_input(...), record_output(...) sequence even
        # when callers omit a request_id.
        if pending is None and request_id is None:
            matching_id = next(
                (
                    key
                    for key, value in reversed(list(self._open.items()))
                    if value["user_id"] == user_id
                ),
                None,
            )
            if matching_id is not None:
                correlation_id = matching_id
                pending = self._open.pop(matching_id)

        completed_at = utc_now_iso()
        latency_ms = 0.0
        if pending is not None:
            latency_ms = max(
                0.0,
                (time.perf_counter() - pending["started_monotonic"]) * 1000,
            )

        entry = {
            "request_id": correlation_id,
            "user_id": user_id,
            "input": pending["input"] if pending else None,
            "output": text,
            "blocked": blocked,
            "layer": layer,
            "started_at": pending["started_at"] if pending else None,
            "completed_at": completed_at,
            "latency_ms": round(latency_ms, 3),
        }
        self.logs.append(entry)
        return entry

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.logs, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return str(path)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
