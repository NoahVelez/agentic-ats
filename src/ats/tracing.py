"""File-based run traces: ``runs/{run_id}/trace.jsonl``, one JSON object per line (SPEC §8).

This is the minimum the MCP server needs to log tool calls. Run ids come from an HTTP header, so
they are validated before they are used in a path, and events are only written for runs whose
folder already exists (whoever starts a run creates it).
"""

from __future__ import annotations

import re
import threading
from pathlib import Path

from ats.models import TraceEvent

RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]{0,63}$")
TRACE_FILE = "trace.jsonl"

_lock = threading.Lock()


def valid_run_id(run_id: str) -> bool:
    return bool(RUN_ID_RE.fullmatch(run_id))


def run_dir(runs_dir: Path, run_id: str) -> Path:
    if not valid_run_id(run_id):
        raise ValueError("invalid run id")
    return Path(runs_dir) / run_id


def append_event(runs_dir: Path, event: TraceEvent) -> bool:
    """Append one event. Returns False, writing nothing, if the run folder does not exist."""
    folder = run_dir(runs_dir, event.run_id)
    if not folder.is_dir():
        return False
    line = event.model_dump_json() + "\n"
    with _lock, (folder / TRACE_FILE).open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(line)
    return True


def read_trace(runs_dir: Path, run_id: str) -> str | None:
    """The trace as JSONL text, or None if the run has no trace."""
    path = run_dir(runs_dir, run_id) / TRACE_FILE
    return path.read_text(encoding="utf-8") if path.is_file() else None
