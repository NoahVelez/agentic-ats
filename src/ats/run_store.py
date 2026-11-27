"""File-based persistence for screening runs."""

from __future__ import annotations

import os
from pathlib import Path

from ats.models import RunState
from ats.tracing import valid_run_id


class RunStore:
    """Interface for saving and loading screening run state."""

    def create(self, state: RunState) -> None:
        raise NotImplementedError

    def get(self, run_id: str) -> RunState:
        raise NotImplementedError

    def save(self, state: RunState) -> None:
        raise NotImplementedError


class FileRunStore(RunStore):
    """Persist each run in runs/{run_id}/state.json."""

    def __init__(self, runs_dir: Path) -> None:
        self.runs_dir = Path(runs_dir)

    def _run_folder(self, run_id: str) -> Path:
        if not valid_run_id(run_id):
            raise ValueError(f"Invalid run ID: {run_id}")
        return self.runs_dir / run_id

    def create(self, state: RunState) -> None:
        folder = self._run_folder(state.run_id)
        folder.mkdir(parents=True, exist_ok=False)
        self._write_state(folder / "state.json", state)

    def get(self, run_id: str) -> RunState:
        path = self._run_folder(run_id) / "state.json"
        if not path.is_file():
            raise FileNotFoundError(f"Run not found: {run_id}")

        return RunState.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, state: RunState) -> None:
        folder = self._run_folder(state.run_id)
        if not folder.is_dir():
            raise FileNotFoundError(f"Run not found: {state.run_id}")

        self._write_state(folder / "state.json", state)

    @staticmethod
    def _write_state(path: Path, state: RunState) -> None:
        # Write then rename so a concurrent reader (the API polled by the
        # dashboard) never sees a half-written state.json.
        temp_path = path.with_suffix(".json.tmp")
        temp_path.write_text(
            state.model_dump_json(indent=2),
            encoding="utf-8",
        )
        os.replace(temp_path, path)
