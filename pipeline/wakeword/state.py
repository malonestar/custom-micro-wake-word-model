"""Step completion markers and a machine-readable run status file.

The pipeline is designed to be killed at any moment — a preempted spot VM, a
closed laptop, an OOM — and then restarted with the exact same command. Each
step writes a marker only after it has fully succeeded, so a restart skips
completed work and resumes at the step that died.

`status.json` is written continuously so you can see progress from another
shell (or paste it into a chat) without reading the whole log.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path


class State:
    def __init__(self, state_dir: Path):
        self.dir = state_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.status_path = self.dir / "status.json"

    # ---- step markers -----------------------------------------------------
    def marker(self, step: str) -> Path:
        return self.dir / f"{step}.done"

    def is_done(self, step: str) -> bool:
        return self.marker(step).exists()

    def mark_done(self, step: str, detail: str = "") -> None:
        self.marker(step).write_text(
            json.dumps({"finished_at": time.time(), "detail": detail}, indent=2)
        )

    def clear(self, step: str) -> None:
        self.marker(step).unlink(missing_ok=True)

    # ---- status file ------------------------------------------------------
    def read_status(self) -> dict:
        if self.status_path.exists():
            try:
                return json.loads(self.status_path.read_text())
            except json.JSONDecodeError:
                pass
        return {}

    def update(self, **fields) -> None:
        status = self.read_status()
        status.update(fields)
        status["updated_at"] = time.time()
        tmp = self.status_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(status, indent=2))
        tmp.replace(self.status_path)  # atomic: never leave a torn status file

    @contextmanager
    def step(self, name: str, description: str):
        """Track one step: records start, success, or the failure message."""
        started = time.time()
        self.update(current_step=name, current_step_description=description,
                    current_step_started_at=started, error=None)
        try:
            yield
        except BaseException as exc:  # includes KeyboardInterrupt / SystemExit
            self.update(
                error=f"{name}: {type(exc).__name__}: {exc}",
                failed_step=name,
                current_step=None,
            )
            raise
        elapsed = time.time() - started
        self.mark_done(name, description)
        completed = self.read_status().get("completed_steps", [])
        if name not in completed:
            completed.append(name)
        self.update(
            completed_steps=completed,
            current_step=None,
            last_step_seconds=round(elapsed, 1),
        )
