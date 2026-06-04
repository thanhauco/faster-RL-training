"""JSONL metric logging."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


class JsonlLogger:
    def __init__(self, path: Optional[str | Path] = None) -> None:
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: dict) -> None:
        if self.path:
            with self.path.open("a") as f:
                f.write(json.dumps(record) + "\n")
