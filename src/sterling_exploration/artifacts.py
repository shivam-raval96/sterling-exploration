from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def canonical_config(config: dict[str, Any]) -> dict[str, Any]:
    """Remove runtime-only fields before calculating resume identity."""
    return {
        key: value
        for key, value in config.items()
        if key not in {"fingerprint", "run_id", "run_mode"}
    }


def config_fingerprint(config: dict[str, Any]) -> str:
    payload = json.dumps(canonical_config(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def make_run_id(description: str, now: datetime | None = None) -> str:
    timestamp = (now or datetime.now(UTC)).astimezone(UTC)
    slug = re.sub(r"[^a-z0-9]+", "-", description.lower()).strip("-")
    if not slug:
        raise ValueError("description must contain an ASCII letter or number")
    return f"{timestamp:%Y-%m-%d_%H%M%SZ}_{slug}"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
