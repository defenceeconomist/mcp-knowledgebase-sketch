from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def load_dotenv(path: str | Path = ".env") -> None:
    """Load KEY=VALUE lines from a dotenv file into os.environ when missing."""
    file_path = Path(path)
    if not file_path.is_file():
        return
    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def decode_redis_value(value: Any) -> Any:
    """Decode Redis bytes payloads into UTF-8 strings when needed."""
    if value is None:
        return None
    return value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else value


def load_env_bool(key: str, default: bool = False) -> bool:
    """Parse a boolean environment variable with a default fallback."""
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def load_env_int(key: str, default: int) -> int:
    """Parse an integer environment variable with default fallback."""
    raw = os.getenv(key)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def load_env_list(key: str) -> list[str]:
    """Parse a comma-separated environment variable into a stripped list."""
    raw = os.getenv(key, "").strip()
    if not raw:
        return []
    return [entry.strip() for entry in raw.split(",") if entry.strip()]
