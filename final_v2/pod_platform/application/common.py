"""Shared helpers for the reference application layer (stdlib + pydantic only).

Nothing here imports pandas/pyarrow or any model provider. All timestamps are
timezone-aware UTC; naive inputs are rejected.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, NoReturn

from ..contracts.base import canonical_digest, require_aware
from ..contracts.errors import ErrorCode, PodPlatformError, ToolResponse

UTC = dt.timezone.utc

SYNTHETIC_LABEL = "SYNTHETIC_FIXTURE_NOT_INVESTMENT_EVIDENCE"
FIXTURE_LIMITATION = "FIXTURE_CERTIFICATE_NOT_FOR_COMPETITION_BOOK"
FIXTURE_MODE_PREFIX = "fixture"
FORMATION_MODE_PREFIX = "formation"

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,99}$")


def now_utc() -> dt.datetime:
    return dt.datetime.now(UTC)


def aware(value: dt.datetime, name: str) -> dt.datetime:
    return require_aware(value, name)


def parse_aware(text: str, name: str = "timestamp") -> dt.datetime:
    """Parse an ISO-8601 timestamp that carries an explicit offset (``Z`` allowed)."""
    try:
        parsed = dt.datetime.fromisoformat(str(text).strip().replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(f"{name} is not an ISO-8601 timestamp") from None
    return require_aware(parsed, name)


def utc_close(day: dt.date) -> dt.datetime:
    """Market-close reference (21:00 UTC), identical to ``orchestration.replay.utc_close``."""
    return dt.datetime(day.year, day.month, day.day, 21, 0, tzinfo=UTC)


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str, allow_nan=False)


def digest(obj: Any) -> str:
    return canonical_digest(json.loads(canonical_json(obj)))


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def file_sha256(path: "str | Path") -> str:
    path = Path(path)
    if not path.is_file():
        fail(ErrorCode.SCHEMA_MISMATCH, f"file does not exist: {path}", "Source hashes need existing files.")
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_sha256(path: "str | Path") -> str:
    """Hash a file, or a directory as sorted (relative path, file hash) pairs."""
    path = Path(path)
    if path.is_file():
        return file_sha256(path)
    if not path.is_dir():
        fail(ErrorCode.SCHEMA_MISMATCH, f"path does not exist: {path}", "Source hashes need existing paths.")
    h = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        rel = item.relative_to(path)
        if any(part.startswith(".") or part == "__pycache__" for part in rel.parts):
            continue
        h.update(rel.as_posix().encode("utf-8"))
        h.update(b"\0")
        h.update(file_sha256(item).encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def atomic_write_bytes(path: "str | Path", payload: bytes) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return sha256_bytes(payload)


def pretty_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, default=str, allow_nan=False)


def atomic_write_json(path: "str | Path", obj: Any) -> str:
    return atomic_write_bytes(path, pretty_json(obj).encode("utf-8"))


def read_json(path: "str | Path") -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def safe_id(value: str, what: str = "identifier") -> str:
    if not isinstance(value, str) or not _SAFE_ID.match(value):
        fail(ErrorCode.SCHEMA_MISMATCH, f"invalid {what}: {value!r}", "Identifiers are plain [A-Za-z0-9_.-] tokens.")
    return value


def fail(code: ErrorCode, summary: str, stop: str, **kwargs: Any) -> NoReturn:
    raise PodPlatformError(ToolResponse.failure(summary=summary, code=code, stop_condition=stop, **kwargs))
