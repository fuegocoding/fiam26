"""Load application settings from the single final_v2 configuration file."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
CONFIG_PATH = ROOT / "config.json"


def load_config(section: str | None = None) -> dict[str, Any]:
    """Read a configuration section, allowing an isolated file for test runs.

    Args:
        section: Optional top-level section name.

    Returns:
        An independent dictionary of configured settings.

    Raises:
        ValueError: If the file or requested section is not a JSON object.
    """
    path = Path(os.environ.get("FIAM_CONFIG", CONFIG_PATH))
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Configuration must be a JSON object.")
    value = payload if section is None else payload.get(section, {})
    if not isinstance(value, dict):
        raise ValueError(
            f"Configuration section {section!r} must be an object."
        )
    return json.loads(
        json.dumps(copy.deepcopy(value)).replace(
            "${project_root}", str(PROJECT_ROOT)
        )
    )


def config_path(value: str | Path) -> Path:
    """Resolve configured paths relative to final_v2 without expanding secrets."""
    path = Path(str(value).replace("${project_root}", str(PROJECT_ROOT)))
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_namespace(
    section: str, defaults: dict[str, Any] | None = None
) -> SimpleNamespace:
    """Support existing callable script functions while settings move to JSON."""
    return SimpleNamespace(**{**(defaults or {}), **load_config(section)})
