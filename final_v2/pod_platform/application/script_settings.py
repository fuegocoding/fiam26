"""Validate configuration for maintained callable utility scripts.

Application runs use final_v2/config.json. Script entrypoints accept an optional
settings mapping for tests or embedding; they do not parse command-line flags.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

from .configuration import config_path, load_config


def load_script_settings(
    section: str,
    specification: dict[str, Any],
    settings: Mapping[str, Any] | None = None,
) -> SimpleNamespace:
    """Load and validate a utility's section of the central configuration."""
    if settings is not None and not isinstance(settings, Mapping):
        raise ValueError(
            "Settings must be a mapping; configure final_v2/config.json instead of command-line flags"
        )
    supplied = {**load_config(section), **(dict(settings) if settings else {})}
    fields = dict(specification["fields"])
    allowed = set(fields)
    for command in specification["commands"].values():
        allowed.update(command["fields"])
    command_field = specification["command_field"]
    if command_field:
        allowed.add(command_field)
    unknown = set(supplied) - allowed
    if unknown:
        raise ValueError(
            f"Unknown {section} configuration fields: {sorted(unknown)}"
        )
    defaults = dict(specification["defaults"])
    if command_field:
        command_name = supplied.get(command_field)
        if command_name not in specification["commands"]:
            raise ValueError(
                f"{section}.{command_field} must select a configured operation"
            )
        command = specification["commands"][command_name]
        fields.update(command["fields"])
        defaults.update(command["defaults"])
        defaults[command_field] = command_name
    values = {key: field["default"] for key, field in fields.items()}
    values.update(
        {key: value for key, value in supplied.items() if key in fields}
    )
    for key, field in fields.items():
        value = values[key]
        if field["required"] and value is None:
            raise ValueError(f"Missing required configuration: {section}.{key}")
        if value is None:
            continue
        many = (
            field["action"] == "append"
            or field["nargs"] in ("+", "*")
            or isinstance(field["nargs"], int)
        )
        if many and not isinstance(value, (list, tuple)):
            raise ValueError(f"{section}.{key} must be a list")
        if field["action"] in ("store_true", "store_false") and not isinstance(
            value, bool
        ):
            raise ValueError(f"{section}.{key} must be boolean")
        converter = field["converter"]
        if converter:
            converters = {
                "int": int,
                "float": float,
                "str": str,
                "bool": bool,
                "config_path": config_path,
                "dt.date.fromisoformat": dt.date.fromisoformat,
                "_dt": dt.datetime.fromisoformat,
            }
            converter = converters[converter]
            value = (
                [converter(item) for item in value]
                if many
                else converter(value)
            )
        choices = field["choices"]
        if choices and any(
            item not in choices for item in (value if many else [value])
        ):
            raise ValueError(
                f"Unsupported configured value for {section}.{key}"
            )
        values[key] = value
    return SimpleNamespace(**{**values, **defaults})
