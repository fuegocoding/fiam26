"""Read named model instructions from the shared, editable prompt file."""

from __future__ import annotations

import re
import sysconfig
from pathlib import Path

PROMPTS_PATH = Path(__file__).resolve().parents[1] / "prompts.txt"
if not PROMPTS_PATH.is_file():
    PROMPTS_PATH = (
        Path(sysconfig.get_path("data"))
        / "share/fiam-pod-platform/prompts.txt"
    )
_SECTION = re.compile(r"^@@ ([a-z][a-z0-9_.-]*)\n", re.MULTILINE)
_FIELD = re.compile(r"\{\{([a-z][a-z0-9_]*)\}\}")


def load_prompt(name: str, **fields: object) -> str:
    """Load one section and substitute its explicit double-brace fields.

    Read on demand so edits affect subsequent requests and their cache hashes.
    Values are inserted once; braces in task data are never reinterpreted.
    """
    source = PROMPTS_PATH.read_text(encoding="utf-8")
    matches = list(_SECTION.finditer(source))
    sections = {}
    for index, match in enumerate(matches):
        key = match.group(1)
        if key in sections:
            raise ValueError(f"Duplicate prompt section: {key}")
        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(source)
        )
        # One blank separator belongs to the file format, not the prompt.
        sections[key] = source[match.end():end].removesuffix("\n\n")
    if name not in sections:
        raise ValueError(f"Unknown prompt section: {name}")
    template = sections[name]
    needed = set(_FIELD.findall(template))
    if needed != set(fields):
        raise ValueError(f"Prompt {name} requires fields {sorted(needed)}")
    return _FIELD.sub(lambda match: str(fields[match.group(1)]), template)
