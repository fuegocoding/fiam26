"""Inspect research capabilities or page through dated local 8-K filings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pod_platform.application.configuration import config_path, load_config
from pod_platform.application.research_access import write_access_audit
from pod_platform.research.filing_access import FilingResearchStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["inventory", "filings"],
        nargs="?",
        default="inventory",
    )
    parser.add_argument("--as-of")
    parser.add_argument("--permno", type=int, action="append")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--text", action="store_true")
    parser.add_argument("--max-text-chars", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_config()
    settings = config["training_lab"]
    store = FilingResearchStore(
        config_path(settings["filings_path"]),
        training_end=settings["end"],
        start=settings["start"],
    )
    if args.command == "inventory":
        destination = (
            config_path(config["application"]["state_dir"]) / "research_access"
        )
        result = write_access_audit(
            config, destination, {"filing_research": store.audit()}
        )
        text = json.dumps(result, indent=2)
    else:
        documents = store.query(
            args.as_of or settings["end"],
            permnos=args.permno,
            offset=args.offset,
            limit=args.limit,
            include_text=args.text,
            max_text_chars=args.max_text_chars,
        )
        text = documents.to_json(orient="records", date_format="iso", indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
        print(args.output)
    else:
        print(text)


if __name__ == "__main__":
    main()
