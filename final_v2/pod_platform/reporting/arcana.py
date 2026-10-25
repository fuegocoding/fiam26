"""Operator-configured Arcana review bundle; no vendor schema is assumed."""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

CUTOFF = dt.date(2026, 8, 31)
FIELDS = ("date", "security_id", "weight", "source_classification")
CLASSES = {"fixture", "research", "formation"}


def _canonical(records, descriptive=False):
    out = []
    seen = set()
    for i, r in enumerate(records):
        try:
            day = dt.date.fromisoformat(str(r["date"]))
        except Exception as e:
            raise ValueError(f"record {i}: invalid date") from e
        if day > CUTOFF and not descriptive:
            raise ValueError(
                f"record {i}: date after {CUTOFF} requires descriptive mode"
            )
        sid = str(r.get("security_id", "")).strip()
        if not sid:
            raise ValueError(f"record {i}: security_id is required")
        if (day, sid) in seen:
            raise ValueError(f"duplicate position for {day} {sid}")
        seen.add((day, sid))
        try:
            weight = float(r["weight"])
        except Exception as e:
            raise ValueError(f"record {i}: invalid weight") from e
        if not math.isfinite(weight):
            raise ValueError(f"record {i}: weight must be finite")
        cls = str(r.get("source_classification", ""))
        if cls not in CLASSES:
            raise ValueError(
                f"record {i}: source_classification must be fixture, research, or formation"
            )
        out.append(
            {
                "date": day.isoformat(),
                "security_id": sid,
                "weight": weight,
                "source_classification": cls,
            }
        )
    return sorted(out, key=lambda r: (r["date"], r["security_id"]))


def make_bundle(
    records: Sequence[Mapping[str, Any]],
    *,
    mapping: Mapping[str, str] | None = None,
    vendor_schema: Mapping[str, str] | None = None,
    weight_unit="decimal",
    source_files: Sequence[str | Path] = (),
    descriptive=False,
):
    if weight_unit not in ("decimal", "percent"):
        raise ValueError("weight_unit must be decimal or percent")
    rows = _canonical(records, descriptive)
    mapping = dict({k: k for k in FIELDS} if mapping is None else mapping)
    if set(mapping) != set(FIELDS):
        raise ValueError(f"mapping must define exactly: {', '.join(FIELDS)}")
    cols = list(mapping.values())
    if any(
        not isinstance(c, str)
        or not c
        or c in (".", "..")
        or "/" in c
        or "\\" in c
        for c in cols
    ):
        raise ValueError("target columns must be nonempty simple names")
    if len(set(cols)) != len(cols):
        raise ValueError("target column collision")
    schema = dict(vendor_schema or {})
    if schema:
        if set(cols) - set(schema):
            raise ValueError(
                f"vendor schema missing mapped column(s): {', '.join(sorted(set(cols) - set(schema)))}"
            )
        types = {
            "date": "date",
            "security_id": "string",
            "weight": "number",
            "source_classification": "string",
        }
        for field, col in mapping.items():
            if schema[col] != types[field]:
                raise ValueError(f"vendor schema type mismatch for {col}")
    exported = []
    for row in rows:
        target = {mapping[k]: row[k] for k in FIELDS}
        if weight_unit == "percent":
            target[mapping["weight"]] *= 100.0
        exported.append(target)
    fingerprints = []
    for f in source_files:
        p = Path(f)
        h = hashlib.sha256()
        with p.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(block)
        fingerprints.append({"path": str(p.resolve()), "sha256": h.hexdigest()})
    schema_ok = bool(schema)
    payload_hash = hashlib.sha256(
        json.dumps(exported, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "status": "vendor_schema_validated_export"
        if schema_ok
        else "canonical_review_only",
        "vendor_compatible": schema_ok,
        "live_uploaded": False,
        "weight_unit": weight_unit,
        "mapping": mapping,
        "vendor_schema": schema or None,
        "source_files": fingerprints,
        "source_classifications": sorted(
            {r["source_classification"] for r in rows}
        ),
        "rows": exported,
        "payload_sha256": payload_hash,
        "notice": "Local validation only; Arcana vendor compatibility and portfolio admission are not established.",
    }


def _csv(f, cols, rows):
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="raise")
    w.writeheader()
    w.writerows(rows)


def write_bundle(bundle, directory, *, overwrite=False):
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    outputs = [
        (
            d / "positions.csv",
            lambda f: _csv(
                f, list(bundle["mapping"].values()), list(bundle["rows"])
            ),
        ),
        (
            d / "bundle.json",
            lambda f: (
                json.dump(bundle, f, indent=2, sort_keys=True),
                f.write("\n"),
            ),
        ),
    ]
    if not overwrite and any(p.exists() for p, _ in outputs):
        raise FileExistsError("bundle output exists; pass --overwrite")
    for p, writer in outputs:
        fd, tmp = tempfile.mkstemp(prefix=f".{p.name}.", dir=d, text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
                writer(f)
                f.flush()
                os.fsync(f.fileno())
            if overwrite:
                os.replace(tmp, p)
            else:
                try:
                    os.link(tmp, p)
                except FileExistsError as e:
                    raise FileExistsError(
                        "bundle output exists; pass --overwrite"
                    ) from e
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    return outputs[0][0], outputs[1][0]


def publish_bundle(
    bundle: Mapping[str, Any],
    *,
    endpoint: str | None,
    key_env: str | None,
    transport: Callable[..., Mapping[str, Any]] | None,
    consent: bool,
    receipt_path: str | Path,
):
    if not consent:
        raise PermissionError("explicit publish consent is required")
    if not endpoint or not endpoint.startswith("https://"):
        raise ValueError("an explicit HTTPS endpoint is required")
    if not key_env or not key_env.replace("_", "").isalnum():
        raise ValueError(
            "an environment variable name for the API key is required"
        )
    secret = os.environ.get(key_env)
    if not secret:
        raise ValueError(
            "configured API key environment variable is unavailable"
        )
    if transport is None:
        raise ValueError("an injected exact-endpoint transport is required")
    receipt = Path(receipt_path)
    config_hash = hashlib.sha256(
        json.dumps(
            {"endpoint": endpoint, "key_env": key_env}, sort_keys=True
        ).encode()
    ).hexdigest()
    payload_hash = str(bundle["payload_sha256"])
    if receipt.exists():
        old = json.loads(receipt.read_text())
        if (
            old.get("config_hash") == config_hash
            and old.get("payload_sha256") == payload_hash
        ):
            return old
        raise ValueError(
            "publish idempotency conflict: receipt path already records a different payload or endpoint configuration"
        )
    try:
        response = transport(
            endpoint=endpoint,
            api_key=secret,
            payload=dict(bundle),
            allow_redirects=False,
        )
    except Exception:
        raise RuntimeError(
            "publish transport failed; response details were suppressed"
        ) from None
    result = {
        "status": "published",
        "endpoint_hash": hashlib.sha256(endpoint.encode()).hexdigest(),
        "config_hash": config_hash,
        "payload_sha256": payload_hash,
        "receipt_id": str(response.get("receipt_id", "")),
    }
    if not result["receipt_id"]:
        raise ValueError("transport response did not include receipt_id")
    receipt.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=f".{receipt.name}.", dir=receipt.parent, text=True
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.link(tmp, receipt)
    except FileExistsError as e:
        raise ValueError(
            "publish receipt raced with another request; inspect receipt before retry"
        ) from e
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return result
