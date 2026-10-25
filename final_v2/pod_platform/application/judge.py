"""Resolvable judge facts, immutable commitments and explicitly bounded acceptance."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from .common import (
    atomic_write_json,
    digest,
    file_sha256,
    now_utc,
    parse_aware,
    read_json,
)


@dataclass(frozen=True)
class FactRef:
    artifact: str
    sha256: str
    pointer: str = ""


class FactResolver:
    def __init__(self, run_dir):
        self.root = Path(run_dir).resolve()

    def resolve(self, ref: FactRef | dict):
        ref = FactRef(**ref) if isinstance(ref, dict) else ref
        path = (self.root / ref.artifact).resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            raise ValueError("fact path is outside run or absent")
        if file_sha256(path) != ref.sha256:
            raise ValueError(f"fact hash mismatch: {ref.artifact}")
        value = read_json(path)
        if ref.pointer:
            if not ref.pointer.startswith("/"):
                raise ValueError("fact pointer must be a JSON pointer")
            for part in ref.pointer[1:].split("/"):
                part = part.replace("~1", "/").replace("~0", "~")
                value = (
                    value[int(part)] if isinstance(value, list) else value[part]
                )
        return value

    def ref(self, artifact, pointer=""):
        ref = FactRef(artifact, file_sha256(self.root / artifact), pointer)
        self.resolve(ref)
        return ref


@dataclass
class JudgeBuild:
    status: str
    package: dict
    missing: list[str]
    facts: dict


def build_judge_package(run_dir, **kwargs):
    root = Path(run_dir)
    resolver = FactResolver(root)
    facts = {}
    missing = []
    for name, artifact in [
        ("performance", "book/performance_report.json"),
        ("risk", "book/risk_reviews.json"),
        ("compliance", "book/compliance_report.json"),
        ("decisions", "book/decision_dossier.json"),
        ("paired_exhibit", "book/twin_exhibit.json"),
        ("candidate_challenge", "candidates.json"),
        ("attribution", "book/attribution_report.json"),
        ("daily", "book/daily_report.json"),
        ("incremental_portfolio", "book/incremental_evidence.json"),
        ("decay", "book/alpha_decay.json"),
    ]:
        if (root / artifact).is_file():
            facts[name] = resolver.ref(artifact).__dict__
        else:
            missing.append(name)
    candidates = (
        read_json(root / "candidates.json")
        if (root / "candidates.json").is_file()
        else []
    )
    rejected = [
        i for i, c in enumerate(candidates) if c.get("outcome") == "REJECTED"
    ]
    if rejected:
        facts["rejected_idea"] = resolver.ref(
            "candidates.json", f"/{rejected[0]}"
        ).__dict__
    else:
        missing.append("rejected_idea")
    twin = root / "book/twin_exhibit.json"
    if twin.is_file() and read_json(twin).get("failures"):
        facts["failed_trade"] = resolver.ref(
            "book/twin_exhibit.json", "/failures/0"
        ).__dict__
    else:
        missing.append("failed_trade")
    # All facts are source-resolved before the package is persisted.
    for ref in facts.values():
        resolver.resolve(ref)
    package = {
        "schema_version": "judge_package.v1",
        "status": "COMPLETE_SOFTWARE_FIXTURE" if not missing else "INCOMPLETE",
        "facts": facts,
        "missing": missing,
        "investment_validation": "UNPROVEN",
        "limitations": [
            "Fixture evidence tests software; real investment performance and LLM skill remain unvalidated."
        ],
    }
    atomic_write_json(root / "judge_package.json", package)
    return JudgeBuild(package["status"], package, missing, facts)


def verify_judge_package(run_dir, package=None):
    package = package or read_json(Path(run_dir) / "judge_package.json")
    resolver = FactResolver(run_dir)
    for ref in package["facts"].values():
        resolver.resolve(ref)
    return {
        "valid": True,
        "facts_verified": len(package["facts"]),
        "status": package["status"],
    }


def build_commitment(
    run_dir, *, frozen_at, prospective_start=None, artifacts=None
):
    root = Path(run_dir)
    stamp = parse_aware(str(frozen_at), "frozen_at")
    names = (
        artifacts
        if artifacts is not None
        else sorted(
            p.relative_to(root).as_posix()
            for p in root.rglob("*")
            if p.is_file()
            and p.name
            not in (
                "commitment.json",
                "acceptance.json",
                "reference_summary.json",
            )
        )
    )
    hashes = {}
    for name in names:
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("commitment path escapes run")
        hashes[name] = file_sha256(path)
    payload = {
        "schema_version": "prospective_commitment.v1",
        "recorded_at": now_utc().isoformat(),
        "frozen_at": stamp.isoformat(),
        "prospective_start": str(prospective_start)
        if prospective_start
        else None,
        "artifacts": hashes,
        "purpose": "content freeze; prospective performance requires separately observed future outcomes",
    }
    commitment = {**payload, "commitment_hash": digest(payload)}
    target = root / "commitment.json"
    if target.exists() and read_json(target) != commitment:
        raise ValueError("commitment already exists with different content")
    atomic_write_json(target, commitment)
    return commitment


def verify_commitment(run_dir, commitment=None):
    root = Path(run_dir)
    commitment = commitment or read_json(root / "commitment.json")
    if (
        digest({k: v for k, v in commitment.items() if k != "commitment_hash"})
        != commitment["commitment_hash"]
    ):
        raise ValueError("commitment record hash mismatch")
    for name, sha in commitment["artifacts"].items():
        path = (root / name).resolve()
        if (
            not path.is_relative_to(root.resolve())
            or not path.is_file()
            or file_sha256(path) != sha
        ):
            raise ValueError(f"committed artifact changed: {name}")
    return {
        "valid": True,
        "artifacts_verified": len(commitment["artifacts"]),
        "commitment_hash": commitment["commitment_hash"],
    }


def score_prospective(run_dir, observations=()):
    verify_commitment(run_dir)
    c = read_json(Path(run_dir) / "commitment.json")
    if not c.get("prospective_start") or not observations:
        return {
            "status": "INCOMPLETE",
            "reason": "No separately observed prospective outcomes after a dated commitment.",
        }
    cutoff = max(
        parse_aware(c["frozen_at"]),
        parse_aware(c["recorded_at"]),
        parse_aware(c["prospective_start"]),
    )
    vals = []
    for row in observations:
        if (
            parse_aware(row["observed_at"]) <= cutoff
            or parse_aware(row["formation_at"]) <= cutoff
        ):
            raise ValueError(
                "prospective scoring requires formation and observation after commitment"
            )
        value = float(row["net_return"])
        if not math.isfinite(value) or value <= -1:
            raise ValueError("invalid prospective return")
        vals.append(value)
    growth = 1.0
    for value in vals:
        growth *= 1 + value
    return {
        "status": "COMPUTED_SUPPLIED_OUTCOMES",
        "observations": len(vals),
        "net_return": growth - 1,
        "limitation": "Operator-supplied outcomes; independent provenance must be reviewed.",
    }


def acceptance_report(run_dir):
    root = Path(run_dir)
    checks = {}
    errors = []
    for name, fn in [
        ("commitment", verify_commitment),
        ("judge_facts", verify_judge_package),
    ]:
        try:
            checks[name] = fn(root)
        except (ValueError, KeyError, FileNotFoundError) as exc:
            errors.append(str(exc))
    package = (
        read_json(root / "judge_package.json")
        if (root / "judge_package.json").is_file()
        else {}
    )
    twin_path = root / "book/twin_exhibit.json"
    if twin_path.exists():
        twin = read_json(twin_path)
        valid = (
            twin.get("schema_version") == "paired_exhibit.v2"
            and twin.get("status") == "COMPLETED"
            and bool(twin.get("periods"))
            and all(p.get("agent") and p.get("fixed") for p in twin["periods"])
        )
        checks["paired_nav"] = {
            "valid": valid,
            "periods": len(twin.get("periods", [])),
            "net_difference": twin.get("paired_nav_difference"),
            "llm_skill_evidence": False,
        }
        if not valid:
            errors.append(
                "Full self-financing paired NAV evidence is incomplete."
            )
    else:
        errors.append("Full paired replay artifact is unavailable.")
    result = {
        "software_status": "PASS"
        if not errors and not package.get("missing")
        else "INCOMPLETE",
        "checks": checks,
        "errors": errors,
        "missing": package.get("missing", ["judge_package"]),
        "investment_validation": "UNPROVEN",
        "prospective": score_prospective(root)
        if not errors
        else {"status": "INCOMPLETE"},
    }
    atomic_write_json(root / "acceptance.json", result)
    return result
