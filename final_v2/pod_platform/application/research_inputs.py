"""Prepare bounded observations from the authorized competition design panel.

The official source card attests month-end availability for characteristics.
Forward returns are shifted to their realization month and are never exposed as
features. Training preparation reads only through 2019-12-31. A separate
reserved call can read 2019 warmup and 2020 validation rows after the parent
has frozen training results and persisted its access reservation.
"""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import math
import sqlite3
from pathlib import Path

try:  # Parquet support is optional for importers that only use other workflows.
    import pandas as pd
except (
    ImportError
):  # pragma: no cover - exercised where the optional extra is absent
    pd = None

from ..contracts.base import require_aware
from ..contracts.sources import SourceCard
from ..research.composition import observation_digest
from ..research.features import FeatureObservation
from .configuration import config_path

SOURCE_ID = "comp_chars_panel"
FEATURE_METRICS = (
    "niq_su",
    "sale_gr1",
    "ni_me",
    "be_me",
    "eqnetis_at",
    "ret_1_0",
    "ret_12_1",
    "dolvol_126d",
)
LABEL_METRIC = "ret_exc_lead1m"
MAX_NAMES = 500
MAX_OBSERVATIONS = 300_000
DEFAULT_SOURCE_MANIFEST = (
    Path(__file__).resolve().parents[2]
    / ".pod_state"
    / "datasets"
    / "competition_design.json"
)
UTC = dt.timezone.utc


def _end_of_day(day: dt.date) -> dt.datetime:
    return dt.datetime.combine(day, dt.time.max, UTC)


def _month_end(day: dt.date) -> dt.date:
    if day.month == 12:
        first_next = dt.date(day.year + 1, 1, 1)
    else:
        first_next = dt.date(day.year, day.month + 1, 1)
    return first_next - dt.timedelta(days=1)


def _next_month_end(day: dt.date) -> dt.date:
    return _month_end(day + dt.timedelta(days=1))


def _previous_month_end(day: dt.date) -> dt.date:
    return day.replace(day=1) - dt.timedelta(days=1)


def _read_design_frame(path: Path, start: dt.date, end: dt.date):
    if pd is None:
        raise RuntimeError(
            "Reading the official Parquet panel requires pandas and pyarrow; "
            "install the repository requirements before preparing live research inputs."
        )
    try:
        # Feature/target measurements retain the full boundary month. The
        # forward-return label is read only where its realization month remains
        # inside this partition, so Dec labels cannot expose Jan held-out returns.
        features = pd.read_parquet(
            path,
            columns=["permno", "eom", "me", *FEATURE_METRICS],
            engine="pyarrow",
            filters=[
                ("eom", ">=", pd.Timestamp(start)),
                ("eom", "<=", pd.Timestamp(end)),
            ],
        )
        labels_end = _previous_month_end(end)
        if labels_end < start:
            features[LABEL_METRIC] = float("nan")
            return features
        labels = pd.read_parquet(
            path,
            columns=["permno", "eom", LABEL_METRIC],
            engine="pyarrow",
            filters=[
                ("eom", ">=", pd.Timestamp(start)),
                ("eom", "<=", pd.Timestamp(labels_end)),
            ],
        )
        return features.merge(
            labels, on=["permno", "eom"], how="left", validate="one_to_one"
        )
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "Reading the official Parquet panel requires pyarrow; "
            "install the repository requirements before preparing live research inputs."
        ) from exc


def _plain_date(value) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime().date()
    return dt.date.fromisoformat(str(value)[:10])


def _finite(value) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_source_card(card: SourceCard) -> None:
    if (
        card.source_id != SOURCE_ID
        or card.status.value != "APPROVED"
        or not card.access_confirmed
    ):
        raise ValueError(
            "The original approved competition source card is required."
        )


def _dates(config: dict) -> dict[str, dt.date]:
    periods = config.get("research", {}).get("periods")
    if not isinstance(periods, dict):
        raise ValueError(
            "Explicit research.periods are required for live input preparation."
        )
    names = (
        "training_start",
        "training_end",
        "validation_start",
        "validation_end",
        "evaluation_start",
    )
    try:
        dates = {name: dt.date.fromisoformat(periods[name]) for name in names}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "research.periods must define ISO training, validation and evaluation boundaries."
        ) from exc
    if (
        dates["training_start"] != dt.date(2015, 1, 1)
        or dates["training_end"] != dt.date(2019, 12, 31)
        or dates["validation_start"] != dt.date(2020, 1, 1)
        or dates["validation_end"] != dt.date(2020, 12, 31)
        or dates["evaluation_start"] != dt.date(2021, 1, 1)
        or config.get("application", {}).get("dataset_id")
        != "comp_chars_training"
    ):
        raise ValueError(
            "Live inputs require dataset_id comp_chars_training and the explicitly authorized "
            "2015–2019 training, 2020 validation, 2021+ evaluation split."
        )
    horizon = (
        config["research"].get("periods", {}).get("purge_horizon_months", 3)
    )
    if horizon != 3:
        raise ValueError(
            "The configured purge horizon must cover the longest three-month native target."
        )
    dates["feature_start"] = dt.date.fromisoformat(
        periods.get("feature_start", dates["training_start"].isoformat())
    )
    if dates["feature_start"] < dt.date(2015, 1, 1):
        raise ValueError(
            "2014 warmup is unavailable in the authorized characteristics panel."
        )
    limit = config.get("research", {}).get("universe_limit", 50)
    if (
        not isinstance(limit, int)
        or isinstance(limit, bool)
        or not 3 <= limit <= MAX_NAMES
    ):
        raise ValueError(
            f"research.universe_limit must be an integer from 3 to {MAX_NAMES}."
        )
    dates["universe_limit"] = limit
    return dates


def _load_authority(
    config: dict,
    dates: dict[str, dt.date],
    now: dt.datetime,
    *,
    write_manifest: bool = True,
) -> tuple[dict, SourceCard, Path, Path, Path]:
    app = config.get("application", {})
    state_dir = config_path(app["state_dir"])
    manifest_path = config_path(
        app.get("manifest", state_dir / "competition_training.json")
    )
    if manifest_path.exists():
        training_manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        if (
            training_manifest.get("manifest_id")
            != "official_live_training_2015_2019"
            or training_manifest.get("research_periods", {}).get(
                "validation_start"
            )
            != dates["validation_start"].isoformat()
            or training_manifest.get("research_periods", {}).get(
                "training_start"
            )
            != dates["training_start"].isoformat()
            or training_manifest.get("research_periods", {}).get("training_end")
            != dates["training_end"].isoformat()
            or training_manifest.get("research_periods", {}).get(
                "validation_end"
            )
            != dates["validation_end"].isoformat()
            or training_manifest.get("research_periods", {}).get(
                "evaluation_start"
            )
            != dates["evaluation_start"].isoformat()
        ):
            raise ValueError(
                "Existing training manifest conflicts with the configured frozen period split."
            )
        entry = training_manifest.get("datasets", {}).get("comp_chars_training")
        if (
            not entry
            or entry.get("source_id") != SOURCE_ID
            or entry.get("authorized_training_end")
            != dates["validation_start"].isoformat()
        ):
            raise ValueError(
                "Frozen training manifest has no authorized competition source."
            )
        parquet_path = config_path(entry["path"])
        card_path = config_path(entry["source_card_path"])
        card = SourceCard.model_validate_json(
            card_path.read_text(encoding="utf-8")
        )
        _validate_source_card(card)
        if _file_sha256(parquet_path) != entry.get("sha256"):
            raise ValueError(
                "Official competition panel bytes do not match the frozen source manifest."
            )
        if write_manifest:
            original_path = config_path(
                config.get("research", {}).get(
                    "source_manifest", str(DEFAULT_SOURCE_MANIFEST)
                )
            )
            original = json.loads(original_path.read_text(encoding="utf-8"))
            original_entry = original.get("datasets", {}).get(
                "comp_chars_design", {}
            )
            if original_entry.get("sha256") != entry.get(
                "sha256"
            ) or original_entry.get("source_snapshot_id") != entry.get(
                "source_snapshot_id"
            ):
                raise ValueError(
                    "Existing training manifest no longer matches the immutable official source."
                )
        return entry, card, parquet_path, state_dir, manifest_path
    if not write_manifest:
        raise ValueError(
            "Reserved validation requires the frozen training manifest."
        )

    source_manifest = config_path(
        config.get("research", {}).get(
            "source_manifest", str(DEFAULT_SOURCE_MANIFEST)
        )
    )
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    old_entry = manifest.get("datasets", {}).get("comp_chars_design")
    if (
        not old_entry
        or old_entry.get("approved") is not True
        or old_entry.get("source_id") != SOURCE_ID
    ):
        raise ValueError(
            "The immutable official competition source manifest is unavailable or changed."
        )
    entry = copy.deepcopy(old_entry)
    entry["partitions"] = copy.deepcopy(entry.get("partitions", {}))
    entry["partitions"]["design"] = {
        "start": dates["training_start"].isoformat(),
        "end": dates["training_end"].isoformat(),
    }
    entry["authorized_training_end"] = dates["validation_start"].isoformat()
    training_manifest = {
        **{key: value for key, value in manifest.items() if key != "datasets"},
        "manifest_id": "official_live_training_2015_2019",
        "approved_by": "user_authorized_live_research_training_extension",
        "approved_at": now.isoformat(),
        "datasets": {"comp_chars_training": entry},
        "research_periods": {
            key: dates[key].isoformat()
            for key in (
                "training_start",
                "training_end",
                "validation_start",
                "validation_end",
                "evaluation_start",
                "feature_start",
            )
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(training_manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    entry = training_manifest["datasets"]["comp_chars_training"]
    parquet_path = config_path(entry["path"])
    card_path = config_path(entry["source_card_path"])
    card = SourceCard.model_validate_json(card_path.read_text(encoding="utf-8"))
    _validate_source_card(card)
    if set(FEATURE_METRICS + (LABEL_METRIC,)) - set(entry.get("columns", {})):
        raise ValueError(
            "Official manifest is missing required granted metrics."
        )
    if _file_sha256(parquet_path) != entry.get("sha256"):
        raise ValueError(
            "Official competition panel bytes do not match the source manifest."
        )
    return entry, card, parquet_path, state_dir, manifest_path


def _select_names(frame, asof: dt.date, limit: int) -> list[str]:
    """Select a bounded universe using only coverage and pre-formation market cap."""
    rows = frame.copy()
    rows["_date"] = rows["eom"].map(_plain_date)
    # The selection snapshot is entirely before the first registered formation.
    asof_rows = rows[rows["_date"] <= asof]
    ranked = []
    for security, group in asof_rows.groupby("permno", dropna=True):
        coverage = len(set(group["_date"]))
        if coverage < 1:
            continue
        latest = group.sort_values("_date").iloc[-1]
        size = _finite(latest.get("me"))
        if size is None or size <= 0:
            continue
        ranked.append(
            (
                str(int(security))
                if float(security).is_integer()
                else str(security),
                coverage,
                size,
            )
        )
    # More pre-formation history first, then formation-date market cap. Outcomes
    # are not read by selection, and this point-in-time sample is frozen once.
    ranked.sort(key=lambda row: (-row[1], -row[2], row[0]))
    selected = [row[0] for row in ranked[:limit]]
    if len(selected) < 3:
        raise ValueError(
            "The authorized design panel has fewer than three eligible names."
        )
    return selected


def _observations(
    frame,
    security_ids: list[str],
    feature_start: dt.date,
    training_end: dt.date,
) -> list[dict]:
    selected = set(security_ids)
    rows = frame.copy()
    rows["_date"] = rows["eom"].map(_plain_date)
    rows = rows[
        (rows["_date"] >= feature_start)
        & (rows["_date"] <= training_end)
        & rows["permno"].map(
            lambda item: (
                (str(int(item)) if float(item).is_integer() else str(item))
                in selected
            )
        )
    ].sort_values(["_date", "permno"])
    result: list[dict] = []
    for row in rows.to_dict(orient="records"):
        day = _plain_date(row["_date"])
        when = _end_of_day(day)
        security = (
            str(int(row["permno"]))
            if float(row["permno"]).is_integer()
            else str(row["permno"])
        )
        for metric in FEATURE_METRICS:
            value = _finite(row.get(metric))
            if value is None:
                continue
            result.append(
                FeatureObservation(
                    observation_id=f"competition:{security}:{day.isoformat()}:{metric}",
                    security_id=security,
                    source_id=SOURCE_ID,
                    metric=metric,
                    period_end=when,
                    available_at=when,
                    version_known_at=when,
                    value=value,
                ).model_dump(mode="json")
            )

        # The source column is the following month's realized excess return.
        # Re-key its measurement to that future month and give it that future
        # knowledge time, so it cannot pass any same-month feature cutoff.
        next_end = _next_month_end(day)
        label = _finite(row.get(LABEL_METRIC))
        if label is not None and next_end <= training_end:
            known = _end_of_day(next_end)
            result.append(
                FeatureObservation(
                    observation_id=f"competition:{security}:{next_end.isoformat()}:{LABEL_METRIC}",
                    security_id=security,
                    source_id=SOURCE_ID,
                    metric=LABEL_METRIC,
                    period_end=known,
                    available_at=known,
                    version_known_at=known,
                    value=label,
                ).model_dump(mode="json")
            )
    result.sort(key=lambda item: item["observation_id"])
    if len(result) > MAX_OBSERVATIONS:
        raise ValueError("Normalized design observations exceed the host cap.")
    if any(
        dt.datetime.fromisoformat(item["period_end"]).date() > training_end
        for item in result
    ):
        raise ValueError(
            "Protected rows cannot be included in design observations."
        )
    return result


def _cached_paper_ids(
    database: Path,
    wanted: dict[str, tuple[tuple[str, str], ...]],
    literature_asof: dt.date,
) -> dict[str, list[str]]:
    """Return exact cached versions available by the configured research cutoff."""
    available = {}
    with sqlite3.connect(database) as conn:
        for paper_id, version_key, version_date, card_json in conn.execute(
            "SELECT paper_id, version_key, version_date, card_json FROM versions"
        ):
            card = json.loads(card_json)
            # A current DOI catalogue abstract is not a pinned historical
            # paper version. It stays discoverable, outside quoted evidence.
            if not card.get("arxiv_id") or not card.get("abstract"):
                continue
            first_public = card.get("first_public_date")
            available[paper_id, version_key] = (version_date, first_public)
    selected = {}
    for pod, refs in wanted.items():
        valid = []
        for paper_id, version_key in refs:
            dates = available.get((paper_id, version_key))
            if dates is None:
                continue
            version_date, first_public = dates
            version_day = dt.datetime.fromisoformat(
                version_date.replace("Z", "+00:00")
            ).date()
            public_day = dt.datetime.fromisoformat(
                first_public.replace("Z", "+00:00")
            ).date()
            if version_day > literature_asof or public_day > literature_asof:
                raise ValueError(
                    f"Selected literature version is post-cutoff for {pod}."
                )
            valid.append(
                f"{paper_id}{version_key}"
                if paper_id.startswith("arxiv:")
                else paper_id
            )
        selected[pod] = valid
    return selected


POD_RESEARCH = {
    "fundamentals": {
        "namespace": "pod_fundamentals",
        "question": "Do lagged earnings surprise and sales growth contain distinct information about the next three months of earnings innovation in this bounded design panel?",
        "reasoning_mode": "own_reasoning",
        "papers": (("arxiv:1611.02090", "v2"),),
        "metrics": ("niq_su", "sale_gr1", "ret_12_1"),
    },
    "events": {
        "namespace": "pod_events",
        "question": "Does lagged net issuance intensity relate to next-month net issuance intensity in the supplied monthly accounting proxy?",
        "reasoning_mode": "own_reasoning",
        "papers": (
            ("doi:10.1111/j.1540-6261.1990.tb02425.x", "pub"),
            ("doi:10.1016/0304-405x(86)90050-4", "pub"),
        ),
        "metrics": ("eqnetis_at", "be_me", "ret_12_1"),
    },
    "valuation": {
        "namespace": "pod_valuation",
        "question": "Does book-to-market contain information about the next month's supplied excess return after controlling for trailing momentum?",
        "reasoning_mode": "mechanism_transfer",
        "papers": (("arxiv:1204.5055", "v2"), ("arxiv:1702.07374", "v1")),
        "metrics": ("be_me", "ret_12_1", "ret_1_0"),
    },
    "dynamics": {
        "namespace": "pod_dynamics",
        "question": "Does twelve-to-one month momentum remain associated with next-month supplied excess returns after accounting for short-horizon reversal and trading volume?",
        "reasoning_mode": "mechanism_transfer",
        "papers": (
            ("arxiv:1702.07374", "v1"),
            ("doi:10.1093/rfs/7.1.179", "pub"),
        ),
        "metrics": ("ret_12_1", "ret_1_0", "dolvol_126d"),
    },
}


def prepare_inputs(config: dict, clock) -> dict:
    """Build the four POD grants and one hash-pinned normalized snapshot.

    The caller injects ``code_digest`` from its active research loop before
    registering each grant. The zero digest here is a schema-safe placeholder;
    it is never accepted by ``CompositionService`` until replaced by that loop.
    """
    now = require_aware(clock(), "clock")
    periods = _dates(config)
    protected_from = dt.datetime.combine(
        periods["validation_start"], dt.time.min, UTC
    )
    manifest_entry, source_card, parquet_path, state_dir, manifest_path = (
        _load_authority(config, periods, now)
    )
    frame = _read_design_frame(
        parquet_path, periods["feature_start"], periods["training_end"]
    )
    selection_asof = _month_end(periods["training_start"])
    security_ids = _select_names(
        frame, selection_asof, periods["universe_limit"]
    )
    observations = _observations(
        frame, security_ids, periods["feature_start"], periods["training_end"]
    )

    # Purge three months from the training formation calendar so every native
    # target resolves inside the training period. Validation/evaluation rows
    # never enter the loader or grant.
    formation_end = periods["training_end"]
    for _ in range(config["research"]["periods"]["purge_horizon_months"]):
        formation_end = formation_end.replace(day=1) - dt.timedelta(days=1)
    formation_end = _month_end(formation_end)
    training_formations = [
        _end_of_day(_month_end(dt.date(year, month, 1)))
        for year in range(
            periods["training_start"].year, formation_end.year + 1
        )
        for month in range(1, 13)
    ]
    training_formations = [
        day
        for day in training_formations
        if periods["training_start"] <= day.date() <= formation_end
    ]
    validation_formations = [
        _end_of_day(
            _month_end(dt.date(periods["validation_start"].year, month, 1))
        )
        for month in range(1, 13)
    ]
    card = source_card.model_dump(mode="json")
    metrics = [*FEATURE_METRICS, LABEL_METRIC]
    grant_rows = []
    for pod, spec in POD_RESEARCH.items():
        grant_rows.append(
            {
                "grant_id": f"competition_training_{pod}",
                "namespace": spec["namespace"],
                "code_digest": "0" * 64,
                "source_metrics": {SOURCE_ID: metrics},
                "source_cards": [card],
                "targets": {
                    "next_month_excess_return": {
                        "source_id": SOURCE_ID,
                        "metric": LABEL_METRIC,
                        "horizon_months": 1,
                        "resolve_lag_days": 0,
                    },
                    "next_quarter_earnings_proxy": {
                        "source_id": SOURCE_ID,
                        "metric": "niq_su",
                        "horizon_months": 3,
                        "resolve_lag_days": 0,
                    },
                    "next_month_issuance_proxy": {
                        "source_id": SOURCE_ID,
                        "metric": "eqnetis_at",
                        "horizon_months": 1,
                        "resolve_lag_days": 0,
                    },
                },
                "formation_dates": [
                    day.isoformat() for day in training_formations
                ],
                "security_ids": security_ids,
                "partition": "design",
                "protected_from": protected_from.isoformat(),
                "authorized_training_end": protected_from.isoformat(),
                "max_observations": MAX_OBSERVATIONS,
            }
        )

    state_dir.mkdir(parents=True, exist_ok=True)
    observation_path = state_dir / "competition_training_observations.json"
    payload = json.dumps(
        observations, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    observation_path.write_text(payload, encoding="utf-8")
    sha256 = hashlib.sha256(observation_path.read_bytes()).hexdigest()
    binding = {
        "kind": "file",
        "source_id": SOURCE_ID,
        "path": str(observation_path),
        "sha256": sha256,
        "observation_digest": observation_digest(observations),
    }

    wanted = {pod: tuple(spec["papers"]) for pod, spec in POD_RESEARCH.items()}
    paper_ids = _cached_paper_ids(
        config_path(config["application"]["papers_db"]),
        wanted,
        periods["training_end"],
    )
    proposals = []
    for pod, spec in POD_RESEARCH.items():
        grant_id = f"competition_training_{pod}"
        proposals.append(
            {
                "namespace": spec["namespace"],
                "question": spec["question"],
                "grant_id": grant_id,
                "paper_ids": paper_ids[pod],
                "reasoning_mode": spec["reasoning_mode"],
                "source_metadata": [
                    {
                        "source_id": SOURCE_ID,
                        "metric": metric,
                        "dataset_id": str(manifest_entry["source_snapshot_id"]),
                        "description": (
                            "Net issuance intensity from an accounting characteristic; a monthly proxy, not an offering date or transaction calendar."
                            if metric == "eqnetis_at"
                            else "Selected pre-2020 literature studies social-media sentiment around earnings announcements; sentiment and announcement dates are absent here, so this is a proxy-based original test."
                            if pod == "fundamentals"
                            else "Competition supplied monthly characteristic. Values are bounded research proxies; they do not establish a trading strategy or historical publication vintage."
                            if metric != LABEL_METRIC
                            else "Next-month excess-return label shifted to its realization month; never an as-of feature."
                        ),
                        "frequency": "monthly",
                        "history_start": periods["feature_start"].isoformat(),
                        "history_end": periods["training_end"].isoformat(),
                        "availability": (
                            "Official manifest attests characteristics known at their month-end eom; "
                            "forward return is available only at the following month end."
                        ),
                    }
                    for metric in spec["metrics"]
                ],
            }
        )

    return {
        "measurement_grants": grant_rows,
        "observation_sources": [binding],
        "proposals": proposals,
        "snapshot": {
            "manifest_id": "official_live_training_2015_2019",
            "source_snapshot_id": manifest_entry.get("source_snapshot_id"),
            "selected_security_ids": security_ids,
            "selection_asof": selection_asof.isoformat(),
            "formation_dates": [day.isoformat() for day in training_formations],
            "observation_count": len(observations),
            "universe_limit": periods["universe_limit"],
            "sample_scope": f"bounded {len(security_ids)}-name research pilot; not the full mandate universe",
            "sha256": sha256,
            "created_at": now.isoformat(),
            "split": {
                "training_start": periods["training_start"].isoformat(),
                "training_end": periods["training_end"].isoformat(),
                "training_formation_end_after_purge": formation_end.isoformat(),
                "training_months": len(training_formations),
                "validation_start": periods["validation_start"].isoformat(),
                "validation_end": periods["validation_end"].isoformat(),
                "validation_months": 12,
                "validation_dates": [
                    day.isoformat() for day in validation_formations
                ],
                "validation_target_maturity_within_design": {
                    "next_month_excess_return": _month_end(
                        periods["validation_end"] - dt.timedelta(days=31)
                    ).isoformat(),
                    "next_quarter_earnings_proxy": _month_end(
                        periods["validation_end"] - dt.timedelta(days=92)
                    ).isoformat(),
                    "next_month_issuance_proxy": _month_end(
                        periods["validation_end"] - dt.timedelta(days=31)
                    ).isoformat(),
                },
                "protected_evaluation": {
                    "start": dt.datetime.combine(
                        periods["evaluation_start"], dt.time.min, UTC
                    ).isoformat(),
                    "accessed": False,
                },
                "note": (
                    "The training snapshot includes observations only through 2019-12-31; "
                    "training formations are purged three months so native targets mature by that date. "
                    "The 2020 validation calendar is recorded but is not loaded by this call. "
                    "The 2021+ evaluation partition is untouched."
                ),
            },
        },
        "manifest_path": str(manifest_path),
    }


def prepare_validation_inputs(
    config: dict,
    security_ids: list[str],
    clock,
    *,
    reservation_id: str,
    freeze_digest: str,
) -> dict:
    """Normalize 2020 validation rows after the trusted parent reserves access.

    The reservation and freeze digest are attestations from the parent evaluation
    workflow. They must be supplied as explicit keyword arguments; model output
    must never be forwarded as either value. This helper does not open the 2021+
    evaluation rows.
    """
    if not isinstance(reservation_id, str) or not reservation_id.strip():
        raise ValueError("A persisted validation reservation ID is required.")
    if (
        not isinstance(freeze_digest, str)
        or len(freeze_digest) != 64
        or any(ch not in "0123456789abcdef" for ch in freeze_digest)
    ):
        raise ValueError("A frozen-training SHA-256 digest is required.")
    if (
        not security_ids
        or len(security_ids) > MAX_NAMES
        or len(set(security_ids)) != len(security_ids)
    ):
        raise ValueError(
            "Validation requires the frozen, unique training security sample."
        )

    now = require_aware(clock(), "clock")
    periods = _dates(config)
    validation_cutoff = _end_of_day(periods["validation_end"])
    if now < validation_cutoff:
        raise ValueError(
            "Validation outcomes are not mature at the supplied clock."
        )
    _, _, parquet_path, state_dir, _ = _load_authority(
        config, periods, now, write_manifest=False
    )

    # The 2019 block provides the lagged feature history for January 2020.
    # 2020 is the only outcome period loaded here; validation targets that would
    # mature in 2021 are dropped by _observations.
    warmup_start = dt.date(periods["validation_start"].year - 1, 1, 1)
    frame = _read_design_frame(
        parquet_path, warmup_start, periods["validation_end"]
    )
    observations = _observations(
        frame,
        security_ids,
        warmup_start,
        periods["validation_end"],
    )
    path = state_dir / (
        "competition_validation_"
        + hashlib.sha256(reservation_id.encode()).hexdigest()[:16]
        + ".json"
    )
    payload = json.dumps(
        observations, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    path.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "observation_source": {
            "kind": "file",
            "source_id": SOURCE_ID,
            "path": str(path),
            "sha256": digest,
            "observation_digest": observation_digest(observations),
        },
        "snapshot": {
            "reservation_id": reservation_id,
            "freeze_digest": freeze_digest,
            "security_ids": list(security_ids),
            "warmup_start": warmup_start.isoformat(),
            "validation_start": periods["validation_start"].isoformat(),
            "validation_end": periods["validation_end"].isoformat(),
            "maximum_target_maturity": periods["validation_end"].isoformat(),
            "protected_evaluation_start": periods[
                "evaluation_start"
            ].isoformat(),
            "evaluation_accessed": False,
            "observation_count": len(observations),
            "sha256": digest,
            "created_at": now.isoformat(),
        },
    }
