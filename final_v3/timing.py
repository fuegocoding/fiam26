"""Checks prediction dates, feature safety and saved input manifests."""

import hashlib
import json
import pathlib

import numpy as np
import pandas as pd

import config

OUTCOMES = {"ret", "ret_exc", "ret_exc_lead1m", "stock_exret", "target", "anndats", "available_date"}
CACHE_FILE = "timing_manifest.json"
PREDICTION_POLICY = "official_data_available_clock_v2"


def task_masks(
    frame: pd.DataFrame, year: int, tasks: tuple[str, ...], first_target: pd.Timestamp
) -> dict[str, np.ndarray]:
    """Training labels must be realized within their fold's cutoff, including EPS."""
    target = frame.eom + pd.offsets.MonthEnd(1)
    train_end = pd.Timestamp(year=year - 3, month=12, day=31)
    validation_end = pd.Timestamp(year=year - 1, month=12, day=31)
    windows = {
        "train": (target.between(first_target, train_end), train_end),
        "validation": (
            target.between(
                pd.Timestamp(year=year - 2, month=1, day=31), validation_end
            ),
            validation_end,
        ),
    }
    observed = np.isfinite(frame[[f"{task}_label" for task in tasks]].to_numpy(float))
    masks = {}
    for name, (rows, cutoff) in windows.items():
        mask = observed & np.asarray(rows)[:, None]
        reported = frame.anndats.notna() & frame.available_date.le(cutoff)
        mask[:, tasks.index("eps")] &= np.asarray(reported)
        masks[name] = mask
    return masks


def validate_features(features: list[str]) -> None:
    """Reject realized outcomes, forward labels and duplicate feature names."""
    forbidden = [
        name
        for name in features
        if name.lower() in OUTCOMES
        or "lead" in name.lower()
        or name.lower().startswith(("target_", "next_"))
        or name.lower().endswith("_label")
    ]
    if forbidden or len(set(features)) != len(features):
        raise ValueError(f"Unsafe or duplicate predictor columns: {forbidden}")


def validate_predictions(predictions: pd.DataFrame) -> None:
    """No outcomes, exact next-month alignment and no future identity evidence."""
    validate_features(list(predictions.columns))
    if (
        "timing_policy" not in predictions
        or not predictions.timing_policy.eq(PREDICTION_POLICY).all()
    ):
        raise ValueError("Unaudited predictions. Rebuild without --reuse.")
    formation = pd.to_datetime(predictions.formation_date)
    target = pd.to_datetime(predictions.return_month)
    if (formation.isna() | target.isna()).any():
        raise ValueError("Missing formation or return date")
    if not target.dt.to_period("M").eq(formation.dt.to_period("M") + 1).all():
        raise ValueError("Predictions must target the month after formation")
    if target.dt.to_period("M").gt(config.LAST_TARGET.to_period("M")).any():
        raise ValueError("Prediction target exceeds configured evaluation window")
    reference = pd.to_datetime(predictions.ticker_name_reference_date)
    labelled = predictions.ticker.notna() & predictions.company_name.notna()
    if (labelled & (reference.isna() | reference.gt(formation))).any():
        raise ValueError("Identity evidence missing or dated after formation")
    if predictions.duplicated(["permno", "return_month"]).any():
        raise ValueError("Duplicate prediction keys")


def prepare_cache(work_dir: pathlib.Path, data_dir: pathlib.Path, device: str) -> None:
    """Refuse pre-audit caches or artifacts built with different code or inputs.

    Source files are hashed; large data files are identified by path, size and
    mtime. The manifest is written before builds so interrupted runs can resume.
    """
    source = hashlib.sha256()
    for path in sorted(config.FINAL.rglob("*.py")):
        if {"output", "submission"} & set(path.relative_to(config.FINAL).parts):
            continue
        if path == config.FINAL / "MAIN.py":
            continue  # Generated single-file copy; the original modules are hashed.
        source.update(str(path.relative_to(config.FINAL)).encode())
        source.update(path.read_bytes())
    inputs = [
        data_dir / name
        for name in (config.PANEL_FILE, config.FACTOR_LIST_FILE, config.FILINGS_FILE)
    ]
    inputs.append(data_dir / "dsf_mini_from_2014.csv")
    inputs.append(config.FINAL.parent / "external_data" / "external.parquet")
    if config.REPORTING_FILE not in inputs:
        inputs.append(config.REPORTING_FILE)
    manifest = {
        "source_sha256": source.hexdigest(),
        "inputs": [
            {
                "path": str(path.resolve()),
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for path in inputs
        ],
        "device": device,
        "years": list(config.TEST_YEARS),
        "last_target": str(config.LAST_TARGET),
        "pretrained_forecasts": False,
    }
    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / CACHE_FILE
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError("Cached artifacts are stale. Rebuild without --reuse.")
    elif any(work_dir.iterdir()):
        raise ValueError("Unaudited cached artifacts. Rebuild without --reuse.")
    else:
        path.write_text(json.dumps(manifest, indent=2) + "\n")
