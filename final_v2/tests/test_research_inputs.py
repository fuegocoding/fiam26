"""Regression checks for the bounded live-research input preparation API."""

import datetime as dt
import hashlib
import json
from pathlib import Path

import pytest

from pod_platform.application import research_inputs as inputs
from pod_platform.research.builtins.manifest import load_manifest
from pod_platform.research.composition import MeasurementGrant
from pod_platform.research.features import FeatureObservation

pd = pytest.importorskip("pandas")


UTC = dt.timezone.utc
PROJECT = Path(__file__).resolve().parents[2]


def _panel(start, end, names=55):
    rows = []
    day = dt.date(start.year, start.month, 1)
    while day <= end:
        month_end = inputs._month_end(day)
        for security in range(1, names + 1):
            row = {
                "permno": security,
                "eom": pd.Timestamp(month_end),
                "me": security * 1_000_000.0,
                "ret_exc_lead1m": security / 10000.0,
            }
            row.update(
                {metric: security / 100.0 for metric in inputs.FEATURE_METRICS}
            )
            rows.append(row)
        if day.month == 12:
            day = dt.date(day.year + 1, 1, 1)
        else:
            day = dt.date(day.year, day.month + 1, 1)
    return pd.DataFrame(rows)


def _config(tmp_path):
    return {
        "application": {
            "manifest": str(tmp_path / "state" / "competition_training.json"),
            "state_dir": str(tmp_path / "state"),
            "dataset_id": "comp_chars_training",
            "papers_db": str(
                PROJECT / "final_v2/.pod_state/year_trial_2025/papers.sqlite"
            ),
        },
        "research": {
            "source_manifest": str(
                PROJECT / "final_v2/.pod_state/datasets/competition_design.json"
            ),
            "periods": {
                "training_start": "2015-01-01",
                "training_end": "2019-12-31",
                "validation_start": "2020-01-01",
                "validation_end": "2020-12-31",
                "evaluation_start": "2021-01-01",
                "purge_horizon_months": 3,
            },
        },
    }


def test_parquet_reader_does_not_project_labels_across_boundary(
    tmp_path, monkeypatch
):
    nov = pd.Timestamp("2019-11-30")
    dec = pd.Timestamp("2019-12-31")
    features = pd.DataFrame(
        [
            {
                "permno": 7,
                "eom": nov,
                "me": 70.0,
                **{m: 1.0 for m in inputs.FEATURE_METRICS},
            },
            {
                "permno": 7,
                "eom": dec,
                "me": 71.0,
                **{m: 2.0 for m in inputs.FEATURE_METRICS},
            },
        ]
    )
    labels = pd.DataFrame(
        [{"permno": 7, "eom": nov, inputs.LABEL_METRIC: 0.125}]
    )
    calls = []

    def read_parquet(path, *, columns, engine, filters):
        calls.append({"columns": columns, "filters": filters, "engine": engine})
        return (
            features.copy()
            if inputs.LABEL_METRIC not in columns
            else labels.copy()
        )

    monkeypatch.setattr(inputs.pd, "read_parquet", read_parquet)
    result = inputs._read_design_frame(
        tmp_path / "panel.parquet", dt.date(2019, 1, 1), dt.date(2019, 12, 31)
    )

    assert len(calls) == 2
    assert inputs.LABEL_METRIC not in calls[0]["columns"]
    assert calls[0]["columns"] == [
        "permno",
        "eom",
        "me",
        *inputs.FEATURE_METRICS,
    ]
    assert calls[1]["columns"] == ["permno", "eom", inputs.LABEL_METRIC]
    assert ("eom", "<=", pd.Timestamp("2019-12-31")) in calls[0]["filters"]
    assert ("eom", "<=", pd.Timestamp("2019-11-30")) in calls[1]["filters"]
    assert result.loc[result["eom"] == nov, inputs.LABEL_METRIC].iloc[
        0
    ] == pytest.approx(0.125)
    assert pd.isna(
        result.loc[result["eom"] == dec, inputs.LABEL_METRIC].iloc[0]
    )
    assert result.loc[result["eom"] == dec, "niq_su"].iloc[0] == pytest.approx(
        2.0
    )


def test_training_inputs_pin_real_design_data_split_and_four_distinct_pods(
    tmp_path, monkeypatch
):
    frame = _panel(dt.date(2015, 1, 1), dt.date(2019, 12, 31))
    monkeypatch.setattr(
        inputs, "_read_design_frame", lambda path, start, end: frame
    )
    result = inputs.prepare_inputs(
        _config(tmp_path), lambda: dt.datetime(2026, 10, 25, tzinfo=UTC)
    )

    grants = result["measurement_grants"]
    tasks = result["proposals"]
    assert len(grants) == len(tasks) == 4
    assert {grant["namespace"] for grant in grants} == {
        "pod_fundamentals",
        "pod_events",
        "pod_valuation",
        "pod_dynamics",
    }
    assert len({task["question"] for task in tasks}) == 4
    assert all(
        pid.startswith("arxiv:") for task in tasks for pid in task["paper_ids"]
    )
    events = next(t for t in tasks if t["namespace"] == "pod_events")
    assert events["reasoning_mode"] == "own_reasoning"
    for grant in grants:
        assert len(grant["security_ids"]) == 50
        assert len(grant["formation_dates"]) == 57
        assert grant["formation_dates"][0].startswith("2015-01-31")
        assert grant["formation_dates"][-1].startswith("2019-09-30")
        assert grant["authorized_training_end"] == "2020-01-01T00:00:00+00:00"
        assert grant["protected_from"] == grant["authorized_training_end"]

    binding = result["observation_sources"][0]
    body = Path(binding["path"]).read_bytes()
    assert hashlib.sha256(body).hexdigest() == binding["sha256"]
    observations = [
        FeatureObservation.model_validate(item) for item in json.loads(body)
    ]
    assert observations
    assert all(
        item.period_end < dt.datetime(2020, 1, 1, tzinfo=UTC)
        for item in observations
    )
    # January's next-month label is timestamped in February, so it is not visible
    # at a January formation cutoff.
    shifted = next(
        item
        for item in observations
        if item.security_id == grants[0]["security_ids"][0]
        and item.metric == inputs.LABEL_METRIC
        and item.period_end.date() == dt.date(2015, 2, 28)
    )
    assert (
        shifted.available_at == shifted.version_known_at == shifted.period_end
    )
    assert shifted.value == pytest.approx(int(shifted.security_id) / 10000.0)
    assert result["snapshot"]["split"]["validation_start"] == "2020-01-01"
    assert result["snapshot"]["split"]["protected_evaluation"][
        "start"
    ].startswith("2021-01-01")
    assert (
        result["snapshot"]["split"]["protected_evaluation"]["accessed"] is False
    )
    manifest = json.loads(Path(result["manifest_path"]).read_text())
    manifest_bytes = Path(result["manifest_path"]).read_bytes()
    assert manifest["datasets"]["comp_chars_training"]["partitions"][
        "design"
    ] == {"start": "2015-01-01", "end": "2019-12-31"}
    assert (
        manifest["datasets"]["comp_chars_training"]["authorized_training_end"]
        == "2020-01-01"
    )
    loaded = load_manifest(result["manifest_path"], allow_research_only=True)
    assert loaded.dataset("comp_chars_training").design_window == (
        dt.date(2015, 1, 1),
        dt.date(2019, 12, 31),
    )
    for grant in grants:
        MeasurementGrant.model_validate(grant)

    inputs.prepare_inputs(
        _config(tmp_path), lambda: dt.datetime(2026, 10, 26, tzinfo=UTC)
    )
    assert Path(result["manifest_path"]).read_bytes() == manifest_bytes


def test_validation_is_a_separate_reserved_call_and_does_not_read_2021(
    tmp_path, monkeypatch
):
    config = _config(tmp_path)
    training_frame = _panel(dt.date(2015, 1, 1), dt.date(2019, 12, 31))
    frame = _panel(dt.date(2019, 1, 1), dt.date(2020, 12, 31))
    reads = []

    monkeypatch.setattr(
        inputs, "_read_design_frame", lambda path, start, end: training_frame
    )
    training = inputs.prepare_inputs(
        config, lambda: dt.datetime(2026, 10, 25, tzinfo=UTC)
    )

    def read(path, start, end):
        reads.append((start, end))
        return frame

    monkeypatch.setattr(inputs, "_read_design_frame", read)
    ids = training["measurement_grants"][0]["security_ids"]
    freeze = "a" * 64
    reservation = inputs.prepare_validation_inputs(
        config,
        ids,
        lambda: dt.datetime(2026, 10, 25, tzinfo=UTC),
        reservation_id="validation_reservation_01",
        freeze_digest=freeze,
    )
    assert reads == [(dt.date(2019, 1, 1), dt.date(2020, 12, 31))]
    snapshot = reservation["snapshot"]
    assert snapshot["reservation_id"] == "validation_reservation_01"
    assert snapshot["freeze_digest"] == freeze
    assert snapshot["evaluation_accessed"] is False
    raw = Path(reservation["observation_source"]["path"]).read_bytes()
    assert (
        hashlib.sha256(raw).hexdigest()
        == reservation["observation_source"]["sha256"]
    )
    observations = [
        FeatureObservation.model_validate(row) for row in json.loads(raw)
    ]
    assert any(item.period_end.year == 2019 for item in observations)  # warmup
    assert any(item.period_end.year == 2020 for item in observations)
    assert all(item.period_end.year <= 2020 for item in observations)

    with pytest.raises(ValueError, match="reservation ID"):
        inputs.prepare_validation_inputs(
            config,
            ids,
            lambda: dt.datetime(2026, 10, 25, tzinfo=UTC),
            reservation_id="",
            freeze_digest=freeze,
        )


def test_training_configuration_cannot_silently_change_holdout_boundaries():
    config = {
        "application": {"dataset_id": "comp_chars_training"},
        "research": {
            "periods": {
                "training_start": "2015-01-01",
                "training_end": "2019-12-31",
                "validation_start": "2020-01-01",
                "validation_end": "2020-12-31",
                "evaluation_start": "2021-01-01",
                "purge_horizon_months": 3,
            }
        },
    }
    assert inputs._dates(config)["evaluation_start"] == dt.date(2021, 1, 1)
    config["research"]["periods"]["evaluation_start"] = "2020-12-31"
    with pytest.raises(ValueError, match="explicitly authorized"):
        inputs._dates(config)
