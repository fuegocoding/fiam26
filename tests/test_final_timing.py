"""Causal regression checks for the current, self-contained final pipeline."""

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "final"))

import borrow
import combiner
import config
import evaluate
import identity
import optimizer
import risk
import timing
from graph import builder, dataset
from sleeves import base, ibes
from sleeves.e7 import panel as e7
from sleeves.filing_events import edgar
from sleeves.filing_events import model as filing_model
from sleeves.gap_micro_best import valuation
from sleeves.xgb_rank import model as rank_model


@pytest.mark.parametrize("model", [rank_model, filing_model])
@pytest.mark.parametrize("year", config.TEST_YEARS)
def test_annual_splits_do_not_read_test_outcomes(model, year):
    months = pd.date_range("2015-02-28", "2026-08-31", freq="ME")
    frame = pd.DataFrame(
        {"target_month": months, "target": np.arange(len(months), dtype=float)}
    )
    train, validation, test = model._split(frame, year)
    assert train.target_month.max() <= pd.Timestamp(f"{year - 3}-12-31")
    assert validation.target_month.max() <= pd.Timestamp(f"{year - 1}-12-31")
    changed = frame.copy()
    changed.loc[changed.target_month.dt.year >= year, "target"] = np.nan
    new_train, new_validation, new_test = model._split(changed, year)
    pd.testing.assert_frame_equal(train, new_train)
    pd.testing.assert_frame_equal(validation, new_validation)
    assert new_test.index.equals(test.index)  # unobserved returns remain eligible


@pytest.mark.parametrize("model", [rank_model, filing_model])
def test_feature_ranks_ignore_later_months(model):
    frame = pd.DataFrame(
        {
            "eom": pd.to_datetime(["2020-12-31"] * 3 + ["2021-01-31"] * 3),
            "signal": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
        }
    )
    expected = model._rank_transform(frame.iloc[:3], ["signal"])
    pd.testing.assert_frame_equal(
        model._rank_transform(frame, ["signal"]).iloc[:3], expected
    )


def test_graph_eps_masks_require_actual_announcement_before_fold_cutoff():
    frame = pd.DataFrame(
        {
            "eom": pd.to_datetime(
                ["2018-11-30", "2018-11-30", "2020-11-30", "2020-11-30", "2020-12-31"]
            ),
            "anndats": pd.to_datetime(
                ["2018-12-31", "2019-01-01", "2020-12-31", "2021-01-01", "2021-02-01"]
            ),
            "return_label": 1.0,
            "eps_label": 1.0,
            "risk_label": 1.0,
        }
    )
    masks = timing.task_masks(frame, 2021, dataset.TASKS, pd.Timestamp("2015-02-28"))
    assert masks["train"][:, 1].tolist() == [True, False, False, False, False]
    assert masks["validation"][:, 1].tolist() == [False, False, True, False, False]
    assert not masks["train"][-1].any() and not masks["validation"][-1].any()


def test_past_eps_features_ignore_next_report():
    panel = pd.DataFrame({"permno": [1], "eom": pd.to_datetime(["2020-12-31"])})
    events = pd.DataFrame(
        {
            "permno": [1, 1, 1, 1],
            "anndats": pd.to_datetime(
                ["2020-09-01", "2020-11-01", "2020-12-31", "2021-02-01"]
            ),
            "next_suescore": [1.0, 2.0, 3.0, 4.0],
        }
    )
    expected = dataset._past_news(panel, events)
    events.loc[events.anndats >= panel.eom.iloc[0], "next_suescore"] = -999.0
    pd.testing.assert_frame_equal(dataset._past_news(panel, events), expected)
    assert expected.eps_last.iloc[0] == 2.0


def test_later_eps_revision_cannot_replace_first_report(monkeypatch):
    raw = pd.DataFrame(
        {
            "TICKER": ["A", "A"],
            "PYEAR": [2020, 2020],
            "PMON": [12, 12],
            "USFIRM": [1, 1],
            "MEASURE": ["EPS", "EPS"],
            "FISCALP": ["QTR", "QTR"],
            "anndats": ["2021-02-01", "2021-04-01"],
            "actual": [1.0, 999.0],
            "surpmean": [0.5, 0.5],
            "suescore": [2.0, 999.0],
        }
    )
    monkeypatch.setattr(ibes.external, "read", lambda _: raw.copy())
    result = ibes.load_surprises()
    assert len(result) == 1 and result.suescore.iloc[0] == 2.0
    raw.loc[1, "anndats"] = "2021-02-01"
    with pytest.raises(ValueError, match="revision timestamps"):
        ibes.load_surprises()


@pytest.mark.parametrize("source", ["consensus", "price"])
def test_conflicting_snapshot_duplicates_fail_closed(source):
    frame = pd.DataFrame(
        {"TICKER": ["A", "A"], "STATPERS": ["2021-01-01"] * 2, "value": [1.0, 2.0]}
    )
    with pytest.raises(ValueError, match="revision timestamps"):
        ibes._unique_records(frame, ["TICKER", "STATPERS"], source)


def test_link_end_dates_are_never_extended_using_sample_maximum(monkeypatch):
    raw = pd.DataFrame(
        {
            "TICKER": ["A"],
            "PERMNO": [1],
            "SCORE": [0],
            "sdate": ["2020-01-01"],
            "edate": ["2020-12-31"],
        }
    )
    monkeypatch.setattr(ibes.external, "read", lambda _: raw.copy())
    grid = pd.DataFrame({"permno": [1], "eom": pd.to_datetime(["2021-01-31"])})
    assert ibes.attach_ticker(grid, ibes.load_link()).empty


def test_e7_following_quarter_is_selected_only_from_known_snapshots(
    tmp_path, monkeypatch
):
    chars = pd.DataFrame(
        {
            "permno": [1],
            "eom": pd.to_datetime(["2021-01-31"]),
            "sector": ["45"],
            "size_grp": ["large"],
            "tradable": [True],
        }
    )
    state = pd.DataFrame(
        {
            "permno": [1],
            "eom": pd.to_datetime(["2020-11-30"]),
            "TICKER": ["A"],
            "target_fpe": pd.to_datetime(["2020-09-30"]),
            "target_ann": pd.to_datetime(["2020-12-10"]),
            "target_surprise_p": [0.1],
        }
    )
    consensus = pd.DataFrame(
        {
            "TICKER": ["A"],
            "STATPERS": pd.to_datetime(["2021-01-15"]),
            "FPEDATS": pd.to_datetime(["2020-12-31"]),
            "NUMUP": [2],
            "NUMDOWN": [1],
            "NUMEST": [10],
        }
    )
    monkeypatch.setattr(ibes, "build_expectation_state", lambda _: state.copy())
    monkeypatch.setattr(ibes, "load_consensus", lambda: consensus.copy())
    context = base.Context(chars, tmp_path, tmp_path / "before", "cpu")
    before = pd.read_parquet(e7.build(context))
    # A later snapshot introduces an earlier following fiscal quarter.
    later = consensus.copy()
    later["STATPERS"] = pd.Timestamp("2021-02-15")
    later["FPEDATS"] = pd.Timestamp("2020-11-30")
    consensus = pd.concat([consensus, later], ignore_index=True)
    after = pd.read_parquet(
        e7.build(base.Context(chars, tmp_path, tmp_path / "after", "cpu"))
    )
    assert len(before) == 1
    pd.testing.assert_frame_equal(before, after)


def _write_identity_panel(path, dates, statuses, references):
    frame = pd.DataFrame(
        {
            "permno": [1] * len(dates),
            "eom": pd.to_datetime(dates),
            "ticker": [
                "A" if status != identity.UNVERIFIED else None for status in statuses
            ],
            "company_name": [
                "Alpha" if status != identity.UNVERIFIED else None
                for status in statuses
            ],
            "ticker_name_status": statuses,
            "ticker_name_reference_date": pd.to_datetime(references),
            "ticker_name_source": "test",
        }
    )
    frame.to_parquet(path)


def test_identity_gap_fill_ignores_later_crsp_coverage_and_future_evidence(tmp_path):
    path = tmp_path / "panel.parquet"
    _write_identity_panel(
        path,
        ["2020-12-31", "2021-01-31"],
        [identity.CRSP_STATUS, identity.UNVERIFIED],
        ["2020-12-31", None],
    )
    expected = identity._carried_labels(path)
    _write_identity_panel(
        path,
        ["2020-12-31", "2021-01-31", "2021-02-28", "2022-12-31"],
        [
            identity.CRSP_STATUS,
            identity.UNVERIFIED,
            identity.CRSP_STATUS,
            identity.CRSP_STATUS,
        ],
        ["2020-12-31", None, "2021-03-01", "2022-12-31"],
    )
    after = identity._carried_labels(path)
    pd.testing.assert_frame_equal(
        after.loc[after.eom <= pd.Timestamp("2021-01-31")], expected
    )
    assert after.loc[after.eom == pd.Timestamp("2021-02-28"), "label_date"].iloc[
        0
    ] == pd.Timestamp("2020-12-31")


def _edgar_inputs(tmp_path):
    panel_path, filing_path = tmp_path / "panel.parquet", tmp_path / "filings.parquet"
    panel = pd.DataFrame(
        {
            "permno": [1],
            "permco": [10],
            "company_name": ["Alpha Mining"],
            "eom": pd.to_datetime(["2020-12-31"]),
            "ticker_name_reference_date": pd.to_datetime(["2020-12-31"]),
        }
    )
    linked = pd.DataFrame(
        {
            "permno": [1],
            "permco": [10],
            "cik": [101],
            "filing_date": pd.to_datetime(["2020-12-01"]),
        }
    )
    index = pd.DataFrame(
        {
            "cik": [101, 202],
            "form_type": ["10-K", "S-3"],
            "company_name": ["Alpha Mining", "Zebra Widgets"],
            "date_filed": pd.to_datetime(["2020-12-01", "2020-12-10"]),
            "accession": ["a", "b"],
        }
    )
    panel.to_parquet(panel_path)
    linked.to_parquet(filing_path)
    return panel_path, filing_path, panel, linked, index


def test_edgar_future_names_and_cik_links_cannot_backfill_old_features(tmp_path):
    panel_path, filing_path, panel, linked, index = _edgar_inputs(tmp_path)
    cutoff = pd.Timestamp("2020-12-31")
    expected = edgar._crosswalk(panel_path, filing_path, index, cutoff)
    later_panel = panel.assign(
        permno=2,
        permco=20,
        company_name="Zebra Widgets",
        eom=pd.Timestamp("2021-02-28"),
        ticker_name_reference_date=pd.Timestamp("2021-02-28"),
    )
    later_link = linked.assign(
        permno=2, permco=20, cik=202, filing_date=pd.Timestamp("2021-02-01")
    )
    pd.concat([panel, later_panel]).to_parquet(panel_path)
    pd.concat([linked, later_link]).to_parquet(filing_path)
    later_index = index.iloc[[1]].assign(
        form_type="10-K", date_filed=pd.Timestamp("2021-02-01"), accession="c"
    )
    after = edgar._crosswalk(
        panel_path, filing_path, pd.concat([index, later_index]), cutoff
    )
    pd.testing.assert_frame_equal(after, expected)
    assert 202 not in set(after.cik)
    assert 202 in set(
        edgar._crosswalk(
            panel_path,
            filing_path,
            pd.concat([index, later_index]),
            pd.Timestamp("2021-02-28"),
        ).cik
    )


def test_edgar_build_counts_only_events_known_before_return_month(
    tmp_path, monkeypatch
):
    panel_path, filing_path, _, _, index = _edgar_inputs(tmp_path)
    event = index.iloc[[0]].assign(form_type="S-3", accession="event")
    future = event.assign(date_filed=pd.Timestamp("2021-01-15"), accession="future")
    raw = pd.concat([index, event, future], ignore_index=True)
    monkeypatch.setattr(edgar.external, "read", lambda _: raw.copy())
    monkeypatch.setattr(edgar, "_FIRST_FEATURE_MONTH", pd.Period("2021-01", "M"))
    monkeypatch.setattr(config, "LAST_TARGET", pd.Timestamp("2021-02-28"))
    result = pd.read_parquet(edgar.build(panel_path, filing_path, tmp_path / "events"))
    assert (
        result.loc[result.return_month == pd.Timestamp("2021-01-31"), "shelf_1m"].iloc[
            0
        ]
        == 1
    )
    assert (
        result.loc[result.return_month == pd.Timestamp("2021-02-28"), "shelf_3m"].iloc[
            0
        ]
        == 2
    )
    assert set(edgar.FEATURE_COLUMNS).issubset(result.columns)


def test_finra_publication_lag_covers_holidays_and_explicit_dates():
    raw = pd.DataFrame({"settlement": pd.to_datetime(["2021-02-12", "2026-06-15"])})
    dates = borrow.public_dates(raw)
    assert dates.iloc[0] == pd.Timestamp("2021-03-01")  # excludes Presidents' Day
    assert dates.iloc[1] >= pd.Timestamp("2026-06-25")  # actual FINRA publication
    raw["publication_date"] = pd.to_datetime(["2021-02-24", "2026-06-25"])
    pd.testing.assert_series_equal(borrow.public_dates(raw), raw.publication_date)


def test_opening_turnover_cannot_depend_on_current_month_rate():
    months = pd.period_range("2021-01", periods=2, freq="M")
    books = {month: {1: 0.5, 2: -0.5} for month in months}
    universe = pd.DataFrame(
        [
            {
                "return_month": month,
                "permno": p,
                "ret_exc_lead1m": r,
                "size_grp": "large",
                "sector": "45",
                "beta_60m": 1.0,
            }
            for month in months
            for p, r in [(1, 0.02), (2, -0.01)]
        ]
    )
    market = pd.DataFrame({"rf": [0.001, 0.002], "spx": [0.01, 0.01]}, index=months)
    before = evaluate.evaluate(books, universe, market)
    market.loc[months[-1], "rf"] = 0.5
    after = evaluate.evaluate(books, universe, market)
    np.testing.assert_array_equal(before.traded, after.traded)
    np.testing.assert_array_equal(before.cost, after.cost)


def test_risk_weights_ignore_future_returns_and_scores():
    months = pd.period_range("2015-01", "2021-02", freq="M")
    panel = pd.DataFrame(
        [
            {
                "permno": p,
                "eom": month,
                "gics": "451010",
                "size_grp": "large",
                "ret_exc": 0.02 * np.sin(i + p),
                "rmax1_21d": 0.03 + 0.01 * np.cos(i + p),
            }
            for i, month in enumerate(months)
            for p in [1, 2]
        ]
    )
    scores = pd.DataFrame(
        [
            {"permno": p, "return_month": month, "final_score": float(p), "prc": 10.0}
            for month in months
            for p in [1, 2]
        ]
    )
    month = pd.Period("2021-01", "M")
    expected = risk.protect({1: 0.5, 2: -0.5}, month, panel, scores)
    panel.loc[panel.eom >= month, ["ret_exc", "rmax1_21d"]] = 999.0
    scores.loc[scores.return_month >= month, "final_score"] = -999.0
    after = risk.protect({1: 0.5, 2: -0.5}, month, panel, scores)
    assert after[0] == expected[0]
    pd.testing.assert_frame_equal(after[1], expected[1])
    for name, value in expected[2].items():
        assert after[2][name] == pytest.approx(value, nan_ok=True)


def test_graph_snapshot_ignores_future_fundamentals_returns_and_filing_text():
    panel = pd.DataFrame(
        {
            "permno": [1, 2, 3],
            "permco": [10, 20, 30],
            "eom": pd.Timestamp("2020-12-31"),
            "ret": [0.1, 0.2, 0.3],
            "me": [100.0, 200.0, 300.0],
            "gics": "451010",
            "be_me": [1.0, 2.0, 3.0],
        }
    )
    text = pd.DataFrame(
        {
            "permco": [10, 20, 30],
            "document_id": ["a", "b", "c"],
            "available_at": pd.to_datetime(["2020-12-01"] * 3, utc=True),
            "business_description": ["cloud software services"] * 3,
            "event_text": ["software launch"] * 3,
        }
    )
    settings = builder.GraphConfig(
        fundamental_columns=("be_me",),
        min_fundamental_overlap=1,
        min_return_overlap=1,
        n_neighbors=2,
    )
    month = pd.Timestamp("2020-12-31")
    expected = builder.EconomicGraphBuilder(panel, text, settings).build_month(month)
    future_panel = panel.assign(eom=pd.Timestamp("2021-01-31"), ret=999.0, be_me=999.0)
    future_text = text.assign(
        available_at=pd.Timestamp("2021-01-01", tz="UTC"),
        business_description="unrelated future news",
    )
    after = builder.EconomicGraphBuilder(
        pd.concat([panel, future_panel]), pd.concat([text, future_text]), settings
    ).build_month(month)
    np.testing.assert_array_equal(
        after.adjacency.toarray(), expected.adjacency.toarray()
    )
    pd.testing.assert_frame_equal(after.nodes, expected.nodes)


def test_valuation_model_selection_ignores_future_pricing_errors(monkeypatch):
    monkeypatch.setattr(config, "TEST_YEARS", (2021,))
    months = pd.to_datetime(["2019-12-31", "2020-12-31", "2021-01-31"])
    xs = pd.DataFrame({"permno": [1, 1, 1], "eom": months, "y": 1.0})
    for name in set(valuation.MODELS) - {"M6", "M8", "M9"}:
        xs[name] = 1.1 if name == "M5" else 2.0
    pooled = pd.DataFrame(
        {"permno": [1, 1, 1], "eom": months, "fold_year": 2021, "M6": 2.0, "M8": 2.0}
    )
    expected = valuation._select_models(xs, pooled)
    xs.loc[xs.eom >= pd.Timestamp("2021-01-01"), "y"] = -999.0
    pooled.loc[pooled.eom >= pd.Timestamp("2021-01-01"), ["M6", "M8"]] = 999.0
    assert valuation._select_models(xs, pooled) == expected
    assert expected["folds"][2021]["best_pricing_model"] == "M5"


def test_optimizer_earlier_book_ignores_future_scores_and_returns(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(optimizer, "N_PER_SIDE", 1)
    predictions = pd.DataFrame(
        [
            {
                "permno": p,
                "return_month": month,
                "formation_date": month - pd.offsets.MonthEnd(1),
                "timing_policy": timing.PREDICTION_POLICY,
                "final_score": float(p),
                "available_count": 1,
                "ticker": f"T{p}",
                "company_name": f"Company {p}",
                "ticker_name_reference_date": month - pd.offsets.MonthEnd(1),
                "ticker_name_source": "test",
                "ticker_name_status": identity.CRSP_STATUS,
                "prc": 10.0,
                "size_grp": "large",
                "dolvol_126d": 1e8,
                "sector": "45",
                "beta_60m": 1.0,
            }
            for month in pd.to_datetime(["2021-01-31", "2021-02-28"])
            for p in range(1, 5)
        ]
    )
    panel = pd.DataFrame(
        [
            {
                "permno": p,
                "eom": month,
                "gics": "451010",
                "size_grp": "large",
                "ret_exc": 0.01 * p,
                "rmax1_21d": 0.02 * p,
            }
            for month in pd.to_datetime(["2020-11-30", "2020-12-31", "2021-01-31"])
            for p in range(1, 5)
        ]
    )
    path = tmp_path / "panel.parquet"
    panel.to_parquet(path)
    before, _ = optimizer.optimize(predictions, path, nav=10_000_000)
    cutoff = pd.Timestamp("2020-12-31")
    panel.loc[panel.eom > cutoff, ["ret_exc", "rmax1_21d"]] = 999.0
    panel.to_parquet(path)
    predictions.loc[predictions.formation_date > cutoff, "final_score"] *= -999.0
    after, _ = optimizer.optimize(predictions, path, nav=10_000_000)
    first = pd.Period("2021-01", "M")
    expected = before.loc[before.return_month == first]
    assert len(expected) == 2
    pd.testing.assert_frame_equal(after.loc[after.return_month == first], expected)


def _predictions():
    return pd.DataFrame(
        {
            "permno": [1],
            "return_month": pd.to_datetime(["2021-01-31"]),
            "formation_date": pd.to_datetime(["2020-12-31"]),
            "timing_policy": timing.PREDICTION_POLICY,
            "ticker": ["A"],
            "company_name": ["Alpha"],
            "ticker_name_reference_date": pd.to_datetime(["2020-12-31"]),
        }
    )


@pytest.mark.parametrize(
    "name", ["ret_exc_lead1m", "target_eps", "next_suescore", "eps_label"]
)
def test_prediction_and_feature_guards_reject_outcomes(name):
    with pytest.raises(ValueError, match="predictor"):
        timing.validate_features([name])
    with pytest.raises(ValueError, match="predictor"):
        timing.validate_predictions(_predictions().assign(**{name: 1.0}))


def test_prediction_guard_rejects_future_identity_and_wrong_horizon():
    timing.validate_predictions(_predictions())
    with pytest.raises(ValueError, match="after formation"):
        timing.validate_predictions(
            _predictions().assign(ticker_name_reference_date=pd.Timestamp("2021-01-01"))
        )
    with pytest.raises(ValueError, match="after formation"):
        timing.validate_predictions(
            _predictions().assign(return_month=pd.Timestamp("2021-02-28"))
        )


def test_optimizer_rejects_legacy_predictions_before_loading_history(tmp_path):
    legacy = _predictions().drop(columns="timing_policy")
    with pytest.raises(ValueError, match="Unaudited predictions"):
        optimizer.optimize(legacy, tmp_path / "missing-panel.parquet")


def test_timing_only_pipeline_excludes_pretrained_and_restores_committed_switch():
    assert "timesfm" not in {s.name for s in combiner.make_sleeves()}
    assert optimizer.ENABLE_MICRO_SLEEVE_SWITCH
    frame = pd.DataFrame(
        {
            "final_score": [0.1],
            "final_score_liquid": [0.9],
            "available_count": [9],
            "available_count_liquid": [7],
        }
    )
    assert optimizer.scored_for_nav(frame, 1e9).final_score.iloc[0] == 0.9


def test_cache_refuses_legacy_artifacts_and_changed_inputs(tmp_path, monkeypatch):
    data_dir, external_dir = tmp_path / "data", tmp_path / "external"
    data_dir.mkdir()
    external_dir.mkdir()
    for name in [config.PANEL_FILE, config.FACTOR_LIST_FILE, config.FILINGS_FILE]:
        (data_dir / name).write_bytes(b"input")
    (external_dir / "external.parquet").write_bytes(b"external")
    monkeypatch.setattr(config, "EXTERNAL_DIR", external_dir)
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "forecasts.parquet").write_bytes(b"old")
    with pytest.raises(ValueError, match="Unaudited"):
        timing.prepare_cache(legacy, data_dir, "cpu")
    work = tmp_path / "work"
    timing.prepare_cache(work, data_dir, "cpu")
    timing.prepare_cache(work, data_dir, "cpu")
    (data_dir / config.PANEL_FILE).write_bytes(b"changed input")
    with pytest.raises(ValueError, match="stale"):
        timing.prepare_cache(work, data_dir, "cpu")
