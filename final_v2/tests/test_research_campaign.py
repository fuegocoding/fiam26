"""Real columnar integration plus causal and financial boundary regressions."""

import datetime as dt
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pod_platform.agents.provider import LLMResponse
from pod_platform.application.configuration import load_config
from pod_platform.application.research_campaign import (
    ResearchCampaign,
    economic_controls,
    grouped_allocation,
)
from pod_platform.research.campaign_metrics import competition_report
from pod_platform.research.input_permissions import COMPETITION_CHARACTERISTICS
from pod_platform.research.strategy_code import (
    CodeStrategy,
    code_signals,
    compile_strategy,
)
from pod_platform.research.training_tactical import DailyMarket


def strategy(**changes):
    return CodeStrategy(
        **{
            "pod": "dynamics",
            "name": "test_signal",
            "mechanism": "A causal trend hypothesis using only trailing observations.",
            "falsification": "Prior settled net returns must remain positive across regimes.",
            "code": "def score(features, history, daily):\n    return features['cash_at']\n",
            "controls": "dollar",
            "review": "monthly",
            "signal_update_months": 1,
            "expected_holding_days": 21,
            **changes,
        }
    )


@pytest.mark.parametrize(
    "code",
    [
        "import os\ndef score(features, history, daily):\n return daily[-1]",
        "def score(features, history, daily):\n return open('cash_at')",
        "def score(features, history, daily):\n return daily.__class__",
        "def score(features, history, daily):\n return features['ret_exc_lead1m']",
        "def score(features, history, daily):\n while True: pass",
        "def score(features, history, daily):\n features['cash_at'] = daily[-1]\n return daily[-1]",
        "def score(features, history, daily):\n return daily[:, None]",
        "def score(features, history, daily):\n return np.stack([daily,daily])",
    ],
)
def test_research_code_cannot_access_io_outcomes_or_expand_dimensions(code):
    with pytest.raises((ValueError, SyntaxError)):
        compile_strategy(code, {"cash_at"})


def test_generated_code_uses_month_end_snapshot_before_next_close_even_on_weekend():
    dates = pd.bdate_range("2020-01-27", "2020-03-04")
    monthly = pd.to_datetime(["2020-01-31", "2020-02-29"])
    panel = SimpleNamespace(
        dates=monthly,
        execution_dates=pd.to_datetime(["2020-02-03", "2020-03-02"]),
        securities=np.arange(4),
        mask=np.ones((2, 4), bool),
        features={"cash_at": np.array([[1, 2, 3, 4], [5, 6, 7, 8]], float)},
    )
    market = DailyMarket(
        dates,
        np.arange(len(dates) * 4).reshape(-1, 4).astype(float),
        np.ones((len(dates), 4), bool),
        monthly.searchsorted(dates, side="right") - 1,
        [],
    )
    s = strategy(
        code="def score(features, history, daily):\n    return features['cash_at'] + daily[-1]\n"
    )
    _, original = code_signals(s, panel, market)
    feb = dates.get_loc("2020-02-03")
    march = dates.get_loc("2020-03-02")
    np.testing.assert_array_equal(
        original[march][1],
        panel.features["cash_at"][1] + market.returns[march - 1],
    )
    market.returns[feb:] *= -1000
    panel.features["cash_at"][1] *= -1e8
    _, changed = code_signals(s, panel, market)
    np.testing.assert_array_equal(original[feb][1], changed[feb][1])
    assert changed[feb][0] == 0


def test_nested_continuous_allocator_enforces_pod_and_fund_caps():
    edges = [0.5, 0.4, 0.3, -0.2]
    capital = grouped_allocation(
        edges, [0, 0, 0, 0], [1e9] * 4, ["a", "a", "b", "b"], 1e9, 0.6
    )
    assert capital.sum() <= 1e9 + 1
    assert capital[:2].sum() <= 6e8 + 1
    assert capital[-1] == 0
    assert capital[0] > capital[1]
    small = grouped_allocation([0.1], [0.001], [1e9], ["a"], 1e9, 0.5)
    assert 0 < small[0] < 1e8


def test_reporting_retains_all_dates_cash_and_official_benchmark_with_explicit_beta_coverage():
    dates = pd.bdate_range("2019-01-02", periods=270)
    market = np.linspace(-0.01, 0.01, len(dates))
    benchmark = pd.DataFrame({"tb3ms": 2.0, "sp_return": market}, index=dates)
    benchmark.iloc[:20, 1] = np.nan
    net = 0.5 * market + 0.0001
    report, daily, monthly = competition_report(dates, net, benchmark)
    assert len(daily) == 270
    np.testing.assert_allclose(daily.total, net + 0.02 / 252)
    assert report["daily"]["beta"] == pytest.approx(0.5)
    assert report["daily"]["regression_observations"] == 250
    assert report["sp500_missing_sessions"] == 20
    expected = np.mean(net - 0.04 / 252) / np.std(net, ddof=1) * np.sqrt(252)
    assert report["daily"]["net_information_ratio"] == pytest.approx(expected)
    assert monthly.market.isna().sum() == 1


class PaperFixture:
    def search(self, *args, **kwargs):
        return SimpleNamespace(cards=[])


class CarbonFixture:
    pass


class ProgrammerFixture:
    name, model, is_mock = "test_fixture", "code_fixture", True

    def __init__(self):
        self.requests = []

    def complete(self, request):
        self.requests.append(json.loads(request.user))
        code = []
        for pod in [self.requests[-1]["pod"]]:
            code.append(
                strategy(
                    pod=pod,
                    name="agent_revision",
                    code="def score(features, history, daily):\n    x = features['cash_at']\n    return x + 0.1 * np.nanmean(daily[-21:], axis=0)\n",
                ).model_dump(mode="json")
            )
        return LLMResponse(
            text=json.dumps(
                {
                    "diagnosis": "Test feedback arrives before the next explicit code revision.",
                    "strategies": code,
                }
            ),
            provider=self.name,
            model=self.model,
            response_hash="a" * 64,
            received_at=dt.datetime.now(dt.timezone.utc),
            latency_s=0,
            total_tokens=200,
        )


def _files(tmp_path):
    rng = np.random.default_rng(19)
    dates = pd.date_range("2017-01-31", "2020-12-31", freq="ME")
    names = 60
    signal = rng.uniform(-1, 1, names)
    parts = []
    for date in dates:
        frame = pd.DataFrame(
            {
                f: signal + rng.normal(0, 0.01, names)
                for f in COMPETITION_CHARACTERISTICS
            }
        )
        frame["permno"] = np.arange(1, names + 1)
        frame["eom"] = date
        frame["me"], frame["prc"], frame["gics"], frame["beta_60m"] = (
            20000.0,
            20.0,
            "10",
            1.0,
        )
        frame["dolvol_126d"], frame["rvol_21d"], frame["bidaskhl_21d"] = (
            1e9,
            0.02,
            0.0,
        )
        parts.append(frame)
    panel = tmp_path / "panel.parquet"
    pd.concat(parts).to_parquet(panel, index=False)
    sessions = pd.bdate_range("2016-01-01", "2020-12-31")
    daily = pd.DataFrame(
        {
            "date": np.repeat(sessions.strftime("%Y%m%d"), names),
            "permno": np.tile(np.arange(1, names + 1), len(sessions)),
            "prc": 20.0,
            "daily_return": np.tile(signal * 0.001, len(sessions)),
        }
    )
    daily_path = tmp_path / "daily.csv"
    daily.to_csv(daily_path, index=False)
    rates = pd.DataFrame(
        {
            "date": pd.date_range("2016-01-01", "2020-12-01", freq="MS"),
            "table": "fred",
            "series": "TB3MS",
            "value": 2.0,
        }
    )
    index = pd.DataFrame(
        {
            "date": sessions,
            "table": "fred",
            "series": "SP500",
            "value": 1000 * np.exp(np.arange(len(sessions)) * 0.0001),
        }
    )
    benchmark = tmp_path / "benchmark.parquet"
    pd.concat([rates, index]).to_parquet(benchmark, index=False)
    return panel, daily_path, benchmark


def test_entire_campaign_codes_revises_owns_funds_executes_replays_and_reports(
    tmp_path,
):
    panel, daily, benchmark = _files(tmp_path)
    config = load_config()
    config["application"].update(
        state_dir=str(tmp_path / "state"),
        papers_db=str(tmp_path / "papers.sqlite"),
        online_papers=False,
    )
    config["training_lab"].update(
        panel_path=str(panel),
        daily_path=str(daily),
        start="2017-01-01",
        daily_warmup_start="2016-01-01",
        filings_research_enabled=False,
        min_positions=20,
        universe_size=60,
        cost_bps_one_way=0,
        large_cap_borrow_bps=0,
        other_borrow_bps=0,
        fund_capital_usd=1e6,
        impact_eta=0,
    )
    config["research_campaign"].update(
        benchmark_fred_path=str(benchmark),
        minimum_edge_months=6,
        edge_uncertainty_haircut=0,
        max_rounds=2,
        max_new_model_calls=8,
    )
    provider = ProgrammerFixture()
    campaign = ResearchCampaign(
        config,
        providers=SimpleNamespace(lead=provider),
        evidence_services={"papers": PaperFixture(), "carbon": CarbonFixture()},
    )
    report = campaign.run()
    assert report["status"] == "integrated_campaign_completed"
    assert len(provider.requests) == 8
    assert provider.requests[1]["feedback"]
    assert report["successful_strategies"] >= len(economic_controls()) + 4
    assert all(1 <= n <= 3 for n in report["owned_strategy_counts"].values())
    assert report["static_target_replay_max_return_difference"] < 1e-10
    assert report["fund"]["daily"]["observations"] > 500
    assert report["execution"]["transaction_cost_usd"] >= 0
    assert report["edge_selection_uses_capital"] is False
    assert report["capital_grid_used"] is False
    assert (campaign.output / "actual_holdings.parquet").exists()
    verifier_spec = importlib.util.spec_from_file_location(
        "campaign_replay_verifier",
        Path(__file__).parents[1] / "verify_campaign.py",
    )
    verifier = importlib.util.module_from_spec(verifier_spec)
    verifier_spec.loader.exec_module(verifier)
    review = verifier.verify(campaign.output / "campaign_report.json")
    assert review["headline_metrics_recomputed"]
    assert review["ineligible_actual_short_increases"] == 0
    assert any(
        row["funded"]
        for row in json.loads(
            (campaign.output / "capital_allocations.json").read_text()
        )
    )
    replay_provider = ProgrammerFixture()
    replayed = ResearchCampaign(
        config,
        providers=SimpleNamespace(lead=replay_provider),
        evidence_services={"papers": PaperFixture(), "carbon": CarbonFixture()},
    ).run()
    assert not replay_provider.requests
    assert replayed["fund"]["cumulative_return"] == pytest.approx(
        report["fund"]["cumulative_return"]
    )


def test_compact_loader_preserves_full_universe_features_with_column_streaming(
    tmp_path,
):
    from pod_platform.research.training_data import prepare_panel

    panel_path, daily_path, _ = _files(tmp_path)
    settings = load_config()["training_lab"]
    settings.update(
        start="2017-01-01",
        daily_warmup_start="2016-01-01",
        min_positions=20,
        universe_size=60,
        filings_research_enabled=False,
    )
    regular = prepare_panel(
        panel_path, daily_path, settings, tmp_path / "regular_daily.parquet"
    )
    compact = prepare_panel(
        panel_path,
        daily_path,
        {**settings, "compact_storage": True},
        tmp_path / "compact_daily.parquet",
    )
    np.testing.assert_array_equal(regular.securities, compact.securities)
    np.testing.assert_array_equal(regular.mask, compact.mask)
    assert len(compact.features) == 147
    for name in regular.features:
        np.testing.assert_array_equal(
            regular.features[name], compact.features[name]
        )
        assert isinstance(compact.features[name], np.memmap)
    np.testing.assert_allclose(regular.returns, compact.returns, atol=1e-6)
    assert set(compact.frames[0]) <= {
        "me",
        "prc",
        "gics",
        "beta_60m",
        "dolvol_126d",
        "rvol_21d",
        "bidaskhl_21d",
    }


def test_heavy_job_lock_rejects_overlapping_work_and_releases(tmp_path):
    from pod_platform.application.memory_control import research_job_lock

    with research_job_lock(tmp_path):
        with pytest.raises(RuntimeError, match="Another data-heavy"):
            with research_job_lock(tmp_path):
                pytest.fail("A second job acquired the same lock")
    with research_job_lock(tmp_path):
        pass


@pytest.mark.parametrize(
    "scores",
    [
        [1.0, 0.8, 0.6, np.nan, np.nan, np.nan],
        [1.0, 0.0, 0.0, 0.0, 0.0, -0.5],
    ],
)
def test_projection_preserves_missing_and_threshold_membership(scores):
    from pod_platform.research.strategy_code import unit_schedule

    scores = np.asarray(scores)
    panel = SimpleNamespace(
        mask=np.ones((1, 6), bool),
        frames=[
            pd.DataFrame(
                {
                    "me": [20000.0] * 6,
                    "gics": ["10"] * 6,
                    "beta_60m": np.arange(6),
                }
            )
        ],
    )
    weights = unit_schedule(
        strategy(controls="sector_beta"),
        {1: (0, scores)},
        panel,
        {"gross_limit": 2.0, "single_name_limit": 0.5, "universe_size": 6},
        6,
    )[1]
    excluded = ~np.isfinite(scores) | (scores == 0)
    np.testing.assert_array_equal(weights[excluded], 0)


def test_daily_cache_keys_universe_dates_and_detects_tampering(tmp_path):
    from pod_platform.research.training_data import read_daily

    source = tmp_path / "daily.csv"
    pd.DataFrame(
        {
            "permno": [1, 2, 1, 2],
            "date": [20200102, 20200102, 20200103, 20200103],
            "prc": 10.0,
            "daily_return": 0.01,
        }
    ).to_csv(source, index=False)
    first = read_daily(
        source,
        [1],
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-12-31"),
        tmp_path / "cache.parquet",
    )
    second = read_daily(
        source,
        [1, 2],
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-12-31"),
        tmp_path / "cache.parquet",
    )
    assert set(first.permno) == {1}
    assert set(second.permno) == {1, 2}
    cache = next(
        p
        for p in tmp_path.glob("cache_*.parquet")
        if len(pd.read_parquet(p)) == 4
    )
    changed = pd.read_parquet(cache)
    changed["daily_return"] = 1.0
    changed.to_parquet(cache, index=False)
    with pytest.raises(ValueError, match="digest changed"):
        read_daily(
            source,
            [1, 2],
            pd.Timestamp("2020-01-01"),
            pd.Timestamp("2020-12-31"),
            tmp_path / "cache.parquet",
        )


def _excessive_array_fixture(features, history, daily):
    return np.ones(100_000_000)


def test_numerical_worker_enforces_hard_extra_memory_limit():
    from pod_platform.research.score_worker import ScoreWorker

    with ScoreWorker(_excessive_array_fixture, extra_memory_mib=32) as worker:
        with pytest.raises(ValueError, match="resource boundary|MemoryError"):
            worker.score({}, {}, np.zeros((1, 4)))


def test_python_container_repetition_is_rejected_before_execution():
    for expression in ("[0] * 10000 * 10000", "'cash_at' * 10000 * 10000"):
        code = f"def score(features, history, daily):\n    x = {expression}\n    return features['cash_at']\n"
        with pytest.raises(ValueError):
            compile_strategy(code, {"cash_at"})


@pytest.mark.parametrize("hedge_policy", ["paired_signal", "liquid_sector"])
def test_exact_monthly_concentration_matches_native_unit_gross_replay(
    tmp_path, hedge_policy
):
    from pod_platform.research.edge_capacity import measure_edge
    from pod_platform.research.forecast_construction import (
        monthly_concentration,
    )
    from pod_platform.research.strategy_code import unit_schedule
    from pod_platform.research.training_data import prepare_panel
    from pod_platform.research.training_tactical import (
        daily_market,
    )

    panel_path, daily_path, _ = _files(tmp_path)
    settings = load_config()["training_lab"]
    settings.update(
        start="2017-01-01",
        daily_warmup_start="2016-01-01",
        min_positions=20,
        universe_size=60,
        filings_research_enabled=False,
        compact_storage=True,
        cost_bps_one_way=0,
        large_cap_borrow_bps=0,
        other_borrow_bps=0,
    )
    panel = prepare_panel(
        panel_path, daily_path, settings, tmp_path / "unit_daily.parquet"
    )
    market = daily_market(panel, settings)
    plan = strategy(
        signal_update_months=3,
        weighting="equal_tail",
        hedge_policy=hedge_policy,
        controls="sector",
    )
    _, decisions = code_signals(plan, panel, market)
    curve = monthly_concentration(plan, decisions, panel, market, settings)
    weights = unit_schedule(plan, decisions, panel, settings, 20)
    daily = measure_edge(market, weights, settings)
    # Concentration labels span execution-to-execution intervals, while fund
    # headlines use calendar months. Reconcile the same valuation clocks.
    actual = np.array(
        [
            np.prod(
                1
                + daily["gross"][
                    (market.dates > execution) & (market.dates <= settlement)
                ]
            )
            - 1
            for execution, settlement in zip(
                panel.execution_dates, panel.settlement_dates
            )
        ]
    )
    expected = curve.loc[curve.names == 20, "controlled_spread"].to_numpy()
    np.testing.assert_allclose(actual, expected, atol=2e-7)


def test_supported_constructor_preserves_ranked_signs_and_exact_controls():
    from pod_platform.research.forecast_construction import (
        prepare_forecast,
        supported_target,
    )
    from pod_platform.research.pod_plans import construction_designs

    rng = np.random.default_rng(194)
    names = 200
    frame = pd.DataFrame(
        {
            "me": np.full(names, 20000.0),
            "gics": np.repeat(["10", "20"], names // 2),
            "beta_60m": rng.uniform(0.5, 1.5, names),
        }
    )
    panel = SimpleNamespace(mask=np.ones((1, names), bool), frames=[frame])
    score = rng.normal(size=names)
    score[:10] = np.nan
    score[10:20] = 0
    design = construction_designs(panel, "sector_beta")[0]
    prepared = prepare_forecast(score, design)
    weights = supported_target(
        score,
        design,
        {"gross_limit": 2.0, "single_name_limit": 0.05},
        maximum=100,
        weighting="equal_tail",
        prepared=prepared,
    )
    np.testing.assert_array_equal(weights[:20], 0)
    np.testing.assert_allclose(design.matrix.T @ weights, 0, atol=1e-7)
    indices, _, ranked, _ = prepared
    assert np.all(weights[indices] * ranked >= -1e-10)
    assert np.count_nonzero(weights) <= 100


def test_local_array_mask_assignment_and_finite_methods_preserve_inputs():
    from pod_platform.research.score_worker import ScoreWorker

    code = "def score(features, history, daily):\n    value = features['cash_at'].copy()\n    mask = np.isnan(value)\n    value[mask] = 0\n    return value - value.mean()\n"
    function = compile_strategy(code, {"cash_at"})
    source = np.array([1.0, np.nan, -1.0, 2.0])
    with ScoreWorker(function) as worker:
        actual = worker.score({"cash_at": source}, {}, np.zeros((1, 4)))
    np.testing.assert_allclose(actual, [0.5, -0.5, -1.5, 1.5])
    assert np.isnan(source[1])
    alias = compile_strategy(
        "def score(features, history, daily):\n    x = features['cash_at']\n    x[0] = 5\n    return x\n",
        {"cash_at"},
    )
    with ScoreWorker(alias) as worker:
        with pytest.raises(ValueError, match="read-only"):
            worker.score({"cash_at": source}, {}, np.zeros((1, 4)))


def test_paper_scope_excludes_physics_and_labor_earnings_from_stock_research():
    from pod_platform.application.campaign_sources import paper_relevant

    assert not paper_relevant(
        "pod_dynamics",
        "Minimal momentum estimation in noncommutative phase space",
    )
    assert not paper_relevant(
        "pod_fundamentals",
        "Heterogeneous Earnings Effects of the Job Corps by Gender Earnings",
    )
    assert paper_relevant(
        "pod_dynamics",
        "Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency",
        "Momentum stock returns and trading strategies",
    )
    assert paper_relevant(
        "pod_fundamentals",
        "Do Stock Prices Fully Reflect Information in Accruals and Cash Flows About Future Earnings?",
    )
    assert paper_relevant(
        "pod_events",
        "Post earnings announcement drift",
        "Stock returns after earnings announcements",
    )


def test_same_month_benchmark_does_not_inherit_lagged_cash_accrual():
    dates = pd.bdate_range("2019-01-02", "2019-02-28")
    reference = np.where(dates.month == 1, 1.0, 3.0)
    benchmark = pd.DataFrame(
        {
            "tb3ms": 0.0,
            "reference_tb3ms": reference,
            "sp_return": np.linspace(-0.001, 0.002, len(dates)),
        },
        index=dates,
    )
    report, daily, monthly = competition_report(
        dates, np.zeros(len(dates)), benchmark, returns_include_cash=True
    )
    np.testing.assert_allclose(daily.cash, reference / 25200)
    np.testing.assert_allclose(monthly.cash, [1 / 1200, 3 / 1200])
    assert report["daily"]["annualized_alpha_se"] is not None


def test_pure_helpers_and_feature_conditionals_execute_without_io_or_recursion():
    from pod_platform.research.score_worker import ScoreWorker

    code = "def score(features, history, daily):\n    def winsor(x):\n        lo = np.nanpercentile(x, 5)\n        hi = np.nanpercentile(x, 95)\n        return np.clip(x, lo, hi)\n    value = features['cash_at']\n    if 'cash_at' in history:\n        value = value + np.nanmean(history['cash_at'][-3:], axis=0)\n    return winsor(value)\n"
    function = compile_strategy(code, {"cash_at"})
    data = np.array([-1.0, 0.0, 1.0, 2.0])
    with ScoreWorker(function) as worker:
        actual = worker.score(
            {"cash_at": data},
            {"cash_at": np.array([data, data])},
            np.zeros((2, 4)),
        )
    np.testing.assert_allclose(
        actual, np.clip(data * 2, *np.nanpercentile(data * 2, [5, 95]))
    )
    for body in (
        "def f(x):\n        return f(x)\n    return f(features['cash_at'])",
        "def f(x):\n        return g(x)\n    def g(x):\n        return f(x)\n    return f(features['cash_at'])",
    ):
        with pytest.raises(ValueError, match="Recursive"):
            compile_strategy(
                "def score(features, history, daily):\n    " + body + "\n",
                {"cash_at"},
            )
