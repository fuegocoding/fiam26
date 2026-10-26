"""Real columnar integration plus causal and financial boundary regressions."""

import datetime as dt
import json
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
