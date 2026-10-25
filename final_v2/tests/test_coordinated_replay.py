"""Coordinated (CIO) mode inside the real ControlledReplay target -> risk -> fills -> ledger pipeline."""

from __future__ import annotations

import csv
import datetime as dt
import importlib.util
import json
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from pod_platform.allocation.coordination import FillFeedback
from pod_platform.allocation.covariance import ReturnRow
from pod_platform.contracts.errors import PodPlatformError
from pod_platform.contracts.forecasts import Forecast
from pod_platform.execution.planner import CostModel, CostModelConfig
from pod_platform.orchestration.replay import (
    COORDINATION_MODE,
    ControlledReplay,
    assert_fees_within_reserve,
    assert_plan_current,
    planned_fee_fraction,
    coordination_hook_from_records,
    default_coordination_inputs,
    fill_state_to_feedback,
    load_fixture,
    planned_order_to_pending,
)
from pod_platform.research.certificates import (
    CertificateEnvironment,
    CertificateScope,
)

from .test_cio_coordination import coordinator, intention, make_inputs, snapshot
from .test_replay import (
    LONGS,
    SHORTS,
    _certificate,
    make_fixture,
    run_controlled,
)

NAMES = LONGS + SHORTS + ["HEDGE"]
FIRST = "2024-01-31"


def risk_rows(
    end=dt.date(2024, 4, 30), start=dt.date(2023, 8, 1), drop=(), extra=None
):
    x, rows, day = 12345, [], start
    while day <= end:
        x = (x * 1103515245 + 12345) % 2**31
        market = (x / 2**31 - 0.5) * 3.46
        vals = {}
        for s in NAMES:
            x = (x * 1103515245 + 12345) % 2**31
            vals[s] = 0.01 * (0.8 * market + (x / 2**31 - 0.5) * 3.46)
        rows.append(
            {
                "observed_at": f"{day.isoformat()}T20:00:00+00:00",
                "returns": {k: v for k, v in vals.items() if k not in drop},
            }
        )
        day += dt.timedelta(days=1)
    return rows + (extra or [])


def policy(vol=100.0, requests=("fam_a",), allow=True, **over):
    pol = {
        "max_ex_ante_vol": vol,
        "max_abs_beta_net": 0.3,
        "max_sector_net": 2.0,
        "max_stress_loss": 5.0,
        "hedge_adv_participation": 1.0,
    }
    pol.update(over)
    return {
        "coordination_policy": pol,
        "covariance_policy": {"min_observations": 60},
        "hedge_allowances": [{"security_id": "HEDGE", "max_abs_weight": 0.25}]
        if allow
        else [],
        "hedge_requests": [
            {
                "requester": r,
                "security_id": "HEDGE",
                "dimension": "beta",
                "max_abs_weight": 0.25,
            }
            for r in requests
        ],
    }


def build(
    tmp_path,
    name,
    *,
    pods=("fam_a",),
    hedge_adv="100000000",
    hedge_rows=True,
    declare=True,
    certs=None,
    namespace=None,
):
    root = make_fixture(tmp_path / f"fx_{name}")
    cfg = json.loads((root / "config.json").read_text())
    cfg["admitted_families"] = list(pods)
    cfg["internal_policy"]["capital_usd"] = 1_000_000
    if declare:
        cfg["coordination"] = {"mode": COORDINATION_MODE}
    if namespace:
        cfg["namespace"] = namespace
    (root / "config.json").write_text(json.dumps(cfg))
    rows = list(csv.DictReader((root / "market.csv").open()))
    out = []
    for r in rows:  # long names carry beta 1.3 and shorts 0.7 so the book has a real beta to hedge
        r["beta"] = "1.3" if r["security_id"] in LONGS else "0.7"
        out.append(r)
        if hedge_rows and r["security_id"] == "AAA":
            out.append(
                {
                    **r,
                    "security_id": "HEDGE",
                    "ticker": "HEDGE",
                    "name": "Index hedge",
                    "beta": "1.0",
                    "sector": "idx",
                    "market_cap_usd": "500000000000",
                    "adv_dollars": hedge_adv,
                    "outcome_excess_return": "0.005",
                }
            )
    with (root / "market.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(out)
    forecasts = [
        Forecast.model_validate(f)
        for f in json.loads((root / "forecasts.json").read_text())
    ]
    extra = [
        f.model_copy(
            update={"strategy_id": p, "forecast_id": f"{f.forecast_id}_{p}"}
        ).sealed()
        for p in pods[1:]
        for f in forecasts
    ]
    (root / "forecasts.json").write_text(
        json.dumps([f.model_dump(mode="json") for f in forecasts + extra])
    )
    if certs is not None:
        (root / "certificates.json").write_text(json.dumps(certs))
    return root


def run(
    tmp_path, name, pol=None, rows=None, hook=None, research_only=False, **kw
):
    root = build(tmp_path, name, **kw)
    hook = hook or coordination_hook_from_records(
        pol or policy(), rows or risk_rows()
    )
    fixture = load_fixture(root)
    outcome = ControlledReplay(
        fixture, tmp_path / name, coordination=hook, research_only=research_only
    ).run()
    return outcome, tmp_path / name


def load(out, name):
    return json.loads((out / name).read_text())


def weights(out, period=FIRST):
    return {
        r["security_id"]: Decimal(r["post_trade_weight"])
        for r in csv.DictReader((out / "holdings_by_period.csv").open())
        if r["period"] == period
    }


def fails(tmp_path, name, text, **kw):
    with pytest.raises(PodPlatformError) as ei:
        run(tmp_path, name, **kw)
    assert text in ei.value.response.summary
    out = tmp_path / name
    assert load(out, "failure.json")["status"] == "FAILED_CLOSED"
    assert (
        load(out, "periods.partial.json") == [] or text == "FILL"
    )  # nothing traded before the rejection
    return out


def test_legacy_fixture_without_coordination_is_unchanged(tmp_path):
    outcome, out = run_controlled(tmp_path)
    assert (
        "coordination" not in outcome["summary"]
        and not (out / "coordination_receipts.json").exists()
    )


def test_declared_mode_cannot_disable_itself_and_hook_cannot_enable_itself(
    tmp_path,
):
    root = build(tmp_path, "nohook")
    with pytest.raises(PodPlatformError) as ei:
        ControlledReplay(load_fixture(root), tmp_path / "o1").run()
    assert (
        "no trusted-host coordination hook" in ei.value.response.summary
        and (tmp_path / "o1" / "failure.json").exists()
    )
    root = build(tmp_path, "nodecl", declare=False)
    with pytest.raises(PodPlatformError) as ei:
        ControlledReplay(
            load_fixture(root),
            tmp_path / "o2",
            coordination=coordination_hook_from_records(policy(), risk_rows()),
        ).run()
    assert "config.coordination is not declared" in ei.value.response.summary
    root = build(tmp_path, "pin")
    fx = load_fixture(root)
    fx.config["coordination"]["policy_digest"] = "0" * 64
    with pytest.raises(PodPlatformError) as ei:
        ControlledReplay(
            fx,
            tmp_path / "o3",
            coordination=coordination_hook_from_records(policy(), risk_rows()),
        ).run()
    assert "policy_digest" in ei.value.response.summary


def test_correlated_risk_cap_changes_actual_holdings_fills_and_is_audited(
    tmp_path,
):
    free, out_a = run(tmp_path, "free", policy(vol=100.0))
    rec_a = load(out_a, "coordination_receipts.json")
    vol_a = rec_a[0]["ex_ante_vol"]
    assert rec_a[0]["vol_scale"] == 1.0 and vol_a > 0
    capped, out_b = run(tmp_path, "capped", policy(vol=0.5 * vol_a))
    rec_b = load(out_b, "coordination_receipts.json")
    assert (
        rec_b[0]["vol_scale"] == pytest.approx(0.5, rel=0.02)
        and rec_b[0]["ex_ante_vol"] <= 0.5 * vol_a + 1e-9
    )
    wa, wb = weights(out_a), weights(out_b)
    gross_a, gross_b = (
        sum(abs(v) for v in wa.values()),
        sum(abs(v) for v in wb.values()),
    )
    assert (
        float(gross_b) == pytest.approx(0.5 * float(gross_a), rel=0.03)
        and wa != wb
    )
    ma, mb = load(out_a, "monthly.json")[0], load(out_b, "monthly.json")[0]
    assert Decimal(mb["traded_dollars"]) < Decimal(
        ma["traded_dollars"]
    ) and Decimal(mb["trading_fees"]) < Decimal(ma["trading_fees"])
    assert (
        load(out_b, "periods.json")[0]["trades"]
        != load(out_a, "periods.json")[0]["trades"]
    )
    # evidence: covariance/correlation/overlap/budgets/netting/original evidence persisted per period
    r = rec_b[0]
    assert (
        len(rec_b) == 4
        and r["risk_snapshot"]["covariance"]["pd_checked"]
        and r["risk_snapshot"]["covariance"]["correlation"]
    )
    assert (
        r["risk_budgets"]["budgets"]
        and "plan" in r["hedge_netting"]
        and r["original_evidence"]["forecasts"][0]["certificate_digest"]
    )
    # existing gates still ran on the coordinated book; bookkeeping reconciled; hashes in manifest/freeze inputs
    assert (
        load(out_b, "compliance_report.json")["all_official_pass"]
        and len(load(out_b, "risk_reviews_actual_fill.json")) == 4
    )
    assert (
        load(out_b, "periods.json")[0]["coordination"]["receipt_sha256"]
        == r["receipt_sha256"]
    )
    manifest = load(out_b, "run_manifest.json")
    assert (
        manifest["config_hashes"][f"coordination:risk_snapshot:{FIRST}"]
        == r["risk_snapshot"]["rows_sha256"]
    )
    assert (
        capped["summary"]["coordination"]["hashes"]["coordination:receipts"]
        == load(out_b, "artifact_hashes.json")["coordination_receipts.json"]
    )
    assert capped["assessment"].frozen == free["assessment"].frozen


def test_fee_reserve_is_hand_checked_bounded_and_covers_post_fee_nav_normalization(
    tmp_path,
):
    # Hand check: NAV 1,000,000; ADV 1e9; daily vol 0.02; eta 0.5; half-spread 2bp; fee 1bp.
    # Trade +10% = 100,000: participation 1e-4, impact 0.5*0.02*sqrt(1e-4)=1e-4 -> 1bp; rate 2+1+1=4bp; cost 40.
    # Trade -5% = 50,000: participation 5e-5, impact 0.5*0.02*sqrt(5e-5)=0.70711bp; rate 3.70711bp; cost 18.5355.
    model = CostModel(
        CostModelConfig(
            model_id="m",
            half_spread_by_size_bp={"large": 2.0},
            eta=0.5,
            explicit_fee_bp=1.0,
        )
    )
    row = SimpleNamespace(
        adv_dollars=Decimal("1000000000"),
        daily_volatility=0.02,
        size_tier="large",
        price=Decimal("10"),
    )
    fraction = planned_fee_fraction(
        model, {"A": 0.10, "B": -0.05}, {"A": row, "B": row}, Decimal("1000000")
    )
    assert fraction == pytest.approx((40.0 + 18.5355) / 1_000_000, rel=1e-4)
    with pytest.raises(
        PodPlatformError
    ):  # no dated liquidity -> no bound -> no trade
        planned_fee_fraction(
            model, {"A": 0.1, "C": 0.1}, {"A": row}, Decimal("1000000")
        )
    # Normalization: fees F reduce NAV to N(1-f); weights scale by 1/(1-f), so vol sized at cap*(1-r) stays <= cap iff f <= r.
    f, r, cap = 58.5355 / 1_000_000, 1.1 * 58.5355 / 1_000_000, 0.04
    assert cap * (1 - r) / (1 - f) <= cap
    assert float(
        Decimal("500000") / (Decimal("1000000") - Decimal("1000"))
    ) == pytest.approx(0.5005005, abs=1e-7)
    assert_fees_within_reserve(Decimal("58.5355"), Decimal("1000000"), r, FIRST)
    with pytest.raises(
        PodPlatformError
    ) as ei:  # actual cost above the planned bound still fails closed
        assert_fees_within_reserve(Decimal("70"), Decimal("1000000"), r, FIRST)
    assert "exceed the planned fee reserve" in ei.value.response.summary
    root = build(tmp_path, "tinyreserve")
    fx = load_fixture(root)
    fx.config["coordination"]["fee_reserve"] = {"max": 1e-9}
    with pytest.raises(PodPlatformError) as ei:
        ControlledReplay(
            fx,
            tmp_path / "tiny",
            coordination=coordination_hook_from_records(policy(), risk_rows()),
        ).run()
    assert "fee reserve" in ei.value.response.summary
    root = build(tmp_path, "highcost")
    fx = load_fixture(root)
    fx.config["cost_model"]["explicit_fee_bp"] = 1000.0
    with pytest.raises(PodPlatformError) as ei:
        ControlledReplay(
            fx,
            tmp_path / "highcost",
            coordination=coordination_hook_from_records(policy(), risk_rows()),
        ).run()
    assert "fee reserve" in ei.value.response.summary
    assert load(tmp_path / "highcost", "periods.partial.json") == []
    assert (
        load(tmp_path / "highcost", "risk_reviews_actual_fill.partial.json")
        == []
    )


def test_reserve_and_capital_follow_actual_compounding_nav_with_asof_inputs(
    tmp_path,
):
    _, out = run(tmp_path, "nav", policy(vol=0.05))
    receipts, monthly = (
        load(out, "coordination_receipts.json"),
        load(out, "monthly.json"),
    )
    assert len(receipts) == len(monthly) >= 2
    for rec, month in zip(receipts, monthly):
        fee = rec["fee_reserve"]
        assert Decimal(fee["nav_open"]) == Decimal(month["nav_open"]) and fee[
            "capital_usd"
        ] == pytest.approx(float(month["nav_open"]))
        assert (
            0.0 <= fee["planned_fee_fraction"] <= fee["reserve"] <= fee["max"]
        )
        assert fee["vol_cap_sized"] == pytest.approx(
            fee["vol_cap_policy"] * (1 - fee["reserve"])
        )
        assert Decimal(month["trading_fees"]) / Decimal(
            month["nav_open"]
        ) <= Decimal(str(fee["reserve"]))  # actual cost inside the bound
        assert (
            rec["ex_ante_vol"] <= fee["vol_cap_sized"] + 1e-12
        )  # sized on the reserve, policy cap itself unchanged
        assert (
            fee["projected_execution_error_vol"] <= fee["execution_vol_buffer"]
        )
        assert rec["ex_ante_vol"] <= fee["target_vol_cap"] + 1e-12
        # Actual post-fee ledger weights, independently multiplied through the persisted covariance.
        cov = rec["risk_snapshot"]["covariance"]
        actual = {
            sid: float(w)
            for sid, w in rec["execution"]["post_trade_weights"].items()
        }
        annual_variance = 252 * sum(
            actual.get(a, 0.0) * actual.get(b, 0.0) * cov["covariance"][i][j]
            for i, a in enumerate(cov["security_ids"])
            for j, b in enumerate(cov["security_ids"])
        )
        assert annual_variance**0.5 <= fee["vol_cap_policy"]
        assert (
            rec["post_fill"]["post_fill_covariance_review"]["decision"]
            == "PASS"
        )
        assert (
            4 <= len(actual) <= 20
            and sum(abs(v) for v in actual.values()) > 0.5
        )
    # April's small resizes hit the no-trade band: fee headroom alone was insufficient.
    assert receipts[-1]["fee_reserve"]["execution_vol_buffer"] > 1e-6
    assert (
        receipts[1]["fee_reserve"]["capital_usd"]
        != receipts[0]["fee_reserve"]["capital_usd"]
    )  # NAV compounded, not static
    assert Decimal(monthly[1]["nav_open"]) == Decimal(monthly[0]["nav_end"])
    assert (
        receipts[0]["risk_snapshot"]["last_observed_at"]
        <= "2024-01-31T21:00:00+00:00"
    )


def test_future_risk_rows_are_never_read_before_a_decision(tmp_path):
    base, out_a = run(tmp_path, "base", policy())
    future = [
        {
            "observed_at": "2024-06-01T20:00:00+00:00",
            "returns": {s: 0.5 for s in NAMES},
        }
    ]
    _, out_b = run(tmp_path, "withfuture", policy(), risk_rows(extra=future))
    ra, rb = (
        load(out_a, "coordination_receipts.json"),
        load(out_b, "coordination_receipts.json"),
    )
    assert [x["risk_snapshot"]["rows_sha256"] for x in ra] == [
        x["risk_snapshot"]["rows_sha256"] for x in rb
    ]
    assert (
        ra[0]["risk_snapshot"]["rows_used"]
        < ra[-1]["risk_snapshot"]["rows_used"]
    )  # each decision sees only its own past
    assert (
        ra[0]["risk_snapshot"]["last_observed_at"]
        <= "2024-01-31T21:00:00+00:00"
    )


def test_shared_hedge_is_not_doubled_by_two_pods_and_cio_and_partial_fills_are_accounted(
    tmp_path,
):
    one, out1 = run(
        tmp_path, "one", policy(requests=("fam_a",)), hedge_adv="100000"
    )
    many, out3 = run(
        tmp_path,
        "many",
        policy(requests=("fam_a", "fam_b", "CIO")),
        pods=("fam_a", "fam_b"),
        hedge_adv="100000",
    )
    r1, r3 = (
        load(out1, "coordination_receipts.json"),
        load(out3, "coordination_receipts.json"),
    )
    plan = r3[0]["hedge_netting"]["plan"]
    assert (
        len(plan["lines"]) == 1
        and len(plan["dispositions"]) == 3
        and len(r3[0]["groups"]) == 1
    )  # twin pods = one family
    assert (
        plan["lines"][0]["target_weight"]
        == r1[0]["hedge_netting"]["plan"]["lines"][0]["target_weight"]
        < 0
    )
    assert weights(out3)["HEDGE"] == weights(out1)["HEDGE"]
    assert [
        t["security_id"] for t in load(out3, "periods.json")[0]["trades"]
    ].count("HEDGE") == 1
    # the thin hedge only partly fills: plan goes stale, residual hedge is recomputed from the held quantity
    target = plan["lines"][0]["target_weight"]
    held = float(weights(out3)["HEDGE"])
    assert target < held < 0
    post = r3[0]["post_fill"]
    assert (
        post["plan_stale_after_fills"]
        and post["residual_orders_placed"] == 0
        and post["cancelled_remainders"]
    )
    assert (
        post["post_fill_covariance_review"]["decision"] == "PASS"
        and post["ledger_gap"] <= 1e-9
    )
    assert r3[0]["fee_reserve"]["projected_execution_error_vol"] > 0
    line = post["residual_hedge_lines"][0]
    assert line["held_weight"] == pytest.approx(held, abs=1e-4) and line[
        "new_order_weight"
    ] == pytest.approx(line["target_weight"] - line["held_weight"], abs=1e-9)
    assert abs(line["new_order_weight"]) < abs(
        target
    )  # the shortfall, not a second full hedge
    nxt = r3[1]["hedge_netting"]["plan"]["lines"][0]
    assert nxt["held_weight"] != 0 and abs(nxt["new_order_weight"]) < abs(
        nxt["target_weight"]
    )  # held counted once next period


@pytest.mark.parametrize(
    "label,rows,text",
    [
        ("stale", risk_rows(end=dt.date(2024, 1, 10)), "STALE"),
        ("unknown", risk_rows(drop=("HEDGE",)), "UNKNOWN_EXPOSURE"),
        (
            "nan",
            [
                {**r, "returns": {**r["returns"], "AAA": float("nan")}}
                if i == 100
                else r
                for i, r in enumerate(risk_rows())
            ],
            "NONFINITE",
        ),
        ("short", risk_rows(start=dt.date(2024, 1, 20)), "TOO_SHORT"),
    ],
)
def test_bad_risk_inputs_fail_closed_before_any_fill(
    tmp_path, label, rows, text
):
    fails(tmp_path, label, text, rows=rows)


def test_future_row_injected_by_a_hostile_factory_is_rejected_and_malformed_records_too(
    tmp_path,
):
    def hostile(ctx, hook):
        inputs = default_coordination_inputs(ctx, hook)
        last = inputs.return_history[-1]
        return replace(
            inputs,
            return_history=inputs.return_history
            + (
                ReturnRow(
                    ctx.decision_at + dt.timedelta(hours=1), last.returns
                ),
            ),
        )

    hook = replace(
        coordination_hook_from_records(policy(), risk_rows()),
        input_factory=hostile,
    )
    fails(tmp_path, "hostile", "COVARIANCE_FUTURE_ROW", hook=hook)
    with pytest.raises(PodPlatformError):
        coordination_hook_from_records(
            policy(), [{"observed_at": "2024-01-01T00:00:00+00:00"}]
        )
    with pytest.raises(PodPlatformError):
        coordination_hook_from_records(
            policy(),
            [{"observed_at": "2024-01-01T00:00:00", "returns": {"AAA": 0.1}}],
        )  # naive time


def test_unavailable_or_unallowed_hedge_and_uncertified_pod_are_rejected(
    tmp_path,
):
    fails(tmp_path, "nohedgerow", "HEDGE_UNAVAILABLE", hedge_rows=False)
    fails(
        tmp_path,
        "notallowed",
        "not in the explicit hedge allowance",
        pol=policy(allow=False),
    )
    fails(
        tmp_path,
        "ghostpod",
        "no active certified forecast",
        pol=policy(requests=("fam_zzz",)),
    )
    scoped = (
        _certificate()
        .model_copy(
            update={
                "scope": CertificateScope(
                    environment=CertificateEnvironment.TEST,
                    namespaces=["ns_other"],
                )
            }
        )
        .sealed()
    )
    fails(
        tmp_path,
        "namespace",
        "namespace",
        certs=[scoped.model_dump(mode="json")],
        namespace="ns_a",
        research_only=True,
    )
    fails(tmp_path, "nocert", "", certs=[])


def test_stale_plan_is_never_executed_and_adapters_use_existing_contracts():
    inp = make_inputs([intention("p1")])
    res = coordinator().coordinate(inp)
    assert_plan_current(res, inp.snapshot, inp.all_hedge_requests())
    moved = snapshot(holdings=[("S01", 0.05)])
    with pytest.raises(PodPlatformError) as ei:
        assert_plan_current(res, moved, inp.all_hedge_requests())
    assert "STALE_COORDINATION_PLAN" in ei.value.response.summary
    order = SimpleNamespace(
        order_id="o1", security_id="S01", order_dollars=Decimal("-1000")
    )
    pending = planned_order_to_pending(order, Decimal("1000000"))
    assert pending.remaining_weight == pytest.approx(-0.001)
    assert (
        fill_state_to_feedback(
            order,
            SimpleNamespace(filled_dollars=Decimal("0")),
            Decimal("1000000"),
        )
        is None
    )
    fb = fill_state_to_feedback(
        order,
        SimpleNamespace(filled_dollars=Decimal("400")),
        Decimal("1000000"),
    )
    assert fb == FillFeedback("fill:o1", "o1", pytest.approx(-0.0004))
    snap = replace(inp.snapshot, pending_orders=(pending,))
    after = snap.apply_fills([fb], as_of=snap.as_of)
    assert after.pending_orders[0].remaining_weight == pytest.approx(-0.0006)
    assert {h.security_id: h.weight for h in after.holdings}[
        "S01"
    ] == pytest.approx(-0.0004)
    assert (
        after.apply_fills([fb], as_of=snap.as_of) == after
    )  # idempotent: no duplicate execution
    assert not res.is_current_for(after, inp.all_hedge_requests())


def _script():
    spec = importlib.util.spec_from_file_location(
        "run_coordinated_replay",
        Path(__file__).resolve().parent.parent
        / "scripts"
        / "run_coordinated_replay.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_operator_cli_runs_with_explicit_inputs_and_fails_closed(
    tmp_path, capsys
):
    root = build(tmp_path, "cli", declare=False)
    (tmp_path / "policy.json").write_text(json.dumps(policy()))
    (tmp_path / "risk.json").write_text(json.dumps(risk_rows()))
    cli, base = (
        _script(),
        {
            "fixture": str(root),
            "coordination_policy": str(tmp_path / "policy.json"),
        },
    )
    assert (
        cli.main(
            {
                **base,
                **{
                    "risk_inputs": str(tmp_path / "risk.json"),
                    "out": str(tmp_path / "cli_out"),
                },
            }
        )
        == 0
    )
    shown = json.loads(capsys.readouterr().out)
    assert (
        shown["coordination"]["mode"] == COORDINATION_MODE
        and shown["llm_called"] is False
    )
    assert load(tmp_path / "cli_out", "run_summary.json")["coordination"][
        "hashes"
    ]["coordination:policy"]
    assert (
        cli.main(
            {
                **base,
                **{
                    "risk_inputs": str(tmp_path / "missing.json"),
                    "out": str(tmp_path / "cli_bad"),
                },
            }
        )
        == 2
    )
    (tmp_path / "stale.json").write_text(
        json.dumps(risk_rows(end=dt.date(2024, 1, 10)))
    )
    assert (
        cli.main(
            {
                **base,
                **{
                    "risk_inputs": str(tmp_path / "stale.json"),
                    "out": str(tmp_path / "cli_stale"),
                },
            }
        )
        == 2
    )
    assert (tmp_path / "cli_stale" / "failure.json").exists()
