import pandas as pd
import pytest

from src.capacity_constraints import CapacityConfig, apply_capacity_constraints


def _rows(**overrides):
    data = {
        "permno": [1],
        "target_weight": [0.10],
        "dolvol_126d": [1_000_000.0],
        "prc": [10.0],
    }
    data.update(overrides)
    return pd.DataFrame(data)


def test_long_is_capped_in_dollars_from_its_own_adv():
    config = CapacityConfig(nav=10_000_000)

    result = apply_capacity_constraints(_rows(), {}, config).iloc[0]

    # 5% participation x $1m ADV x 5 days = $250k capacity.
    assert result["requested_dollars"] == pytest.approx(1_000_000)
    assert result["position_cap_dollars"] == pytest.approx(250_000)
    assert result["final_dollars"] == pytest.approx(250_000)
    assert result["final_weight"] == pytest.approx(0.025)
    assert result["unfilled_dollars"] == pytest.approx(750_000)
    assert result["bind_reason"] == "long_liquidity"


def test_short_uses_tighter_of_adv_capacity_and_confirmed_locate():
    config = CapacityConfig(nav=10_000_000, require_locate=True)
    rows = _rows(target_weight=[-0.10], locate_dollars=[60_000.0])

    result = apply_capacity_constraints(rows, {}, config).iloc[0]

    # The short market-volume cap is $100k, but only $60k is located.
    assert result["market_position_cap_dollars"] == pytest.approx(100_000)
    assert result["position_cap_dollars"] == pytest.approx(60_000)
    assert result["final_dollars"] == pytest.approx(-60_000)
    assert result["borrow_status"] == "confirmed"
    assert result["bind_reason"] == "locate"


def test_live_mode_blocks_a_short_without_a_confirmed_locate():
    config = CapacityConfig(nav=10_000_000, require_locate=True)

    result = apply_capacity_constraints(
        _rows(target_weight=[-0.10]), {}, config
    ).iloc[0]

    assert result["final_dollars"] == 0
    assert result["borrow_status"] == "blocked_missing"
    assert result["bind_reason"] == "missing_locate"


def test_research_mode_allows_short_volume_proxy_but_flags_it():
    config = CapacityConfig(nav=10_000_000, require_locate=False)

    result = apply_capacity_constraints(
        _rows(target_weight=[-0.10]), {}, config
    ).iloc[0]

    assert result["final_dollars"] == pytest.approx(-100_000)
    assert result["borrow_status"] == "proxy_unverified"
    assert result["bind_reason"] == "short_liquidity"


def test_rebalance_capacity_applies_to_change_not_full_holding():
    config = CapacityConfig(nav=10_000_000)
    rows = _rows(target_weight=[0.025])

    result = apply_capacity_constraints(rows, {1: 200_000.0}, config).iloc[0]

    assert result["requested_dollars"] == pytest.approx(250_000)
    assert result["requested_trade_dollars"] == pytest.approx(50_000)
    assert result["executed_trade_dollars"] == pytest.approx(50_000)
    assert result["final_dollars"] == pytest.approx(250_000)
    assert result["bind_reason"] == "none"


def test_sign_flip_uses_remaining_exit_budget_for_opposite_side():
    config = CapacityConfig(nav=10_000_000)
    rows = _rows(target_weight=[-0.10], locate_dollars=[100_000.0])

    result = apply_capacity_constraints(rows, {1: 200_000.0}, config).iloc[0]

    # The $250k close/trade budget first exits the $200k long, leaving $50k
    # available to initiate the located short in the same execution window.
    assert result["final_dollars"] == pytest.approx(-50_000)
    assert result["executed_trade_dollars"] == pytest.approx(-250_000)
    assert result["unfilled_dollars"] == pytest.approx(-950_000)
    assert result["bind_reason"] == "crossing_trade_capacity"


def test_missing_adv_blocks_new_capacity():
    config = CapacityConfig(nav=10_000_000)

    result = apply_capacity_constraints(
        _rows(dolvol_126d=[float("nan")]), {}, config
    ).iloc[0]

    assert result["final_dollars"] == 0
    assert result["bind_reason"] == "missing_adv"


def test_expensive_borrow_blocks_short_when_fee_limit_is_set():
    config = CapacityConfig(
        nav=10_000_000, require_locate=True, max_borrow_fee_bps=500
    )
    rows = _rows(
        target_weight=[-0.10], locate_dollars=[100_000.0],
        borrow_fee_bps=[750.0],
    )

    result = apply_capacity_constraints(rows, {}, config).iloc[0]

    assert result["final_dollars"] == 0
    assert result["borrow_status"] == "fee_blocked"
    assert result["bind_reason"] == "borrow_fee"


def test_missing_locate_does_not_block_reducing_an_existing_short():
    config = CapacityConfig(nav=10_000_000, require_locate=True)
    rows = _rows(target_weight=[-0.002])

    result = apply_capacity_constraints(rows, {1: -200_000.0}, config).iloc[0]

    assert result["requested_dollars"] == pytest.approx(-20_000)
    # Covering uses the long/ordinary market-impact allowance, not the stricter
    # fresh-short allowance, and it does not need a new locate.
    assert result["executed_trade_dollars"] == pytest.approx(180_000)
    assert result["final_dollars"] == pytest.approx(-20_000)
    assert result["borrow_status"] == "blocked_missing"
    assert result["bind_reason"] == "none"


def test_fee_block_does_not_block_reducing_an_existing_short():
    config = CapacityConfig(
        nav=10_000_000, require_locate=True, max_borrow_fee_bps=500
    )
    rows = _rows(
        target_weight=[-0.002], locate_dollars=[100_000.0],
        borrow_fee_bps=[750.0],
    )

    result = apply_capacity_constraints(rows, {1: -80_000.0}, config).iloc[0]

    assert result["executed_trade_dollars"] == pytest.approx(60_000)
    assert result["final_dollars"] == pytest.approx(-20_000)
    assert result["borrow_status"] == "fee_blocked"
    assert result["bind_reason"] == "none"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"nav": 0},
        {"nav": 1, "execution_days": 0},
        {"nav": 1, "long_participation": 1.1},
        {"nav": 1, "short_participation": -0.1},
    ],
)
def test_invalid_capacity_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        CapacityConfig(**kwargs)
