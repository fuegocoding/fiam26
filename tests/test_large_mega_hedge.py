import pandas as pd
import pytest

from src.large_mega_hedge import badness_scores, combine_weights, one_way_cost


def test_badness_direction_is_distress_issuance_and_weak_payout():
    frame = pd.DataFrame({
        "target_month": [pd.Period("2025-01", "M")] * 3,
        "sector": ["10"] * 3,
        "o_score": [1.0, 2.0, 3.0],
        "chcsho_12m": [1.0, 2.0, 3.0],
        "eqnpo_12m": [3.0, 2.0, 1.0],
    })
    score = badness_scores(frame)
    assert score.iloc[2] > score.iloc[1] > score.iloc[0]


def test_replacement_preserves_half_notional_on_each_side():
    combined = combine_weights({1: 0.5}, {2: 0.5}, {3: 1.0}, 0.25)
    assert combined[1] == pytest.approx(0.5)
    assert combined[2] == pytest.approx(-0.375)
    assert combined[3] == pytest.approx(-0.125)
    assert sum(v for v in combined.values() if v > 0) == pytest.approx(0.5)
    assert -sum(v for v in combined.values() if v < 0) == pytest.approx(0.5)


def test_cost_uses_last_tier_when_a_position_exits():
    cost = one_way_cost({}, {7: -0.25}, {}, {7: "nano"})
    assert cost == pytest.approx(0.25 * 60.0 / 1e4)
