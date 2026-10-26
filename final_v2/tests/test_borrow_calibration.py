"""Current cost calibration must keep fee units, tails and outcome isolation."""

import pandas as pd
import pytest

from pod_platform.application.common import atomic_write_json, file_sha256
from pod_platform.research import borrow_calibration as module
from verify_campaign import verify_cost_lineage


def test_quote_calibration_reads_only_identifiers_size_and_retains_expensive_tail(
    tmp_path, monkeypatch
):
    stocks = pd.DataFrame(
        {
            "permno": range(240),
            "eom": pd.Timestamp("2026-08-31"),
            "me": [20000] * 120 + [2000] * 120,
            "ticker": [f"T{i}" for i in range(240)],
            "ret_exc_lead1m": [999.0] * 240,
        }
    )
    panel = tmp_path / "panel.parquet"
    stocks.to_parquet(panel, index=False)
    quotes = tmp_path / "quotes.txt"
    quotes.write_text(
        "#BOF|2026.10.26|11:45:39\n#SYM|CUR|NAME|FEERATE|AVAILABLE|\n"
        + "".join(
            f"T{i}|USD|Issuer{i}|{0.25 if i < 120 else 0.75 if i < 239 else 30}|0|\n"
            for i in range(240)
        )
    )
    original = module.pq.read_table

    def observed(path, **kwargs):
        assert set(kwargs["columns"]) == {"permno", "eom", "me", "ticker"}
        return original(path, **kwargs)

    monkeypatch.setattr(module.pq, "read_table", observed)
    result = module.calibrate_borrow(quotes, panel, "2026-08-31")
    assert result["return_columns_read"] == []
    assert result["parameters"] == {
        "large_cap_borrow_bps": 25,
        "other_borrow_bps": 100,
    }
    assert result["matched_groups"]["other_eligible"][
        "mean_bps"
    ] == pytest.approx((119 * 75 + 3000) / 120)
    assert result["matched_groups"]["other_eligible"]["names"] == 120


def test_saved_cost_model_cannot_disagree_with_execution_fees_or_report(
    tmp_path,
):
    receipt = {
        "parameters": {"large_cap_borrow_bps": 50, "other_borrow_bps": 100}
    }
    path = tmp_path / "borrow_fee_calibration.json"
    atomic_write_json(path, receipt)
    experiment = {
        "borrow_fee_calibration": receipt,
        "settings": {
            **receipt["parameters"],
            "borrow_fee_calibration_sha256": file_sha256(path),
        },
    }
    report = {"borrow_fee_model": receipt}
    assert verify_cost_lineage(tmp_path, experiment, report)
    experiment["settings"]["other_borrow_bps"] = 500
    with pytest.raises(ValueError, match="execution fees"):
        verify_cost_lineage(tmp_path, experiment, report)
    experiment["settings"]["other_borrow_bps"] = 100
    report["borrow_fee_model"] = {"parameters": {"other_borrow_bps": 25}}
    with pytest.raises(ValueError, match="lineage differs"):
        verify_cost_lineage(tmp_path, experiment, report)
