"""Full supplied-input access must retain causal clocks and explicit source limitations."""

import csv
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from pod_platform.application.research_access import describe_research_access
from pod_platform.research.filing_access import FilingResearchStore
from pod_platform.research.input_permissions import (
    COMPETITION_CHARACTERISTICS,
    FILING_RESEARCH_FEATURES,
)
from pod_platform.research.pod_plans import SignalTerm
from pod_platform.research.training_data import prepare_panel


def _filings(path):
    records = [
        ("a", 1, "2019-01-10", None, 100, False, "First disclosure."),
        (
            "b",
            1,
            "2019-01-20",
            "2019-01-21T15:00:00+00:00",
            200,
            False,
            "Accepted after the nominal filing date.",
        ),
        (
            "amendment",
            1,
            "2019-01-15",
            "2019-01-15T19:00:00+00:00",
            50,
            True,
            "Amendment.",
        ),
        (
            "later_company",
            2,
            "2019-03-01",
            None,
            30,
            False,
            "This does not establish January coverage.",
        ),
        (
            "unmapped",
            None,
            "2019-01-12",
            None,
            20,
            False,
            "Unmapped text remains research-accessible.",
        ),
        (
            "future",
            1,
            "2021-01-01",
            None,
            999999,
            False,
            "This document must never reach training.",
        ),
    ]
    rows = []
    for identity, permno, date, accepted, words, amendment, text in records:
        rows.append(
            {
                "document_id": identity,
                "permno": permno,
                "filing_date": dt.date.fromisoformat(date),
                "filing_timestamp_utc": pd.Timestamp(
                    date, tz="America/New_York"
                ).to_pydatetime(),
                "sec_accepted_at_utc": pd.Timestamp(accepted).to_pydatetime()
                if accepted
                else None,
                "filing_time_precision": "date_only",
                "is_amendment": amendment,
                "word_count": words,
                "form_type": "8-K/A" if amendment else "8-K",
                "text": text,
                "company_name": "Research company " + str(permno),
            }
        )
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=2)
    return rows


def test_all_147_official_characteristics_are_available():
    assert (
        len(COMPETITION_CHARACTERISTICS)
        == len(set(COMPETITION_CHARACTERISTICS))
        == 147
    )
    dictionary = (
        Path(__file__).resolve().parents[2] / "data/factor_char_list.csv"
    )
    if dictionary.exists():
        with dictionary.open() as handle:
            expected = [row["variable"] for row in csv.DictReader(handle)]
        assert list(COMPETITION_CHARACTERISTICS) == expected
    for feature in (
        "dsale_dinv",
        "dsale_dsga",
        "qmj",
        "ret_60_12",
        "filings_8k_count_90d",
    ):
        assert (
            SignalTerm(
                feature=feature,
                other_feature=None,
                operation="identity",
                sign=1,
            ).feature
            == feature
        )
    with pytest.raises(ValidationError):
        SignalTerm(
            feature="ret_exc_lead1m",
            other_feature=None,
            operation="identity",
            sign=1,
        )


def test_filing_text_obeys_publication_time_and_keeps_unmapped_docs(tmp_path):
    path = tmp_path / "filings.parquet"
    _filings(path)
    store = FilingResearchStore(path)
    assert store.query("2019-01-10T12:00:00-05:00").empty
    documents = store.query("2019-01-10", include_text=True)
    assert documents.document_id.tolist() == ["a"]
    assert documents.text.iloc[0] == "First disclosure."
    assert documents.company_name.iloc[0] == "Research company 1"
    assert documents.source_research_only.all()
    assert not documents.historical_identity_verified.any()
    assert "b" not in store.query("2019-01-20").document_id.tolist()
    assert "b" in store.query("2019-01-21T16:00:00+00:00").document_id.tolist()
    assert "unmapped" in store.query("2019-01-31").document_id.tolist()
    assert "future" not in store.query("2020-12-31").document_id.tolist()
    with pytest.raises(ValueError, match="training"):
        store.query("2021-01-01")


def test_filing_pages_cover_the_whole_authorized_corpus(tmp_path):
    path = tmp_path / "filings.parquet"
    _filings(path)
    store = FilingResearchStore(path)
    all_documents = store.query("2020-12-31", limit=200)
    paged = pd.concat(
        [
            store.query("2020-12-31", offset=n, limit=1)
            for n in range(len(all_documents))
        ]
    )
    assert paged.document_id.tolist() == all_documents.document_id.tolist()
    preview = store.query("2019-01-10", max_text_chars=5)
    assert preview.text.iloc[0] == "First"
    assert preview.text_preview_truncated.iloc[0]


def test_filing_features_do_not_condition_on_future_coverage(tmp_path):
    path = tmp_path / "filings.parquet"
    rows = _filings(path)
    dates = pd.to_datetime(["2019-01-31", "2019-02-28", "2019-03-31"])
    original = FilingResearchStore(path).feature_arrays(
        dates, np.array([1, 2, 3])
    )
    assert original["filings_8k_count_30d"][0, 0] == 3
    assert original["filings_8k_words_90d"][0, 0] == 350
    assert original["filings_8k_amendment_share_365d"][0, 0] == pytest.approx(
        1 / 3
    )
    assert np.isnan(original["filings_8k_count_30d"][:2, 1]).all()
    assert np.isnan(original["filings_8k_count_30d"][:, 2]).all()
    assert original["filings_8k_count_30d"][2, 0] == 0
    rows[-1]["word_count"] = 10**12
    rows[-1]["text"] = "Changed future text."
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=2)
    changed = FilingResearchStore(path).feature_arrays(
        dates, np.array([1, 2, 3])
    )
    for feature in original:
        np.testing.assert_array_equal(original[feature], changed[feature])


def test_active_panel_loads_all_features_and_filing_inputs(tmp_path):
    filing_path = tmp_path / "filings.parquet"
    _filings(filing_path)
    dates = pd.date_range("2019-01-31", periods=3, freq="ME")
    rows = []
    for day in dates:
        for permno in (1, 2, 3):
            row = {name: 0.2 for name in COMPETITION_CHARACTERISTICS}
            row.update(
                permno=permno,
                eom=day,
                me=1000,
                prc=10,
                dolvol_126d=1e7,
                gics="10",
            )
            rows.append(row)
    panel_path = tmp_path / "panel.parquet"
    pd.DataFrame(rows).to_parquet(panel_path, index=False)
    daily_path = tmp_path / "daily.csv"
    pd.DataFrame(
        [
            {
                "permno": p,
                "date": int(d.strftime("%Y%m%d")),
                "prc": 10,
                "daily_return": 0,
            }
            for d in pd.bdate_range("2019-01-01", "2019-03-31")
            for p in (1, 2, 3)
        ]
    ).to_csv(daily_path, index=False)
    panel = prepare_panel(
        panel_path,
        daily_path,
        {
            "start": "2019-01-01",
            "end": "2019-03-31",
            "min_market_cap_millions": 0,
            "min_price": 0,
            "min_daily_dollar_volume": 0,
            "min_positions": 3,
            "daily_warmup_start": "2019-01-01",
            "filings_research_enabled": True,
            "filings_path": str(filing_path),
        },
        tmp_path / "cache.parquet",
    )
    assert set(COMPETITION_CHARACTERISTICS) | set(
        FILING_RESEARCH_FEATURES
    ) == set(panel.features)
    assert panel.metadata["filing_research"]["raw_text_query_available"]
    assert np.isnan(panel.features["filings_8k_count_30d"][0, 1])
    audit = describe_research_access(
        {
            "application": {"mode": "edge_diagnostics"},
            "training_lab": {"filings_research_enabled": True},
        },
        panel.metadata,
    )
    assert audit["sources"]["supplied_characteristics"]["count"] == 147
    assert audit["sources"]["supplied_8k"]["metadata_and_features_enabled"]
