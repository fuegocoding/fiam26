"""Ported strategy signals from the first-round sleeves, on competition data.

Exact ports where the input is the chars panel (inventory_retail,
oplev_software, runway_micro). PIT-disciplined ports of the IBES sleeves
(e7 surprise+breadth, conf_lm_gated revisions) built from the same
ibes_surprise / ibes_consensus / ibes_link tables with the same clock rules:
announcements only after their report month, consensus snapshots only on or
before the month end, dated ticker links with SCORE <= 2.

Every score is the old sleeve's formula, re-expressed through the platform's
registered feature names.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

MAX_LINK_SCORE = 2


@dataclass(frozen=True)
class SignalResult:
    name: str
    frame: pd.DataFrame
    eligible: pd.Series


TABLE_COLUMNS = {
    "ibes_link": ["TICKER", "PERMNO", "sdate", "edate", "SCORE"],
    "ibes_surprise": ["TICKER", "anndats", "suescore", "actual"],
    "ibes_consensus": ["TICKER", "STATPERS", "FPEDATS", "NUMUP", "NUMDOWN", "NUMEST", "ANNDATS_ACT", "MEANEST", "FPI"],
    "ibes_price": ["TICKER", "STATPERS", "PRICE"],
}


def _load_external(path, table: str) -> pd.DataFrame:
    ext = pd.read_parquet(path, columns=TABLE_COLUMNS[table] + ["table"])
    return ext[ext["table"] == table][TABLE_COLUMNS[table]].reset_index(drop=True)


class PortedSignals:
    def __init__(self, external_path):
        self._external_path = external_path
        self._cache: dict[str, pd.DataFrame] = {}

    def _table(self, table: str) -> pd.DataFrame:
        if table not in self._cache:
            self._cache[table] = _load_external(self._external_path, table)
        return self._cache[table]

    def inventory_retail(self, month: pd.DataFrame) -> SignalResult:
        frame = month.assign(score=month["dsale_dinv"])
        eligible = (
            month["size_grp"].isin(["small", "large", "mega"])
            & month["gics"].astype("string").str[:4].isin(["2550", "2520"])
        )
        return SignalResult("inventory_retail", frame, eligible.where(eligible, False))

    def oplev_software(self, month: pd.DataFrame) -> SignalResult:
        frame = month.assign(score=month["dsale_dsga"])
        eligible = (
            month["size_grp"].isin(["small", "large", "mega"])
            & month["gics"].astype("string").str[:4].eq("4510")
        )
        return SignalResult("oplev_software", frame, eligible.where(eligible, False))

    def runway_micro(self, month: pd.DataFrame) -> SignalResult:
        frame = month.assign(
            score=(month["cash_at"] / -month["ocf_at"]).where(month["ocf_at"] < 0)
        )
        eligible = month["size_grp"].eq("micro") & frame["score"].notna()
        return SignalResult("runway_micro", frame, eligible.where(eligible, False))

    def _load_link(self) -> pd.DataFrame:
        link = self._table("ibes_link")
        link = link.dropna(subset=["PERMNO"])
        link = link[link["SCORE"] <= MAX_LINK_SCORE].copy()
        link["permno"] = link["PERMNO"].astype(int)
        link["sdate"] = pd.to_datetime(link["sdate"])
        link["edate"] = pd.to_datetime(link["edate"])
        return link[["TICKER", "permno", "sdate", "edate"]]

    def _ticker_for(self, month: pd.DataFrame, eom: pd.Timestamp) -> pd.Series:
        link = self._load_link()
        grid = month[["permno"]].copy()
        grid["eom"] = eom
        merged = grid.merge(link, on="permno", how="left")
        valid = merged[(merged["sdate"] <= eom) & (merged["edate"] >= eom)]
        return valid.drop_duplicates("permno").set_index("permno")["TICKER"]

    def e7_surprise_breadth(self, month: pd.DataFrame, eom: pd.Timestamp) -> SignalResult:
        ticker = self._ticker_for(month, eom)
        surprises = self._table("ibes_surprise")
        surprises["anndats"] = pd.to_datetime(surprises["anndats"])
        announced = surprises[surprises["anndats"] <= eom].copy()
        announced["report_eom"] = announced["anndats"] + pd.offsets.MonthEnd(1)
        delayed = announced[announced["report_eom"] == eom]
        latest = delayed.sort_values("anndats").groupby("TICKER").tail(1)

        consensus = self._table("ibes_consensus")
        consensus["STATPERS"] = pd.to_datetime(consensus["STATPERS"])
        consensus["FPEDATS"] = pd.to_datetime(consensus["FPEDATS"])
        post_rows = latest[["TICKER", "anndats"]].merge(consensus, on="TICKER")
        post_rows = post_rows[
            (post_rows["STATPERS"] > post_rows["anndats"]) & (post_rows["STATPERS"] <= eom)
        ]
        post = (
            post_rows.sort_values("STATPERS")
            .groupby("TICKER")
            .head(1)
            .assign(post_breadth=lambda d: (d["NUMUP"] - d["NUMDOWN"]) / d["NUMEST"].where(d["NUMEST"] > 0))
        )

        joined = latest[["TICKER", "suescore"]].merge(
            post[["TICKER", "post_breadth"]], on="TICKER", how="left"
        )
        joined["score"] = joined["suescore"].fillna(0.0) + joined["post_breadth"].fillna(0.0)
        by_ticker = joined.set_index("TICKER")["score"]
        frame = month.copy()
        mapped = frame["permno"].map(ticker)
        frame["score"] = mapped.map(by_ticker)
        eligible = mapped.isin(by_ticker.index) & frame["score"].notna()
        frame["score"] = frame["score"].fillna(0.0)
        return SignalResult("e7_surprise_breadth", frame, pd.Series(eligible.values, index=frame.index))

    def conf_revision_gated(self, month: pd.DataFrame, eom: pd.Timestamp) -> SignalResult:
        ticker = self._ticker_for(month, eom)
        consensus = self._table("ibes_consensus")
        consensus = consensus[consensus["FPI"].astype(str).isin(["6", "7", "8", "9"])].copy()
        consensus["STATPERS"] = pd.to_datetime(consensus["STATPERS"])
        consensus["FPEDATS"] = pd.to_datetime(consensus["FPEDATS"])
        consensus["ANNDATS_ACT"] = pd.to_datetime(consensus["ANNDATS_ACT"])
        announced = consensus["ANNDATS_ACT"].notna() & (consensus["ANNDATS_ACT"] <= consensus["STATPERS"])
        consensus = consensus.loc[~announced]
        consensus = consensus[consensus["STATPERS"] <= eom].copy()
        consensus["snap_month"] = consensus["STATPERS"].dt.to_period("M")
        last = consensus.groupby(["TICKER", "snap_month"])["STATPERS"].transform("max")
        consensus = consensus[consensus["STATPERS"] == last]

        prices = self._table("ibes_price")
        prices["STATPERS"] = pd.to_datetime(prices["STATPERS"])
        consensus = consensus.merge(prices, on=["TICKER", "STATPERS"], how="inner")

        lagged = consensus[["TICKER", "FPEDATS", "snap_month", "MEANEST"]].copy()
        lagged["snap_month"] = lagged["snap_month"] + 3
        lagged = lagged.rename(columns={"MEANEST": "lag3"})
        consensus = consensus.merge(lagged, on=["TICKER", "FPEDATS", "snap_month"], how="left")
        consensus["rev3_near"] = (consensus["MEANEST"] - consensus["lag3"]) / consensus["PRICE"]

        near = (
            consensus.sort_values(["TICKER", "snap_month", "FPEDATS"])
            .drop_duplicates(["TICKER"], keep="last")
        )
        rev_by_ticker = near.dropna(subset=["rev3_near"]).set_index("TICKER")["rev3_near"]

        frame = month.copy()
        frame["rev3_near"] = frame["permno"].map(ticker).map(rev_by_ticker)
        agree = (
            frame["ret_3_1"].apply(lambda v: 1 if v > 0 else (-1 if v < 0 else 0))
            == frame["rev3_near"].apply(lambda v: 1 if v > 0 else (-1 if v < 0 else 0))
        ) & frame["rev3_near"].ne(0) & frame["rev3_near"].notna()
        frame["score"] = frame["rev3_near"].where(agree)
        eligible = (
            agree
            & month["size_grp"].isin(["large", "mega"])
            & frame["score"].notna()
        )
        return SignalResult("conf_revision_gated", frame, eligible.where(eligible, False))


def scaled_rank(result: SignalResult) -> pd.Series:
    scores = result.frame["score"]
    ranked = scores[result.eligible.reindex(scores.index).fillna(False).astype(bool)].rank(pct=True)
    scaled = (ranked - 0.5) * (12 ** 0.5)
    out = pd.Series(0.0, index=result.frame.index)
    out.loc[scaled.index] = scaled.fillna(0.0)
    return out
