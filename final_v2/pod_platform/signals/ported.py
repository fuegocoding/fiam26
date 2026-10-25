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
from pathlib import Path

import numpy as np
import pandas as pd

from . import ibes_state

MAX_LINK_SCORE = 2


@dataclass(frozen=True)
class SignalResult:
    name: str
    frame: pd.DataFrame
    eligible: pd.Series


TABLE_COLUMNS = {
    "ibes_link": ["TICKER", "PERMNO", "sdate", "edate", "SCORE"],
    "ibes_surprise": ["TICKER", "anndats", "suescore", "actual"],
    "ibes_consensus": [
        "TICKER",
        "STATPERS",
        "FPEDATS",
        "NUMUP",
        "NUMDOWN",
        "NUMEST",
        "ANNDATS_ACT",
        "MEANEST",
        "FPI",
    ],
    "ibes_price": ["TICKER", "STATPERS", "PRICE"],
}


def _load_external(path, table: str) -> pd.DataFrame:
    ext = pd.read_parquet(path, columns=TABLE_COLUMNS[table] + ["table"])
    return ext[ext["table"] == table][TABLE_COLUMNS[table]].reset_index(
        drop=True
    )


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
        eligible = month["size_grp"].isin(["small", "large", "mega"]) & month[
            "gics"
        ].astype("string").str[:4].isin(["2550", "2520"])
        return SignalResult(
            "inventory_retail", frame, eligible.where(eligible, False)
        )

    def oplev_software(self, month: pd.DataFrame) -> SignalResult:
        frame = month.assign(score=month["dsale_dsga"])
        eligible = month["size_grp"].isin(["small", "large", "mega"]) & month[
            "gics"
        ].astype("string").str[:4].eq("4510")
        return SignalResult(
            "oplev_software", frame, eligible.where(eligible, False)
        )

    def runway_micro(self, month: pd.DataFrame) -> SignalResult:
        frame = month.assign(
            score=(month["cash_at"] / -month["ocf_at"]).where(
                month["ocf_at"] < 0
            )
        )
        eligible = month["size_grp"].eq("micro") & frame["score"].notna()
        return SignalResult(
            "runway_micro", frame, eligible.where(eligible, False)
        )

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

    def e7_surprise_breadth(
        self, month: pd.DataFrame, eom: pd.Timestamp
    ) -> SignalResult:
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
            (post_rows["STATPERS"] > post_rows["anndats"])
            & (post_rows["STATPERS"] <= eom)
        ]
        post = (
            post_rows.sort_values("STATPERS")
            .groupby("TICKER")
            .head(1)
            .assign(
                post_breadth=lambda d: (
                    (d["NUMUP"] - d["NUMDOWN"])
                    / d["NUMEST"].where(d["NUMEST"] > 0)
                )
            )
        )

        joined = latest[["TICKER", "suescore"]].merge(
            post[["TICKER", "post_breadth"]], on="TICKER", how="left"
        )
        joined["score"] = joined["suescore"].fillna(0.0) + joined[
            "post_breadth"
        ].fillna(0.0)
        by_ticker = joined.set_index("TICKER")["score"]
        frame = month.copy()
        mapped = frame["permno"].map(ticker)
        frame["score"] = mapped.map(by_ticker)
        eligible = mapped.isin(by_ticker.index) & frame["score"].notna()
        frame["score"] = frame["score"].fillna(0.0)
        return SignalResult(
            "e7_surprise_breadth",
            frame,
            pd.Series(eligible.values, index=frame.index),
        )

    def conf_revision_gated(
        self, month: pd.DataFrame, eom: pd.Timestamp
    ) -> SignalResult:
        ticker = self._ticker_for(month, eom)
        consensus = self._table("ibes_consensus")
        consensus = consensus[
            consensus["FPI"].astype(str).isin(["6", "7", "8", "9"])
        ].copy()
        consensus["STATPERS"] = pd.to_datetime(consensus["STATPERS"])
        consensus["FPEDATS"] = pd.to_datetime(consensus["FPEDATS"])
        consensus["ANNDATS_ACT"] = pd.to_datetime(consensus["ANNDATS_ACT"])
        announced = consensus["ANNDATS_ACT"].notna() & (
            consensus["ANNDATS_ACT"] <= consensus["STATPERS"]
        )
        consensus = consensus.loc[~announced]
        consensus = consensus[consensus["STATPERS"] <= eom].copy()
        consensus["snap_month"] = consensus["STATPERS"].dt.to_period("M")
        last = consensus.groupby(["TICKER", "snap_month"])[
            "STATPERS"
        ].transform("max")
        consensus = consensus[consensus["STATPERS"] == last]

        prices = self._table("ibes_price")
        prices["STATPERS"] = pd.to_datetime(prices["STATPERS"])
        consensus = consensus.merge(
            prices, on=["TICKER", "STATPERS"], how="inner"
        )

        lagged = consensus[
            ["TICKER", "FPEDATS", "snap_month", "MEANEST"]
        ].copy()
        lagged["snap_month"] = lagged["snap_month"] + 3
        lagged = lagged.rename(columns={"MEANEST": "lag3"})
        consensus = consensus.merge(
            lagged, on=["TICKER", "FPEDATS", "snap_month"], how="left"
        )
        consensus["rev3_near"] = (
            consensus["MEANEST"] - consensus["lag3"]
        ) / consensus["PRICE"]

        near = consensus.sort_values(
            ["TICKER", "snap_month", "FPEDATS"]
        ).drop_duplicates(["TICKER"], keep="last")
        rev_by_ticker = near.dropna(subset=["rev3_near"]).set_index("TICKER")[
            "rev3_near"
        ]

        frame = month.copy()
        frame["rev3_near"] = frame["permno"].map(ticker).map(rev_by_ticker)
        agree = (
            (
                frame["ret_3_1"].apply(
                    lambda v: 1 if v > 0 else (-1 if v < 0 else 0)
                )
                == frame["rev3_near"].apply(
                    lambda v: 1 if v > 0 else (-1 if v < 0 else 0)
                )
            )
            & frame["rev3_near"].ne(0)
            & frame["rev3_near"].notna()
        )
        frame["score"] = frame["rev3_near"].where(agree)
        eligible = (
            agree
            & month["size_grp"].isin(["large", "mega"])
            & frame["score"].notna()
        )
        return SignalResult(
            "conf_revision_gated", frame, eligible.where(eligible, False)
        )


def scaled_rank(result: SignalResult) -> pd.Series:
    scores = result.frame["score"]
    ranked = scores[
        result.eligible.reindex(scores.index).fillna(False).astype(bool)
    ].rank(pct=True)
    scaled = (ranked - 0.5) * (12**0.5)
    out = pd.Series(0.0, index=result.frame.index)
    out.loc[scaled.index] = scaled.fillna(0.0)
    return out


def scale_scores_broad_native(
    broad: pd.DataFrame, native: pd.DataFrame
) -> pd.DataFrame:
    """Uniform monthly ranks; non-native stocks are ranked against natives (empirical CDF)."""
    broad = broad.copy()
    broad["broad_score"] = 0.0
    references = {
        month: np.sort(group["score"].to_numpy(float))
        for month, group in native.groupby("return_month", sort=False)
    }
    by_month = broad.groupby("return_month", sort=False).indices
    for month, positions in by_month.items():
        values = broad.iloc[positions]["score"].to_numpy(float)
        reference = references.get(month)
        if reference is None or not len(reference):
            ranks = pd.Series(values).rank(pct=True).to_numpy(float)
        else:
            below = np.searchsorted(reference, values, side="left")
            above = np.searchsorted(reference, values, side="right")
            ranks = (below + above) / (2.0 * len(reference))
        broad.iloc[positions, broad.columns.get_loc("broad_score")] = (
            ranks - 0.5
        ) * np.sqrt(12.0)
    native = native.copy()
    groups = native.groupby("return_month")["score"]
    native["native_score"] = (
        (groups.rank(pct=True) - 0.5) * np.sqrt(12.0)
    ).where(groups.transform("size") > 1, 0.0)
    return broad[["return_month", "permno", "broad_score"]].merge(
        native[["return_month", "permno", "native_score"]],
        on=["return_month", "permno"],
        how="left",
    )


def build_e7_delayed_panel(
    chars: pd.DataFrame,
    external_path: str | Path,
    cache_path: Path | None = None,
) -> pd.DataFrame:
    """Delayed-reporter panel of price-scaled earnings surprise and post-report breadth."""
    if cache_path is not None and cache_path.exists():
        df = pd.read_parquet(cache_path)
        df["eom"] = pd.to_datetime(df["eom"])
        return df
    ibes_state.set_external_path(str(external_path))
    state = ibes_state.build_expectation_state(chars[["permno", "eom"]])
    state["eom"] = pd.to_datetime(state["eom"])
    events = state.dropna(subset=["target_ann", "target_surprise_p"])
    events = events.loc[events["target_ann"] > events["eom"]]
    events = (
        events.sort_values("eom")
        .drop_duplicates(["permno", "target_fpe"], keep="last")
        .copy()
    )
    events["report_eom"] = events["eom"] + pd.offsets.MonthEnd(1)
    events = events.loc[events["target_ann"] <= events["report_eom"]].rename(
        columns={"eom": "forecast_eom", "target_surprise_p": "sue"}
    )
    consensus = ibes_state.load_consensus()[
        ["TICKER", "STATPERS", "FPEDATS", "NUMUP", "NUMDOWN", "NUMEST"]
    ]
    key = ["permno", "target_fpe"]
    snapshots = consensus.sort_values("STATPERS").set_index(
        "STATPERS", drop=False
    )
    formation = ibes_state.month_end(
        events["target_ann"]
    ) + pd.offsets.MonthEnd(1)
    parts = []
    for cutoff, rows in events.groupby(formation, sort=True):
        known = snapshots.loc[rows["target_ann"].min() : cutoff].reset_index(
            drop=True
        )
        pairs = rows[["permno", "TICKER", "target_fpe", "target_ann"]].merge(
            known, on="TICKER"
        )
        following = pairs.loc[
            (pairs["FPEDATS"] > pairs["target_fpe"])
            & (pairs["FPEDATS"] <= pairs["target_fpe"] + pd.offsets.MonthEnd(4))
            & (pairs["STATPERS"] > pairs["target_ann"])
        ]
        post = following.loc[
            following["FPEDATS"]
            == following.groupby(key)["FPEDATS"].transform("min")
        ]
        parts.append(post.sort_values("STATPERS").groupby(key).head(1))
    post = pd.concat(parts, ignore_index=True)
    post["post_breadth"] = (post["NUMUP"] - post["NUMDOWN"]) / post[
        "NUMEST"
    ].where(post["NUMEST"] > 0)
    events = events.merge(
        post[key + ["post_breadth", "STATPERS"]].rename(
            columns={"STATPERS": "post_snapshot"}
        ),
        on=key,
        how="left",
    )
    delayed = events.assign(eom=events["report_eom"] + pd.offsets.MonthEnd(1))
    delayed = delayed.loc[delayed["post_snapshot"] <= delayed["eom"]].merge(
        chars[["permno", "eom", "sector", "size_grp", "tradable"]],
        on=["permno", "eom"],
        how="inner",
    )
    columns = [
        "permno",
        "eom",
        "sector",
        "size_grp",
        "tradable",
        "sue",
        "post_breadth",
    ]
    out = delayed[columns].reset_index(drop=True)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(cache_path, index=False)
    return out


def build_conf_revisions_panel(
    chars: pd.DataFrame,
    external_path: str | Path,
    cache_path: Path | None = None,
) -> pd.DataFrame:
    """Three-month analyst revision of the nearest open fiscal quarter per stock."""
    if cache_path is not None and cache_path.exists():
        df = pd.read_parquet(cache_path)
        df["eom"] = pd.to_datetime(df["eom"])
        return df
    ibes_state.set_external_path(str(external_path))
    consensus = ibes_state.load_consensus()
    consensus = consensus.loc[
        consensus["FPI"].astype(str).isin(("6", "7", "8", "9"))
    ].copy()
    consensus["eom"] = consensus["snap_month"].dt.to_timestamp("M")
    announced = consensus["ANNDATS_ACT"].notna() & (
        consensus["ANNDATS_ACT"] <= consensus["STATPERS"]
    )
    consensus = consensus.loc[~announced].merge(
        ibes_state.load_prices(),
        on=["TICKER", "STATPERS"],
        how="inner",
    )
    lagged = consensus[["TICKER", "FPEDATS", "snap_month", "MEANEST"]].rename(
        columns={"MEANEST": "lag3"}
    )
    lagged["snap_month"] = lagged["snap_month"] + 3
    consensus = consensus.merge(
        lagged, on=["TICKER", "FPEDATS", "snap_month"], how="left"
    )
    consensus["rev3_near"] = (
        consensus["MEANEST"] - consensus["lag3"]
    ) / consensus["PRICE"]
    near = consensus.sort_values(["TICKER", "eom", "FPEDATS"]).drop_duplicates(
        ["TICKER", "eom"]
    )
    tickers = ibes_state.attach_ticker(
        chars[["permno", "eom"]], ibes_state.load_link()
    )
    revisions = tickers.merge(
        near[["TICKER", "eom", "rev3_near"]], on=["TICKER", "eom"], how="inner"
    )
    out = revisions[["permno", "eom", "rev3_near"]].reset_index(drop=True)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(cache_path, index=False)
    return out


def _formation_output(
    frame: pd.DataFrame, test_years: list[int]
) -> pd.DataFrame:
    frame = frame.loc[np.isfinite(frame["score"])].copy()
    frame["return_month"] = pd.to_datetime(frame["eom"]).dt.to_period("M") + 1
    return frame.loc[
        frame["return_month"].dt.year.isin(test_years),
        ["return_month", "permno", "score"],
    ].reset_index(drop=True)


def score_sleeves_table(
    chars: pd.DataFrame,
    e7_delayed: pd.DataFrame,
    conf_revisions: pd.DataFrame,
    xgb_fc: pd.DataFrame,
    filing_fc: pd.DataFrame,
    *,
    test_years: list[int],
    gap_df: pd.DataFrame | None = None,
    tfm_fc: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Scores all sleeves across the test window using broad-vs-native empirical CDF scaling."""
    universe = chars[["permno", "eom"]].copy()
    universe["return_month"] = (
        pd.to_datetime(universe.pop("eom")).dt.to_period("M") + 1
    )
    universe = (
        universe.loc[
            universe["return_month"].dt.year.isin(test_years),
            ["return_month", "permno"],
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    wide = universe.copy()

    def _attach(
        name: str, broad_df: pd.DataFrame, native_df: pd.DataFrame
    ) -> None:
        parts = []
        for yr in test_years:
            b_yr = broad_df.loc[broad_df["return_month"].dt.year == yr]
            n_yr = native_df.loc[native_df["return_month"].dt.year == yr]
            if not b_yr.empty:
                parts.append(scale_scores_broad_native(b_yr, n_yr))
        scaled = (
            pd.concat(parts, ignore_index=True)
            if parts
            else pd.DataFrame(
                columns=[
                    "return_month",
                    "permno",
                    "broad_score",
                    "native_score",
                ]
            )
        )
        dense = universe.merge(
            scaled, on=["return_month", "permno"], how="left"
        )
        score = dense["native_score"].fillna(dense["broad_score"]).fillna(0.0)
        wide[f"score_{name}"] = score.to_numpy(float)
        wide[f"available_{name}"] = dense["broad_score"].notna().to_numpy()
        wide[f"native_{name}"] = dense["native_score"].notna().to_numpy()

    # 1. e7_lm & e7_small_gated
    for name, sgrps in (
        ("e7_lm", ("large", "mega")),
        ("e7_small_gated", ("small",)),
    ):
        b_f = e7_delayed.copy()
        g_b = b_f.groupby(["eom", "sector"])
        b_f["score"] = g_b["sue"].rank(pct=True) + g_b["post_breadth"].rank(
            pct=True
        )
        n_f = e7_delayed.loc[
            e7_delayed["tradable"] & e7_delayed["size_grp"].isin(sgrps)
        ].copy()
        g_n = n_f.groupby(["eom", "sector"])
        n_f["score"] = g_n["sue"].rank(pct=True) + g_n["post_breadth"].rank(
            pct=True
        )
        _attach(
            name,
            _formation_output(b_f, test_years),
            _formation_output(n_f, test_years),
        )

    # 2. gap_micro_best (optional)
    if gap_df is not None and not gap_df.empty:
        g_merged = chars.merge(
            gap_df[["permno", "eom", "gap_score"]],
            on=["permno", "eom"],
            how="inner",
        )
        b_f = g_merged.assign(score=g_merged["gap_score"])
        n_f = b_f.loc[b_f["tradable"] & b_f["size_grp"].eq("micro")]
        _attach(
            "gap_micro_best",
            _formation_output(b_f, test_years),
            _formation_output(n_f, test_years),
        )

    # 3. runway_micro
    burn = (chars["cash_at"] / -chars["ocf_at"]).where(chars["ocf_at"] < 0)
    b_run = chars.assign(score=burn)
    n_run = b_run.loc[b_run["tradable"] & b_run["size_grp"].eq("micro")]
    _attach(
        "runway_micro",
        _formation_output(b_run, test_years),
        _formation_output(n_run, test_years),
    )

    # 4. conf_lm_gated
    c_merged = chars.merge(conf_revisions, on=["permno", "eom"], how="inner")
    agree = np.sign(c_merged["ret_3_1"]).eq(
        np.sign(c_merged["rev3_near"])
    ) & c_merged["rev3_near"].ne(0)
    b_conf = c_merged.assign(score=c_merged["rev3_near"])
    n_conf = c_merged.assign(score=c_merged["rev3_near"].where(agree))
    n_conf = n_conf.loc[
        n_conf["tradable"] & n_conf["size_grp"].isin(("large", "mega"))
    ]
    _attach(
        "conf_lm_gated",
        _formation_output(b_conf, test_years),
        _formation_output(n_conf, test_years),
    )

    # 5. inventory_retail
    b_inv = chars.assign(score=chars["dsale_dinv"])
    n_inv = b_inv.loc[
        b_inv["tradable"]
        & b_inv["size_grp"].isin(("small", "large", "mega"))
        & b_inv["gics"].astype("string").str[:4].isin(("2550", "2520"))
    ]
    _attach(
        "inventory_retail",
        _formation_output(b_inv, test_years),
        _formation_output(n_inv, test_years),
    )

    # 6. oplev_software
    b_op = chars.assign(score=chars["dsale_dsga"])
    n_op = b_op.loc[
        b_op["tradable"]
        & b_op["size_grp"].isin(("small", "large", "mega"))
        & b_op["gics"].astype("string").str[:4].isin(("4510",))
    ]
    _attach(
        "oplev_software",
        _formation_output(b_op, test_years),
        _formation_output(n_op, test_years),
    )

    # 7. xgb_rank
    x_f = (
        xgb_fc[["permno", "eom", "target_month", "pred_xgb_rank"]]
        .rename(columns={"pred_xgb_rank": "score"})
        .copy()
    )
    x_f["eom"] = pd.to_datetime(x_f["eom"])
    x_f["target_month"] = pd.to_datetime(x_f["target_month"])
    x_f = x_f.loc[
        x_f["target_month"]
        .dt.to_period("M")
        .eq(x_f["eom"].dt.to_period("M") + 1)
    ]
    x_f = x_f.merge(
        chars[["permno", "eom", "prc"]], on=["permno", "eom"], how="left"
    )
    _attach(
        "xgb_rank",
        _formation_output(x_f, test_years),
        _formation_output(x_f.loc[x_f["prc"] >= 1.0], test_years),
    )

    # 8. filing_events
    fe_f = (
        filing_fc[["permno", "eom", "target_month", "pred_filing_events"]]
        .rename(columns={"pred_filing_events": "score"})
        .copy()
    )
    fe_f["eom"] = pd.to_datetime(fe_f["eom"])
    fe_f["target_month"] = pd.to_datetime(fe_f["target_month"])
    fe_f = fe_f.loc[
        fe_f["target_month"]
        .dt.to_period("M")
        .eq(fe_f["eom"].dt.to_period("M") + 1)
    ]
    fe_f = fe_f.merge(
        chars[["permno", "eom", "prc"]], on=["permno", "eom"], how="left"
    )
    _attach(
        "filing_events",
        _formation_output(fe_f, test_years),
        _formation_output(fe_f.loc[fe_f["prc"] >= 1.0], test_years),
    )

    # 9. timesfm_monthly (optional)
    if tfm_fc is not None and not tfm_fc.empty:
        t_f = (
            tfm_fc[["permno", "eom", "target_month", "tfm_forecast"]]
            .rename(columns={"tfm_forecast": "score"})
            .copy()
        )
        t_f["eom"] = pd.to_datetime(t_f["eom"])
        t_f["target_month"] = pd.to_datetime(t_f["target_month"])
        t_f = t_f.loc[
            t_f["target_month"]
            .dt.to_period("M")
            .eq(t_f["eom"].dt.to_period("M") + 1)
        ]
        t_f = t_f.merge(
            chars[["permno", "eom", "prc", "tradable"]],
            on=["permno", "eom"],
            how="left",
        )
        _attach(
            "timesfm_monthly",
            _formation_output(t_f, test_years),
            _formation_output(
                t_f.loc[t_f["tradable"].fillna(False)], test_years
            ),
        )

    return wide.sort_values(["return_month", "permno"]).reset_index(drop=True)
