"""Walk-forward trained sleeves: xgb_rank and filing_events
(POD_FRAMEWORK_IMPLEMENTATION_SPEC.md 13.3, 14; ports of first-round recipes).

Both sleeves fit per test year Y on targets through Y-3, early-stop on
Y-2..Y-1, predict Y — the same folds as the first round's timing masks.
Four seeds, heavily regularized XGBoost; the target is a cross-sectional
rank of the next-month excess return. Filing-event features use the
8-K corpus's dated cik->permno links (evidence-based) rather than name
normalization, and each month's features use only filings known before
that month (acceptance time).
"""

from __future__ import annotations

try:
    import xgboost as xgb
except ImportError:
    xgb = None
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

XGB_PARAMS = {
    "learning_rate": 0.02,
    "max_depth": 4,
    "min_child_weight": 500,
    "subsample": 0.7,
    "colsample_bytree": 0.5,
    "reg_lambda": 10.0,
    "n_estimators": 3000,
    "early_stopping_rounds": 200,
    "tree_method": "hist",
    "eval_metric": "rmse",
}
SEEDS = (0, 1, 2, 3)
FIRST_TRAIN_TARGET = pd.Timestamp("2015-02-01")

FILING_FAMILIES = {
    "act13d": ["SC 13D", "SCHEDULE 13D"],
    "act13d_a": ["SC 13D/A", "SCHEDULE 13D/A"],
    "g13": ["SC 13G", "SCHEDULE 13G"],
    "g13_a": ["SC 13G/A", "SCHEDULE 13G/A"],
    "nt": ["NT 10-K", "NT 10-Q"],
    "comment": ["UPLOAD"],
    "corresp": ["CORRESP"],
    "offer": ["424B5", "424B4", "424B3"],
    "shelf": ["S-3", "S-3ASR", "S-1", "S-1/A"],
    "proxy": ["DEFC14A", "PREC14A", "DFAN14A", "PX14A6G"],
    "tender_i": ["SC TO-I"],
    "tender_t": ["SC TO-T", "SC 14D9"],
    "mergerp": ["DEFM14A", "PREM14A"],
    "dereg": ["15-12G", "15-12B", "15-15D", "25-NSE"],
    "amend": ["10-K/A", "10-Q/A"],
    "f144": ["144"],
    "formd": ["D", "D/A"],
    "s4": ["S-4"],
    "spin": ["10-12B", "10-12G"],
    "succ": ["8-K12B", "8-K12G3"],
    "sc13e3": ["SC 13E3"],
    "form25": ["25"],
    "effect": ["EFFECT"],
    "defa14a": ["DEFA14A"],
    "posam": ["POS AM"],
}
WINDOWS = {"1m": 1, "3m": 3, "12m": 12}


@dataclass
class FoldAudit:
    year: int
    train_end: str
    validation_window: str
    test_window: str
    train_rows: int
    validation_rows: int
    test_rows: int
    best_iterations: list


def _rank_transform(table: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    ranks = table.groupby("eom")[features].rank(method="dense")
    span = ranks.groupby(table["eom"]).transform("max") - 1.0
    return (ranks - 1.0) / span.replace(0.0, np.nan) * 2.0 - 1.0


def _split(table: pd.DataFrame, year: int, last_target: pd.Timestamp):
    months = table["target_month"]
    train_end = pd.Timestamp(f"{year - 3}-12-31")
    train = table.loc[months.between(FIRST_TRAIN_TARGET, train_end)]
    validation = table.loc[
        months.between(
            pd.Timestamp(f"{year - 2}-01-01"), pd.Timestamp(f"{year - 1}-12-31")
        )
    ]
    test = table.loc[
        months.between(
            pd.Timestamp(f"{year}-01-01"),
            min(pd.Timestamp(f"{year}-12-31"), last_target),
        )
    ]
    assert train["target_month"].max() <= train_end, (
        "training labels leak past the cutoff"
    )
    return (
        train.dropna(subset=["target"]),
        validation.dropna(subset=["target"]),
        test,
    )


def train_forecasts(
    table: pd.DataFrame,
    features: list[str],
    *,
    test_years: list[int],
    last_target: pd.Timestamp,
    target_column: str,
    output_column: str,
    device: str = "cpu",
    on_fold=None,
) -> tuple[pd.DataFrame, list[FoldAudit]]:
    if xgb is None:
        raise ImportError("Training charter models requires xgboost")

    audits: list[FoldAudit] = []
    parts = []
    for year in test_years:
        train, validation, test = _split(table, year, last_target)
        predictions = []
        best_iterations = []
        for seed in SEEDS:
            model = xgb.XGBRegressor(
                random_state=seed, device=device, **XGB_PARAMS
            )
            model.fit(
                train[features].values,
                train[target_column].values,
                eval_set=[
                    (
                        validation[features].values,
                        validation[target_column].values,
                    )
                ],
                verbose=False,
            )
            best_iterations.append(int(model.best_iteration))
            predictions.append(model.predict(test[features].values))
        audits.append(
            FoldAudit(
                year=year,
                train_end=str(train["target_month"].max().date())
                if len(train)
                else "empty",
                validation_window=f"{year - 2}..{year - 1}",
                test_window=str(year),
                train_rows=len(train),
                validation_rows=len(validation),
                test_rows=len(test),
                best_iterations=best_iterations,
            )
        )
        if on_fold:
            on_fold(audits[-1])
        part = test[["permno", "eom", "target_month"]].copy()
        part[output_column] = np.mean(predictions, axis=0)
        parts.append(part)
    return pd.concat(parts, ignore_index=True), audits


def build_xgb_rank(
    panel: pd.DataFrame,
    feature_columns: list[str],
    *,
    test_years: list[int],
    last_target: pd.Timestamp,
    device: str = "cpu",
    on_fold=None,
) -> tuple[pd.DataFrame, list[FoldAudit]]:
    columns = ["permno", "eom", "ret_exc_lead1m", *feature_columns]
    table = panel[columns].copy()
    table["eom"] = pd.to_datetime(table["eom"])
    table["target_month"] = table["eom"] + pd.offsets.MonthEnd(1)
    table[feature_columns] = _rank_transform(table, feature_columns).astype(
        "float32"
    )
    table["target"] = (
        table.groupby("eom")["ret_exc_lead1m"].rank(pct=True) - 0.5
    )
    return train_forecasts(
        table,
        feature_columns,
        test_years=test_years,
        last_target=last_target,
        target_column="target",
        output_column="pred_xgb_rank",
        device=device,
        on_fold=on_fold,
    )


def build_filing_event_features(
    filings_index: pd.DataFrame,
    links: pd.DataFrame,
    months: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Monthly filing-event counts per permno, using only filings known
    before each month (date_filed <= previous month end), keyed through
    dated cik->permno links."""
    index = filings_index.copy()
    index["date_filed"] = pd.to_datetime(index["date_filed"])
    index = index.dropna(subset=["cik"])
    index["cik"] = index["cik"].astype(str).str.lstrip("0")
    form_map = {}
    for family, forms in FILING_FAMILIES.items():
        for form in forms:
            form_map[form] = family
    index["family"] = index["form_type"].map(form_map)
    index = index.dropna(subset=["family"])
    index = index.drop(columns=["permno"], errors="ignore")

    links = links.copy()
    links["cik"] = links["cik"].astype(str).str.lstrip("0")
    links["permno"] = links["permno"].astype(int)

    rows = []
    for month_end in months:
        cutoff = month_end - pd.offsets.MonthEnd(1)
        known = index.loc[index["date_filed"] <= cutoff]
        merged = known.merge(links, on="cik", how="inner")
        merged = merged[merged["link_date"] <= cutoff]
        counts_frames = {}
        for window, months_back in WINDOWS.items():
            start = (
                cutoff
                - pd.offsets.DateOffset(months=months_back)
                + pd.offsets.Day(1)
            )
            windowed = merged[merged["date_filed"] >= start]
            wcounts = (
                windowed.groupby(["permno", "family"])
                .size()
                .unstack(fill_value=0)
                .reindex(columns=list(FILING_FAMILIES), fill_value=0)
            )
            wcounts.columns = [
                f"{family}_{window}" for family in wcounts.columns
            ]
            counts_frames[window] = wcounts
        counts = (
            counts_frames["1m"]
            .join(counts_frames["3m"], how="outer")
            .join(counts_frames["12m"], how="outer")
            .fillna(0)
        )
        counts["permno"] = counts.index
        counts["target_month"] = month_end
        rows.append(counts.reset_index(drop=True))
    return pd.concat(rows, ignore_index=True)


def build_filing_events(
    panel: pd.DataFrame,
    event_features: pd.DataFrame,
    *,
    test_years: list[int],
    last_target: pd.Timestamp,
    device: str = "cpu",
    on_fold=None,
) -> tuple[pd.DataFrame, list[FoldAudit]]:
    columns = ["permno", "eom", "ret_exc_lead1m", "size_grp", "gics"]
    table = panel[columns].copy()
    table["eom"] = pd.to_datetime(table["eom"])
    table["target_month"] = table["eom"] + pd.offsets.MonthEnd(1)
    features = [
        c for c in event_features.columns if c not in ("permno", "target_month")
    ]
    table = table.merge(
        event_features, on=["permno", "target_month"], how="left"
    )
    table[features] = table[features].fillna(0.0)
    table[features] = _rank_transform(table, features).astype("float32")
    peers = [
        table["eom"],
        table["size_grp"].astype(str),
        table["gics"].astype("string").str[:2].fillna("na"),
    ]
    table["target"] = (
        table.groupby(peers)["ret_exc_lead1m"].rank(pct=True) - 0.5
    )
    return train_forecasts(
        table,
        features,
        test_years=test_years,
        last_target=last_target,
        target_column="target",
        output_column="pred_filing_events",
        device=device,
        on_fold=on_fold,
    )


_NAME_FORMS = (
    "10-K",
    "10-Q",
    "8-K",
    "10-K/A",
    "10-Q/A",
    "S-1",
    "S-1/A",
    "424B4",
    "424B5",
    "S-3",
    "10-KT",
    "DEF 14A",
    "8-A12B",
    "10-12B",
)
_LOCKUP_DAYS = 180
_FIRST_FEATURE_MONTH = pd.Period("2015-02", "M")


def _normalise_company_name(name: str | None) -> str:
    if not isinstance(name, str):
        return ""
    suffix = re.compile(
        r"\b(INCORPORATED|INC|CORPORATION|CORP|COMPANY|CO|LIMITED|LTD|LLC|LP|L P"
        r"|PLC|NV|N V|SA|S A|AG|HOLDINGS|HOLDING|HLDGS|HLDG|GROUP|GRP|THE|NEW|DEL"
        r"|COM|CLASS [A-Z]|CL [A-Z]|ORD|ADR|SPONSORED|COMMON|STOCK|SHS|SHARES"
        r"|TRUST|FUND|INTERNATIONAL|INTL|TECHNOLOGIES|TECHNOLOGY|TECH|SYSTEMS"
        r"|THERAPEUTICS|PHARMACEUTICALS|PHARMA|BANCORP|BANCORPORATION|FINANCIAL"
        r"|ENTERPRISES|INDUSTRIES|BIOSCIENCES|BIOPHARMA|SCIENCES|LABORATORIES"
        r"|LABS|ENERGY|RESOURCES)\b"
    )
    text = name.upper().replace("&", " AND ")
    text = re.sub(r"/[A-Z]{2,3}/?", " ", text)
    text = re.sub(r"[^A-Z0-9 ]", " ", text)
    for _ in range(2):
        text = suffix.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_edgar_crosswalk_features(
    panel_path: Path,
    filings_path: Path,
    external_path: Path,
    last_target: pd.Timestamp = pd.Timestamp("2026-08-31"),
    cache_path: Path | None = None,
) -> pd.DataFrame:
    """Point-in-time EDGAR filing event features + 180-day IPO lockup indicator."""
    if cache_path is not None and cache_path.exists():
        df = pd.read_parquet(cache_path)
        if "return_month" in df.columns:
            df = df.rename(columns={"return_month": "target_month"})
        df["target_month"] = pd.to_datetime(df["target_month"])
        return df

    ext = pd.read_parquet(
        external_path,
        columns=[
            "table",
            "form_type",
            "date_filed",
            "cik",
            "company_name",
            "accession",
        ],
    )
    index = ext[ext["table"] == "edgar"].drop(columns=["table"]).copy()
    index["date_filed"] = pd.to_datetime(index["date_filed"])

    linked = pd.read_parquet(
        filings_path, columns=["permno", "permco", "cik", "filing_date"]
    ).dropna(subset=["permno"])
    linked["filing_date"] = pd.to_datetime(linked["filing_date"])
    linked["permno"] = linked["permno"].astype(int)
    linked["cik"] = pd.to_numeric(linked["cik"], errors="coerce")
    linked = (
        linked.dropna(subset=["cik"]).astype({"cik": "int64"}).drop_duplicates()
    )

    panel = pd.read_parquet(
        panel_path,
        columns=[
            "permno",
            "permco",
            "company_name",
            "eom",
            "ticker_name_reference_date",
        ],
    ).dropna(subset=["company_name"])
    panel["eom"] = pd.to_datetime(panel["eom"])
    ref = pd.to_datetime(panel["ticker_name_reference_date"])
    panel = panel.loc[ref.notna() & ref.le(panel["eom"])].copy()
    panel["key"] = panel["company_name"].map(_normalise_company_name)
    panel = panel[panel["key"].str.len() >= 4]

    columns = [
        f"{family}_{window}" for family in FILING_FAMILIES for window in WINDOWS
    ] + ["lockup_m"]
    family_of = {
        form: family
        for family, forms in FILING_FAMILIES.items()
        for form in forms
    }

    parts = []
    for month in pd.period_range(
        _FIRST_FEATURE_MONTH, last_target.to_period("M"), freq="M"
    ):
        cutoff = (month - 1).to_timestamp("M")
        known = index.loc[index["date_filed"] <= cutoff]
        p_cut = panel.loc[panel["eom"] <= cutoff].drop_duplicates(
            ["permno", "company_name"]
        )
        l_cut = linked.loc[
            linked["filing_date"] <= cutoff, ["permno", "permco", "cik"]
        ].drop_duplicates()
        filers = known[known["form_type"].isin(_NAME_FORMS)]
        names = filers.groupby(["cik", "company_name"]).size().reset_index()
        names["key"] = names["company_name"].map(_normalise_company_name)
        names = names[names["key"].str.len() >= 4]
        one_company = p_cut.groupby("key")["permco"].nunique()
        one_filer = names.groupby("key")["cik"].nunique()
        matched = (
            p_cut[p_cut["key"].isin(one_company[one_company == 1].index)]
            .merge(
                names[names["key"].isin(one_filer[one_filer == 1].index)],
                on="key",
            )
            .drop_duplicates(["permno", "cik"])
        )
        matched = matched[~matched["permno"].isin(l_cut["permno"])]
        matched["cik"] = matched["cik"].astype("int64")
        crosswalk = pd.concat(
            [l_cut, matched[["permno", "permco", "cik"]]], ignore_index=True
        )

        recent = known.loc[known["date_filed"] >= (month - 12).start_time]
        ev = recent[recent["form_type"].isin(family_of)].copy()
        ev["family"] = ev["form_type"].map(family_of)
        ev["cik"] = pd.to_numeric(ev["cik"], errors="coerce")
        ev = (
            ev.dropna(subset=["cik"])
            .astype({"cik": "int64"})
            .merge(crosswalk, on="cik")
        )
        companies = ev.groupby(["accession", "family"])["permco"].nunique()
        ambiguous = companies[companies > 1].index
        ev = ev.set_index(["accession", "family"])
        ev = ev.loc[~ev.index.isin(ambiguous)].reset_index()
        ev["month"] = ev["date_filed"].dt.to_period("M")
        ev = ev.drop_duplicates(["permno", "accession", "family"])[
            ["permno", "family", "month"]
        ]

        periodic = known[known["form_type"].isin(["10-K", "10-Q", "8-K"])]
        first_report = (
            periodic.groupby("cik")["date_filed"].min().rename("first")
        )
        offering = known[known["form_type"] == "424B4"]
        ipo = offering.groupby("cik")["date_filed"].min().rename("ipo")
        both = pd.concat([ipo, first_report], axis=1).dropna(subset=["ipo"])
        new_issuer = both["first"].isna() | (
            both["first"] >= both["ipo"] - pd.Timedelta(days=5)
        )
        both = both[new_issuer & (both["ipo"] >= pd.Timestamp("2014-06-01"))]
        both = both.rename_axis("cik").reset_index()
        both["cik"] = pd.to_numeric(both["cik"], errors="coerce")
        both = (
            both.dropna(subset=["cik"])
            .astype({"cik": "int64"})
            .merge(crosswalk[["permno", "cik"]], on="cik")
        )
        both = both.drop_duplicates("permno")
        both["return_month"] = (
            both["ipo"] + pd.Timedelta(days=_LOCKUP_DAYS)
        ).dt.to_period("M")
        lockups = both.assign(lockup_m=1)[
            ["permno", "return_month", "lockup_m"]
        ]

        wide = None
        for family, group in ev.groupby("family"):
            counts = (
                group.groupby(["permno", "month"])
                .size()
                .rename("n")
                .reset_index()
            )
            for label, m_back in WINDOWS.items():
                sub = []
                for shift in range(m_back):
                    part = counts.copy()
                    part["return_month"] = part["month"] + 1 + shift
                    sub.append(part[["permno", "return_month", "n"]])
                col = (
                    pd.concat(sub)
                    .groupby(["permno", "return_month"])["n"]
                    .sum()
                    .rename(f"{family}_{label}")
                )
                wide = (
                    col.to_frame()
                    if wide is None
                    else wide.join(col, how="outer")
                )
        if wide is None:
            wide = pd.DataFrame(
                {
                    "permno": pd.Series(dtype="int64"),
                    "return_month": pd.Series(dtype="period[M]"),
                }
            )
        else:
            wide = wide.reset_index()
        wide = wide.merge(lockups, on=["permno", "return_month"], how="outer")
        wide = wide.reindex(
            columns=["permno", "return_month", *columns], fill_value=0
        ).fillna(0)
        parts.append(wide.loc[wide["return_month"] == month])

    out = pd.concat(parts, ignore_index=True)
    out["target_month"] = out.pop("return_month").dt.to_timestamp("M")
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(cache_path, index=False)
    return out
