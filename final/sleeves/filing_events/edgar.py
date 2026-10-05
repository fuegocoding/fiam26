"""SEC EDGAR filing events as monthly stock features.

The quarterly EDGAR form index lists every filing once per related company
(a Schedule 13D appears under the target and under the filer).  Each filing
form belongs to one event family; a stock's features count its family events
over the last 1, 3 and 12 months.  A filing made in month m is usable from
month m + 1, the same rule the characteristics panel follows.

Companies are linked to the panel by CIK from the supplied 8-K file, then by
exact unique name matches between panel and EDGAR company names (99.6%
precise against the 8-K links where both exist).  An accession whose related
companies resolve to two different panel companies is dropped as ambiguous.
"""

import pathlib
import re

import pandas as pd

import config
import external

FAMILIES = {
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
# Forms read only to learn company names, first filings and IPO dates.
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
FEATURE_COLUMNS = [f"{family}_{window}" for family in FAMILIES for window in WINDOWS]
FEATURE_COLUMNS.append("lockup_m")
_SUFFIX = re.compile(
    r"\b(INCORPORATED|INC|CORPORATION|CORP|COMPANY|CO|LIMITED|LTD|LLC|LP|L P"
    r"|PLC|NV|N V|SA|S A|AG|HOLDINGS|HOLDING|HLDGS|HLDG|GROUP|GRP|THE|NEW|DEL"
    r"|COM|CLASS [A-Z]|CL [A-Z]|ORD|ADR|SPONSORED|COMMON|STOCK|SHS|SHARES"
    r"|TRUST|FUND|INTERNATIONAL|INTL|TECHNOLOGIES|TECHNOLOGY|TECH|SYSTEMS"
    r"|THERAPEUTICS|PHARMACEUTICALS|PHARMA|BANCORP|BANCORPORATION|FINANCIAL"
    r"|ENTERPRISES|INDUSTRIES|BIOSCIENCES|BIOPHARMA|SCIENCES|LABORATORIES"
    r"|LABS|ENERGY|RESOURCES)\b"
)


def build(
    panel_path: pathlib.Path,
    filings_path: pathlib.Path,
    work_dir: pathlib.Path,
) -> pathlib.Path:
    """Writes features.parquet: permno, return_month and one count per column."""
    output = work_dir / "features.parquet"
    if output.exists():
        return output
    work_dir.mkdir(parents=True, exist_ok=True)
    index = external.read("edgar")
    index["date_filed"] = pd.to_datetime(index.date_filed)
    panel, linked = _link_evidence(panel_path, filings_path)
    families = FAMILIES
    columns = [f"{family}_{window}" for family in families for window in WINDOWS] + [
        "lockup_m"
    ]
    parts = []
    for month in pd.period_range(
        _FIRST_FEATURE_MONTH, config.LAST_TARGET.to_period("M"), freq="M"
    ):
        cutoff = (month - 1).to_timestamp("M")
        known = index.loc[index.date_filed <= cutoff]
        crosswalk = _crosswalk_frames(panel, linked, known, cutoff)
        recent = known.loc[known.date_filed >= (month - 12).start_time]
        features = _monthly(
            _events(recent, crosswalk, families), _lockups(known, crosswalk), columns
        )
        parts.append(features.loc[features.return_month == month])
    features = pd.concat(parts, ignore_index=True)
    features["return_month"] = features.return_month.dt.to_timestamp("M")
    features.to_parquet(output, index=False)
    return output


def normalise(name: str | None) -> str:
    """Upper-case alphanumerics with corporate-form words removed."""
    if not isinstance(name, str):
        return ""
    text = name.upper().replace("&", " AND ")
    text = re.sub(r"/[A-Z]{2,3}/?", " ", text)
    text = re.sub(r"[^A-Z0-9 ]", " ", text)
    for _ in range(2):
        text = _SUFFIX.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def kept_forms() -> set[str]:
    """Form types the sleeve counts, plus those used to learn names."""
    forms = {form for group in FAMILIES.values() for form in group}
    return forms | set(_NAME_FORMS)


def _link_evidence(
    panel_path: pathlib.Path, filings_path: pathlib.Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Dated evidence retained for each month's independent company mapping."""
    linked = pd.read_parquet(
        filings_path, columns=["permno", "permco", "cik", "filing_date"]
    ).dropna(subset=["permno"])
    linked["filing_date"] = pd.to_datetime(linked.filing_date)
    linked["permno"] = linked.permno.astype(int)
    linked["cik"] = pd.to_numeric(linked.cik, errors="coerce")
    linked = linked.dropna(subset=["cik"]).astype({"cik": "int64"})
    linked = linked.drop_duplicates()
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
    panel["eom"] = pd.to_datetime(panel.eom)
    reference = pd.to_datetime(panel.ticker_name_reference_date)
    panel = panel.loc[reference.notna() & reference.le(panel.eom)].copy()
    panel["key"] = panel.company_name.map(normalise)
    panel = panel[panel.key.str.len() >= 4]
    return panel, linked


def _crosswalk(
    panel_path: pathlib.Path,
    filings_path: pathlib.Path,
    index: pd.DataFrame,
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    """Company links known at cutoff; later names and 8-Ks cannot backfill."""
    panel, linked = _link_evidence(panel_path, filings_path)
    return _crosswalk_frames(panel, linked, index, cutoff)


def _crosswalk_frames(
    panel: pd.DataFrame, linked: pd.DataFrame, index: pd.DataFrame, cutoff: pd.Timestamp
) -> pd.DataFrame:
    panel = panel.loc[panel.eom <= cutoff].drop_duplicates(["permno", "company_name"])
    linked = linked.loc[linked.filing_date <= cutoff, ["permno", "permco", "cik"]]
    linked = linked.drop_duplicates()
    index = index.loc[index.date_filed <= cutoff]
    filers = index[index.form_type.isin(_NAME_FORMS)]
    names = filers.groupby(["cik", "company_name"]).size().reset_index()
    names["key"] = names.company_name.map(normalise)
    names = names[names.key.str.len() >= 4]
    one_company = panel.groupby("key").permco.nunique()
    one_filer = names.groupby("key").cik.nunique()
    matched = panel[panel.key.isin(one_company[one_company == 1].index)].merge(
        names[names.key.isin(one_filer[one_filer == 1].index)], on="key"
    )
    matched = matched.drop_duplicates(["permno", "cik"])
    matched = matched[~matched.permno.isin(linked.permno)]
    matched["cik"] = matched.cik.astype("int64")
    return pd.concat([linked, matched[["permno", "permco", "cik"]]], ignore_index=True)


def _events(
    index: pd.DataFrame, crosswalk: pd.DataFrame, families=None
) -> pd.DataFrame:
    """One row per stock, filing and family, dated by filing month."""
    families = FAMILIES if families is None else families
    family_of = {form: family for family, forms in families.items() for form in forms}
    events = index[index.form_type.isin(family_of)].copy()
    events["family"] = events.form_type.map(family_of)
    events["cik"] = events.cik.astype("int64")
    events = events.merge(crosswalk, on="cik")
    companies = events.groupby(["accession", "family"]).permco.nunique()
    ambiguous = companies[companies > 1].index
    events = events.set_index(["accession", "family"])
    events = events.loc[~events.index.isin(ambiguous)].reset_index()
    events["month"] = events.date_filed.dt.to_period("M")
    return events.drop_duplicates(["permno", "accession", "family"])[
        ["permno", "family", "month"]
    ]


def _lockups(index: pd.DataFrame, crosswalk: pd.DataFrame) -> pd.DataFrame:
    """Stock-months in which a 180-day IPO lock-up expires."""
    periodic = index[index.form_type.isin(["10-K", "10-Q", "8-K"])]
    first_report = periodic.groupby("cik").date_filed.min().rename("first")
    offering = index[index.form_type == "424B4"]
    ipo = offering.groupby("cik").date_filed.min().rename("ipo")
    both = pd.concat([ipo, first_report], axis=1).dropna(subset=["ipo"])
    new_issuer = both["first"].isna() | (
        both["first"] >= both.ipo - pd.Timedelta(days=5)
    )
    both = both[new_issuer & (both.ipo >= pd.Timestamp("2014-06-01"))]
    both = both.rename_axis("cik").reset_index().astype({"cik": "int64"})
    both = both.merge(crosswalk[["permno", "cik"]], on="cik")
    both = both.drop_duplicates("permno")
    both["return_month"] = (both.ipo + pd.Timedelta(days=_LOCKUP_DAYS)).dt.to_period(
        "M"
    )
    return both.assign(lockup_m=1)[["permno", "return_month", "lockup_m"]]


def _monthly(events: pd.DataFrame, lockups: pd.DataFrame, columns=None) -> pd.DataFrame:
    """Event counts over the last 1, 3 and 12 months, by return month."""
    wide = None
    for family, group in events.groupby("family"):
        counts = group.groupby(["permno", "month"]).size().rename("n")
        counts = counts.reset_index()
        for label, months in WINDOWS.items():
            parts = []
            for shift in range(months):
                part = counts.copy()
                part["return_month"] = part.month + 1 + shift
                parts.append(part[["permno", "return_month", "n"]])
            column = pd.concat(parts).groupby(["permno", "return_month"]).n
            column = column.sum().rename(f"{family}_{label}")
            wide = column.to_frame() if wide is None else wide.join(column, how="outer")
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
        columns=[
            "permno",
            "return_month",
            *(FEATURE_COLUMNS if columns is None else columns),
        ],
        fill_value=0,
    )
    keep = wide.return_month.between(
        _FIRST_FEATURE_MONTH, config.LAST_TARGET.to_period("M")
    )
    return wide.loc[keep].fillna(0)
