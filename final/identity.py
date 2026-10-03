"""Identity agent: point-in-time ticker and company-name labels.

The optimizer only trades stocks it can name, since the deck must identify
every holding by ticker and full company name. The panel has CRSP labels up
to its last CRSP month (12/2025); after that it labels a stock-month only
when the company filed an 8-K in that same month. Dropping the rest would
restrict the 2026 book to that month's 8-K filers, about 60% of the stocks.

For every unlabelled stock-month after the last CRSP month, with formation
month-end E:
  1. take the issuer's last verified panel label dated on or before E, if it
     is at most MAX_AGE_DAYS old;
  2. if the issuer's newest 8-K filed on or before E is newer than that label
     and its cover page lists the same trading symbol, date the label to that
     filing and take the SEC issuer name on it (a different symbol is more
     likely another listed security, such as notes, so the label is kept);
  3. record the label's source and reference date.
Nothing dated after E is used and no later name is ever substituted.
"""

import pathlib
import re

import pandas as pd
import pyarrow.dataset as ds

UNVERIFIED = "no_verified_label_for_observation_period"
CRSP_STATUS = "verified_in_observation_month"
CARRIED = "carried_forward_last_verified"
COVER = "verified_in_earlier_8k_cover"
MAX_AGE_DAYS = 366
LABELS = [
    "ticker",
    "company_name",
    "ticker_name_status",
    "ticker_name_reference_date",
    "ticker_name_source",
]
_COVER_TABLE = re.compile(r"Trading\s+Symbol", re.I)
_SYMBOL = re.compile(
    r"\b([A-Z]{1,5}(?:\.[A-Z])?)\s+(?:on\s+)?(?:[Tt]he\s+)?"
    r"(?:Nasdaq|NASDAQ|New York Stock Exchange|NYSE|Cboe|CBOE)"
)
_NOT_SYMBOL = {
    "LLC",
    "INC",
    "THE",
    "NYSE",
    "PAR",
    "USD",
    "N/A",
    "NONE",
    "ADS",
    "ADR",
    "CLASS",
    "A",
    "B",
}


def cover_symbol(text: str) -> str | None:
    """First trading symbol in an 8-K cover page's securities table."""
    table = _COVER_TABLE.search(text[:6000])
    if not table:
        return None
    for match in _SYMBOL.finditer(text[table.end() : table.end() + 1500]):
        if match.group(1) not in _NOT_SYMBOL:
            return match.group(1)
    return None


def _carried_labels(panel_path: pathlib.Path) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Unlabelled post-CRSP stock-months with the last verified label."""
    panel = pd.read_parquet(panel_path, columns=["permno", "eom", *LABELS])
    panel["eom"] = pd.to_datetime(panel.eom)
    panel = panel.sort_values(["permno", "eom"]).reset_index(drop=True)
    verified = (
        panel.ticker.notna()
        & panel.company_name.notna()
        & panel.ticker_name_status.ne(UNVERIFIED)
    )
    last_crsp = panel.loc[panel.ticker_name_status.eq(CRSP_STATUS), "eom"].max()

    known = panel[["ticker", "company_name"]].where(verified)
    known["label_date"] = panel.eom.where(verified)
    known["permno"] = panel.permno
    known = known.groupby("permno")[["ticker", "company_name", "label_date"]]
    known = known.ffill()

    need = ~verified & (panel.eom > last_crsp)
    carried = panel.loc[need, ["permno", "eom"]].join(known.loc[need])
    age = (carried.eom - carried.label_date).dt.days
    fresh = carried.label_date.notna() & (age <= MAX_AGE_DAYS)
    return carried[fresh], last_crsp


def _cover_symbols(
    filings_path: pathlib.Path, permnos: list[int], since: pd.Timestamp
) -> pd.DataFrame:
    """8-K filings since `since` whose cover page lists a trading symbol."""
    filings = (
        ds.dataset(filings_path)
        .to_table(
            filter=(ds.field("filing_date") >= since.date())
            & ds.field("permno").isin(permnos),
            columns=["permno", "filing_date", "text", "company_name", "document_id"],
        )
        .to_pandas()
    )
    filings["symbol"] = filings.text.map(cover_symbol)
    filings = filings.drop(columns="text").dropna(subset=["symbol"])
    filings["filing_date"] = pd.to_datetime(filings.filing_date)
    return filings.rename(columns={"company_name": "sec_name"})


def labels(panel_path: pathlib.Path, filings_path: pathlib.Path) -> pd.DataFrame:
    """Agent labels keyed by (permno, eom) for unlabelled post-CRSP months."""
    carried, last_crsp = _carried_labels(panel_path)
    if carried.empty:
        return carried.assign(**{column: None for column in LABELS})
    since = last_crsp - pd.Timedelta(days=MAX_AGE_DAYS)
    filings = _cover_symbols(filings_path, carried.permno.unique().tolist(), since)

    out = pd.merge_asof(
        carried.assign(eom=carried.eom.astype("datetime64[ns]")).sort_values("eom"),
        filings.assign(
            filing_date=filings.filing_date.astype("datetime64[ns]")
        ).sort_values("filing_date"),
        left_on="eom",
        right_on="filing_date",
        by="permno",
        direction="backward",
    )
    cover = (
        out.filing_date.notna()
        & (out.filing_date > out.label_date)
        & out.symbol.eq(out.ticker)
    )
    label_day = out.label_date.dt.strftime("%Y-%m-%d")
    document = out.document_id.astype(str)
    out["company_name"] = out.company_name.where(~cover, out.sec_name)
    out["ticker_name_status"] = CARRIED
    out.loc[cover, "ticker_name_status"] = COVER
    out["ticker_name_reference_date"] = out.label_date.where(~cover, out.filing_date)
    out["ticker_name_source"] = "panel verified label of " + label_day
    out.loc[cover, "ticker_name_source"] = (
        "8-K cover trading symbol, document " + document
    )
    return out[["permno", "eom", *LABELS]]


def label(
    predictions: pd.DataFrame,
    panel_path: pathlib.Path,
    filings_path: pathlib.Path,
) -> pd.DataFrame:
    """Fills the unlabelled rows of a predictions frame."""
    found = labels(panel_path, filings_path).rename(columns={"eom": "formation"})
    found["formation"] = found.formation.astype("datetime64[ns]")
    out = predictions.assign(
        formation=pd.to_datetime(predictions.formation_date).astype("datetime64[ns]")
    )
    out = out.merge(
        found,
        on=["permno", "formation"],
        how="left",
        suffixes=("", "_agent"),
        validate="many_to_one",
    )
    fill = out.ticker_agent.notna() & out.ticker_name_status.eq(UNVERIFIED)
    for column in LABELS:
        values = out[column].astype(object)
        values[fill] = out.loc[fill, column + "_agent"]
        if column == "ticker_name_reference_date":
            values = pd.to_datetime(values)
        out[column] = values.astype(predictions[column].dtype)
    print(f"Identity agent: labelled {int(fill.sum()):,} stock-months")
    return out[predictions.columns]
