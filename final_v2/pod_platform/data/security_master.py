"""Security master and dated entity resolution (POD_FRAMEWORK_IMPLEMENTATION_SPEC.md 9.5-9.6)."""

from __future__ import annotations

import re

try:
    import pandas as pd
except ImportError:
    pd = None
try:
    import pyarrow.dataset as ds
except ImportError:
    ds = None
try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None


import datetime as dt
import enum
from typing import Optional

from pydantic import Field, model_validator

from ..contracts.base import DurableModel, StrictModel, require_aware
from ..contracts.errors import ErrorCode, PodPlatformError, ToolResponse


class ResolutionState(str, enum.Enum):
    EXACT_DATED = "EXACT_DATED"
    REVIEWED_DATED = "REVIEWED_DATED"
    AMBIGUOUS = "AMBIGUOUS"
    UNRESOLVED = "UNRESOLVED"


class IdentifierType(str, enum.Enum):
    PERMNO = "permno"
    CIK = "cik"
    TICKER = "ticker"
    SHARE_CLASS = "share_class"
    ISSUER = "issuer"
    BRAND = "brand"


class IdentifierLink(DurableModel):
    ID_FIELD = "link_id"

    link_id: str = Field(min_length=1)
    security_id: str = Field(min_length=1)
    identifier_type: IdentifierType
    identifier_value: str = Field(min_length=1)
    valid_from: dt.datetime
    valid_to: dt.datetime
    first_known_at: dt.datetime
    resolution_state: ResolutionState
    evidence_id: Optional[str] = None

    @model_validator(mode="after")
    def _check_interval(self) -> "IdentifierLink":
        require_aware(self.valid_from, "valid_from")
        require_aware(self.valid_to, "valid_to")
        require_aware(self.first_known_at, "first_known_at")
        if self.valid_to <= self.valid_from:
            raise ValueError("valid_to must lie after valid_from")
        if self.resolution_state in (
            ResolutionState.EXACT_DATED,
            ResolutionState.REVIEWED_DATED,
        ):
            if not self.evidence_id:
                raise ValueError("a dated resolution requires evidence")
        return self

    def valid_at(self, when: dt.datetime) -> bool:
        when = require_aware(when, "when")
        return self.valid_from <= when < self.valid_to

    def known_by(self, when: dt.datetime) -> bool:
        return self.first_known_at <= require_aware(when, "when")


class MappingProposal(StrictModel):
    source_identifier_type: IdentifierType
    source_identifier_value: str = Field(min_length=1)
    candidate_security_ids: list[str] = Field(min_length=1)
    valid_from: dt.datetime
    valid_to: dt.datetime
    evidence_id: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    review_status: ResolutionState

    @model_validator(mode="after")
    def _check(self) -> "MappingProposal":
        require_aware(self.valid_from, "valid_from")
        require_aware(self.valid_to, "valid_to")
        if self.valid_to <= self.valid_from:
            raise ValueError("valid_to must lie after valid_from")
        return self

    @property
    def admissible_for_formation(self) -> bool:
        return (
            self.review_status
            in (ResolutionState.EXACT_DATED, ResolutionState.REVIEWED_DATED)
            and len(self.candidate_security_ids) == 1
        )


class SecurityMaster:
    def __init__(self) -> None:
        self._links: dict[str, IdentifierLink] = {}

    def add_link(self, link: IdentifierLink) -> str:
        if link.link_id in self._links:
            raise ValueError(f"duplicate link_id {link.link_id}")
        self._links[link.link_id] = link
        return link.link_id

    def candidates(
        self,
        identifier_type: IdentifierType,
        identifier_value: str,
        at: dt.datetime,
    ) -> list[IdentifierLink]:
        at = require_aware(at, "at")
        return [
            link
            for link in self._links.values()
            if link.identifier_type == identifier_type
            and link.identifier_value == identifier_value
            and link.valid_at(at)
            and link.known_by(at)
        ]

    def resolve(
        self,
        identifier_type: IdentifierType,
        identifier_value: str,
        at: dt.datetime,
    ) -> str:
        links = self.candidates(identifier_type, identifier_value, at)
        usable = [
            link
            for link in links
            if link.resolution_state
            in (ResolutionState.EXACT_DATED, ResolutionState.REVIEWED_DATED)
        ]
        if not links:
            raise PodPlatformError(
                ToolResponse.failure(
                    summary=f"no dated link for {identifier_type.value}={identifier_value!r} at {at.isoformat()}",
                    code=ErrorCode.IDENTITY_AMBIGUOUS,
                    stop_condition="Unresolved identifiers stay out of formation until mapped.",
                )
            )
        if not usable:
            raise PodPlatformError(
                ToolResponse.failure(
                    summary=f"identifier {identifier_value!r} exists only in ambiguous/unresolved links",
                    code=ErrorCode.IDENTITY_AMBIGUOUS,
                    stop_condition="Ambiguous records stay out of formation until resolved.",
                )
            )
        security_ids = {link.security_id for link in usable}
        if len(security_ids) > 1:
            raise PodPlatformError(
                ToolResponse.failure(
                    summary=f"identifier {identifier_value!r} maps to multiple securities at {at.isoformat()}: {sorted(security_ids)}",
                    code=ErrorCode.IDENTITY_AMBIGUOUS,
                    stop_condition="A one-to-many mapping must be resolved to a single dated share class before use.",
                )
            )
        return usable[0].security_id

    def issuer_of(self, security_id: str, at: dt.datetime) -> Optional[str]:
        at = require_aware(at, "at")
        for link in self._links.values():
            if (
                link.security_id == security_id
                and link.identifier_type == IdentifierType.ISSUER
                and link.valid_at(at)
                and link.known_by(at)
            ):
                return link.identifier_value
        return None

    def security_link_valid(self, security_id: str, at: dt.datetime) -> bool:
        at = require_aware(at, "at")
        return any(
            link.security_id == security_id
            and link.resolution_state
            in (ResolutionState.EXACT_DATED, ResolutionState.REVIEWED_DATED)
            and link.valid_at(at)
            and link.known_by(at)
            for link in self._links.values()
        )

    def __len__(self) -> int:
        return len(self._links)


UNVERIFIED_LABEL = "no_verified_label_for_observation_period"
CARRIED_LABEL = "carried_forward_last_verified"
COVER_LABEL = "verified_in_earlier_8k_cover"
MAX_LABEL_AGE_DAYS = 366
LABEL_COLUMNS = [
    "ticker",
    "company_name",
    "ticker_name_status",
    "ticker_name_reference_date",
    "ticker_name_source",
]


def _cover_symbol(text: str) -> Optional[str]:
    not_sym = {
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
    tbl = re.search(r"Trading\s+Symbol", text[:6000], re.I)
    if not tbl:
        return None
    sym_re = re.compile(
        r"\b([A-Z]{1,5}(?:\.[A-Z])?)\s+(?:on\s+)?(?:[Tt]he\s+)?"
        r"(?:Nasdaq|NASDAQ|New York Stock Exchange|NYSE|Cboe|CBOE)"
    )
    for m in sym_re.finditer(text[tbl.end() : tbl.end() + 1500]):
        if m.group(1) not in not_sym:
            return m.group(1)
    return None


def load_formation_fields(panel_path, filings_path=None) -> "pd.DataFrame":
    """Loads formation fields with point-in-time verified ticker and company_name labels."""
    if pd is None or ds is None or pq is None:
        raise ImportError("Formation-field loading requires pandas and pyarrow")

    cols = [
        "permno",
        "permco",
        "eom",
        "prc",
        "me",
        "size_grp",
        "dolvol_126d",
        "gics",
        "beta_60m",
        *LABEL_COLUMNS,
    ]
    panel = pq.read_table(panel_path, columns=cols).to_pandas()
    panel["eom_ts"] = pd.to_datetime(panel["eom"])
    panel["eom"] = panel["eom_ts"].dt.to_period("M")
    panel["return_month"] = panel["eom"] + 1
    panel["prc"] = panel["prc"].abs()
    panel["sector"] = panel["gics"].astype("string").str[:2].fillna("na")
    ref = pd.to_datetime(panel["ticker_name_reference_date"])
    panel["ticker_name_reference_date"] = ref
    future = ref.notna() & ref.gt(panel["eom"].dt.to_timestamp("M"))
    panel.loc[future, ["ticker", "company_name"]] = None
    panel.loc[future, "ticker_name_status"] = UNVERIFIED_LABEL

    if filings_path is not None:
        p_sorted = (
            panel[["permno", "eom_ts", *LABEL_COLUMNS]]
            .rename(columns={"eom_ts": "eom"})
            .sort_values(["permno", "eom"])
            .reset_index(drop=True)
        )
        r_sorted = pd.to_datetime(p_sorted["ticker_name_reference_date"])
        verified = (
            p_sorted["ticker"].notna()
            & p_sorted["company_name"].notna()
            & p_sorted["ticker_name_status"].ne(UNVERIFIED_LABEL)
            & r_sorted.notna()
            & r_sorted.le(p_sorted["eom"])
        )
        known = p_sorted[["ticker", "company_name"]].where(verified)
        known["label_date"] = r_sorted.where(verified)
        known["permno"] = p_sorted["permno"]
        known = known.groupby("permno")[
            ["ticker", "company_name", "label_date"]
        ].ffill()
        need = ~verified
        carried = p_sorted.loc[need, ["permno", "eom"]].join(known.loc[need])
        age = (carried["eom"] - carried["label_date"]).dt.days
        carried = carried[
            carried["label_date"].notna() & (age <= MAX_LABEL_AGE_DAYS)
        ]
        if not carried.empty:
            since = carried["label_date"].min()
            filings = (
                ds.dataset(filings_path)
                .to_table(
                    filter=(ds.field("filing_date") >= since.date())
                    & ds.field("permno").isin(
                        carried["permno"].unique().tolist()
                    ),
                    columns=[
                        "permno",
                        "filing_date",
                        "text",
                        "company_name",
                        "document_id",
                    ],
                )
                .to_pandas()
            )
            filings["symbol"] = filings["text"].map(_cover_symbol)
            filings = filings.drop(columns="text").dropna(subset=["symbol"])
            filings["filing_date"] = pd.to_datetime(filings["filing_date"])
            filings = filings.rename(columns={"company_name": "sec_name"})
            out = pd.merge_asof(
                carried.assign(
                    eom=carried["eom"].astype("datetime64[ns]")
                ).sort_values("eom"),
                filings.assign(
                    filing_date=filings["filing_date"].astype("datetime64[ns]")
                ).sort_values("filing_date"),
                left_on="eom",
                right_on="filing_date",
                by="permno",
                direction="backward",
            )
            cover = (
                out["filing_date"].notna()
                & (out["filing_date"] > out["label_date"])
                & out["symbol"].eq(out["ticker"])
            )
            label_day = out["label_date"].dt.strftime("%Y-%m-%d")
            doc = out["document_id"].astype(str)
            out["company_name"] = out["company_name"].where(
                ~cover, out["sec_name"]
            )
            out["ticker_name_status"] = CARRIED_LABEL
            out.loc[cover, "ticker_name_status"] = COVER_LABEL
            out["ticker_name_reference_date"] = out["label_date"].where(
                ~cover, out["filing_date"]
            )
            out["ticker_name_source"] = "panel verified label of " + label_day
            out.loc[cover, "ticker_name_source"] = (
                "8-K cover trading symbol, document " + doc
            )
            found = out[["permno", "eom", *LABEL_COLUMNS]].rename(
                columns={"eom": "eom_ts"}
            )
            panel = panel.merge(
                found,
                on=["permno", "eom_ts"],
                how="left",
                suffixes=("", "_verified"),
            )
            fill = panel["ticker_verified"].notna() & panel[
                "ticker_name_status"
            ].eq(UNVERIFIED_LABEL)
            for col in LABEL_COLUMNS:
                vals = panel[col].astype(object)
                vals[fill] = panel.loc[fill, col + "_verified"]
                if col == "ticker_name_reference_date":
                    vals = pd.to_datetime(vals)
                panel[col] = vals
                panel = panel.drop(columns=[col + "_verified"])
    return panel.drop(columns=["eom_ts"])
