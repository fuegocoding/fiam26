"""Connect evidence services and dated feature intake to the research campaign."""

from __future__ import annotations

import datetime as dt
import json
import re

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ..data.carbonarc_requests import (
    CarbonArcOAuthTransport,
    CarbonArcPolicy,
    CarbonArcRequests,
)
from ..papers import PaperService, PaperStore
from ..papers.transport import UrllibTransport
from ..research.filing_access import as_of_timestamp
from .common import atomic_write_json, file_sha256, now_utc
from .configuration import PROJECT_ROOT, config_path


def paper_relevant(namespace, title, abstract=""):
    text = (title + " " + (abstract or "")).lower()
    if re.search(
        r"noncommutative|quantum field|neutrino|galax|particle physics", text
    ):
        return False
    finance = re.search(
        r"stock.{0,25}(return|price|market)|equity|asset pric|portfolio|corporate financ|securities|earnings.?announcement|cash flow|financial (market|crash)",
        text,
    )
    focus = {
        "pod_fundamentals": r"earnings|profit|account|cash|quality",
        "pod_events": r"issuance|issue|repurchase|event|announcement|earnings|financing|filing",
        "pod_valuation": r"value|valuation|book.to.market|cash flow|profitab|dividend|pricing",
        "pod_dynamics": r"momentum|trend|reversal|risk|volatility|trading",
    }.get(namespace, r".")
    return bool(finance and re.search(focus, text))


def discover_evidence(config, state, *, papers=None, carbon=None):
    application = config["application"]
    papers = papers or PaperService(
        PaperStore(config_path(application["papers_db"])),
        clock=now_utc,
        online=application.get("online_papers", False),
        transport=UrllibTransport()
        if application.get("online_papers")
        else None,
        timeout=15,
        max_version_lookups=2,
    )
    policy = config["carbon"]
    carbon = carbon or CarbonArcRequests(
        state / "carbon",
        clock=now_utc,
        transport=CarbonArcOAuthTransport(
            PROJECT_ROOT, online=policy["allow_remote"]
        ),
        policy=CarbonArcPolicy.model_validate(
            {
                k: v
                for k, v in policy.items()
                if k not in {"capabilities", "requests"}
            }
        ),
    )
    evidence = {
        "papers": [],
        "alternative_data": [],
        "errors": [],
        "excluded_papers": [],
    }
    cache = state / "campaign_evidence.json"
    # Persist discovery separately: a completed campaign can replay with no network.
    identity = {
        "relevance_policy": "finance_and_pod_scope_v1",
        "online_papers": application.get("online_papers", False),
        "papers_per_query": application.get("papers_per_query", 2),
        "queries": application.get("paper_queries", []),
        "paper_ids": application.get("paper_ids", []),
        "cutoff": config["training_lab"]["end"],
        "capabilities": policy["capabilities"],
        "requests": policy["requests"],
        "discovery": config["research_campaign"].get("alt_discovery", []),
    }
    if cache.exists():
        previous = json.loads(cache.read_text())
        failed = previous.get("errors") or any(
            r.get("status") in {"failed", "uncertain", "remote_not_configured"}
            for r in previous.get("alternative_data", [])
        )
        if (
            previous.get("identity") == identity
            and not failed
            and not config["research_campaign"].get("refresh_evidence", False)
        ):
            return previous, carbon
    cutoff = dt.datetime.combine(
        dt.date.fromisoformat(config["training_lab"]["end"]),
        dt.time.max,
        dt.timezone.utc,
    )
    cards = []
    for query in application.get("paper_queries", []):
        try:
            result = papers.search(
                query["query"],
                as_of=cutoff,
                limit=min(12, application.get("papers_per_query", 2) * 4),
            )
            relevant = []
            for card in result.cards:
                if paper_relevant(
                    query["namespace"], card.title, card.abstract
                ):
                    relevant.append(card)
                else:
                    evidence["excluded_papers"].append(
                        {
                            "paper_id": card.paper_id,
                            "title": card.title,
                            "namespace": query["namespace"],
                            "reason": "outside_finance_or_pod_scope",
                        }
                    )
            cards.extend(
                (query["namespace"], card)
                for card in relevant[: application.get("papers_per_query", 2)]
            )
        except Exception as exc:
            evidence["errors"].append(
                {
                    "stage": "paper_search",
                    "query": query["query"],
                    "error": type(exc).__name__,
                    "detail": str(exc)[:400],
                }
            )
    for paper_id in application.get("paper_ids", []):
        try:
            cards.append(("all", papers.get(paper_id, as_of=cutoff)))
        except Exception as exc:
            evidence["errors"].append(
                {
                    "stage": "paper_get",
                    "paper_id": paper_id,
                    "error": type(exc).__name__,
                }
            )
    for namespace, card in cards:
        text = (
            papers.text_of(card, "full_text")
            or papers.text_of(card, "abstract")
            or card.abstract
            or ""
        )
        evidence["papers"].append(
            {
                "evidence_id": card.paper_id + ":" + card.version_key,
                "namespace": namespace,
                "title": card.title,
                "version_date": card.version_date.isoformat()
                if card.version_date
                else None,
                "source": card.source,
                "text": text[:16000],
                "text_scope": "full_text_excerpt"
                if papers.text_of(card, "full_text")
                else "abstract",
                "claims": [c.model_dump(mode="json") for c in card.claims],
            }
        )
    for capability in policy["capabilities"]:
        carbon.register_capability(capability)
        try:
            evidence["alternative_data"].append(
                carbon.describe(
                    capability["namespace"], capability["capability_id"]
                )
            )
        except Exception as exc:
            evidence["errors"].append(
                {
                    "stage": "alternative_metadata",
                    "error": type(exc).__name__,
                    "detail": str(exc)[:400],
                }
            )
    for discovery in config["research_campaign"].get("alt_discovery", []):
        try:
            evidence["alternative_data"].append(
                carbon.discover_stock(
                    discovery["namespace"],
                    discovery["request"],
                    retry_failed=True,
                    max_attempts=3,
                )
            )
        except Exception as exc:
            evidence["errors"].append(
                {
                    "stage": "alternative_discovery",
                    "error": type(exc).__name__,
                    "detail": str(exc)[:400],
                }
            )
    for request in policy["requests"]:
        try:
            if request["request"]["end"] > config["training_lab"]["end"]:
                raise ValueError(
                    "Alternative research request crosses training end"
                )
            evidence["alternative_data"].append(
                carbon.request(request["namespace"], request["request"])
            )
        except Exception as exc:
            evidence["errors"].append(
                {
                    "stage": "alternative_request",
                    "error": type(exc).__name__,
                    "detail": str(exc)[:400],
                }
            )
    evidence["identity"] = identity
    atomic_write_json(cache, evidence)
    return evidence, carbon


def attach_alternative_features(panel, bindings, carbon):
    """Join actual receipts, retaining retrieval-time knowledge absent vintage proof."""
    audit = []
    instants = np.array([as_of_timestamp(d).value for d in panel.dates])
    locations = {int(p): i for i, p in enumerate(panel.securities)}
    for binding in bindings:
        name = binding["feature_name"]
        if name in panel.features:
            raise ValueError("Alternative feature names must be distinct")
        receipt = carbon.read(binding["namespace"], binding["request_id"])
        if receipt["status"] != "data_ready":
            raise ValueError("A ready provider receipt is required")
        if file_sha256(receipt["data_path"]) != receipt["data_sha256"]:
            raise ValueError("Alternative data receipt hash changed")
        frame = pd.read_csv(receipt["data_path"])
        observed = pd.to_datetime(frame["date"], utc=True)
        available = pd.to_datetime(
            frame["available_at"], utc=True, errors="coerce"
        )
        # Receipt retrieval is the conservative knowledge boundary. A raw vendor
        # release column does not establish that historical revisions were known.
        retrieved = pd.Timestamp(receipt["retrieved_at"])
        available = pd.concat(
            [available.rename("release"), observed.rename("period")], axis=1
        ).max(axis=1)
        available = available.clip(lower=retrieved)
        frame = frame.assign(_known=available).sort_values("_known")
        values = np.full(panel.mask.shape, np.nan, dtype=np.float32)
        column = locations.get(int(binding["permno"]))
        if column is not None:
            times = frame._known.astype("int64").to_numpy()
            selected = times.searchsorted(instants, side="right") - 1
            valid = selected >= 0
            values[valid, column] = pd.to_numeric(
                frame.value, errors="coerce"
            ).to_numpy()[selected[valid]]
        panel.features[name] = values
        audit.append(
            {
                "feature": name,
                "request_id": binding["request_id"],
                "research_observations": int(np.isfinite(values).sum()),
                "knowledge_policy": "max(period, release, receipt_retrieval)",
                "data_sha256": receipt["data_sha256"],
            }
        )
    return audit


def load_reporting_benchmarks(path, dates):
    """Read only FRED benchmark rows; never expose the external table to research.

    The legacy container is an explicitly named reporting-only input. IBES,
    FINRA and other rows cannot cross this projection/filter boundary.
    """
    chunks = []
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(
        columns=["table", "series", "date", "value"], batch_size=32768
    ):
        selected = batch.filter(
            pc.and_(
                pc.equal(batch.column("table"), "fred"),
                pc.is_in(
                    batch.column("series"),
                    value_set=pa.array(["TB3MS", "SP500"]),
                ),
            )
        )
        if len(selected):
            chunks.append(selected.to_pandas())
    if not chunks:
        raise ValueError("Official benchmark series are absent")
    source = pd.concat(chunks, ignore_index=True)
    source["date"] = pd.to_datetime(source.date)
    dates = pd.DatetimeIndex(dates)
    source = source.loc[source.date <= dates.max()]
    rate = (
        source.loc[source.series == "TB3MS"]
        .set_index("date")
        .value.dropna()
        .astype(float)
        .sort_index()
    )
    # The challenge benchmark uses the rate for the SAME return month. It is
    # an ex-post reporting reference, never an alpha input. Cash accrual uses
    # a separate lagged observable proxy so capital sizing stays causal.
    reference = rate.reindex(dates.to_period("M").to_timestamp()).set_axis(
        dates
    )
    # Monthly observations become known on the first day of the following month.
    rate.index = rate.index + pd.offsets.MonthBegin(1)
    cash = (
        rate.reindex(rate.index.union(dates))
        .sort_index()
        .ffill()
        .reindex(dates)
    )
    prices = (
        source.loc[source.series == "SP500"]
        .set_index("date")
        .value.dropna()
        .astype(float)
        .sort_index()
    )
    market = prices.pct_change(fill_method=None).reindex(dates)
    if cash.isna().any():
        raise ValueError("TB3MS reporting coverage is incomplete")
    if reference.isna().any():
        raise ValueError("Same-month official benchmark rates are incomplete")
    return pd.DataFrame(
        {"tb3ms": cash, "reference_tb3ms": reference, "sp_return": market},
        index=dates,
    )
