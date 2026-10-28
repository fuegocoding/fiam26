"""Formation-time panel and observed-price return intervals for development."""

from __future__ import annotations

import hashlib
import json
import mmap
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .filing_access import FilingResearchStore
from .input_permissions import COMPETITION_CHARACTERISTICS

POD_FEATURES = {
    "fundamentals": [
        "niq_su",
        "saleq_su",
        "niq_at_chg1",
        "niq_be_chg1",
        "sale_gr1",
        "saleq_gr1",
        "gp_at",
        "cop_at",
        "op_at",
        "ocf_at",
        "ocf_at_chg1",
        "qmj_prof",
        "qmj_growth",
        "f_score",
        "ni_inc8q",
        "earnings_variability",
        "ni_ivol",
        "cash_at",
    ],
    "events": [
        "eqnetis_at",
        "netis_at",
        "dbnetis_at",
        "chcsho_12m",
        "eqnpo_12m",
        "eqpo_me",
        "eqnpo_me",
        "oaccruals_at",
        "taccruals_at",
        "at_gr1",
        "inv_gr1a",
        "capx_gr1",
        "capex_abn",
        "emp_gr1",
        "noa_gr1a",
        "mispricing_mgmt",
        "debt_gr3",
        "sale_emp_gr1",
    ],
    "valuation": [
        "be_me",
        "ni_me",
        "sale_me",
        "ocf_me",
        "fcf_me",
        "ebitda_mev",
        "ebit_bev",
        "sale_bev",
        "bev_mev",
        "div12m_me",
        "rd_me",
        "intrinsic_value",
        "at_me",
        "aliq_mat",
        "debt_me",
        "netdebt_me",
        "eq_dur",
    ],
    "dynamics": [
        "ret_1_0",
        "ret_3_1",
        "ret_6_1",
        "ret_9_1",
        "ret_12_1",
        "ret_12_7",
        "resff3_12_1",
        "resff3_6_1",
        "seas_1_1an",
        "seas_2_5an",
        "ivol_capm_21d",
        "ivol_capm_252d",
        "rvol_21d",
        "rmax5_21d",
        "rmax5_rvol_21d",
        "prc_highprc_252d",
        "turnover_126d",
        "dolvol_var_126d",
        "bidaskhl_21d",
        "qmj_safety",
    ],
}


@dataclass
class TrainingPanel:
    """Arrays share formation dates and security order; no future features."""

    dates: pd.DatetimeIndex
    securities: np.ndarray
    frames: list[pd.DataFrame]
    mask: np.ndarray
    features: dict[str, np.ndarray]
    execution_dates: pd.DatetimeIndex
    settlement_dates: pd.DatetimeIndex
    returns: np.ndarray
    exit_mask: np.ndarray
    entry_mask: np.ndarray
    metadata: dict
    daily: pd.DataFrame | None = None


def market_return_series(panel):
    """Equal-weight the actual contemporaneous research universe for diagnostics."""
    active = panel.mask[:-1] & panel.entry_mask
    return np.divide(
        (panel.returns * active).sum(axis=1),
        active.sum(axis=1),
        out=np.zeros(len(panel.returns)),
        where=active.sum(axis=1) > 0,
    )


def read_panel(path: Path, settings: dict) -> pd.DataFrame:
    """Read authorized development rows, never projecting forward labels."""
    start = pd.Timestamp(settings["start"])
    end = pd.Timestamp(settings["end"])
    if start > end:
        raise ValueError("Development start must not follow its declared cutoff")
    columns = list(
        dict.fromkeys(
            ["permno", "eom", "me", "prc", "gics", "beta_60m", "dolvol_126d"]
            + list(COMPETITION_CHARACTERISTICS)
        )
    )
    frame = pd.read_parquet(
        path,
        columns=columns,
        filters=[("eom", ">=", start), ("eom", "<=", end)],
    )
    frame["eom"] = pd.to_datetime(frame.eom)
    if frame.empty or frame.duplicated(["permno", "eom"]).any():
        raise ValueError(
            "Nonempty, unique security/month observations required"
        )
    return frame.sort_values(["eom", "permno"])


def _file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_daily(path, securities, start, end, cache, *, compact=False):
    """Cache only development prices/returns for the formation-time universe."""
    identity = {
        "source": str(Path(path).resolve()),
        "source_sha256": _file_digest(path),
        "source_size": Path(path).stat().st_size,
        "source_mtime_ns": Path(path).stat().st_mtime_ns,
        "securities": [int(p) for p in securities],
        "start": pd.Timestamp(start).isoformat(),
        "end": pd.Timestamp(end).isoformat(),
        "compact": compact,
        "version": "dated_security_complete_v2",
    }
    signature = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()[:20]
    cache = cache.with_name(cache.stem + "_" + signature + cache.suffix)
    if cache.exists():
        receipt_path = cache.with_suffix(cache.suffix + ".identity.json")
        if not receipt_path.exists():
            raise ValueError("Daily cache has no identity receipt")
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("identity") != identity or receipt.get(
            "data_sha256"
        ) != _file_digest(cache):
            raise ValueError("Daily cache identity or content digest changed")
        daily = pd.read_parquet(
            cache, columns=["permno", "date", "prc", "daily_return"]
        )
        if (
            daily.date.min() < start
            or daily.date.max() > end
            or not set(daily.permno.unique()) <= set(securities)
        ):
            raise ValueError("Daily cache violates its requested identity")
        if compact:
            daily = daily.astype(
                {"permno": "int32", "prc": "float32", "daily_return": "float32"}
            )
        return daily
    chunks = []
    low, high = int(start.strftime("%Y%m%d")), int(end.strftime("%Y%m%d"))
    for chunk in pd.read_csv(
        path,
        chunksize=100_000,
        usecols=["permno", "date", "prc", "daily_return"],
        dtype={
            "permno": "int32",
            "date": "int32",
            "prc": "float32",
            "daily_return": "float32",
        }
        if compact
        else None,
    ):
        selected = chunk.loc[
            chunk.permno.isin(securities) & chunk.date.between(low, high)
        ].copy()
        if len(selected):
            chunks.append(selected)
    if not chunks:
        raise ValueError("No supplied daily observations in development range")
    daily = pd.concat(chunks, ignore_index=True)
    daily["date"] = pd.to_datetime(daily.date.astype(str), format="%Y%m%d")
    daily["daily_return"] = pd.to_numeric(daily.daily_return, errors="coerce")
    daily["prc"] = pd.to_numeric(daily.prc, errors="coerce").abs()
    if daily.duplicated(["permno", "date"]).any():
        raise ValueError("Duplicate daily security/date observations")
    cache.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(cache, index=False)
    cache.with_suffix(cache.suffix + ".identity.json").write_text(
        json.dumps(
            {"identity": identity, "data_sha256": _file_digest(cache)},
            sort_keys=True,
        )
    )
    return daily


def interval_returns(daily, dates, securities, end):
    """Fill at the first observed close on or after the decision date.

    A missing held quote/return triggers exit at the last known price and cash
    for the rest of the interval, following the operator's existing exit rule.
    Entry requires an observed execution close. The final interval liquidates
    on the last supplied development session, without reading January 2021.
    """
    prices = daily.pivot(index="date", columns="permno", values="prc")
    returns = daily.pivot(index="date", columns="permno", values="daily_return")
    calendar = prices.index.sort_values()
    prices = prices.reindex(index=calendar, columns=securities).to_numpy()
    returns = returns.reindex(index=calendar, columns=securities).to_numpy()
    executions, settlements, realized, exits, entries = [], [], [], [], []
    for day in dates:
        begin = calendar.searchsorted(day, side="left")
        next_formation = day + pd.offsets.MonthEnd(1)
        finish = calendar.searchsorted(next_formation, side="left")
        finish = min(finish, len(calendar) - 1)
        if begin >= finish or calendar[finish] > end:
            raise ValueError("Insufficient post-execution development sessions")
        entry = np.isfinite(prices[begin]) & (prices[begin] > 0)
        path = returns[begin + 1 : finish + 1].astype(np.float64, copy=False)
        valid = np.isfinite(path) & (path >= -1)
        alive = np.logical_and.accumulate(valid, axis=0) & entry[None, :]
        growth = np.prod(np.where(alive, 1 + path, 1), axis=0) - 1
        realized.append(growth)
        exits.append(entry & ~alive[-1])
        entries.append(entry)
        executions.append(calendar[begin])
        settlements.append(calendar[finish])
    return (
        pd.DatetimeIndex(executions),
        pd.DatetimeIndex(settlements),
        np.asarray(realized),
        np.asarray(exits),
        np.asarray(entries),
    )


def prepare_panel(panel_path, daily_path, settings, cache):
    """Select each month's universe using only contemporaneous characteristics."""
    compact = settings.get("compact_storage", False)
    if pd.Timestamp(settings["start"]) > pd.Timestamp(settings["end"]):
        raise ValueError("Development start must not follow its declared cutoff")
    if compact:
        context_columns = [
            "permno",
            "eom",
            "me",
            "prc",
            "gics",
            "beta_60m",
            "dolvol_126d",
            "rvol_21d",
            "bidaskhl_21d",
        ]
        schema = pq.read_schema(panel_path).names
        context_columns += [c for c in ('ticker', 'company_name') if c in schema]
        raw = pd.read_parquet(
            panel_path,
            columns=context_columns,
            filters=[
                ("eom", ">=", pd.Timestamp(settings["start"])),
                ("eom", "<=", pd.Timestamp(settings["end"])),
            ],
        )
        raw["eom"] = pd.to_datetime(raw.eom)
        if raw.empty or raw.duplicated(["permno", "eom"]).any():
            raise ValueError(
                "Nonempty, unique security/month observations required"
            )
    else:
        raw = read_panel(panel_path, settings)
    observed_start, observed_end = raw.eom.min(), raw.eom.max()
    selected = raw.loc[
        (raw.me >= settings["min_market_cap_millions"])
        & (raw.prc.abs() >= settings["min_price"])
        & (raw.dolvol_126d >= settings["min_daily_dollar_volume"])
    ].copy()
    selected = selected.sort_values(
        ["eom", "me", "permno"], ascending=[True, False, True]
    )
    limit = settings.get("research_universe_size")
    if limit is not None:
        selected = selected.groupby("eom", sort=True).head(limit)
    dates = pd.DatetimeIndex(sorted(selected.eom.unique()))
    expected = pd.date_range(dates[0], dates[-1], freq="ME")
    if not dates.equals(expected):
        raise ValueError("Development panel must have every calendar month")
    counts = selected.groupby("eom").size()
    if counts.min() < settings["min_positions"]:
        raise ValueError("Insufficient contemporaneously eligible names")
    securities = np.sort(selected.permno.unique())
    frames = [
        g.set_index("permno").reindex(securities)
        for _, g in selected.groupby("eom", sort=True)
    ]
    mask = np.asarray([g.me.notna().to_numpy() for g in frames])
    features = {}
    feature_identity = {
        "panel": str(Path(panel_path).resolve()),
        "panel_sha256": _file_digest(panel_path),
        "source_size": Path(panel_path).stat().st_size,
        "source_mtime_ns": Path(panel_path).stat().st_mtime_ns,
        "securities": [int(p) for p in securities],
        "dates": [str(d) for d in dates],
        "mask": hashlib.sha256(mask.tobytes()).hexdigest(),
        "version": "ranked_compact_v2",
    }
    feature_signature = hashlib.sha256(
        json.dumps(feature_identity, sort_keys=True).encode()
    ).hexdigest()[:20]
    feature_directory = cache.parent / (
        cache.stem + "_" + feature_signature + "_features"
    )
    if compact:
        feature_directory.mkdir(parents=True, exist_ok=True)
    for name in COMPETITION_CHARACTERISTICS:
        if compact:
            feature_path = feature_directory / (name + ".npy")
            feature_receipt = feature_directory / (name + ".sha256")
            if feature_path.exists() and feature_receipt.exists():
                if (
                    _file_digest(feature_path)
                    != feature_receipt.read_text().strip()
                ):
                    raise ValueError(
                        "Ranked feature cache content digest changed"
                    )
                stored = np.load(feature_path, mmap_mode="r")
                if stored.shape != mask.shape or stored.dtype != np.dtype(
                    "float32"
                ):
                    raise ValueError("Ranked feature cache shape changed")
                features[name] = stored
                continue
            observation = pd.read_parquet(
                panel_path,
                columns=["permno", "eom", name],
                filters=[
                    ("eom", ">=", pd.Timestamp(settings["start"])),
                    ("eom", "<=", pd.Timestamp(settings["end"])),
                ],
            )
            observation["eom"] = pd.to_datetime(observation.eom)
            values = (
                observation.pivot(index="eom", columns="permno", values=name)
                .reindex(index=dates, columns=securities)
                .where(mask)
            )
            if name == "prc":
                values = values.abs()
            ranked = (
                values.replace([np.inf, -np.inf], np.nan)
                .rank(axis=1, pct=True)
                .to_numpy()
                * 2
                - 1
            ).astype(np.float32)
            temporary = feature_directory / (
                name + "." + uuid.uuid4().hex + ".tmp.npy"
            )
            destination = np.lib.format.open_memmap(
                temporary,
                mode="w+",
                dtype=np.float32,
                shape=ranked.shape,
            )
            destination[:] = ranked
            destination.flush()
            if hasattr(destination._mmap, "madvise"):
                destination._mmap.madvise(mmap.MADV_DONTNEED)
            del destination
            os.replace(temporary, feature_path)
            feature_receipt.write_text(_file_digest(feature_path))
            features[name] = np.load(feature_path, mmap_mode="r")
            del observation, values, ranked
        else:
            # Rank before any arithmetic; missing alpha is neutral, never an outcome.
            features[name] = np.asarray(
                [
                    (
                        (g[name].abs() if name == "prc" else g[name])
                        .replace([np.inf, -np.inf], np.nan)
                        .rank(pct=True)
                        * 2
                        - 1
                    ).to_numpy()
                    for g in frames
                ],
                dtype=np.float32,
            )
    source_rows, source_names, selected_rows = (
        len(raw),
        int(raw.permno.nunique()),
        len(selected),
    )
    filing_audit = None
    if settings.get("filings_research_enabled", False):
        if not settings.get("filings_path"):
            raise ValueError(
                "Enabled filing research requires the supplied corpus path"
            )
        store = FilingResearchStore(
            settings["filings_path"],
            training_end=settings["end"],
            start=settings["start"],
        )
        raw_filings = store.feature_arrays(
            dates.tz_localize("America/New_York"), securities
        )
        for name, values in raw_filings.items():
            features[name] = (
                pd.DataFrame(values)
                .where(mask)
                .rank(axis=1, pct=True)
                .to_numpy()
                * 2
                - 1
            ).astype(np.float32)
        filing_audit = store.audit()
    daily = read_daily(
        daily_path,
        securities,
        pd.Timestamp(settings.get("daily_warmup_start", "2014-01-01")),
        pd.Timestamp(settings["end"]),
        cache,
        compact=settings.get("compact_storage", False),
    )
    if settings.get("compact_storage", False):
        context = [
            "me",
            "prc",
            "gics",
            "beta_60m",
            "dolvol_126d",
            "rvol_21d",
            "bidaskhl_21d",
            "ticker",
            "company_name",
        ]
        frames = [
            frame[[c for c in context if c in frame]].copy() for frame in frames
        ]
        del raw, selected
    # Last formation is kept for a frozen next-period target, not scored using
    # outcomes beyond the development boundary.
    scored_dates = dates[:-1]
    execution, settlement, realized, exits, entries = interval_returns(
        daily, scored_dates, securities, pd.Timestamp(settings["end"])
    )
    return TrainingPanel(
        dates,
        securities,
        frames,
        mask,
        features,
        execution,
        settlement,
        realized,
        exits,
        entries,
        {
            "available_start": str(observed_start.date()),
            "raw_characteristic_source": str(Path(panel_path).resolve()),
            "source_stock_month_rows": source_rows,
            "source_unique_stocks": source_names,
            "rows_without_required_execution_context": source_rows
            - selected_rows,
            "available_end": str(observed_end.date()),
            "requested_start": settings["start"],
            "development_end": settings["end"],
            "formation_months": len(dates),
            "scored_months": len(scored_dates),
            "universe_min": int(counts.min()),
            "universe_max": int(counts.max()),
            "unique_securities": len(securities),
            "evaluation_outcomes_used": False,
            "label": "supplied_daily_total_return_after_observed_price_fill",
            "missing_return_rule": "exit_at_last_known_price_then_cash",
            "missing_entry_quotes": int((mask[:-1] & ~entries).sum()),
            "terminal_exit_observations": int((mask[:-1] & exits).sum()),
            "availability_assumption": "competition_month_end_characteristics_attestation",
            "available_characteristics": list(COMPETITION_CHARACTERISTICS),
            "available_numeric_research_features": sorted(features),
            "filing_research": filing_audit,
        },
        daily,
    )
