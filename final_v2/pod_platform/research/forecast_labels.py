"""Observed forecast labels kept separate from monthly execution accounting."""

from __future__ import annotations

from dataclasses import replace
import hashlib

import numpy as np
import pandas as pd


ISSUER_FOLD_POLICY = "issuer_identity_blake2b_v1"


def prediction_horizon(strategy):
    """Resolve the declared forecast life in observed market sessions."""
    explicit = getattr(strategy, "forecast_horizon_days", None)
    horizon = explicit if explicit is not None else strategy.expected_holding_days
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or not 1 <= horizon <= 504:
        raise ValueError("Forecast horizon must be an integer from 1 to 504 sessions")
    return int(horizon)


def forecast_training_panel(panel, market, horizon, *, target="raw_return", universe=None):
    """Label complete close-to-close horizons without extending partial paths.

    Entry requires the observed execution quote. Missing held returns exit at
    the last known price, matching the normal replay; the remaining interval is
    cash. A label is usable only after its full horizon has been observed, even
    when that issuer exited earlier. The input monthly accounting is untouched.
    The panel retains raw paths; peer-relative transformations are deferred
    until each fit has excluded its validation issuers from the training cohort.
    """
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or not 1 <= horizon <= 504:
        raise ValueError("Forecast horizon must be an integer from 1 to 504 sessions")
    if target not in {"raw_return", "peer_relative_return"}:
        raise ValueError("Forecast target must be raw_return or peer_relative_return")
    horizon = int(horizon)
    rows, names = panel.returns.shape
    if market.returns.shape != market.tradable.shape or market.returns.shape[1] != names:
        raise ValueError("Forecast labels require the same observed market security order")
    if len(panel.execution_dates) != rows or len(panel.mask) < rows:
        raise ValueError("Forecast labels require one execution close per label row")
    labels = np.full((rows, names), np.nan, dtype=np.float64)
    entries = np.zeros((rows, names), dtype=bool)
    exits = np.zeros((rows, names), dtype=bool)
    settlements = np.full(rows, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    for k, execution in enumerate(panel.execution_dates):
        begin = int(market.dates.searchsorted(execution))
        finish = begin + horizon
        if begin >= len(market.dates) or market.dates[begin] != execution:
            raise ValueError("Forecast entry must use an observed execution close")
        if finish >= len(market.dates):
            continue
        settlements[k] = market.dates[finish].to_datetime64()
        entry = panel.mask[k] & market.tradable[begin]
        columns = np.flatnonzero(entry)
        if not len(columns):
            continue
        path = np.asarray(market.returns[begin + 1 : finish + 1, columns], dtype=np.float64)
        alive = np.logical_and.accumulate(np.isfinite(path) & (path >= -1), axis=0)
        labels[k, columns] = np.prod(np.where(alive, 1 + path, 1), axis=0) - 1
        entries[k] = entry
        exits[k, columns] = ~alive[-1]
    contract = {
        "horizon_sessions": horizon,
        "origin": "observed execution close",
        "settlement": "close after the full declared number of observed sessions",
        "entry": "declared formation cohort and observed positive quote",
        "missing_held_return": "exit at last known price and remain cash",
        "incomplete_horizon": "missing label and missing settlement; never truncate",
    }
    metadata = {**panel.metadata, "forecast_horizon_days": horizon,
        "forecast_label_contract": contract, "forecast_target": target}
    if target == "peer_relative_return":
        universe = universe if universe is not None else panel.metadata.get("strategy_universe", {})
        prefixes = (universe.get("gics_prefixes", []) if isinstance(universe, dict)
            else universe.gics_prefixes)
        metadata["forecast_peer_policy"] = {
            "industry_digits": 4 if any(len(prefix) >= 4 for prefix in prefixes) else 2,
            "size_bins": 3, "minimum_peer_names": 8,
            "size_order": "formation market cap, stable first rank",
            "benchmark": "leave-one-out equal-weight industry and size peer return",
            "cohort": "finite settled labels after training issuer exclusions",
        }
    else:
        metadata.pop("forecast_peer_policy", None)
    return replace(panel, returns=labels, settlement_dates=pd.DatetimeIndex(settlements),
        entry_mask=entries, exit_mask=exits, metadata=metadata)


def training_target(panel, formation, labels, eligible_mask):
    """Build a settled target using only the actually admitted training rows.

    Raw horizon returns remain on the panel so validation folds can construct
    their own peer benchmarks. Held-out issuers must be removed in the supplied
    mask before either size bins or peer outcomes are computed. The returned
    values follow the security order selected by the returned full-width mask.
    """
    labels = np.asarray(labels, dtype=float)
    mask = np.asarray(eligible_mask, dtype=bool).copy() & np.isfinite(labels)
    target = panel.metadata.get("forecast_target", "raw_return")
    if target == "raw_return":
        return mask, labels[mask]
    if target != "peer_relative_return":
        raise ValueError("Unknown forecast training target")
    policy = panel.metadata["forecast_peer_policy"]
    raw = panel.frames[formation]
    cap = raw.me.to_numpy(dtype=float)
    mask &= np.isfinite(cap)
    columns = np.flatnonzero(mask)
    if not len(columns):
        return mask, np.empty(0, dtype=float)
    sectors = raw.gics.fillna("").astype(str).str.replace(r"\.0$", "", regex=True)
    data = pd.DataFrame({"column": columns, "cap": cap[columns],
        "industry": sectors.str[:policy["industry_digits"]].to_numpy()[columns],
        "outcome": labels[columns]})
    data["size_bin"] = data.groupby("industry").cap.transform(
        lambda values: pd.qcut(values.rank(method="first"), policy["size_bins"],
            labels=False, duplicates="drop"))
    groups = data.groupby(["industry", "size_bin"])
    count = groups.outcome.transform("size")
    total = groups.outcome.transform("sum")
    supported = count >= policy["minimum_peer_names"]
    selected = data.loc[supported, "column"].to_numpy(dtype=int)
    mask[:] = False
    mask[selected] = True
    outcome = data.outcome.to_numpy()
    effect = outcome - np.divide(total.to_numpy() - outcome, count.to_numpy() - 1,
        out=np.full(len(data), np.nan), where=count.to_numpy() > 1)
    return mask, effect[supported.to_numpy()]


def issuer_validation_masks(securities, *, cohort=None):
    """Give each issuer a stable fold, unaffected by future cohort additions."""
    securities = np.asarray(securities)
    if len(np.unique(securities)) != len(securities) or len(securities) < 2:
        raise ValueError("Issuer validation requires at least two distinct issuers")
    cohort = np.ones(len(securities), dtype=bool) if cohort is None else np.asarray(cohort, dtype=bool)
    if cohort.shape != securities.shape or np.count_nonzero(cohort) < 2:
        raise ValueError("Issuer validation requires at least two issuers in the declared cohort")
    folds = np.fromiter((int.from_bytes(hashlib.blake2b(
        ("issuer_fold_v1/" + str(issuer)).encode("utf-8"), digest_size=8).digest(),
        "big") % 2 for issuer in securities), dtype=np.int8, count=len(securities))
    if not np.any((folds == 0) & cohort) or not np.any((folds == 1) & cohort):
        raise ValueError("Issuer validation requires nonempty cohorts in both identity hash folds")
    return tuple(folds == fold for fold in (0, 1))


def label_metadata(panel):
    """Copy label provenance into fitted state without embedding label arrays."""
    return {name: panel.metadata[name] for name in
        ("forecast_horizon_days", "forecast_label_contract", "forecast_target", "forecast_peer_policy")
        if name in panel.metadata}
