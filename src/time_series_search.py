"""Resumable, CPU-only FIAM time-series hypothesis screen; see docs/research/time_series_search.md."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse
from scipy.stats import t as student_t

from src.graph import expectations as ibes
from src.graph.walk_forward import hac_mean, split_masks

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger(__name__)
VERSION = 1
UNIVERSES = {"large_mega": {"large", "mega"}, "investable": {"small", "large", "mega"}, "micro": {"micro"}}
# Directions are economic hypotheses, fixed before inspecting these experiment results.
IDEAS = [
    ("01", "peer_residual_momentum", "Continue a three-month return trend relative to economic peers."),
    ("02", "return_acceleration", "Prefer stocks whose peer-relative trend is accelerating."),
    ("03", "trend_consistency", "Prefer persistent peer-relative winners over one-month jumps."),
    ("04", "drawdown_recovery", "Prefer stocks recovering from a deep trailing drawdown."),
    ("05", "volatility_contraction", "Test whether a positive trend with contracting volatility persists."),
    ("06", "downside_asymmetry", "Prefer trends with less downside than upside variation."),
    ("07", "liquidity_improvement", "Test whether improving trading liquidity predicts returns."),
    ("08", "attention_confirmation", "Test whether increasing turnover confirms a price trend."),
    ("09", "profitability_trajectory", "Prefer profitability improving over six months."),
    ("10", "cash_generation_trajectory", "Prefer improving operating cash flow relative to assets."),
    ("11", "deleveraging_trajectory", "Prefer falling leverage, controlling for current characteristics."),
    ("12", "accrual_cleanup", "Prefer declining operating accruals."),
    ("13", "valuation_own_history", "Prefer valuation cheaper than its trailing twelve-month norm."),
    ("14", "quality_value_divergence", "Prefer improving quality while valuation remains cheap."),
    ("15", "consensus_acceleration", "Prefer accelerating same-quarter analyst EPS revisions."),
    ("16", "revision_persistence", "Prefer consistent positive same-quarter revisions."),
    ("17", "disagreement_resolution", "Test revision direction confirmed by falling dispersion."),
    ("18", "disagreement_expansion", "Test revision direction accompanied by increasing dispersion."),
    ("19", "coverage_confirmation", "Test whether expanding analyst coverage confirms revisions."),
    ("20", "earnings_error_innovation", "Trade the latest earnings error relative to earlier errors."),
    ("21", "earnings_error_consistency", "Trade repeated beats/misses, discounting unstable error histories."),
    ("22", "peer_news_innovation", "Trade new peer earnings information relative to its trailing level."),
    ("23", "revision_vs_peers", "Trade revisions improving faster than economic peers' revisions."),
    ("24", "peer_information_wave", "Trade peer surprise direction when new peers are reporting."),
    ("25", "expectations_price_divergence", "Trade analyst revisions less reflected in peer-relative prices."),
    ("26", "timesfm_consistency", "Prefer TimesFM forecasts with a persistent recent direction."),
    ("27", "timesfm_innovation", "Trade changes in TimesFM forecasts relative to their recent level."),
    ("28", "timesfm_uncertainty", "Scale TimesFM's signal by its forecast interval width."),
    ("29", "return_ridge_snapshot", "Pooled next-return Ridge on current company/expectation state."),
    ("30", "return_ridge_trajectory", "Same Ridge plus the temporal feature block; compare with 29."),
    ("31", "surprise_ridge_snapshot", "Pooled next-announced-surprise Ridge on current state."),
    ("32", "surprise_ridge_trajectory", "Same surprise Ridge plus temporal features; compare with 31."),
    ("33", "ewma_risk_sizing", "Size a TimesFM book by trailing EWMA return volatility."),
    ("34", "dynamic_beta_sizing", "Tilt equity weights toward beta neutrality using trailing return betas."),
    ("35", "peer_reversal_exit", "Exit post-earnings positions when new peer news contradicts them."),
    ("36", "peer_completion_exit", "Exit post-earnings positions after comparable peers have mostly reported."),
]
BASELINES = [("B1", "momentum_baseline", "Current twelve-month momentum."),
             ("B2", "revision_baseline", "Current three-month same-quarter EPS revision."),
             ("B3", "timesfm_baseline", "Cached univariate TimesFM forecast."),
             ("B4", "earnings_baseline", "Last announced earnings surprise.")]
TIMESFM_DIAGNOSTICS = {"26", "27", "28", "33", "34", "B3"}
CONTROLS = ["market_equity", "be_me", "gp_at", "at_gr1", "ret_1_0", "ret_12_1", "ivol_capm_252d", "beta_60m"]
SNAPSHOT = CONTROLS + ["niq_at", "ocf_at", "debt_me", "oaccruals_at", "cons_p", "disp_p", "rev1_p", "rev3_p",
                       "breadth", "NUMEST", "hist_sue4_p", "hist_sue8_p", "beat_rate8", "peer_surprise_aligned"]


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=str, allow_nan=True) + "\n")
    tmp.replace(path)


def fingerprint(args):
    sources = [ibes.CHARACTERISTICS_FILE, ibes.CONSENSUS_FILE, ibes.PRICING_FILE,
               ibes.SURPRISE_FILE, ibes.RECOMMENDATION_FILE, ibes.LINK_FILE,
               ROOT / "output/preds_timesfm_all.parquet", ROOT / "data/external/SP500.csv", ROOT / "data/external/TB3MS.csv"]
    stats = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in sources]
    for p in sorted(ibes.GRAPH_ROOT.glob("*/nodes.parquet")) + sorted(ibes.GRAPH_ROOT.glob("*/adjacency.npz")):
        stats.append((str(p), p.stat().st_size, p.stat().st_mtime_ns))
    return hashlib.sha256(json.dumps({"version": VERSION, "code": Path(__file__).read_text(),
        "sources": stats, "cost_bp": args.cost_bp, "universes": args.universes,
        "holds": args.holds, "modes": args.modes}, sort_keys=True).encode()).hexdigest()


def calendar_lag(frame, columns, months, same_quarter=False):
    """Exact calendar lag. Missing months stay missing; quarterly targets never silently roll."""
    keys = ["permno", "eom"] + (["target_fpe"] if same_quarter else [])
    old = frame[keys + columns].copy()
    old["eom"] += pd.offsets.MonthEnd(months)
    return frame[keys].merge(old, on=keys, how="left", validate="one_to_one")[columns].set_axis(frame.index)


def rank_month(frame, values):
    return pd.Series(values, index=frame.index).groupby(frame.eom).rank(pct=True) - 0.5


def trailing(frame, column, window, operation="mean", minimum=None):
    """Includes current public observation, respects exact monthly gaps."""
    wide = frame.pivot(index="eom", columns="permno", values=column)
    wide = wide.reindex(pd.date_range(frame.eom.min(), frame.eom.max(), freq="ME"))
    rolling = wide.rolling(window, min_periods=minimum or max(2, window // 2))
    value = getattr(rolling, operation)()
    lookup = value.stack(future_stack=True)
    lookup.index.names = ["eom", "permno"]
    return lookup.reindex(pd.MultiIndex.from_frame(frame[["eom", "permno"]])).to_numpy()


def graph_features(panel):
    output = []
    by_month = dict(tuple(panel.groupby("eom", sort=False)))
    for directory in sorted(ibes.GRAPH_ROOT.glob("????-??")):
        eom = pd.Timestamp(directory.name) + pd.offsets.MonthEnd(0)
        if eom not in by_month:
            continue
        nodes = pd.read_parquet(directory / "nodes.parquet", columns=["permno"])
        matrix = sparse.load_npz(directory / "adjacency.npz").tocsr()
        f = nodes.merge(by_month[eom], on="permno", how="left", validate="one_to_one")
        now = float(ibes.to_days([eom])[0])
        ann = ibes.to_days(f.last_ann)
        last_fpe, target_fpe = ibes.to_days(f.last_fpe), ibes.to_days(f.target_fpe)
        sue = f.sue_last_p.to_numpy(float)
        fresh = np.isfinite(ann) & (ann <= now) & (ann > now - 92)
        shocks = ibes.peer_surprise(matrix, ibes.winsorize(np.where(fresh, sue, np.nan)),
                                    last_fpe, ann, target_fpe, now)
        coo = matrix.tocoo()
        rr, cc, ww = coo.row, coo.col, coo.data
        # Completion uses the issuer's already-announced quarter, not its future announcement.
        comparable = np.isfinite(last_fpe[rr]) & np.isfinite(last_fpe[cc]) & (abs(last_fpe[cc] - last_fpe[rr]) <= 46)
        complete = comparable & (ann[cc] <= now) & (ann[cc] > now - 180)
        new = comparable & (ann[cc] <= now) & (ann[cc] > now - 31) & np.isfinite(sue[cc])
        mass = np.bincount(rr, ww * new, minlength=len(f))
        news = np.bincount(rr, ww * new * np.nan_to_num(sue[cc]), minlength=len(f))
        output.append(pd.DataFrame({"permno": f.permno, "eom": eom,
            "peer_return": ibes.weighted_peer_mean(matrix, f.ret_exc),
            "peer_surprise_aligned": shocks["aligned"], "peer_surprise_recent": shocks["recent"],
            "peer_revision3": ibes.weighted_peer_mean(matrix, f.rev3_p),
            "peer_completion": np.bincount(rr, ww * complete, minlength=len(f)),
            "peer_new_mass": mass,
            "peer_new_news": np.divide(news, mass, out=np.full(len(f), np.nan), where=mass > 0)}))
    if not output:
        raise ValueError("No economic graph months found; build the graph first.")
    return pd.concat(output, ignore_index=True)


def prepare_panel(output):
    cache = output / "panel.parquet"
    if cache.exists():
        LOG.info("Loading cached temporal panel")
        return pd.read_parquet(cache)
    columns = list(dict.fromkeys(["permno", "eom", "size_grp", "gics", "common", "primary_sec", "exch_main",
        "prc", "me", "dolvol", "ret_exc", "ret_exc_lead1m", "ami_126d", "turnover_126d", "cash_at", *SNAPSHOT[:12]]))
    p = pq.read_table(ibes.CHARACTERISTICS_FILE, columns=columns).to_pandas()
    p.eom = pd.to_datetime(p.eom).astype("datetime64[ns]")
    p = p.sort_values(["permno", "eom"]).reset_index(drop=True)
    if p.duplicated(["permno", "eom"]).any():
        raise ValueError("Duplicate stock-months")
    LOG.info("Building point-in-time I/B/E/S state (no temporary-folder dependency)")
    state = ibes.build_expectation_state(p[["permno", "eom"]])
    state = state.merge(ibes.analyst_history(state), on=["permno", "eom"], validate="one_to_one")
    p = p.merge(state, on=["permno", "eom"], how="left", validate="one_to_one")
    LOG.info("Computing graph news clocks and peer-relative histories")
    p = p.merge(graph_features(p), on=["permno", "eom"], how="left", validate="one_to_one")
    tfm = pd.read_parquet(ROOT / "output/preds_timesfm_all.parquet",
                          columns=["permno", "eom", "tfm_forecast", "tfm_spread"])
    tfm.eom = pd.to_datetime(tfm.eom).astype("datetime64[ns]")
    p = p.merge(tfm, on=["permno", "eom"], how="left", validate="one_to_one").reset_index(drop=True)
    p["sector"] = p.gics.astype("string").str[:2]
    # Future target availability is deliberately absent from the eligibility rule.
    p["eligible"] = (p.prc.abs().ge(5) & p.dolvol.groupby(p.eom).rank(pct=True).ge(.2)
                     & p.common.eq(1) & p.primary_sec.eq(1) & p.exch_main.eq(1) & p.sector.notna())
    p["residual_return"] = p.ret_exc - p.peer_return
    p["residual_3"] = trailing(p, "residual_return", 3)
    p["residual_6"] = trailing(p, "residual_return", 6)
    p["residual_12"] = trailing(p, "residual_return", 12)
    p["return_vol3"] = trailing(p, "ret_exc", 3, "std", 3)
    p["return_vol12"] = trailing(p, "ret_exc", 12, "std", 6)
    p["up_month"] = np.where(p.residual_return.notna(), (p.residual_return > 0).astype(float), np.nan)
    p["up_frequency"] = trailing(p, "up_month", 6)
    p["down_square"] = p.ret_exc.clip(upper=0) ** 2
    p["up_square"] = p.ret_exc.clip(lower=0) ** 2
    p["down_risk"] = trailing(p, "down_square", 12)
    p["up_risk"] = trailing(p, "up_square", 12)
    # Local drawdown uses only twelve realized monthly returns, with no global future maximum.
    wide = p.pivot(index="eom", columns="permno", values="ret_exc").reindex(pd.date_range(p.eom.min(), p.eom.max(), freq="ME"))
    def drawdown(a):
        if not np.isfinite(a).all():
            return np.nan
        wealth = np.cumprod(1 + a)
        return wealth[-1] / max(1., float(wealth.max())) - 1
    dd = wide.rolling(12, min_periods=12).apply(drawdown, raw=True).stack(future_stack=True)
    dd.index.names = ["eom", "permno"]
    p["drawdown12"] = dd.reindex(pd.MultiIndex.from_frame(p[["eom", "permno"]])).to_numpy()
    ew = wide.ewm(alpha=2/7, min_periods=6, adjust=False).std().stack(future_stack=True)
    ew.index.names = ["eom", "permno"]
    p["ewma_vol"] = ew.reindex(pd.MultiIndex.from_frame(p[["eom", "permno"]])).to_numpy()
    for col in ["niq_at", "ocf_at", "debt_me", "oaccruals_at", "gp_at", "cash_at", "ami_126d", "turnover_126d"]:
        p[col + "_delta6"] = p[col] - calendar_lag(p, [col], 6)[col]
    p["value_gap"] = p.be_me - trailing(p, "be_me", 12)
    lag1 = calendar_lag(p, ["MEANEST", "STDEV", "NUMEST"], 1, same_quarter=True)
    lag2 = calendar_lag(p, ["MEANEST"], 2, same_quarter=True)
    p["revision_accel"] = (p.MEANEST - 2 * lag1.MEANEST + lag2.MEANEST) / p.PRICE
    p["dispersion_delta"] = (p.STDEV - lag1.STDEV) / p.PRICE
    p["coverage_delta"] = p.NUMEST - lag1.NUMEST
    p["revision_step"] = (p.MEANEST - lag1.MEANEST) / p.PRICE
    p["revision_step_prev"] = (lag1.MEANEST - lag2.MEANEST) / p.PRICE
    p["peer_innovation"] = p.peer_surprise_recent - trailing(p, "peer_surprise_recent", 3)
    p["tfm_mean3"] = trailing(p, "tfm_forecast", 3)
    p["tfm_delta"] = p.tfm_forecast - p.tfm_mean3
    # Past return beta only: the current realized return is known at formation month-end.
    market, _ = external_series()
    m = market.reindex(wide.index)
    cov = wide.rolling(24, min_periods=12).cov(m)
    betas = cov.div(m.rolling(24, min_periods=12).var(), axis=0).stack(future_stack=True)
    betas.index.names = ["eom", "permno"]
    p["dynamic_beta"] = betas.reindex(pd.MultiIndex.from_frame(p[["eom", "permno"]])).to_numpy()
    p.replace([np.inf, -np.inf], np.nan, inplace=True)
    p.to_parquet(cache, index=False)
    LOG.info("Temporal panel cached: %s rows", len(p))
    return p


def external_series():
    bills = pd.read_csv(ROOT / "data/external/TB3MS.csv")
    bills.index = pd.to_datetime(bills.observation_date) + pd.offsets.MonthEnd(0)
    cash = pd.to_numeric(bills.TB3MS, errors="coerce") / 1200
    sp = pd.read_csv(ROOT / "data/external/SP500.csv")
    sp.index = pd.to_datetime(sp.observation_date)
    price = pd.to_numeric(sp.SP500, errors="coerce").resample("ME").last()
    market = price.pct_change(fill_method=None) - cash.reindex(price.index)
    # Market is a price-index excess-return proxy, not an S&P total-return series.
    return market, cash


def signal_columns(p):
    r = lambda v: rank_month(p, v)
    trend = r(p.residual_3)
    return {
        "01": p.residual_3,
        "02": p.residual_3 - p.residual_6,
        "03": trend * (2 * p.up_frequency - 1).abs(),
        "04": r(-p.drawdown12) + r(p.residual_return),
        "05": trend * (1 + r(p.return_vol12 - p.return_vol3)),
        "06": trend * (1 + r(p.up_risk - p.down_risk)),
        "07": -p.ami_126d_delta6,
        "08": trend * (1 + r(p.turnover_126d_delta6)),
        "09": p.niq_at_delta6,
        "10": p.ocf_at_delta6,
        "11": -p.debt_me_delta6,
        "12": -p.oaccruals_at_delta6,
        "13": p.value_gap,
        "14": r(p.gp_at_delta6) + r(p.be_me),
        "15": p.revision_accel,
        "16": r(p.revision_step + p.revision_step_prev).where(p.revision_step * p.revision_step_prev > 0),
        "17": r(p.revision_step) * (1 + r(-p.dispersion_delta)),
        "18": r(p.revision_step) * (1 + r(p.dispersion_delta)),
        "19": r(p.revision_step) * (1 + r(p.coverage_delta)),
        # hist_sue4 includes the latest surprise; subtracting its mean rescales the innovation.
        "20": p.sue_last_p - p.hist_sue4_p,
        "21": p.hist_sue4_p / (p.sue_vol8_p + .001),
        "22": p.peer_innovation,
        "23": p.rev3_p - p.peer_revision3,
        "24": p.peer_surprise_aligned * p.peer_new_mass,
        "25": r(p.rev3_p) - r(p.residual_3),
        "26": p.tfm_mean3,
        "27": p.tfm_delta,
        "28": p.tfm_forecast / p.tfm_spread.clip(lower=.01),
        "33": p.tfm_forecast,
        "34": p.tfm_forecast,
        "35": p.sue_last_p.where(p.days_since_ann <= 31),
        "36": p.sue_last_p.where(p.days_since_ann <= 31),
        "B1": p.ret_12_1, "B2": p.rev3_p, "B3": p.tfm_forecast, "B4": p.sue_last_p,
    }


def temporal_features(p):
    return ["residual_3", "residual_6", "residual_12", "up_frequency", "drawdown12", "return_vol3", "return_vol12",
            "niq_at_delta6", "ocf_at_delta6", "debt_me_delta6", "oaccruals_at_delta6", "gp_at_delta6", "cash_at_delta6",
            "value_gap", "revision_accel", "dispersion_delta", "coverage_delta", "revision_step", "revision_step_prev",
            "peer_innovation", "peer_new_mass", "peer_completion"]


def ridge_solve(x, y, penalty):
    design = np.column_stack([np.ones(len(x)), x])
    reg = np.eye(design.shape[1]) * penalty * len(x)
    reg[0, 0] = 0
    return np.linalg.solve(design.T @ design + reg, design.T @ y)


def ridge_forecasts(p, idea, output):
    """Each year's model scores its own validation/test rows. No generic cached OOS validation."""
    path = output / f"model_{idea}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    features = SNAPSHOT + (temporal_features(p) if idea in {"30", "32"} else [])
    x = (p[features].groupby(p.eom).rank(pct=True) - .5).fillna(0).to_numpy(float)
    x = np.column_stack([x, p[features].isna().to_numpy(float)])
    target = "target_surprise_p" if idea in {"31", "32"} else "ret_exc_lead1m"
    y = p[target].to_numpy(float)
    frames, audit = [], []
    for year in range(2021, 2027):
        masks = split_masks(p.eom, 1, year, label_dates=p.target_ann if target.startswith("target_") else None)
        train = masks["train"] & np.isfinite(y)
        val = masks["validation"] & np.isfinite(y)
        if train.sum() < 500 or val.sum() < 100:
            raise ValueError(f"Insufficient {target} train/validation observations for {year}")
        # For surprise learning, rank labels within the admissible fold rows only.
        train_y = y[train]
        val_y = y[val]
        if target.startswith("target_"):
            train_y = rank_month(p.loc[train], train_y).to_numpy()
            val_y = rank_month(p.loc[val], val_y).to_numpy()
        best = None
        for penalty in [.001, .01, .1, 1.]:
            coef = ridge_solve(x[train], train_y, penalty)
            pred = coef[0] + x[val] @ coef[1:]
            loss = float(np.mean((val_y - pred) ** 2))
            if best is None or loss < best[0]:
                best = (loss, penalty, coef)
        # Label-admissibility controls fitting/scoring loss, not who receives a historical forecast.
        # In particular, late validation labels must not remove stocks from validation portfolios.
        forecast_windows = split_masks(p.eom, 1, year)
        keep = forecast_windows["validation"] | forecast_windows["test"]
        f = p.loc[keep, ["permno", "eom"]].copy()
        f["prediction"] = best[2][0] + x[keep] @ best[2][1:]
        f["fold"] = year
        f["window"] = np.where(forecast_windows["test"][keep], "test", "validation")
        frames.append(f)
        test = masks["test"] & np.isfinite(y)
        test_pred = best[2][0] + x[test] @ best[2][1:]
        audit.append({"fold": year, "target": target, "features": features, "penalty": best[1],
            "train_rows": int(train.sum()), "validation_rows": int(val.sum()),
            "r2_oos_zero": float(1 - np.sum((y[test] - test_pred) ** 2) / np.sum(y[test] ** 2)) if target == "ret_exc_lead1m" else None,
            "test_rank_ic": ibes.monthly_rank_ic(p.loc[test, "eom"], y[test], test_pred)})
        LOG.info("Model %s fold %s fitted", idea, year)
    result = pd.concat(frames, ignore_index=True)
    result.to_parquet(path, index=False)
    atomic_json(output / f"model_{idea}_audit.json", audit)
    return result


def allocate_quotas(capacity, seats):
    """Deterministic proportional allocation, never exceeding per-sector capacity."""
    capacity = np.asarray(capacity, int)
    seats = min(int(seats), int(capacity.sum()))
    if seats == 0:
        return np.zeros_like(capacity)
    ideal = capacity / capacity.sum() * seats
    q = np.floor(ideal).astype(int)
    while q.sum() < seats:
        available = q < capacity
        j = np.argmax(np.where(available, ideal - q, -np.inf))
        q[j] += 1
    return q


def weights_from_score(score, eligible, sectors, controls, mode):
    """Same sector seat quotas for both legs; target outcomes never used."""
    good = eligible & np.isfinite(score) & (sectors >= 0)
    out = np.zeros(len(score))
    if good.sum() < 100:
        return out
    s = score.copy()
    if mode == "factor":
        x = controls[good]
        sec = sectors[good]
        dummies = np.column_stack([(sec == j).astype(float) for j in np.unique(sec)])
        x = np.column_stack([x, dummies])
        s[good] -= x @ np.linalg.lstsq(x, s[good], rcond=None)[0]
    labels = np.unique(sectors[good])
    groups = [np.flatnonzero(good & (sectors == j)) for j in labels]
    quotas = allocate_quotas([len(g) // 2 for g in groups], 50)
    if quotas.sum() < 50:
        return out
    for g, q in zip(groups, quotas):
        if q == 0:
            continue
        # Position index follows sorted PERMNO. Fixed hash avoids listing-age ties.
        tie = ((g.astype(np.uint64) * np.uint64(2654435761)) % np.uint64(1000003)).astype(float)
        order = np.lexsort((tie, s[g]))
        out[g[order[:q]]] = -1 / 50
        out[g[order[-q:]]] = 1 / 50
    return out


def constrain_weights(weights, sectors, vol=None, beta=None):
    """Cap union at 500; enforce equal sector long/short dollars after optional risk tilts."""
    w = weights.copy()
    long, short = w > 0, w < 0
    labels = np.unique(sectors[long | short])
    capacity = [min(np.sum(long & (sectors == j)), np.sum(short & (sectors == j))) for j in labels]
    if sum(capacity) < 50:
        return np.zeros(len(w))
    quotas = allocate_quotas(capacity, 250)
    keep = np.zeros(len(w), bool)
    for j, q in zip(labels, quotas):
        for side in [long, short]:
            ix = np.flatnonzero(side & (sectors == j))
            order = np.argsort(-np.abs(w[ix]), kind="stable")
            keep[ix[order[:q]]] = True
    w[~keep] = 0
    if vol is not None:
        usable = vol[np.isfinite(vol) & (vol > 0)]
        floor = max(.02, float(np.quantile(usable, .1))) if len(usable) else .02
        w /= np.where(np.isfinite(vol), np.maximum(vol, floor), floor)
    if beta is not None:
        b = np.where(np.isfinite(beta), np.clip(beta, -3, 5), 1.)
        # Bounded exponential tilt, then restore sector/dollar neutrality at each iteration.
        for _ in range(12):
            w = normalize_sectors(w, sectors)
            gap = float(w @ b)
            if abs(gap) < .01:
                break
            w *= np.exp(np.clip(-.3 * gap * b * np.sign(w), -.2, .2))
    return normalize_sectors(w, sectors)


def normalize_sectors(w, sectors):
    out = np.zeros(len(w))
    labels = np.unique(sectors[w != 0])
    mass = []
    for j in labels:
        pos = w[(sectors == j) & (w > 0)].sum()
        neg = -w[(sectors == j) & (w < 0)].sum()
        mass.append(min(pos, neg))
    total = sum(mass)
    if total == 0:
        return out
    for j, target in zip(labels, mass):
        for sign in [1, -1]:
            mask = (sectors == j) & (w * sign > 0)
            denominator = np.abs(w[mask]).sum()
            if denominator > 0:
                out[mask] = w[mask] / denominator * target / total
    return out


class Market:
    def __init__(self, p):
        self.months = pd.date_range(p.eom.min(), p.eom.max(), freq="ME")
        self.ids = np.sort(p.permno.unique())
        self.p = p
        self.sector_labels = sorted(p.sector.dropna().unique())
        self.maps = {label: j for j, label in enumerate(self.sector_labels)}
        self.shape = (len(self.months), len(self.ids))
        self.returns = self.wide(p.ret_exc)
        # Provided forward targets can exist even when next month's characteristic row is absent.
        # Use these outcomes for P&L, and current-month returns only for historical features/risk.
        self.forward_returns = self.wide(p.ret_exc_lead1m)
        self.eligible = self.wide(p.eligible.astype(float), fill=0).astype(bool)
        self.sectors = self.wide(p.sector.map(self.maps), fill=-1).astype(int)
        self.controls = np.stack([self.wide(rank_month(p, p[c]), fill=0) for c in CONTROLS], axis=-1)
        self.vol = self.wide(p.ewma_vol)
        self.beta = self.wide(p.dynamic_beta)
        self.news = self.wide(p.peer_new_news)
        self.mass = self.wide(p.peer_new_mass, fill=0)
        self.completion = self.wide(p.peer_completion)
        self.quarters = self.wide(pd.to_datetime(p.last_fpe).map(lambda x: x.toordinal() if pd.notna(x) else np.nan))
        m, cash = external_series()
        self.market = m.reindex(self.months).to_numpy(float)
        self.cash = cash.reindex(self.months).to_numpy(float)

    def wide(self, values, fill=np.nan):
        f = self.p[["permno", "eom"]].assign(value=np.asarray(values))
        return f.pivot(index="eom", columns="permno", values="value").reindex(index=self.months, columns=self.ids).fillna(fill).to_numpy(float)


def simulate(m, scores, hold, mode, idea, cost_bp):
    """Mark actual holdings monthly; outcome gaps invalidate a month, never silently become zero."""
    cohorts = deque()
    previous = np.zeros(len(m.ids))
    records = []
    history = []
    for k in range(len(m.months) - 1):
        sec = m.sectors[k]
        # Current eligibility is known at formation; disappeared names retain prior sector labels.
        if k:
            sec = np.where(sec >= 0, sec, m.sectors[k-1])
        while cohorts and cohorts[0]["expiry"] <= k:
            cohorts.popleft()
        for cohort in cohorts:
            w = cohort["w"]
            if idea == "35":
                exit_now = (m.news[k] * w < 0) & (m.mass[k] >= .2)
                w[exit_now] = 0
            elif idea == "36":
                # Completion must relate to the entry quarter; rollover is also an information exit.
                exit_now = (m.completion[k] >= .8) | (m.quarters[k] != cohort["quarter"])
                w[exit_now & (w != 0)] = 0
            w[~m.eligible[k]] = 0
        entry = weights_from_score(scores[k], m.eligible[k], sec, m.controls[k], mode)
        cohorts.append({"w": entry, "expiry": k + hold, "quarter": m.quarters[k].copy()})
        summed = sum((c["w"] for c in cohorts), start=np.zeros(len(m.ids)))
        w = constrain_weights(summed, sec, vol=m.vol[k] if idea == "33" else None,
                              beta=m.beta[k] if idea == "34" else None)
        history.append(w.astype(np.float32))
        ret = m.forward_returns[k]
        held = w != 0
        missing = held & ~np.isfinite(ret)
        invalid = bool(missing.any())
        traded = float(np.abs(w - previous).sum())
        cost = traded * cost_bp / 10000
        gross = float(w @ np.nan_to_num(ret)) if not invalid else np.nan
        long = float(w.clip(min=0) @ np.nan_to_num(ret)) if not invalid else np.nan
        short = float(w.clip(max=0) @ np.nan_to_num(ret)) if not invalid else np.nan
        records.append({"date": m.months[k+1], "gross_excess": gross, "net_excess": gross - cost,
            "cost": cost, "turnover": traded, "n_long": int((w > 0).sum()), "n_short": int((w < 0).sum()),
            "gross_exposure": float(abs(w).sum()), "net_exposure": float(w.sum()), "missing_held_returns": int(missing.sum()),
            "long_pnl": long, "short_pnl": short, "max_position": float(abs(w).max()),
            "market_excess": m.market[k+1], "cash": m.cash[k+1],
            "exante_beta": float(w @ np.nan_to_num(m.beta[k], nan=1.)),
            "short_me": float(np.average(m.wide_me[k][w < 0], weights=-w[w < 0])) if (w < 0).any() else np.nan})
        # Drift holdings through realized returns for trading costs; normalize on portfolio NAV.
        # An unknown held outcome makes following turnover accounting uncertain and is flagged below.
        if not invalid:
            nav = 1 + m.cash[k+1] + gross - cost
            previous = w * (1 + np.nan_to_num(ret) + m.cash[k+1]) / nav if nav > 0 else np.full(len(w), np.nan)
        else:
            previous = w.copy()
        records[-1]["turnover_uncertain"] = invalid or bool(k and records[-2]["missing_held_returns"])
    result = pd.DataFrame(records)
    result.attrs["weights"] = np.asarray(history)
    return result


def account_selected_book(m, book, weights, cost_bp):
    """Recompute trades on the actual selected path, including annual switches and cash exits."""
    previous = np.zeros(len(m.ids))
    uncertain = False
    for i, row in book.iterrows():
        k = m.months.get_loc(row.date)
        w = weights[i].astype(float)
        traded = float(np.abs(w - previous).sum())
        cost = traded * cost_bp / 10000
        book.loc[i, "turnover"] = traded
        book.loc[i, "cost"] = cost
        book.loc[i, "net_excess"] = row.gross_excess - cost
        invalid = row.missing_held_returns > 0
        book.loc[i, "turnover_uncertain"] = uncertain or invalid
        if invalid:
            previous = w.copy()
        else:
            nav = 1 + row.cash + row.gross_excess - cost
            previous = w * (1 + np.nan_to_num(m.forward_returns[k-1]) + row.cash) / nav if nav > 0 else np.full(len(w), np.nan)
        uncertain = uncertain or invalid
    return book


def metrics(book):
    if book.empty:
        return {"months": 0, "information_ratio": None}
    net = book.net_excess
    active = net - .04 / 12  # Stated convention: NAV collateral earns this same TB3MS cash return.
    active_std = active.std(ddof=1)
    net_std = net.std(ddof=1)
    valid = net.notna()
    stats = hac_mean(net, maxlags=6)
    aligned = book.dropna(subset=["net_excess", "market_excess"])
    beta = alpha = alpha_t = beta_se = None
    if len(aligned) >= 12 and aligned.market_excess.std() > 0:
        x = np.column_stack([np.ones(len(aligned)), aligned.market_excess])
        y = aligned.net_excess.to_numpy()
        coef = np.linalg.lstsq(x, y, rcond=None)[0]
        u = y - x @ coef
        z = x * u[:, None]
        meat = z.T @ z
        for lag in range(1, min(7, len(y))):
            c = z[lag:].T @ z[:-lag]
            meat += (1 - lag / 7) * (c + c.T)
        bread = np.linalg.pinv(x.T @ x)
        se = np.sqrt(np.maximum(np.diag(bread @ meat @ bread), 0))
        alpha, beta = float(coef[0] * 12), float(coef[1])
        alpha_t = float(coef[0] / se[0]) if se[0] else None
        beta_se = float(se[1])
    total = 1 + book.cash + net
    wealth = total.cumprod(skipna=False)
    # Do not bridge an unknown return when reporting CAGR or drawdown.
    dd = wealth / wealth.cummax().clip(lower=1) - 1
    complete = valid.all() and book.cash.notna().all()
    names = book.n_long + book.n_short
    return {"months": len(book), "valid_months": int(valid.sum()), "missing_return_months": int((~valid).sum()),
        "annual_net_excess": float(net.mean() * 12), "annual_gross_excess": float(book.gross_excess.mean() * 12),
        "information_ratio": float(active.mean() / active_std * np.sqrt(12)) if active_std > 1e-12 else None,
        "sharpe": float(net.mean() / net_std * np.sqrt(12)) if net_std > 1e-12 else None,
        "hac_t": stats["hac_tstat"], "alpha_price_proxy": alpha, "alpha_hac_t": alpha_t, "beta_price_proxy": beta, "beta_se": beta_se,
        "total_cagr": float(wealth.iloc[-1] ** (12 / len(book)) - 1) if complete and wealth.iloc[-1] > 0 else None,
        "max_drawdown": float(dd.min()) if complete else None,
        "mean_turnover": float(book.turnover.mean()), "turnover_uncertain_months": int(book.turnover_uncertain.sum()),
        "active_months": int((book.gross_exposure > 0).sum()), "position_compliant_share": float(names.between(100, 500).mean()),
        "average_gross": float(book.gross_exposure.mean()), "maximum_gross": float(book.gross_exposure.max()),
        "maximum_abs_net": float(book.net_exposure.abs().max()), "maximum_position": float(book.max_position.max()),
        "average_short_me_millions": float(book.short_me.mean()),
        "long_pnl_annual": float(book.long_pnl.mean() * 12), "short_pnl_annual": float(book.short_pnl.mean() * 12),
        "annual_net_at_10_25_50bp": {str(bp): float((book.gross_excess - book.turnover * bp / 10000).mean() * 12) for bp in [10, 25, 50]},
        "hit_rate_active": float((active[valid] > 0).mean()),
        "annual_by_year": {str(y): float(g.net_excess.mean() * 12) for y, g in book.groupby(book.date.dt.year)}}


def select_validation(books, year):
    """Exit horizon chosen using that year's rolling validation only; fixed signal direction."""
    choices = []
    for hold, b in books.items():
        v = b[(b.date.dt.year >= year - 2) & (b.date.dt.year < year)]
        stat = metrics(v)
        ir = stat.get("information_ratio")
        if len(v) == 24 and stat["missing_return_months"] == 0 and ir is not None:
            choices.append((ir, -hold, hold, stat))
    if not choices:
        return None, None
    best = max(choices, key=lambda t: t[:2])
    return (best[2] if best[0] > 0 else None), best[3]


def summarize(output):
    rows = []
    for path in sorted((output / "results").glob("*.json")):
        data = json.loads(path.read_text())
        s = data["summary"]
        row = {"experiment": data["experiment"], "name": data["name"], "universe": data["universe"], "mode": data["mode"],
               "timesfm_cache_diagnostic": data["experiment"] in TIMESFM_DIAGNOSTICS, **s}
        row["annual_by_year"] = json.dumps(row["annual_by_year"])
        rows.append(row)
    if not rows:
        return
    f = pd.DataFrame(rows)
    # Rank is retrospective; it never feeds back into model/strategy choices.
    f = f.sort_values("information_ratio", ascending=False, na_position="last")
    f.to_csv(output / "leaderboard.csv", index=False)
    clean = f[(f.missing_return_months == 0) & (f.position_compliant_share == 1) & (f.beta_price_proxy.abs() <= .1) & ~f.timesfm_cache_diagnostic]
    clean.to_csv(output / "implementation_candidates.csv", index=False)
    paired = []
    for row in rows:
        idea = row["experiment"]
        if idea.startswith("B"):
            continue
        baseline = "29" if idea == "30" else "31" if idea == "32" else (
            "B3" if idea in {"26", "27", "28", "33", "34"} else
            "B2" if idea in {"15", "16", "17", "18", "19", "23", "25"} else
            "B4" if idea in {"20", "21", "22", "24", "31", "35", "36"} else "B1")
        a = output / "books" / f"{idea}_{row['universe']}_{row['mode']}.csv"
        b = output / "books" / f"{baseline}_{row['universe']}_{row['mode']}.csv"
        if not b.exists():
            continue
        pair = pd.read_csv(a, usecols=["date", "net_excess"]).merge(
            pd.read_csv(b, usecols=["date", "net_excess"]), on="date", suffixes=("_idea", "_baseline")).dropna()
        delta = pair.net_excess_idea - pair.net_excess_baseline
        stats = hac_mean(delta, maxlags=6)
        tstat = stats["hac_tstat"]
        pvalue = float(2 * student_t.sf(abs(tstat), max(1, len(delta)-1))) if tstat is not None else 1.
        paired.append({"experiment": idea, "baseline": baseline, "universe": row["universe"], "mode": row["mode"],
            "paired_months": len(delta), "incremental_annual_net": stats["mean"] * 12 if stats["mean"] is not None else None,
            "incremental_hac_t": tstat, "approximate_p": pvalue})
    if paired:
        compare = pd.DataFrame(paired).sort_values("approximate_p")
        compare["holm_adjusted_p"] = np.minimum(1, np.maximum.accumulate(
            compare.approximate_p.to_numpy() * np.arange(len(compare), 0, -1)))
        compare.to_csv(output / "paired_comparisons.csv", index=False)
    lines = ["# Time-series search results", "", "Retrospective hypothesis screen; not an untouched holdout or submission-ready strategy.", "",
             f"Completed combinations: {len(f)}. Implementation candidates: {len(clean)}.", "",
             "Returns use TB3MS collateral plus stock excess P&L, net of the stated one-way trading cost; borrow is not modeled.",
             "Market alpha/beta use an S&P price-index proxy. Review missing-return, cash-month, and position-count flags.", "",
             "26/27/28/33/34/B3 are cache diagnostics: the old TimesFM generator filtered on future outcome availability; regenerate safely before claiming an edge.", "",
             "| ID | Universe | Mode | Net excess / year | IR | Beta | Positions compliant | Missing months |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for _, r in f.head(40).iterrows():
        ir = f"{r.information_ratio:.2f}" if pd.notna(r.information_ratio) else "undefined"
        beta = f"{r.beta_price_proxy:.2f}" if pd.notna(r.beta_price_proxy) else "unavailable"
        lines.append(f"| {r.experiment} {r['name']} | {r.universe} | {r['mode']} | {r.annual_net_excess:.1%} | {ir} | {beta} | {r.position_compliant_share:.0%} | {r.missing_return_months} |")
    lines += ["", "Inspect paired_comparisons.csv for incremental net returns and approximate Holm-adjusted p-values across completed comparisons.",
              "These HAC p-values are approximate and do not correct for earlier research, the designed menu, or uncompleted comparisons.",
              "Inspect model_*_audit.json for prediction R²/IC, each result's validation choices, and monthly books for tail risk.",
              "A leaderboard winner is a new hypothesis requiring confirmation; correlated tests and historical research overlap limit inference."]
    (output / "report.md").write_text("\n".join(lines) + "\n")


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[logging.StreamHandler(), logging.FileHandler(output / "run.log")])
    digest = fingerprint(args)
    manifest = output / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text())["fingerprint"] != digest:
        raise ValueError("Inputs/settings/code changed. Use a new --output directory to avoid mixing experiments.")
    atomic_json(manifest, {"fingerprint": digest, "cost_bp_one_way": args.cost_bp, "holds": args.holds,
        "universes": args.universes, "modes": args.modes, "experiments": IDEAS, "baselines": BASELINES,
        "cash_convention": "TB3MS collateral + dollar-neutral stock excess P&L", "test_period": "2021-01 through 2026-08",
        "timesfm_cache_diagnostics": sorted(TIMESFM_DIAGNOSTICS),
        "timesfm_cache_warning": "Old generator excluded unavailable future targets; cached-forecast tests cannot establish clean alpha."})
    (output / "results").mkdir(exist_ok=True)
    (output / "books").mkdir(exist_ok=True)
    p = prepare_panel(output)
    m = Market(p)
    m.wide_me = m.wide(p.me)
    signals = signal_columns(p)
    selections = args.experiments.split(",") if args.experiments else [x[0] for x in IDEAS + BASELINES]
    unknown = set(selections) - {x[0] for x in IDEAS + BASELINES}
    if unknown:
        raise ValueError(f"Unknown experiment IDs: {unknown}")
    failures = []
    start = time.monotonic()
    for idea, name, description in IDEAS + BASELINES:
        if idea not in selections:
            continue
        try:
            model = ridge_forecasts(p, idea, output) if idea in {"29", "30", "31", "32"} else None
            wide_score = m.wide(signals[idea]) if model is None else None
            for universe in args.universes:
                eligibility = m.wide(p.size_grp.isin(UNIVERSES[universe]).astype(float), fill=0).astype(bool)
                original = m.eligible.copy()
                m.eligible &= eligibility
                try:
                    for mode in args.modes:
                        token = f"{idea}_{universe}_{mode}"
                        result_path = output / "results" / f"{token}.json"
                        if result_path.exists():
                            LOG.info("Resume: already complete %s", token)
                            continue
                        selected_books, selected_weights, choices = [], [], []
                        fixed_summary = {}
                        if model is None:
                            books = {hold: simulate(m, wide_score, hold, mode, idea, args.cost_bp) for hold in args.holds}
                            fixed_summary = {str(h): metrics(b[b.date.dt.year.between(2021, 2026)]) for h, b in books.items()}
                        for year in range(2021, 2027):
                            if model is not None:
                                current = model[model.fold == year].set_index(["permno", "eom"]).prediction
                                score = current.reindex(pd.MultiIndex.from_frame(p[["permno", "eom"]])).to_numpy()
                                wide = m.wide(score)
                                # Validation starts without older model-derived holdings, conservatively; no future model in historical books.
                                books = {hold: simulate(m, wide, hold, mode, idea, args.cost_bp) for hold in args.holds}
                            hold, val = select_validation(books, year)
                            template = next(iter(books.values()))
                            selected = (books[hold] if hold is not None else template).loc[template.date.dt.year == year].copy()
                            chosen_w = books[hold].attrs["weights"][selected.index] if hold is not None else np.zeros((len(selected), len(m.ids)), np.float32)
                            selected.attrs = {}
                            if hold is None:
                                for c in ["gross_excess", "net_excess", "cost", "turnover", "n_long", "n_short", "gross_exposure", "net_exposure", "missing_held_returns", "long_pnl", "short_pnl", "max_position", "exante_beta"]:
                                    selected[c] = 0
                                selected["short_me"] = np.nan
                                selected["turnover_uncertain"] = False
                            choices.append({"year": year, "chosen_hold": hold, "validation": val})
                            selected_books.append(selected)
                            selected_weights.append(chosen_w)
                        book = pd.concat(selected_books, ignore_index=True)
                        weights = np.concatenate(selected_weights)
                        book = account_selected_book(m, book, weights, args.cost_bp)
                        book.to_csv(output / "books" / f"{token}.csv", index=False)
                        month_ix, stock_ix = np.nonzero(weights)
                        pd.DataFrame({"date": book.date.to_numpy()[month_ix], "permno": m.ids[stock_ix],
                                      "weight": weights[month_ix, stock_ix]}).to_parquet(output / "books" / f"{token}_holdings.parquet", index=False)
                        result = {"experiment": idea, "name": name, "description": description, "universe": universe, "mode": mode,
                                  "summary": metrics(book), "choices": choices, "fixed_holds_diagnostic": fixed_summary,
                                  "timesfm_cache_diagnostic": idea in TIMESFM_DIAGNOSTICS}
                        atomic_json(result_path, result)
                        summarize(output)
                        LOG.info("Complete %s: IR=%s; missing months=%s", token, result["summary"]["information_ratio"], result["summary"]["missing_return_months"])
                finally:
                    m.eligible = original
        except Exception as exc:
            LOG.exception("Experiment %s failed; continuing", idea)
            failures.append({"experiment": idea, "error": repr(exc)})
            atomic_json(output / "failures.json", failures)
    summarize(output)
    LOG.info("Finished in %.1f minutes; %s failures. Results: %s", (time.monotonic()-start)/60, len(failures), output)
    return bool(failures)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=ROOT / "output/time_series_search")
    ap.add_argument("--experiments", help="Comma-separated IDs, e.g. 01,15,29,B3; default all 36 + four baselines")
    ap.add_argument("--universes", nargs="+", choices=list(UNIVERSES), default=["large_mega", "investable", "micro"])
    ap.add_argument("--holds", nargs="+", type=int, choices=[1, 3, 6], default=[1, 3, 6])
    ap.add_argument("--modes", nargs="+", choices=["raw", "factor"], default=["raw", "factor"])
    ap.add_argument("--cost-bp", type=float, default=25, help="One-way cost per dollar traded, excluding borrow")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    if args.cost_bp < 0:
        ap.error("--cost-bp must be nonnegative")
    if args.report_only:
        summarize(args.output)
    else:
        raise SystemExit(int(run(args)))


if __name__ == "__main__":
    main()
