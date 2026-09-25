"""Test a liquid large/mega-cap replacement for part of the v3 short book.

The existing v3 alpha book gets most of its return from nano/micro shorts.  A
broad large-cap index is a poor substitute for that stock-selection exposure,
so this module tests a deliberately narrow replacement:

* keep the frozen v3 long legs unchanged;
* replace a fixed fraction of the combined short notional;
* short 100 large/mega names with high distress (``o_score``), high issuance
  (``chcsho_12m``), and weak net payout (low ``eqnpo_12m``), ranked within GICS
  sector; and
* compare it with a cap-weighted top-200 large/mega basket as a broad control.

All signals and size labels are observed in the formation month.  Returns are
the supplied next-month excess returns, with missing held-name returns credited
zero, matching the frozen book.  The script does not refit either alpha engine.

Optional public-factor validation accepts Kenneth French's official monthly
factor ZIP (``F-F_Research_Data_Factors_CSV.zip``) via ``--ff3-zip``.
"""
from __future__ import annotations

import argparse
import io
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "chars_final_with_names.parquet"
OUT = ROOT / "output"
DEFAULT_OUTPUT = OUT / "large_mega_hedge_20260924"
N_SIDE = 100
N_BROAD = 200
T = 3
PRICE_FLOOR_CORE = 1.0
PRICE_FLOOR_HEDGE = 5.0
HEDGE_FRACTIONS = (0.0, 0.10, 0.25, 0.40, 0.50, 1.0)
SIZE_GROUPS = ("mega", "large", "small", "micro", "nano", "unknown")
NAME_COST_BP = {"mega": 5.0, "large": 10.0, "small": 20.0,
                "micro": 40.0, "nano": 60.0, "unknown": 25.0}


def _target_period(values) -> pd.PeriodIndex:
    return pd.DatetimeIndex(pd.to_datetime(values)).to_period("M") + 1


def staggered_sets(preds: pd.DataFrame, score: str, t: int = T,
                   n_side: int = N_SIDE) -> tuple[dict, dict]:
    """Frozen v3 T-month staggered top/bottom sets."""
    months = sorted(pd.to_datetime(preds["target_month"]).unique())
    state = {c: (set(), set()) for c in range(t)}
    longs, shorts = {}, {}
    per, extra = divmod(n_side, t)
    for i, month in enumerate(months):
        g = preds[pd.to_datetime(preds["target_month"]) == month]
        names = g.sort_values(score, ascending=False)["permno"].tolist()
        c = i % t
        k = per + int(c < extra)
        state[c] = (set(names[:k]), set(names[-k:]))
        longs[pd.Period(month, freq="M")] = set().union(
            *(state[j][0] for j in range(t)))
        shorts[pd.Period(month, freq="M")] = set().union(
            *(state[j][1] for j in range(t)))
    return longs, shorts


def badness_scores(frame: pd.DataFrame) -> pd.Series:
    """Sector-relative, equal-weighted distress/issuance/weak-payout score."""
    keys = [frame["target_month"], frame["sector"]]
    distress = frame.groupby(keys, sort=False)["o_score"].rank(
        pct=True, method="average")
    issuance = frame.groupby(keys, sort=False)["chcsho_12m"].rank(
        pct=True, method="average")
    payout = frame.groupby(keys, sort=False)["eqnpo_12m"].rank(
        pct=True, method="average")
    return (distress + issuance + (1.0 - payout)) / 3.0


def combine_weights(long_weights: dict, short_weights: dict,
                    hedge_weights: dict, fraction: float) -> dict:
    """Replace ``fraction`` of the 0.5x short leg with the liquid hedge."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")
    out = defaultdict(float)
    for key, value in long_weights.items():
        out[key] += value
    for key, value in short_weights.items():
        out[key] -= (1.0 - fraction) * value
    for key, value in hedge_weights.items():
        out[key] -= 0.5 * fraction * value
    return {key: value for key, value in out.items() if abs(value) > 1e-15}


def one_way_cost(current: dict, previous: dict, current_tiers: dict,
                 previous_tiers: dict) -> float:
    """Per-name one-way cost, using the last observed tier on exits."""
    total = 0.0
    for key in set(current) | set(previous):
        tier = current_tiers.get(key, previous_tiers.get(key, "unknown"))
        bp = NAME_COST_BP.get(str(tier).lower(), NAME_COST_BP["unknown"])
        total += abs(current.get(key, 0.0) - previous.get(key, 0.0)) * bp / 1e4
    return total


def _load_panel() -> pd.DataFrame:
    columns = ["permno", "eom", "prc", "me", "size_grp", "gics",
               "ret_exc_lead1m", "o_score", "chcsho_12m", "eqnpo_12m"]
    panel = pq.read_table(RAW, columns=columns).to_pandas()
    panel["eom"] = pd.to_datetime(panel["eom"])
    panel["target_month"] = _target_period(panel["eom"])
    panel["size_grp"] = panel["size_grp"].astype(str).str.lower()
    panel["sector"] = panel["gics"].astype("string").str[:2].fillna("UNK")
    return panel


def _load_predictions(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    prc = panel[["permno", "eom", "prc"]]
    tfm = pd.read_parquet(OUT / "preds_timesfm_all.parquet")
    tfm["eom"] = pd.to_datetime(tfm["eom"])
    tfm["target_month"] = pd.to_datetime(tfm["target_month"])
    xgb = pd.concat([
        pd.read_csv(OUT / "preds_xgb_timesfm_backcast.csv"),
        pd.read_csv(OUT / "preds_xgb_timesfm.csv"),
    ], ignore_index=True)
    xgb["eom"] = pd.to_datetime(xgb["eom"])
    xgb["target_month"] = pd.to_datetime(xgb["target_month"])
    xgb = xgb.rename(columns={"pred_xgb_tfm": "tfm_forecast"})
    tfm = tfm.merge(prc, on=["permno", "eom"], how="left",
                    validate="many_to_one")
    xgb = xgb.merge(prc, on=["permno", "eom"], how="left",
                    validate="many_to_one")
    # The production loader excludes missing forecasts before ranking.  Keeping
    # them would silently put NaNs at the bottom of the short book.
    tfm = tfm.dropna(subset=["tfm_forecast"])
    xgb = xgb.dropna(subset=["tfm_forecast"])
    return (tfm.loc[tfm["prc"] >= PRICE_FLOOR_CORE].copy(),
            xgb.loc[xgb["prc"] >= PRICE_FLOOR_CORE].copy())


def _leg_returns(preds: pd.DataFrame, sets: dict) -> pd.Series:
    maps = {
        pd.Period(month, freq="M"): g.set_index("permno")["ret_exc_lead1m"]
        for month, g in preds.groupby("target_month", sort=True)
    }
    values = {}
    for month, names in sets.items():
        values[month] = float(maps[month].reindex(sorted(names)).fillna(0.0).mean())
    return pd.Series(values, dtype=float).sort_index()


def _core(panel: pd.DataFrame):
    tfm, xgb = _load_predictions(panel)
    lt, st = staggered_sets(tfm, "tfm_forecast")
    lx, sx = staggered_sets(xgb, "tfm_forecast")
    common = sorted(set(lt) & set(lx))
    common = [m for m in common if pd.Period("2018-01") <= m <= pd.Period("2026-08")]

    long_ret = 0.25 * _leg_returns(tfm, lt).reindex(common)
    long_ret += 0.25 * _leg_returns(xgb, lx).reindex(common)
    short_ret = 0.25 * _leg_returns(tfm, st).reindex(common)
    short_ret += 0.25 * _leg_returns(xgb, sx).reindex(common)

    long_weights, short_weights = {}, {}
    for month in common:
        lw, sw = defaultdict(float), defaultdict(float)
        for names in (lt[month], lx[month]):
            for permno in names:
                lw[permno] += 0.25 / len(names)
        for names in (st[month], sx[month]):
            for permno in names:
                sw[permno] += 0.25 / len(names)
        long_weights[month], short_weights[month] = dict(lw), dict(sw)
    return common, long_ret, short_ret, long_weights, short_weights


def _hedges(panel: pd.DataFrame, months: list[pd.Period]):
    eligible = panel[
        panel["target_month"].isin(months)
        & panel["size_grp"].isin(["large", "mega"])
        & (panel["prc"] >= PRICE_FLOOR_HEDGE)
        & (panel["me"] > 0)
    ].copy()
    complete = eligible[["o_score", "chcsho_12m", "eqnpo_12m"]].notna().all(axis=1)
    eligible["badness"] = np.nan
    eligible.loc[complete, "badness"] = badness_scores(eligible.loc[complete])

    weights = {"bad100": {}, "bad50_50": {}, "top200_cap": {}}
    returns = {"bad100": {}, "bad50_50": {}, "top200_cap": {}}
    holdings = []
    for month, g in eligible.groupby("target_month", sort=True):
        bad = g.dropna(subset=["badness"]).sort_values(
            ["badness", "permno"], ascending=[False, True]).head(N_SIDE).copy()
        bad["weight"] = 1.0 / len(bad)
        balanced = pd.concat([
            g[(g["size_grp"] == tier) & g["badness"].notna()].sort_values(
                ["badness", "permno"], ascending=[False, True]).head(50)
            for tier in ("mega", "large")
        ], ignore_index=True)
        balanced["weight"] = 1.0 / len(balanced)
        broad = g.sort_values(["me", "permno"], ascending=[False, True]).head(N_BROAD).copy()
        broad["weight"] = broad["me"] / broad["me"].sum()
        for label, chosen in (("bad100", bad), ("bad50_50", balanced),
                              ("top200_cap", broad)):
            weights[label][month] = dict(zip(chosen["permno"], chosen["weight"]))
            returns[label][month] = float(
                (chosen["weight"] * chosen["ret_exc_lead1m"].fillna(0.0)).sum())
            keep = ["permno", "eom", "target_month", "size_grp", "sector", "me",
                    "prc", "o_score", "chcsho_12m", "eqnpo_12m", "badness", "weight"]
            h = chosen[keep].copy()
            h.insert(0, "hedge", label)
            holdings.append(h)
    series = {key: pd.Series(value, dtype=float).sort_index()
              for key, value in returns.items()}
    return weights, series, pd.concat(holdings, ignore_index=True)


def _market() -> pd.DataFrame:
    sp = pd.read_csv(ROOT / "data" / "external" / "SP500.csv",
                     parse_dates=["observation_date"]).dropna()
    sp = sp.set_index("observation_date")["SP500"].resample("ME").last().pct_change()
    sp.index = sp.index.to_period("M")
    tb = pd.read_csv(ROOT / "data" / "external" / "TB3MS.csv",
                     parse_dates=["observation_date"]).dropna()
    tb["month"] = tb["observation_date"].dt.to_period("M")
    rf = tb.groupby("month")["TB3MS"].last() / 100.0 / 12.0
    out = pd.DataFrame({"spx": sp, "rf": rf}).dropna()
    out["mkt_rf"] = out["spx"] - out["rf"]
    out["benchmark"] = out["rf"] + 0.04 / 12.0
    return out


def _max_drawdown(ret: pd.Series) -> float:
    wealth = (1.0 + ret).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min())


def _nw_regression(y: pd.Series, x: pd.DataFrame, lags: int = 3) -> dict:
    joined = pd.concat([y.rename("y"), x], axis=1).dropna()
    yy = joined.pop("y").to_numpy(float)
    xx = np.column_stack([np.ones(len(joined)), joined.to_numpy(float)])
    beta = np.linalg.lstsq(xx, yy, rcond=None)[0]
    resid = yy - xx @ beta
    n = len(yy)
    bread = np.linalg.inv(xx.T @ xx / n)
    scores = xx * resid[:, None]
    meat = scores.T @ scores / n
    for lag in range(1, min(lags, n - 1) + 1):
        gamma = scores[lag:].T @ scores[:-lag] / n
        meat += (1.0 - lag / (lags + 1.0)) * (gamma + gamma.T)
    cov = bread @ meat @ bread / n
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    names = ["alpha"] + list(joined.columns)
    out = {name: float(value) for name, value in zip(names, beta)}
    out.update({f"{name}_t": float(value / err) if err > 0 else np.nan
                for name, value, err in zip(names, beta, se)})
    out["alpha_ann"] = out["alpha"] * 12.0
    out["months"] = n
    return out


def _metrics(ret: pd.Series, market: pd.DataFrame) -> dict:
    ret = ret.dropna()
    bench = market["benchmark"].reindex(ret.index)
    active = ret - bench
    reg = _nw_regression(ret, market[["mkt_rf"]].reindex(ret.index))
    years = len(ret) / 12.0
    return {
        "months": len(ret), "ann_return": float(ret.mean() * 12.0),
        "cagr": float((1.0 + ret).prod() ** (1.0 / years) - 1.0),
        "sharpe": float(ret.mean() / ret.std(ddof=1) * np.sqrt(12.0)),
        "ir": float(active.mean() / active.std(ddof=1) * np.sqrt(12.0)),
        "max_drawdown": _max_drawdown(ret),
        "beta": reg["mkt_rf"], "beta_t": reg["mkt_rf_t"],
        "alpha_ann": reg["alpha_ann"], "alpha_t": reg["alpha_t"],
    }


def load_ff3(path: Path) -> pd.DataFrame:
    """Read the official Kenneth French monthly factor ZIP."""
    with zipfile.ZipFile(path) as archive:
        raw = archive.read(archive.namelist()[0]).decode("latin1")
    lines = raw.splitlines()
    rows = [line for line in lines if len(line.split(",")[0].strip()) == 6
            and line.split(",")[0].strip().isdigit()]
    frame = pd.read_csv(io.StringIO("date,mkt_rf,smb,hml,rf\n" + "\n".join(rows)))
    frame["month"] = pd.PeriodIndex(frame.pop("date").astype(str), freq="M")
    frame = frame.set_index("month") / 100.0
    return frame


def _size_exposure(short_weights: dict, hedge_weights: dict, tiers: dict,
                   fraction: float) -> dict:
    totals = defaultdict(float)
    for month, weights in short_weights.items():
        for permno, weight in weights.items():
            totals[tiers[month].get(permno, "unknown")] += (1.0 - fraction) * weight
        for permno, weight in hedge_weights[month].items():
            totals[tiers[month].get(permno, "unknown")] += 0.5 * fraction * weight
    denom = sum(totals.values())
    return {group: totals[group] / denom for group in SIZE_GROUPS}


def run(output_dir: Path = DEFAULT_OUTPUT, ff3_zip: Path | None = None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    panel = _load_panel()
    months, long_ret, short_ret, long_w, short_w = _core(panel)
    hedge_w, hedge_ret, holdings = _hedges(panel, months)
    market = _market().reindex(months)
    frozen = pd.read_csv(OUT / "repro" / "final_config_monthly.csv", index_col=0)
    frozen.index = pd.PeriodIndex(frozen.index, freq="M")
    frozen_core = 0.25 * (frozen["ls_tfm_buffered"] + frozen["ls_xgb_raw"])
    frozen_core = frozen_core.reindex(months)

    panel_key = panel.set_index(["target_month", "permno"])
    tiers = {}
    for month in months:
        g = panel_key.loc[month]
        tiers[month] = g["size_grp"].to_dict()

    monthly = pd.DataFrame(index=pd.PeriodIndex(months, freq="M"))
    monthly.index.name = "target_month"
    monthly["core_long"] = long_ret
    monthly["core_short_underlying"] = short_ret
    monthly["frozen_core_gross"] = frozen_core
    for hedge, values in hedge_ret.items():
        monthly[f"{hedge}_underlying"] = values.reindex(months)

    summaries, exposures = [], []
    factor_rows = []
    ff3 = load_ff3(ff3_zip) if ff3_zip is not None else None
    for hedge in ("bad100", "bad50_50", "top200_cap"):
        for fraction in HEDGE_FRACTIONS:
            if fraction == 0.0 and hedge != "bad100":
                continue
            label = "core" if fraction == 0.0 else f"{hedge}_{int(fraction * 100):02d}pct"
            # Anchor the zero-replacement case to the frozen submitted artifact.
            # XGBoost contains many exact score ties and the historical script used
            # an unstable default sort, so reconstructing its legs can differ by a
            # few names across pandas builds.  Only the replacement delta uses the
            # currently reconstructed short leg.
            gross = frozen_core + fraction * (
                short_ret - 0.5 * hedge_ret[hedge].reindex(months))
            costs, previous, previous_tiers = [], {}, {}
            for month in months:
                current = combine_weights(long_w[month], short_w[month],
                                          hedge_w[hedge][month], fraction)
                current_tiers = {key: tiers[month].get(key, "unknown") for key in current}
                costs.append(one_way_cost(current, previous, current_tiers, previous_tiers))
                previous, previous_tiers = current, current_tiers
            cost = pd.Series(costs, index=gross.index)
            net = gross - cost
            monthly[f"{label}_gross"] = gross
            monthly[f"{label}_cost"] = cost
            monthly[f"{label}_net"] = net

            exposure = _size_exposure(short_w, hedge_w[hedge], tiers, fraction)
            exposures.append({"candidate": label, **exposure,
                              "micro_nano": exposure["micro"] + exposure["nano"]})
            for window, mask in (("full", net.index >= pd.Period("2018-01")),
                                 ("oos", net.index >= pd.Period("2021-01"))):
                for basis, values in (("gross", gross[mask]), ("net", net[mask])):
                    row = {"candidate": label, "hedge": hedge,
                           "replacement_fraction": fraction, "window": window,
                           "basis": basis, **_metrics(values, market)}
                    row["avg_monthly_cost"] = float(cost[mask].mean())
                    row["short_micro_nano_share"] = exposure["micro"] + exposure["nano"]
                    summaries.append(row)
            if ff3 is not None:
                for window, mask in (("full", net.index >= pd.Period("2018-01")),
                                     ("oos", net.index >= pd.Period("2021-01"))):
                    reg = _nw_regression(net[mask], ff3[["mkt_rf", "smb", "hml"]])
                    factor_rows.append({"candidate": label, "window": window, **reg})

    summary = pd.DataFrame(summaries)
    exposure_df = pd.DataFrame(exposures).drop_duplicates("candidate")
    holdings.to_csv(output_dir / "hedge_holdings.csv", index=False)
    monthly.to_csv(output_dir / "monthly_returns.csv")
    summary.to_csv(output_dir / "summary.csv", index=False)
    exposure_df.to_csv(output_dir / "short_cap_exposure.csv", index=False)
    if factor_rows:
        pd.DataFrame(factor_rows).to_csv(output_dir / "ff3_regressions.csv", index=False)

    balanced_sets = [set(g["permno"]) for _, g in
                     holdings[holdings["hedge"] == "bad50_50"]
                     .groupby("target_month", sort=True)]
    balanced_turnover = np.mean([
        len(current - previous) / len(current)
        for previous, current in zip(balanced_sets, balanced_sets[1:])
    ])
    checks = {
        "months": len(months),
        "first_month": str(months[0]), "last_month": str(months[-1]),
        "bad100_min_names": int(holdings[holdings["hedge"] == "bad100"]
                                .groupby("target_month").size().min()),
        "bad100_all_large_mega": bool(holdings[holdings["hedge"] == "bad100"]
                                       ["size_grp"].isin(["large", "mega"]).all()),
        "bad50_50_min_names": int(holdings[holdings["hedge"] == "bad50_50"]
                                  .groupby("target_month").size().min()),
        "bad50_50_exact_cap_split": bool(
            holdings[holdings["hedge"] == "bad50_50"]
            .groupby(["target_month", "size_grp"]).size().eq(50).all()),
        "bad50_50_avg_one_side_turnover": float(balanced_turnover),
        "ff3_zip": str(ff3_zip) if ff3_zip else None,
    }
    (output_dir / "run_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    return {"summary": summary, "exposure": exposure_df, "monthly": monthly,
            "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ff3-zip", type=Path, default=None,
                        help="Official F-F_Research_Data_Factors_CSV.zip")
    args = parser.parse_args()
    result = run(args.output_dir, args.ff3_zip)
    view = result["summary"]
    print(view[(view["window"] == "oos") & (view["basis"] == "net")][
        ["candidate", "ann_return", "ir", "max_drawdown", "beta",
         "avg_monthly_cost", "short_micro_nano_share"]].to_string(index=False))
    print(json.dumps(result["checks"], indent=2))


if __name__ == "__main__":
    main()
