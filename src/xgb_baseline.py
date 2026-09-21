"""Initial XGBoost baseline with default parameters.

Spec: expanding window by TARGET month (ret_exc_lead1m). Train targets [Feb2015, Dec(Y-3)],
validation [Jan(Y-2), Dec(Y-1)], test year Y for Y in 2021..2026 (last fold clipped to 08/2026).
Features: the 147 characteristics from factor_char_list.csv, cross-sectionally rank-transformed
to [-1, 1] within each characteristic month (NaN left as NaN; XGBoost handles missing natively).
Universe policy (brief: target availability must not define the prediction universe): every row
with characteristics is scored and is eligible for the book, regardless of whether the future
target exists; train/validation folds use rows with targets only. Held names without a recorded
forward return are credited 0 excess return (delisting returns are absent from the panel); a
-30% haircut on those holdings is reported as a sensitivity.
Portfolio: long top-100 / short bottom-100 by prediction, equal weights, dollar-neutral,
monthly rebalanced; metrics per the competition brief (IR on active returns =
excess spread - 4%/12, collateral convention per src/comp_metrics.py).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "output"
EXT = ROOT / "data" / "external"
OUT.mkdir(parents=True, exist_ok=True)
EXT.mkdir(parents=True, exist_ok=True)

FIRST_EOM = pd.Timestamp("2015-01-31")
LAST_TARGET = pd.Timestamp("2026-08-31")
N_SIDE = 100
RANK_MIN, RANK_MAX = -1.0, 1.0


def load_model_table():
    chars_path = RAW / "chars_final_with_names.parquet"
    features = list(pd.read_csv(RAW / "factor_char_list.csv")["variable"])
    cols = ["permno", "eom", "ret_exc_lead1m"] + features
    table = pq.read_table(chars_path, columns=cols).to_pandas()
    table["eom"] = pd.to_datetime(table["eom"])
    table = table.reset_index(drop=True)
    table["target_month"] = table["eom"] + pd.offsets.MonthEnd(1)
    return table, features


def rank_transform(table, features):
    grp = table.groupby("eom")[features]
    ranks = grp.rank(method="dense")
    span = grp.transform(lambda s: s.rank(method="dense").max()) - 1.0
    out = (ranks - 1.0) / span.replace(0.0, np.nan)
    return out * (RANK_MAX - RANK_MIN) + RANK_MIN


def expanding_folds(table):
    target_years = sorted(table["target_month"].dt.year.unique())
    test_years = [y for y in target_years if y >= 2021]
    folds = []
    for y in test_years:
        train_hi = pd.Timestamp(f"{y - 3}-12-31")
        val_lo = pd.Timestamp(f"{y - 2}-01-31")
        val_hi = pd.Timestamp(f"{y - 1}-12-31")
        test_lo = pd.Timestamp(f"{y}-01-31")
        test_hi = min(pd.Timestamp(f"{y}-12-31"), LAST_TARGET)
        folds.append((train_hi, val_lo, val_hi, test_lo, test_hi))
    return folds


def fit_fold(train, val, test, features):
    model = xgb.XGBRegressor(
        early_stopping_rounds=50,
        eval_metric="rmse",
        tree_method="hist",
    )
    model.fit(
        train[features].values,
        train["ret_exc_lead1m"].values,
        eval_set=[(val[features].values, val["ret_exc_lead1m"].values)],
        verbose=False,
    )
    pred = model.predict(test[features].values)
    return pred, model


def fetch_fred(series_id):
    cache = EXT / f"{series_id}.csv"
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["observation_date"])
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    df = pd.read_csv(url, parse_dates=["observation_date"])
    df.to_csv(cache, index=False)
    return df


def monthly_benchmark():
    tb = fetch_fred("TB3MS")
    tb = tb.rename(columns={"observation_date": "date", "TB3MS": "tb3ms"})
    tb = tb.dropna()
    tb["month"] = tb["date"].dt.to_period("M")
    tb = tb.groupby("month")["tb3ms"].last().reset_index()
    tb["benchmark"] = tb["tb3ms"] / 100.0 / 12.0 + 0.04 / 12.0
    return tb


def market_series():
    sp = fetch_fred("SP500")
    sp = sp.rename(columns={"observation_date": "date", "SP500": "sp500"}).dropna()
    sp = sp.set_index("date").resample("ME").last()
    mkt = sp.pct_change().dropna().reset_index()
    mkt = mkt.rename(columns={"sp500": "sp500_ret"})
    mkt["month"] = mkt["date"].dt.to_period("M")
    tb = monthly_benchmark_rf()
    mkt = mkt.merge(tb, on="month", how="inner")
    mkt["mkt_rf"] = mkt["sp500_ret"] - mkt["tb3ms_dec"]
    return mkt


def monthly_benchmark_rf():
    tb = monthly_benchmark()
    tb["tb3ms_dec"] = tb["tb3ms"] / 100.0 / 12.0
    return tb[["month", "tb3ms_dec"]]


def build_portfolio(preds, missing_value=0.0):
    """Equal-weight long/short book. Held names without a recorded forward return
    are credited `missing_value` excess return (0.0 primary; -0.30 haircut
    sensitivity). Missing counts are reported per month."""
    preds = preds.sort_values(["target_month", "permno"])
    rows = []
    long_sets = {}
    short_sets = {}
    for month, g in preds.groupby("target_month"):
        g = g.sort_values("pred_xgb", ascending=False)
        n_long = min(N_SIDE, len(g) // 2)
        n_short = min(N_SIDE, len(g) - n_long)
        top = g.head(n_long)
        bottom = g.tail(n_short)
        long_sets[month] = set(top["permno"])
        short_sets[month] = set(bottom["permno"])
        rows.append({
            "target_month": month,
            "n_long": len(top),
            "n_short": len(bottom),
            "n_long_missing": int(top["ret_exc_lead1m"].isna().sum()),
            "n_short_missing": int(bottom["ret_exc_lead1m"].isna().sum()),
            "ret_long": top["ret_exc_lead1m"].fillna(missing_value).mean(),
            "ret_short": bottom["ret_exc_lead1m"].fillna(missing_value).mean(),
        })
        g2 = g.copy()
        try:
            g2["decile"] = pd.qcut(g2["pred_xgb"].rank(method="first"), 10, labels=False)
            dec10 = g2[g2["decile"] == 9]["ret_exc_lead1m"].mean()
            dec1 = g2[g2["decile"] == 0]["ret_exc_lead1m"].mean()
        except ValueError:
            dec10 = dec1 = np.nan
        rows[-1]["ret_decile10"] = dec10
        rows[-1]["ret_decile1"] = dec1
    port = pd.DataFrame(rows)
    port["ret_ls"] = port["ret_long"] - port["ret_short"]
    port["ret_decile_ls"] = port["ret_decile10"] - port["ret_decile1"]
    return port, long_sets, short_sets


def turnover_rate(month_sets):
    rates = []
    months = sorted(month_sets)
    for prev, cur in zip(months, months[1:]):
        prev_set = month_sets[prev]
        cur_set = month_sets[cur]
        if len(cur_set) == 0:
            continue
        rates.append(len(cur_set - prev_set) / len(cur_set))
    return float(np.mean(rates)) if rates else np.nan


def max_drawdown(series):
    cum = (1.0 + series).cumprod()
    peak = cum.cummax()
    return float((cum / peak - 1.0).min())


def evaluate(port, long_sets, short_sets):
    port = port.reset_index(drop=True)
    port["month"] = port["target_month"].dt.to_period("M")
    res = {}
    res["months"] = len(port)
    res["avg_monthly_ls"] = port["ret_ls"].mean()
    res["ann_ls"] = res["avg_monthly_ls"] * 12
    res["sharpe_ls"] = port["ret_ls"].mean() / port["ret_ls"].std(ddof=1) * np.sqrt(12)
    res["avg_monthly_decile_ls"] = port["ret_decile_ls"].mean()
    res["sharpe_decile_ls"] = (
        port["ret_decile_ls"].mean() / port["ret_decile_ls"].std(ddof=1) * np.sqrt(12)
    )
    res["long_leg_ann"] = port["ret_long"].mean() * 12
    res["short_leg_ann"] = port["ret_short"].mean() * 12
    res["avg_turnover_long"] = turnover_rate(long_sets)
    res["avg_turnover_short"] = turnover_rate(short_sets)
    res["best_month"] = port.loc[port["ret_ls"].idxmax(), ["target_month", "ret_ls"]].to_dict()
    res["worst_month"] = port.loc[port["ret_ls"].idxmin(), ["target_month", "ret_ls"]].to_dict()
    res["max_drawdown"] = max_drawdown(port["ret_ls"])
    res["hit_rate_ls"] = (port["ret_ls"] > 0).mean()
    try:
        tb = monthly_benchmark()
        port = port.merge(tb[["month", "benchmark"]], on="month", how="left")
        # Brief convention: total return = excess spread + collateral at TB3MS;
        # active = R_total - (TB3MS + 4%/12) = spread - 4%/12 (cash cancels).
        active = port["ret_ls"] - 0.04 / 12.0
        res["ir_benchmark"] = (
            np.sqrt(12) * active.mean() / active.std(ddof=1)
        )
        res["avg_benchmark_monthly"] = port["benchmark"].mean()
        res["ann_active"] = active.mean() * 12
    except Exception as e:
        res["ir_benchmark"] = None
        print(f"benchmark unavailable: {e}")
    try:
        mkt = market_series()
        port2 = port.merge(mkt[["month", "mkt_rf"]], on="month", how="left").dropna(
            subset=["mkt_rf"]
        )
        reg_data = pd.DataFrame({
            "port_rf": port2["ret_ls"] - port2["benchmark"],
            "mkt_rf": port2["mkt_rf"],
        }).dropna()
        ols = smf.ols("port_rf ~ mkt_rf", data=reg_data).fit(
            cov_type="HAC", cov_kwds={"maxlags": 3}
        )
        res["alpha_monthly"] = ols.params["Intercept"]
        res["alpha_ann"] = ols.params["Intercept"] * 12
        res["alpha_tstat"] = ols.tvalues["Intercept"]
        res["beta"] = ols.params["mkt_rf"]
        res["beta_se"] = ols.bse["mkt_rf"]
        res["corr_sp500"] = port2["ret_ls"].corr(port2["mkt_rf"])
    except Exception as e:
        print(f"market regression unavailable: {e}")
    return res


def main():
    table, features = load_model_table()
    print(f"model table: {len(table):,} rows, {len(features)} features, "
          f"{table['permno'].nunique():,} permnos")
    print("rank-transforming features within each month ...")
    transformed = rank_transform(table, features)
    for f in features:
        table[f] = transformed[f].values
    del transformed

    folds = expanding_folds(table)
    all_preds = []
    for i, (train_hi, val_lo, val_hi, test_lo, test_hi) in enumerate(folds):
        train = table[(table["target_month"] <= train_hi)].dropna(subset=["ret_exc_lead1m"])
        val = table[(table["target_month"] >= val_lo) & (table["target_month"] <= val_hi)].dropna(
            subset=["ret_exc_lead1m"]
        )
        test = table[(table["target_month"] >= test_lo) & (table["target_month"] <= test_hi)]
        print(f"fold {i}: train<={train_hi.date()} ({len(train):,}) "
              f"val {val_lo.date()}..{val_hi.date()} ({len(val):,}) "
              f"test {test_lo.date()}..{test_hi.date()} ({len(test):,})")
        pred, model = fit_fold(train, val, test, features)
        out = test[["permno", "eom", "target_month", "ret_exc_lead1m"]].copy()
        out["pred_xgb"] = pred
        all_preds.append(out)
    preds = pd.concat(all_preds, ignore_index=True)

    evaluated = preds.dropna(subset=["ret_exc_lead1m"])
    y = evaluated["ret_exc_lead1m"].values
    yhat = evaluated["pred_xgb"].values
    r2 = 1.0 - np.sum((y - yhat) ** 2) / np.sum(y ** 2)
    print(f"\nOOS R2 (XGBoost defaults, zero benchmark; {len(evaluated):,} scored rows with targets): {r2:.5f}")
    evaluated["year"] = evaluated["target_month"].dt.year
    for yr, g in evaluated.groupby("year"):
        gy, gyh = g["ret_exc_lead1m"].values, g["pred_xgb"].values
        r2y = 1.0 - np.sum((gy - gyh) ** 2) / np.sum(gy ** 2)
        print(f"  {yr}: R2 {r2y:.5f}  (n={len(g):,})")

    port, long_sets, short_sets = build_portfolio(preds)
    metrics = evaluate(port, long_sets, short_sets)
    print("\n--- portfolio metrics (long top-100 / short bottom-100, equal weight, missing returns credited 0) ---")
    for k, v in metrics.items():
        if k in ("best_month", "worst_month"):
            print(f"{k}: {v['target_month'].date()} ret={v['ret_ls']:.4f}")
        else:
            print(f"{k}: {v}")

    port_h, _, _ = build_portfolio(preds, missing_value=-0.30)
    metrics_h = evaluate(port_h, long_sets, short_sets)
    print("\n--- sensitivity: missing returns credited -30% (delisting haircut) ---")
    for k in ("avg_monthly_ls", "ann_ls", "sharpe_ls", "ir_benchmark", "max_drawdown", "avg_turnover_long", "avg_turnover_short"):
        print(f"{k}: {metrics_h[k]}")

    preds_out = preds[["permno", "eom", "target_month", "ret_exc_lead1m", "pred_xgb"]]
    preds_out.to_csv(OUT / "preds_xgb_default.csv", index=False)
    port_out = port.reset_index(drop=True)
    port_out["weight_long"] = 1.0 / port_out["n_long"]
    port_out["weight_short"] = -1.0 / port_out["n_short"]
    port_out.to_csv(OUT / "port_monthly_xgb_default.csv", index=False)
    print(f"\nsaved: {OUT/'preds_xgb_default.csv'}, {OUT/'port_monthly_xgb_default.csv'}")


if __name__ == "__main__":
    main()
