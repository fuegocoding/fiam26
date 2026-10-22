"""Analytics the judges' Arcana review will ask for, computed before the event.

Instructions 11 lists what the economic review looks at: factor versus
idiosyncratic return, sizing versus selection, hit rate against win/loss ratio,
and alpha decay. Nothing here trades; it decomposes what the book already did,
so the answers are ready before a judge asks for them.

    python -m arcana            # from the final/ directory

Writes `output/report/arcana.json`.
"""

import json
import pathlib

import numpy as np
import pandas as pd

import config
import evaluate
import report

HORIZONS = (1, 2, 3, 6, 12)


def factor_split(
    r: pd.DataFrame, factor_returns: pd.DataFrame
) -> dict:
    """How much of the return is known factor exposure and how much is residual.

    Excess return is regressed on market, size, value and momentum. The fitted
    part is the factor contribution; the residual is the idiosyncratic part.
    A market-neutral mandate is supposed to earn the second, not the first.
    """
    f = pd.DataFrame(
        {
            "y": r.total - r.rf,
            "mkt": r.spx - r.rf,
            "smb": factor_returns.smb,
            "hml": factor_returns.hml,
            "mom": factor_returns.mom,
        }
    ).dropna()
    x = f[["mkt", "smb", "hml", "mom"]].to_numpy()
    y = f.y.to_numpy()
    b, _, _ = evaluate.ols_hac(y, x)
    # The intercept is alpha, which belongs to the idiosyncratic side. Only the
    # factor realizations belong to the factor side; without this split the
    # intercept absorbs the whole mean and the residual looks like zero.
    factor_part = x @ b[1:]
    idio_part = y - factor_part
    total_var = float(np.var(y, ddof=1))
    mean = float(y.mean()) if y.mean() else np.nan
    return {
        "months": int(len(f)),
        "factor_share_of_mean_return": float(factor_part.mean() / mean),
        "idiosyncratic_share_of_mean_return": float(idio_part.mean() / mean),
        "factor_share_of_variance": float(np.var(factor_part, ddof=1) / total_var),
        "idiosyncratic_share_of_variance": float(np.var(idio_part, ddof=1) / total_var),
        "factor_contribution_ann": float(factor_part.mean() * 12),
        "idiosyncratic_contribution_ann": float(idio_part.mean() * 12),
        "alpha_ann_from_decomposition": float(b[0] * 12),
        "loadings": {
            name: float(v)
            for name, v in zip(["mkt", "smb", "hml", "mom"], b[1:], strict=True)
        },
    }


def sizing_vs_selection(p: pd.DataFrame) -> dict:
    """Did the book earn from picking names or from sizing them?

    The same names are re-weighted equally within each side. Everything the
    equal-weight version earns is selection; the difference from the actual
    book is sizing. A result carried by sizing means the construction layer is
    doing the work and the signal is weaker than the headline suggests.
    """
    q = p.copy()
    q["side_w"] = q.target_weight.abs()
    q["equal_w"] = q.groupby(["return_month", "side"]).side_w.transform("mean")
    q["equal_w"] = np.where(q.side == "short", -q.equal_w, q.equal_w)
    q["actual_pnl"] = q.target_weight * q.ret
    q["select_pnl"] = q.equal_w * q.ret
    q["size_pnl"] = q.actual_pnl - q.select_pnl
    monthly = q.groupby("return_month")[["actual_pnl", "select_pnl", "size_pnl"]].sum()
    out = {}
    for col, name in (
        ("actual_pnl", "total"),
        ("select_pnl", "selection"),
        ("size_pnl", "sizing"),
    ):
        s = monthly[col]
        out[f"{name}_pnl_ann"] = float(s.mean() * 12)
        out[f"{name}_share_of_pnl"] = float(s.sum() / monthly.actual_pnl.sum())
    out["selection_t"] = float(
        monthly.select_pnl.mean() / monthly.select_pnl.std() * np.sqrt(len(monthly))
    )
    out["sizing_t"] = float(
        monthly.size_pnl.mean() / monthly.size_pnl.std() * np.sqrt(len(monthly))
    )
    return out


def hit_rate_win_loss(p: pd.DataFrame, r: pd.DataFrame) -> dict:
    """Right how often, and how much bigger wins are than losses.

    A book that is right 48% of the time with a 1.6 win/loss ratio fails
    differently from one right 58% of the time at 1.0, even at the same
    information ratio (Instructions 11).
    """
    q = p.copy()
    q["win"] = q.target_weight * q.ret
    pos = q.loc[q.win > 0, "win"]
    neg = q.loc[q.win < 0, "win"]
    active = r.active
    n = int((active < 0).sum())
    return {
        "position_hit_rate": float((q.win > 0).mean()),
        "position_win_loss_ratio": float(pos.mean() / abs(neg.mean()))
        if len(neg)
        else np.nan,
        "month_hit_rate": float((active > 0).mean()),
        "month_win_loss_ratio": float(
            active[active > 0].mean() / abs(active[active < 0].mean())
        )
        if n
        else np.nan,
        "winning_positions": int(len(pos)),
        "losing_positions": int(len(neg)),
    }


def alpha_decay(
    panel_path: pathlib.Path, predictions: pd.DataFrame, score: str
) -> dict:
    """How fast the signal's edge dissipates after formation.

    Stocks are sorted into quintiles on `score` each month and the top-minus-
    bottom spread of realized excess return is measured at 1, 2, 3, 6 and 12
    months. This is the empirical answer to the rebalancing-frequency question:
    an edge that is gone in three months does not pay for daily trading.
    """
    ret = pd.read_parquet(panel_path, columns=["permno", "eom", "ret_exc"])
    ret["eom"] = pd.to_datetime(ret.eom)
    ret = ret.dropna(subset=["ret_exc"]).drop_duplicates(["permno", "eom"])
    ret = ret.sort_values(["permno", "eom"]).reset_index(drop=True)
    # The exact forward h-month return: cumulative product at t+h over t.
    cum = ret.groupby("permno").ret_exc.transform(lambda s: (1 + s).cumprod())
    ret["cum"] = cum
    for h in HORIZONS:
        ret[f"fwd_{h}"] = (
            ret.groupby("permno").cum.transform(lambda s, k=h: s.shift(-k) / s) - 1
        )
    # Join on the formation month, not its date: `eom` is the last business day
    # of the month, so a timestamp join would miss every row.
    ret["formation"] = ret.eom.dt.to_period("M")
    d = predictions[["return_month", "permno", score]].rename(columns={score: "score"})
    d["formation"] = pd.to_datetime(d.return_month).dt.to_period("M") - 1
    d = d.merge(
        ret.drop(columns=["eom", "cum"]), on=["permno", "formation"], how="left"
    )
    out = {}
    for h in HORIZONS:
        col = f"fwd_{h}"
        g = d.dropna(subset=[col, "score"])
        if g.empty:
            out[f"quintile_spread_{h}m_months"] = 0
            continue
        q = g.groupby("return_month").score.transform(
            lambda s: pd.qcut(s.rank(method="first"), 5, labels=False)
        )
        g = g.assign(quintile=q)
        spread = (
            g[g.quintile == 4].groupby("return_month")[col].mean()
            - g[g.quintile == 0].groupby("return_month")[col].mean()
        )
        spread = spread.replace([np.inf, -np.inf], np.nan).dropna()
        out[f"quintile_spread_{h}m_ann"] = float(spread.mean() * 12 / h)
        out[f"quintile_spread_{h}m_t"] = (
            float(spread.mean() / spread.std() * np.sqrt(len(spread)))
            if len(spread) > 1 and spread.std() > 0
            else np.nan
        )
        out[f"quintile_spread_{h}m_months"] = int(len(spread))
    return out


def run() -> dict:
    """Computes every lens and writes output/report/arcana.json."""
    panel = config.DATA_DIR / config.PANEL_FILE
    portfolio = pd.read_parquet(config.WORK_DIR / "portfolio.parquet")
    predictions = pd.read_parquet(config.WORK_DIR / "predictions.parquet")
    universe = evaluate.load_universe(panel)
    market = evaluate.market()
    p = report.positions(portfolio, universe)
    books = {
        month: dict(zip(g.permno.astype(int), g.target_weight, strict=True))
        for month, g in p.groupby("return_month")
    }
    r = evaluate.evaluate(books, universe, market)
    score = "final_score_liquid" if config.NAV > 5e7 else "final_score"
    out = {
        "score_column": score,
        "nav": config.NAV,
        **factor_split(r, evaluate.factors(universe)),
        **sizing_vs_selection(p),
        **hit_rate_win_loss(p, r),
        **alpha_decay(panel, predictions, score),
    }
    dest = config.WORK_DIR / "report" / "arcana.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, default=str) + "\n")
    return out


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, default=str))
