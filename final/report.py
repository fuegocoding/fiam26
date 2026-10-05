"""Writes holdings, returns, performance tables and charts."""

import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import evaluate

FIRST = pd.Period(f"{config.TEST_YEARS[0]}-01", "M")
LAST = config.LAST_TARGET.to_period("M")
HARD_TO_BORROW = 0.10  # short interest over shares outstanding (borrow.py)
OUTCOMES = {"ret", "ret_exc", "ret_exc_lead1m", "stock_exret"}


def _period(values: pd.Series) -> pd.Series:
    """Monthly periods from periods or timestamps."""
    if isinstance(values.dtype, pd.PeriodDtype):
        return values
    return pd.to_datetime(values).dt.to_period("M")


def _compound(returns: pd.Series) -> float:
    return float((1 + returns).prod() - 1)


def _annualized(returns: pd.Series) -> float:
    return float((1 + returns).prod() ** (12 / len(returns)) - 1)


def positions(portfolio: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    """Each position with its realized next-month return and P&L."""
    p = portfolio.copy()
    p["return_month"] = _period(p.return_month)
    outcome = universe[
        [
            "return_month",
            "permno",
            "ret_exc_lead1m",
            "me",
            *[c for c in ("exit_price", "exit_price_date") if c in universe],
        ]
    ]
    p = p.merge(outcome, on=["return_month", "permno"], how="left")
    rates = evaluate.market().rf
    p["ret"] = p.ret_exc_lead1m
    for month, ids in p.groupby("return_month").groups.items():
        p.loc[ids, "ret"] = evaluate.position_returns(
            p.loc[ids], float(rates.get(month, np.nan))
        )[0]
    p["pnl"] = p.target_weight * p.ret
    return p


def holdings_table(p: pd.DataFrame) -> pd.DataFrame:
    """The monthly holdings in the exact submission schema."""
    start = p.return_month.dt.to_timestamp(how="start")
    out = pd.DataFrame(
        {
            "Date": start.dt.strftime("%Y-%m-%d"),
            "PERMNO": p.permno.astype(int),
            "TICKER": p.ticker,
            "COMPANY NAME": p.company_name,
            "WEIGHT": p.target_weight * 100,
        }
    )
    return out.sort_values(["Date", "WEIGHT", "PERMNO"], ascending=[True, False, True])


def oos_r2(
    predictions: pd.DataFrame,
    universe: pd.DataFrame,
    holdings: pd.DataFrame,
    score: str = "final_score",
) -> dict:
    """Zero-benchmark out-of-sample R2 (Gu, Kelly and Xiu 2020) of the score.

    The score is a cross-sectional rank, so it is mapped to an expected
    excess return r_hat = b_t * (score - 0.5), where b_t is the average
    monthly cross-sectional slope of realized returns on the centred score
    over the months already realized before month t (b = 0 in the first
    month). R2 = 1 - sum((r - r_hat)^2) / sum(r^2) over stocks with an
    observed return, for all scored stocks and for the book's holdings.
    """
    keys = ["return_month", "permno"]
    d = predictions[[*keys, score]].rename(columns={score: "score"})
    d["return_month"] = _period(d.return_month)
    d = d.merge(universe[[*keys, "ret_exc_lead1m"]], on=keys, how="left")
    d = d.dropna()
    d["x"] = d.score - 0.5
    slope = d.groupby("return_month").apply(
        lambda g: float((g.x * g.ret_exc_lead1m).sum() / (g.x**2).sum()),
        include_groups=False,
    )
    prior = slope.expanding().mean().shift(1).fillna(0.0)
    d["r_hat"] = d.return_month.map(prior) * d.x
    held = d.merge(holdings[keys], on=keys)
    out = {}
    for name, part in (("universe", d), ("holdings", held)):
        r, f = part.ret_exc_lead1m, part.r_hat
        out[f"oos_r2_{name}"] = float(1 - ((r - f) ** 2).sum() / (r**2).sum())
        out[f"oos_r2_{name}_obs"] = int(len(part))
    ic = d.groupby("return_month").apply(
        lambda g: g.score.corr(g.ret_exc_lead1m, method="spearman"),
        include_groups=False,
    )
    out["rank_ic_mean"] = float(ic.mean())
    out["rank_ic_t"] = float(ic.mean() / ic.std() * np.sqrt(len(ic)))
    return out


def return_stats(r: pd.DataFrame) -> dict:
    """Return statistics for the strategy, the benchmark and the S&P 500."""
    total = r.total
    best, worst = total.idxmax(), total.idxmin()
    long_leg = r.pnl_long / r.long_notional
    short_leg = r.pnl_short / r.short_notional
    return {
        "period": f"{r.index.min()}..{r.index.max()}",
        "returns": "net of trading costs and borrow fees",
        "mean_monthly_return": float(total.mean()),
        "ann_arithmetic_return": float(total.mean() * 12),
        "cagr": _annualized(total),
        "cumulative_return": _compound(total),
        "benchmark_cagr": _annualized(r.benchmark),
        "benchmark_cumulative": _compound(r.benchmark),
        "spx_cagr": _annualized(r.spx),
        "spx_cumulative": _compound(r.spx),
        "best_month": str(best),
        "best_month_return": float(total[best]),
        "worst_month": str(worst),
        "worst_month_return": float(total[worst]),
        "hit_rate_vs_benchmark": float((r.active > 0).mean()),
        "long_leg_pnl_ann": float(r.pnl_long.mean() * 12),
        "short_leg_pnl_ann": float(r.pnl_short.mean() * 12),
        "long_leg_return_ann": float(long_leg.mean() * 12),
        "short_leg_return_ann": float(short_leg.mean() * 12),
    }


def exposure_stats(r: pd.DataFrame, p: pd.DataFrame) -> dict:
    """Holdings, exposure, concentration and turnover."""
    size = p.target_weight.abs()
    top10 = size.groupby(p.return_month).apply(lambda x: x.nlargest(10).sum() / x.sum())
    return {
        "avg_long_names": float(r.longs.mean()),
        "avg_short_names": float(r.shorts.mean()),
        "gross_avg": float(r.gross.mean()),
        "gross_min": float(r.gross.min()),
        "gross_max": float(r.gross.max()),
        "net_avg": float(r.net.mean()),
        "net_min": float(r.net.min()),
        "net_max": float(r.net.max()),
        "position_avg": float(size.mean()),
        "position_max": float(size.max()),
        "top10_share_of_gross_avg": float(top10.mean()),
        "turnover_avg": float(r.traded.mean()),
        "turnover_min": float(r.traded.min()),
        "turnover_max": float(r.traded.max()),
        "missing_next_month_returns": int(r.missing.sum()),
    }


def short_book_stats(p: pd.DataFrame, short_interest: pd.DataFrame | None) -> dict:
    """Size, liquidity and borrow difficulty of the short positions."""
    shorts = p[p.target_weight < 0].copy()
    hard = np.nan
    if short_interest is not None:
        crowded = short_interest[["permno", "return_month", "short_pct"]]
        crowded = crowded.assign(return_month=_period(crowded.return_month))
        shorts = shorts.merge(crowded, on=["permno", "return_month"], how="left")
        hard = float((shorts.short_pct >= HARD_TO_BORROW).mean())
    micro = shorts.size_grp.isin(["micro", "nano"])
    small = shorts.size_grp.isin(["small", "micro", "nano"])
    return {
        "short_avg_market_cap_musd": float(shorts.me.mean()),
        "short_median_market_cap_musd": float(shorts.me.median()),
        "short_avg_dollar_volume_musd": float(shorts.dolvol_126d.mean() / 1e6),
        "short_share_micro_nano": float(micro.mean()),
        "short_share_small_or_below": float(small.mean()),
        "short_share_short_interest_ge_10pct": hard,
    }


def calendar_years(r: pd.DataFrame) -> pd.DataFrame:
    """Compounded returns by calendar year."""
    by_year = r.groupby(r.index.year)
    return pd.DataFrame(
        {
            "strategy": by_year.total.apply(_compound),
            "benchmark": by_year.benchmark.apply(_compound),
            "sp500": by_year.spx.apply(_compound),
            "active": by_year.active.sum(),
            "months": by_year.size(),
        }
    )


def comparison_metrics(r: pd.DataFrame) -> pd.DataFrame:
    """Competition statistics on the same monthly NAV basis for each series."""
    columns = {}
    for name, values in (("Strategy", r.total), ("T-bill + 4%", r.benchmark),
                         ("S&P 500", r.spx)):
        curve = (1 + values).cumprod()
        excess, active = values - r.rf, values - r.benchmark
        columns[name] = {
            "mean_monthly_return": float(values.mean()),
            "geometric_monthly_return": float(
                (1 + values).prod() ** (1 / len(values)) - 1
            ),
            "annual_arithmetic_return": float(values.mean() * 12),
            "cagr": _annualized(values),
            "cumulative_return": _compound(values),
            "annual_volatility": float(values.std() * np.sqrt(12)),
            "max_drawdown_month_end_nav": float(
                (curve / curve.cummax().clip(lower=1.0) - 1).min()
            ),
            "best_month": str(values.idxmax()),
            "best_month_return": float(values.max()),
            "worst_month": str(values.idxmin()),
            "worst_month_return": float(values.min()),
            "hit_rate_vs_hurdle": float((active > 0).mean()),
            "information_ratio": float(np.sqrt(12) * active.mean() / active.std())
            if active.std() > 1e-12 else None,
            "sharpe_above_t_bill": float(np.sqrt(12) * excess.mean() / excess.std())
            if excess.std() > 1e-12 else None,
        }
    deployed = (1 + (r.total - r.rf) / r.gross).cumprod()
    columns["Strategy"]["max_drawdown_deployed_capital_gross_normalized"] = float(
        (deployed / deployed.cummax().clip(lower=1.0) - 1).min()
    )
    return pd.DataFrame(columns).rename_axis("metric")


def label_history(p: pd.DataFrame) -> pd.DataFrame:
    """Last label held, plus every earlier label the stock carried while
    held, so a later name never stands in for an earlier holding period."""
    held = p.sort_values("return_month")
    last = held.drop_duplicates("permno", keep="last").set_index("permno")
    label = held.ticker.astype(str) + " (" + held.company_name.astype(str) + ")"
    spans = (
        held.assign(label=label)
        .groupby(["permno", "label"])
        .return_month.agg(["min", "max"])
        .reset_index()
        .sort_values(["permno", "min"])
    )
    spans["text"] = (
        spans.label + " " + spans["min"].astype(str) + ".." + spans["max"].astype(str)
    )
    history = spans.groupby("permno").text.apply(
        lambda x: "; ".join(x) if len(x) > 1 else ""
    )
    labels = last[["ticker", "company_name", "ticker_name_source"]]
    return labels.join(history.rename("labels_while_held"))


def top_holdings(p: pd.DataFrame, months: int) -> pd.DataFrame:
    """Top 10 longs and shorts by weight averaged over every month."""
    by_stock = p.groupby("permno")
    out = pd.DataFrame(
        {
            "avg_weight_pct": by_stock.target_weight.sum() / months * 100,
            "months_held": by_stock.size(),
        }
    ).join(label_history(p))
    longs = out[out.avg_weight_pct > 0].nlargest(10, "avg_weight_pct")
    shorts = out[out.avg_weight_pct < 0].nsmallest(10, "avg_weight_pct")
    return pd.concat(
        [
            longs.assign(side="long"),
            shorts.assign(side="short"),
        ]
    ).reset_index()


def contributors(p: pd.DataFrame) -> pd.DataFrame:
    """Top 10 positive and negative contributors to cumulative P&L."""
    pnl = p.groupby("permno").pnl.sum().rename("pnl_pct_nav") * 100
    out = pnl.to_frame().join(label_history(p))
    return pd.concat(
        [
            out.nlargest(10, "pnl_pct_nav").assign(kind="top"),
            out.nsmallest(10, "pnl_pct_nav").assign(kind="bottom"),
        ]
    ).reset_index()


def compliance(
    r: pd.DataFrame, holdings: pd.DataFrame, predictions: pd.DataFrame
) -> dict:
    """The competition's trading rules, checked on every month."""
    months = list(r.index)
    labels = holdings[["TICKER", "COMPANY NAME"]]
    checks = {
        "covers_the_test_window": months[0] == FIRST and months[-1] == LAST,
        "no_missing_month": len(months) == len(pd.period_range(FIRST, LAST)),
        "names_between_100_and_500": bool(r.names.between(100, 500).all()),
        "gross_at_most_200pct": bool((r.gross <= 2.0).all()),
        "net_within_50pct": bool((r.net.abs() <= 0.5).all()),
        "every_holding_has_ticker_and_name": bool(labels.notna().all().all()),
        "no_duplicate_holdings": bool(
            not holdings.duplicated(["Date", "PERMNO"]).any()
        ),
        "predictions_carry_no_outcomes": not any(
            c in OUTCOMES or c.startswith("ret_") for c in predictions.columns
        ),
    }
    checks["all_pass"] = all(checks.values())
    return checks


def _growth(r: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Strategy": (1 + r.total).cumprod(),
            "T-bill + 4%": (1 + r.benchmark).cumprod(),
            "S&P 500": (1 + r.spx.fillna(0)).cumprod(),
        }
    )


def _save(fig, path: pathlib.Path, title: str, period: str) -> None:
    fig.suptitle(f"{title}\n{period}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_cumulative(r, x, out, period) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, series in _growth(r).items():
        ax.plot(x, series, label=name)
    ax.set_ylabel("Growth of $1")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, out / "cumulative.png", "Cumulative return", period)


def _plot_underwater(r, x, out, period) -> None:
    growth = _growth(r)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name in ("Strategy", "S&P 500"):
        drawdown = growth[name] / growth[name].cummax().clip(lower=1.0) - 1
        ax.fill_between(x, drawdown, 0, alpha=0.4, label=name)
    ax.set_ylabel("Drawdown")
    ax.legend()
    ax.grid(alpha=0.3)
    title = "Underwater (drawdown) chart"
    _save(fig, out / "underwater.png", title, period)


def _plot_rolling(r, x, out, period) -> None:
    active = r.active.rolling(12)
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    top.plot(x, active.sum())
    top.set_ylabel("12-m active return")
    bottom.plot(x, np.sqrt(12) * active.mean() / active.std())
    bottom.set_ylabel("12-m information ratio")
    for ax in (top, bottom):
        ax.axhline(0, color="k", lw=0.5)
        ax.grid(alpha=0.3)
    title = "Rolling 12-month active return and IR (vs T-bill + 4%)"
    _save(fig, out / "rolling_active_ir.png", title, period)


def _plot_beta(r, x, out, period) -> None:
    excess, market = r.total - r.rf, r.spx - r.rf
    beta = excess.rolling(12).cov(market) / market.rolling(12).var()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, beta)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("12-m beta to S&P 500")
    ax.grid(alpha=0.3)
    title = "Rolling 12-month beta to the S&P 500"
    _save(fig, out / "rolling_beta.png", title, period)


def _plot_histogram(r, out, period) -> None:
    hurdle = r.benchmark.mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(r.total, bins=25, alpha=0.8)
    label = f"monthly hurdle (avg {hurdle:.2%})"
    ax.axvline(hurdle, color="r", ls="--", label=label)
    ax.set_xlabel("Monthly total return")
    ax.legend()
    ax.grid(alpha=0.3)
    title = "Distribution of monthly returns"
    _save(fig, out / "histogram.png", title, period)


def _plot_contributors(contrib, out, period) -> None:
    c = contrib.sort_values("pnl_pct_nav")
    names = [
        f"{ticker} ({name})"
        for ticker, name in zip(c.ticker, c.company_name, strict=True)
    ]
    colors = np.where(c.pnl_pct_nav > 0, "tab:green", "tab:red")
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.barh(names, c.pnl_pct_nav, color=colors)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_xlabel("Cumulative P&L contribution, % of NAV")
    ax.grid(alpha=0.3)
    title = "Top 10 positive and negative contributors"
    _save(fig, out / "contributors.png", title, period)


def charts(r: pd.DataFrame, contrib: pd.DataFrame, out: pathlib.Path) -> None:
    """The six charts the competition lists, as PNG files."""
    plt.switch_backend("Agg")
    out.mkdir(parents=True, exist_ok=True)
    x = r.index.to_timestamp()
    period = f"{r.index.min()} to {r.index.max()}, monthly, net of costs and borrow"
    _plot_cumulative(r, x, out, period)
    _plot_underwater(r, x, out, period)
    _plot_rolling(r, x, out, period)
    _plot_beta(r, x, out, period)
    _plot_histogram(r, out, period)
    _plot_contributors(contrib, out, period)


def write_submission(
    p: pd.DataFrame, r: pd.DataFrame, out_dir: pathlib.Path
) -> pd.DataFrame:
    """The two required CSVs, label sources and the monthly detail."""
    holdings = holdings_table(p)
    holdings.to_csv(out_dir / "holdings.csv", index=False)
    sources = p.assign(Date=holdings.Date)[
        [
            "Date",
            "permno",
            "ticker",
            "company_name",
            "ticker_name_status",
            "ticker_name_reference_date",
            "ticker_name_source",
        ]
    ]
    sources.to_csv(out_dir / "holdings_label_sources.csv", index=False)
    monthly = pd.DataFrame(
        {
            "Date": r.index.to_timestamp(how="start").strftime("%Y-%m-%d"),
            "RETURN": r.total.to_numpy(),
        }
    )
    monthly.to_csv(out_dir / "returns.csv", index=False)
    r.to_csv(out_dir / "monthly_detail.csv")
    return holdings


def run(
    portfolio: pd.DataFrame,
    predictions: pd.DataFrame,
    universe: pd.DataFrame,
    market: pd.DataFrame,
    short_interest: pd.DataFrame | None,
    out_dir: pathlib.Path,
    score: str = "final_score",
) -> dict:
    """Evaluates the book, writes every report file, returns the statistics."""
    out_dir.mkdir(parents=True, exist_ok=True)
    p = positions(portfolio, universe)
    books = {
        month: dict(zip(g.permno.astype(int), g.target_weight, strict=True))
        for month, g in p.groupby("return_month")
    }
    r = evaluate.evaluate(books, universe, market)
    r.attrs["liquidations"].to_csv(out_dir / "liquidations.csv", index=False)
    holdings = write_submission(p, r, out_dir)

    stats = evaluate.metrics(r, evaluate.factors(universe))
    stats.update(return_stats(r))
    stats.update(exposure_stats(r, p))
    stats.update(short_book_stats(p, short_interest))
    stats.update(oos_r2(predictions, universe, p, score))
    comparison = comparison_metrics(r)
    stats["max_dd_deployed_capital"] = comparison.loc[
        "max_drawdown_deployed_capital_gross_normalized", "Strategy"
    ]
    stats["score_column"] = score
    (out_dir / "performance.json").write_text(
        json.dumps(stats, indent=2, default=str) + "\n"
    )
    calendar_years(r).to_csv(out_dir / "calendar_years.csv")
    comparison.to_csv(out_dir / "comparison_metrics.csv")
    top_holdings(p, len(r)).to_csv(out_dir / "top_average_holdings.csv", index=False)
    contrib = contributors(p)
    contrib.to_csv(out_dir / "contributors.csv", index=False)
    checks = compliance(r, holdings, predictions)
    (out_dir / "compliance.json").write_text(json.dumps(checks, indent=2) + "\n")
    charts(r, contrib, out_dir / "charts")
    return {**stats, "compliance_all_pass": checks["all_pass"]}
