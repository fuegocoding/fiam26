"""REPRO SCRIPT 3/3 - memo section 8: final competition configuration.

Final config = buffered book (k=1500) run at 1x gross (0.5 long + 0.5 short)
+ 0.5x S&P 500 overlay + ratchet sizing, evaluated 2016-10 .. 2026-08
(119 months; FRED SP500 series starts 2016-09, first return month 2016-10).

  SPX overlay return = SPX PRICE return (FRED SP500 month-end, un-excessed);
  ratchet scales the combined book+SPX return (variant A).

Variant B (ratchet on book only) is computed for comparison and does not
match. Items that do NOT reproduce from the memo text (flagged):
  - realized beta: memo +0.14; reproduced +0.27 full window (se 0.08),
    +0.44 in the graded OOS 2021-26 window. Compliance-relevant: beta is
    materially nonzero and higher exactly where the competition grades.
  - hit-rate counts: memo 72/119 positive and 4 months < -5%; reproduced
    83/119 and 8 months < -5%.
  - the memo SS8 prose quotes cushioned crash months 2020-04 -18.1%,
    2019-01 -12.1%, 2023-01 -14.9%; the reproduced series gives -4.6%,
    -4.1%, -5.2% (and the quoted values are arithmetically impossible
    given SPX was UP ~+13% in 2020-04 against a <=1x-ratcheted 0.5x SPX
    sleeve) - stale text from an earlier draft, not from the submitted
    configuration.

Memo reference values:
  full: ann +21.8%, Sharpe 1.62, maxDD -24.1%, beta +0.14
  2016-20: ann +16.7%, Sharpe 1.24, DD -21.2% | 2021-26: ann +25.7%, Sharpe 1.90, DD -6.5%
  yearly rows in YEARLY_REF; 72/119 months positive; worst month -8.9% (2021-01).

Also reproduces the equity-curve chart (output/repro/timesfm_final_equity_curve_repro.png):
growth of $1 (log) for final vs book-only ratchet vs raw buffered book vs SPX
excess vs T-bill+4%, plus the drawdown panel.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from repro_lib import (book_returns, load_preds, max_drawdown, monthly_market,
                       monthly_sets, perf_stats, ratchet_scales, yearly_table)

MEMO = {"ann": 0.218, "sharpe": 1.62, "maxdd": -0.241, "beta": 0.14}
SUB = {"2016-20": {"ann": 0.167, "sharpe": 1.24, "maxdd": -0.212},
       "2021-26": {"ann": 0.257, "sharpe": 1.90, "maxdd": -0.065}}
YEARLY_REF = {
    2016: (0.200, None), 2017: (0.241, 1.63), 2018: (0.382, 3.22),
    2019: (0.198, 1.57), 2020: (-0.161, -1.27), 2021: (0.079, 0.59),
    2022: (0.159, 1.04), 2023: (0.323, 2.09), 2024: (0.348, 2.61),
    2025: (0.300, 2.81), 2026: (0.371, 2.82),
}

REPRO_OUT = Path(__file__).resolve().parents[2] / "output" / "repro"
REPRO_OUT.mkdir(parents=True, exist_ok=True)


def final_config_returns(ls, mkt, variant):
    """Monthly excess returns for the final config under a ratchet variant.

    ls     : buffered book L/S excess return (Period-indexed)
    mkt    : monthly_market() frame (spx_ret, rf_dec, mkt_rf, benchmark)
    variant: "A" ratchet on combined book+SPX; "B" ratchet on book only

    The SPX sleeve earns the SPX price return (the convention that
    reproduces the memo table); book legs are excess over the pipeline RF.
    """
    idx = ls.index.intersection(mkt.index)
    ls, mkt = ls.reindex(idx), mkt.reindex(idx)
    book_1x = 0.5 * ls          # book at 1x gross, dollar-neutral
    combined = book_1x + 0.5 * mkt["spx_ret"]  # SPX overlay at 0.5x, price return
    if variant == "A":
        scales = ratchet_scales(combined)
        ret = scales * combined
        gross = scales * 1.0 + 0.5           # book 100% (ratcheted) + SPX 50%
        net = scales * 0.0 + 0.5
    elif variant == "B":
        scales = ratchet_scales(book_1x)
        ret = scales * book_1x + 0.5 * mkt["mkt_rf"]
        gross = scales * 1.0 + 0.5           # SPX sleeve not ratcheted
        net = 0.5
    else:
        raise ValueError(variant)
    return ret, scales, gross, net, mkt


def beta_with_se(port_ret, mkt):
    df = pd.DataFrame({"port": port_ret, "mkt": mkt["mkt_rf"]}).dropna()
    x, y = df["mkt"].values, df["port"].values
    beta = float(np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1))
    resid = y - beta * x
    n = len(x)
    se = float(np.sqrt(np.sum(resid ** 2) / (n - 2) / np.sum((x - x.mean()) ** 2)))
    return beta, se


def subwindow_table(ret, mkt):
    df = pd.DataFrame({"ret": ret, "mkt": mkt["mkt_rf"]}, index=ret.index)
    rows = []
    for label, m in (("full", df.index.year >= 2016),
                     ("2016-20", df.index.year <= 2020),
                     ("2021-26", df.index.year >= 2021)):
        g = df.loc[m, "ret"]
        rows.append({
            "window": label,
            "ann": g.mean() * 12,
            "sharpe": g.mean() / g.std(ddof=1) * np.sqrt(12),
            "maxdd": max_drawdown(g),
        })
    return pd.DataFrame(rows).set_index("window")


def main():
    preds = load_preds()
    long_sets, short_sets = monthly_sets(preds, buffer_k=1500)
    port = book_returns(preds, long_sets, short_sets)
    idx = pd.PeriodIndex(port["target_month"], freq="M")
    ls = pd.Series(port["ret_ls"].values, index=idx)
    mkt = monthly_market()

    results = {}
    for variant in ("A", "B"):
        ret, scales, gross, net, mkt_v = final_config_returns(ls, mkt, variant)
        stats = perf_stats(ret)
        beta, beta_se = beta_with_se(ret, mkt_v)
        results[variant] = dict(ret=ret, scales=scales, gross=gross, net=net,
                                mkt=mkt_v, stats=stats, beta=beta, beta_se=beta_se)
        yt = yearly_table(ret, ret.index)
        score = sum(
            abs(yt.loc[y, "ann_ret"] - ref[0]) for y, ref in YEARLY_REF.items()
            if y in yt.index
        )
        results[variant]["yearly_err"] = score
        print(f"variant {variant}: ann {stats['ann_ret']:+.1%}  Sharpe {stats['sharpe']:.2f}  "
              f"maxDD {stats['maxdd']:.1%}  beta {beta:+.2f}  yearly-err {score:.4f}")
    print(f"memo target   : ann {MEMO['ann']:+.1%}  Sharpe {MEMO['sharpe']:.2f}  "
          f"maxDD {MEMO['maxdd']:.1%}  beta {MEMO['beta']:+.2f}\n")

    best = min(results, key=lambda v: results[v]["yearly_err"])
    res = results[best]
    ret, scales, gross, net, mkt_v = (res["ret"], res["scales"], res["gross"],
                                      res["net"], res["mkt"])
    stats = res["stats"]
    print(f">>> variant {best} reproduces the memo table (yearly err {res['yearly_err']:.4f})\n")

    sw = subwindow_table(ret, mkt_v)
    print("window  |    ann | Sharpe | maxDD  | memo (ann, Sharpe, maxDD)")
    for w, row in sw.iterrows():
        m = {"full": MEMO, **SUB}[w]
        print(f"{w:7s} | {row['ann']:+6.1%} | {row['sharpe']:5.2f} | {row['maxdd']:6.1%} | "
              f"({m['ann']:+.1%}, {m['sharpe']:.2f}, {m['maxdd']:.1%})")

    b, bse = res["beta"], res["beta_se"]
    mask_oos = ret.index.year >= 2021
    beta_oos, _ = beta_with_se(ret[mask_oos], mkt_v[mask_oos])
    print(f"\nrealized beta: {b:+.2f} (se {bse:.2f}); OOS 2021-26: {beta_oos:+.2f}   memo +0.14 (does not reproduce)")

    yt = yearly_table(ret, ret.index)
    print("\nyear |  ann | Sharpe | memo (ann, Sharpe)")
    for yr, row in yt.iterrows():
        ref = YEARLY_REF.get(int(yr), (np.nan, np.nan))
        print(f"{yr} | {row['ann_ret']:+6.1%} | {row['sharpe']:5.2f} | "
              f"({ref[0]:+.3f}, {ref[1] if ref[1] is not None else float('nan'):.2f})")

    pos = (ret > 0).sum()
    worst = ret.idxmin()
    deep = (ret < -0.05).sum()
    cum = float((1 + ret).prod())
    print(f"\npositive months: {pos}/{len(ret)} (memo 72/119)")
    print(f"worst month    : {worst} {ret.min():+.1%} (memo 2021-01 -8.9%)")
    print(f"months < -5%   : {deep} (memo 4)")
    print(f"cumulative     : {cum:.1f}x (memo 7.8x)")
    print(f"exposure       : gross avg {gross.mean():.0%} max {gross.max():.0%}, "
          f"net {net.mean():+.0%} (memo nominal 150% / +50%)")
    print(f"ratchet scale  : avg {scales.mean():.2f}, min {scales.min():.2f}, max {scales.max():.2f}")

    out = pd.DataFrame({
        "ret_final": ret, "ratchet_scale": scales,
        "gross_exposure": gross, "net_exposure": net,
        "mkt_rf": mkt_v["mkt_rf"], "benchmark": mkt_v["benchmark"],
    })
    out.to_csv(REPRO_OUT / "final_config_monthly.csv")
    print(f"\nsaved: {REPRO_OUT / 'final_config_monthly.csv'}")

    make_chart(ls, results, best)


def make_chart(ls, results, best_variant):
    mkt = monthly_market()
    idx = ls.index.intersection(mkt.index)
    ls, mkt = ls.reindex(idx), mkt.reindex(idx)

    raw = (1 + ls).cumprod()
    book_ratchet = (1 + ratchet_scales(ls) * ls).cumprod()
    final = (1 + results[best_variant]["ret"]).cumprod()
    spx = (1 + mkt["mkt_rf"]).cumprod()
    bench = (1 + mkt["benchmark"]).cumprod()

    x = [p.to_timestamp() for p in final.index]

    def dd(cum):
        return (cum / cum.cummax() - 1) * 100

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 9), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(x, final, color="#1f77b4", lw=2.8,
             label=f"Final: book 1x gross + 0.5x SPX + ratchet "
                   f"(Sharpe {results[best_variant]['stats']['sharpe']:.2f}, "
                   f"maxDD {results[best_variant]['stats']['maxdd']:.0%})")
    ax1.plot(x, book_ratchet, color="#7f7f7f", lw=1.4,
             label="Book only (2x gross) + ratchet")
    ax1.plot(x, raw, color="#7f7f7f", lw=0.9, ls="--", label="Raw book, k=1500 buffer")
    ax1.plot(x, spx, color="#ff7f0e", lw=1.4, label="S&P 500 excess (context)")
    ax1.plot(x, bench, color="#2ca02c", lw=1.2, ls=":", label="Benchmark: T-bill + 4%")
    ax1.set_yscale("log")
    ax1.set_ylabel("Growth of $1 (log)")
    ax1.set_title("TimesFM strategy - full competition window 2016-2026 (reproduction)")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.fill_between(x, dd(final), 0, color="#d62728", alpha=0.45)
    ax2.set_ylabel("Drawdown (%)")
    ax2.annotate(f"maxDD {results[best_variant]['stats']['maxdd']:.0%}",
                 xy=(x[int(np.argmin(dd(final).values))], dd(final).min()),
                 xytext=(10, -10), textcoords="offset points",
                 color="#d62728", fontsize=9)
    ax2.grid(alpha=0.3)

    years = [p.year for p in final.index]
    tick_idx = [i for i in range(len(years)) if i == 0 or years[i] != years[i - 1]]
    ax2.set_xticks([x[i] for i in tick_idx])
    ax2.set_xticklabels([years[i] for i in tick_idx])

    fig.tight_layout()
    path = REPRO_OUT / "timesfm_final_equity_curve_repro.png"
    fig.savefig(path, dpi=150)
    print(f"saved: {path}")


if __name__ == "__main__":
    main()
