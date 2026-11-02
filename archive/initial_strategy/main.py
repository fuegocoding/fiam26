"""Runs signal training, portfolio construction and reporting."""

import argparse
import json
import shutil

import pandas as pd

import borrow
import combiner
import config
import evaluate
import identity
import optimizer
import report
import timing
from sleeves import base

HEADLINE = [
    "ir",
    "cagr",
    "ann_active",
    "sharpe",
    "max_dd",
    "alpha_ann",
    "alpha_t",
    "beta",
    "beta_se",
    "alpha4_ann",
    "alpha4_t_hac",
    "turnover_avg",
    "oos_r2_universe",
    "compliance_all_pass",
]


def check_inputs() -> None:
    names = [
        config.PANEL_FILE,
        config.FACTOR_LIST_FILE,
        config.FILINGS_FILE,
        evaluate.DSF_FILE,
    ]
    missing = [n for n in names if not (config.DATA_DIR / n).exists()]
    if missing:
        raise SystemExit(
            f"Missing in {config.DATA_DIR}: {', '.join(missing)}. See data/README.md."
        )


def build_predictions() -> pd.DataFrame:
    """Steps 1-3: monthly stock scores with point-in-time labels."""
    context = base.Context(
        chars=base.load_panel(config.DATA_DIR),
        data_dir=config.DATA_DIR,
        work_dir=config.WORK_DIR,
        device=config.DEVICE,
    )
    predictions = combiner.predict(context)
    labelled = identity.label(
        predictions,
        config.DATA_DIR / config.PANEL_FILE,
        config.DATA_DIR / config.FILINGS_FILE,
    )

    timing.validate_predictions(labelled)
    return labelled


def nav_schedule(
    portfolio: pd.DataFrame,
    universe: pd.DataFrame,
    market: pd.DataFrame,
    endowment: float,
) -> pd.Series:
    """Capital at the formation of each holding month.

    The mandate endows US$1bn at the start of the out-of-sample period and the
    fund compounds from there (Instructions 3.1, 3.3): month t is sized on the
    endowment grown by every prior month's net return, so a successful fund
    tightens its own liquidity limits.
    """
    books = {
        month: dict(zip(g.permno.astype(int), g.target_weight, strict=True))
        for month, g in portfolio.groupby("return_month")
    }
    returns = evaluate.evaluate(books, universe, market).total
    return (1 + returns).cumprod().shift(1).fillna(1.0) * float(endowment)


def score_column(predictions: pd.DataFrame) -> str:
    """The score the optimizer ranks on at this fund size."""
    liquid = (
        optimizer.ENABLE_MICRO_SLEEVE_SWITCH
        and config.NAV > optimizer.MICRO_SLEEVES_MAX_NAV
        and "final_score_liquid" in predictions
    )
    return "final_score_liquid" if liquid else "final_score"


def main() -> None:
    parser = argparse.ArgumentParser(description="FIAM 2026 book, data to report.")
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="keep the models and data already in output/ instead of rebuilding",
    )
    args = parser.parse_args()
    check_inputs()
    if not args.reuse and config.WORK_DIR.exists():
        shutil.rmtree(config.WORK_DIR)
    config.WORK_DIR.mkdir(parents=True, exist_ok=True)
    timing.prepare_cache(config.WORK_DIR, config.DATA_DIR, config.DEVICE)
    panel = config.DATA_DIR / config.PANEL_FILE

    predictions = build_predictions()
    predictions.to_parquet(config.WORK_DIR / "predictions.parquet", index=False)

    crowded = pd.read_parquet(borrow.build(config.WORK_DIR, panel))
    universe = evaluate.load_universe(panel)
    market = evaluate.market()

    # Dollar limits follow the compounding fund, so the book and the capital
    # path have to agree. Iterate until the book stops changing; in practice
    # this settles in about nine passes.
    nav: float | pd.Series = config.NAV
    previous = None
    for _ in range(16):
        portfolio, diagnostics = optimizer.optimize(
            predictions, panel, nav=nav, short_interest=crowded
        )
        key = portfolio[
            ["return_month", "permno", "target_weight"]
        ].reset_index(drop=True)
        nav = nav_schedule(portfolio, universe, market, config.NAV)
        if previous is not None and key.equals(previous):
            break
        previous = key
    optimizer.write(portfolio, diagnostics, config.WORK_DIR, nav=config.NAV)

    stats = report.run(
        portfolio,
        predictions,
        universe,
        market,
        crowded,
        config.WORK_DIR / "report",
        score_column(predictions),
        nav=nav,
        data_dir=config.DATA_DIR,
    )
    headline = {key: stats.get(key) for key in HEADLINE}
    print(json.dumps(headline, indent=2, default=str))


if __name__ == "__main__":
    main()
