"""FIAM 2026 long/short U.S. equity book, from data to report.

    python final/main.py           # rebuild every model and recompute everything
    python final/main.py --reuse   # keep what is already in output/

Settings and paths are in config.py. The competition files go in data/;
everything else ships in external_data/. Results are written to output/,
the submission files and performance pack to output/report/.

For every holding month from January 2021 to August 2026:

1. Sleeves (combiner.py, sleeves/). Nine return signals score every stock at
   month-end t for the return of month t+1, from data available at t.
   Learned sleeves refit once a year on target months up to Y-3, validate on
   Y-2..Y-1 and predict year Y. Scores become monthly ranks, averaged over
   the sleeves that cover a stock.
2. Graph overlay (graph/). A point-in-time company graph and a
   peer-attention model forecast the next EPS surprise; 5% rank tilt.
3. Identity agent (identity.py). Each stock-month gets a point-in-time
   ticker and company name with its source.
4. Borrow screen (borrow.py). No new short at 15% short interest or above;
   use public dates or a conservative ten-business-day publication delay.
5. Optimizer and risk rules (optimizer.py, risk.py). 100 longs and 100
   shorts, sector-matched, with bounded beta sizing and liquidity caps, then the
   tail-risk rules.
6. Evaluation and report (evaluate.py, report.py). Net of trading costs and
   borrow fees; cash earns the T-bill; benchmark T-bill + 4%.
"""

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
    names = [config.PANEL_FILE, config.FACTOR_LIST_FILE, config.FILINGS_FILE]
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
    portfolio, diagnostics = optimizer.optimize(
        predictions, panel, nav=config.NAV, short_interest=crowded
    )
    optimizer.write(portfolio, diagnostics, config.WORK_DIR, nav=config.NAV)

    stats = report.run(
        portfolio,
        predictions,
        evaluate.load_universe(panel),
        evaluate.market(),
        crowded,
        config.WORK_DIR / "report",
        score_column(predictions),
    )
    headline = {key: stats.get(key) for key in HEADLINE}
    print(json.dumps(headline, indent=2, default=str))


if __name__ == "__main__":
    main()
