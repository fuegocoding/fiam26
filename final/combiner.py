"""Runs every sleeve and the graph, blends their scores and sizes the book."""

import argparse
import json
import pathlib

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from graph import builder
from graph import business_text
from graph import dataset
from graph import overlay
from graph import train
import optimizer
from sleeves import base
from sleeves import inventory_retail
from sleeves import oplev_software
from sleeves import runway_micro
from sleeves.conf_lm_gated import sleeve as conf_lm_gated
from sleeves.e7 import sleeve as e7
from sleeves.gap_micro_best import sleeve as gap_micro_best
from sleeves.timesfm import forecast as timesfm_forecast
from sleeves.timesfm import sleeve as timesfm
from sleeves.xgb_rank import sleeve as xgb_rank

_REPO = pathlib.Path(__file__).resolve().parents[1]
FILINGS_FILE = "8k_20150101_20260831_identified.parquet"
GRAPH_SEEDS = (7, 19, 43)
_LABELS = [
    "ticker", "company_name", "ticker_name_reference_date",
    "ticker_name_source", "ticker_name_status",
]
_FORMATION_COLUMNS = [
    "permno", "permco", "eom", "prc", "size_grp", "dolvol_126d", "gics",
    "beta_60m", *_LABELS,
]
_PREDICTION_COLUMNS = [
    "return_month", "formation_date", "permno", "permco", *_LABELS, "prc",
    "size_grp", "sector", "beta_60m", "dolvol_126d", "available_count",
    "sleeve_score",
    "graph_attention_eps", "graph_available", "sleeve_rank", "graph_rank",
    "final_score",
]


def make_sleeves() -> tuple[base.Sleeve, ...]:
    """Every sleeve in blend order; the order fixes floating-point ties."""
    return (
        e7.E7("e7_lm", ("large", "mega")),
        e7.E7("e7_small_gated", ("small",)),
        gap_micro_best.GapMicroBest(),
        runway_micro.RunwayMicro(),
        conf_lm_gated.ConfLmGated(),
        inventory_retail.InventoryRetail(),
        oplev_software.OplevSoftware(),
        timesfm.TimesFm(),
        xgb_rank.XgbRank(),
    )


def score_sleeves(
    context: base.Context, sleeves: tuple[base.Sleeve, ...]
) -> pd.DataFrame:
    """Scaled score, coverage and native flag per sleeve and stock-month.

    A stock outside a sleeve's native universe is ranked against that month's
    native scores; a stock the sleeve does not cover at all scores zero.
    """
    universe = context.chars[["permno", "eom"]].copy()
    universe["return_month"] = universe.pop("eom").dt.to_period("M") + 1
    universe = universe.loc[
        universe.return_month.dt.year.isin(base.TEST_YEARS),
        ["return_month", "permno"],
    ].drop_duplicates().reset_index(drop=True)
    wide = universe.copy()
    for sleeve in sleeves:
        sleeve.build(context)
        sleeve.load(context)
        scaled = pd.concat(
            [
                base.scale_scores(
                    sleeve.score(year, broad=True), sleeve.score(year)
                )
                for year in base.TEST_YEARS
            ],
            ignore_index=True,
        )
        dense = universe.merge(
            scaled, on=["return_month", "permno"], how="left"
        )
        score = dense.native_score.fillna(dense.broad_score).fillna(0.0)
        wide[f"score_{sleeve.name}"] = score.to_numpy(float)
        wide[f"available_{sleeve.name}"] = dense.broad_score.notna().to_numpy()
        wide[f"native_{sleeve.name}"] = dense.native_score.notna().to_numpy()
        print(
            f"Sleeve {sleeve.name}: {int(dense.broad_score.notna().sum()):,}"
            " signals",
            flush=True,
        )
    return wide.sort_values(["return_month", "permno"]).reset_index(drop=True)


def blend(wide: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Equal-weight average over the sleeves that cover each stock-month."""
    score = wide[[f"score_{name}" for name in names]].to_numpy(float)
    available = wide[[f"available_{name}" for name in names]].to_numpy(bool)
    count = available.sum(axis=1)
    # Keep the explicit weighted sum: its floating-point ties feed the ranks.
    weights = np.divide(
        available,
        count[:, None],
        out=np.zeros_like(score),
        where=count[:, None] > 0,
    )
    blended = wide[["return_month", "permno"]].copy()
    blended["available_count"] = count
    blended["sleeve_score"] = (weights * score).sum(axis=1)
    return blended


def graph_forecasts(context: base.Context) -> pd.DataFrame:
    """Builds filing text, monthly graphs and tokens, then trains the model."""
    graph_dir = context.work_dir / "graph"
    text = business_text.build_profiles(
        context.data_dir / FILINGS_FILE, graph_dir / "business_text.parquet"
    )
    snapshots = builder.build_snapshots(
        context.panel_path, text, graph_dir / "snapshots"
    )
    tokens = dataset.build(
        graph_dir / "tokens",
        context.panel_path,
        snapshots,
        timesfm_forecast.build(context),
        context.ibes_dir,
    )
    runs = train.train(
        tokens, graph_dir / "runs", context.device, base.TEST_YEARS,
        GRAPH_SEEDS,
    )
    return train.ensemble(runs, base.TEST_YEARS, GRAPH_SEEDS)


def formation_fields(panel_path: pathlib.Path) -> pd.DataFrame:
    """Price, liquidity and labels known at formation, by return month."""
    panel = pq.read_table(panel_path, columns=_FORMATION_COLUMNS).to_pandas()
    panel["eom"] = pd.to_datetime(panel.eom).dt.to_period("M")
    panel["return_month"] = panel.eom + 1
    panel["prc"] = panel.prc.abs()
    panel["sector"] = panel.gics.astype("string").str[:2].fillna("na")
    reference = pd.to_datetime(panel.ticker_name_reference_date)
    panel["ticker_name_reference_date"] = reference
    # A name dated after formation cannot make a stock eligible at formation.
    future = reference.notna() & reference.gt(panel.eom.dt.to_timestamp("M"))
    panel.loc[future, ["ticker", "company_name"]] = None
    panel.loc[future, "ticker_name_status"] = (
        "no_verified_label_for_observation_period"
    )
    return panel


def predict(context: base.Context) -> pd.DataFrame:
    """Final NAV-independent stock scores for every test month."""
    sleeves = make_sleeves()
    wide = score_sleeves(context, sleeves)
    stored = wide.assign(return_month=wide.return_month.dt.to_timestamp("M"))
    stored.to_parquet(context.work_dir / "sleeves.parquet", index=False)
    blended = blend(wide, [sleeve.name for sleeve in sleeves])
    forecasts = graph_forecasts(context)
    forecasts.to_parquet(
        context.work_dir / "graph_forecasts.parquet", index=False
    )
    formation = formation_fields(context.panel_path)
    graph = overlay.company_forecasts(forecasts, formation)
    graph_months = graph.return_month.unique()
    blended = blended.loc[blended.return_month.isin(graph_months)]
    scores = blended.merge(
        formation[["return_month", "sector", *_FORMATION_COLUMNS]],
        on=["return_month", "permno"],
        how="left",
    )
    scored = overlay.apply(scores, graph).rename(
        columns={"eom": "formation_date"}
    )
    scored = scored[_PREDICTION_COLUMNS].sort_values(["return_month", "permno"])
    scored = scored.reset_index(drop=True)
    scored["return_month"] = scored.return_month.dt.to_timestamp("M")
    scored["formation_date"] = scored.formation_date.dt.to_timestamp("M")
    return scored


def main() -> None:
    """Runs the full pipeline and writes predictions and the book."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=pathlib.Path, default=_REPO / "data" / "raw",
        help="characteristics panel, factor list and 8-K filings",
    )
    parser.add_argument(
        "--ibes-dir", type=pathlib.Path, default=_REPO / "local",
        help="I/B/E/S consensus, price, surprise and link extracts",
    )
    parser.add_argument(
        "--work-dir", type=pathlib.Path, default=_REPO / "output" / "final",
        help="fitted models and outputs; existing artifacts are reused",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    context = base.Context(
        chars=base.load_panel(args.data_dir),
        data_dir=args.data_dir,
        ibes_dir=args.ibes_dir,
        work_dir=args.work_dir,
        device=args.device,
    )
    predictions = predict(context)
    predictions.to_parquet(args.work_dir / "predictions.parquet", index=False)
    # Council hook: final/council will adjust final_score here.
    portfolio, diagnostics = optimizer.optimize(
        predictions, context.panel_path
    )
    config = optimizer.write(portfolio, diagnostics, args.work_dir)
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
