"""Combines stock signals and graph forecasts into monthly scores."""

import pathlib

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import config
import timing
from graph import builder, business_text, dataset, overlay
from sleeves import base, inventory_retail, oplev_software, runway_micro
from sleeves.conf_lm_gated import sleeve as conf_lm_gated
from sleeves.e7 import sleeve as e7
from sleeves.filing_events import sleeve as filing_events
from sleeves.gap_micro_best import sleeve as gap_micro_best
from sleeves.xgb_rank import sleeve as xgb_rank

GRAPH_SEEDS = (7, 19, 43)
_LABELS = [
    "ticker",
    "company_name",
    "ticker_name_reference_date",
    "ticker_name_source",
    "ticker_name_status",
]
_FORMATION_COLUMNS = [
    "permno",
    "permco",
    "eom",
    "prc",
    "me",
    "size_grp",
    "dolvol_126d",
    "gics",
    "beta_60m",
    *_LABELS,
]
# Exclude these sleeves from the blend above the fixed fund-size cutoff.
MICRO_SLEEVES = ("gap_micro_best", "runway_micro")
_PREDICTION_COLUMNS = [
    "return_month",
    "formation_date",
    "timing_policy",
    "permno",
    "permco",
    *_LABELS,
    "prc",
    "me",
    "size_grp",
    "sector",
    "beta_60m",
    "dolvol_126d",
    "available_count",
    "sleeve_score",
    "graph_attention_eps",
    "graph_available",
    "sleeve_rank",
    "graph_rank",
    "final_score",
    "final_score_liquid",
    "available_count_liquid",
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
        xgb_rank.XgbRank(),
        filing_events.FilingEvents(),
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
    universe = (
        universe.loc[
            universe.return_month.dt.year.isin(config.TEST_YEARS),
            ["return_month", "permno"],
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    wide = universe.copy()
    for sleeve in sleeves:
        sleeve.build(context)
        sleeve.load(context)
        scaled = pd.concat(
            [
                base.scale_scores(sleeve.score(year, broad=True), sleeve.score(year))
                for year in config.TEST_YEARS
            ],
            ignore_index=True,
        )
        dense = universe.merge(scaled, on=["return_month", "permno"], how="left")
        score = dense.native_score.fillna(dense.broad_score).fillna(0.0)
        wide[f"score_{sleeve.name}"] = score.to_numpy(float)
        wide[f"available_{sleeve.name}"] = dense.broad_score.notna().to_numpy()
        wide[f"native_{sleeve.name}"] = dense.native_score.notna().to_numpy()
        print(
            f"Sleeve {sleeve.name}: {int(dense.broad_score.notna().sum()):,} signals",
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
    cached = context.work_dir / "graph_forecasts.parquet"
    if cached.exists():
        return pd.read_parquet(cached)
    from graph import train

    graph_dir = context.work_dir / "graph"
    text = business_text.build_profiles(
        context.data_dir / config.FILINGS_FILE, graph_dir / "business_text.parquet"
    )
    snapshots = builder.build_snapshots(
        context.panel_path, text, graph_dir / "snapshots"
    )
    tokens = dataset.build(
        graph_dir / "tokens",
        context.panel_path,
        snapshots,
    )
    runs = train.train(
        tokens,
        graph_dir / "runs",
        context.device,
        config.TEST_YEARS,
        GRAPH_SEEDS,
    )
    return train.ensemble(runs, config.TEST_YEARS, GRAPH_SEEDS)


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
    panel.loc[future, "ticker_name_status"] = "no_verified_label_for_observation_period"
    return panel


def predict(context: base.Context) -> pd.DataFrame:
    """Final NAV-independent stock scores for every test month."""
    timing.prepare_cache(context.work_dir, context.data_dir, context.device)
    sleeves = make_sleeves()
    wide = score_sleeves(context, sleeves)
    stored = wide.assign(return_month=wide.return_month.dt.to_timestamp("M"))
    stored.to_parquet(context.work_dir / "sleeves.parquet", index=False)
    blended = blend(wide, [sleeve.name for sleeve in sleeves])
    forecasts = graph_forecasts(context)
    forecasts = forecasts.loc[
        pd.to_datetime(forecasts.target_month).le(config.LAST_TARGET)
    ]
    forecasts.to_parquet(context.work_dir / "graph_forecasts.parquet", index=False)
    formation = formation_fields(context.panel_path)
    graph = overlay.company_forecasts(forecasts, formation)
    graph_months = graph.return_month.unique()
    blended = blended.loc[blended.return_month.isin(graph_months)]
    scores = blended.merge(
        formation[["return_month", "sector", *_FORMATION_COLUMNS]],
        on=["return_month", "permno"],
        how="left",
    )
    liquid = blend(
        wide,
        [s.name for s in sleeves if s.name not in MICRO_SLEEVES],
    )[["return_month", "permno", "sleeve_score", "available_count"]]
    scores = scores.merge(
        liquid.rename(
            columns={
                "sleeve_score": "sleeve_score_liquid",
                "available_count": "available_count_liquid",
            }
        ),
        on=["return_month", "permno"],
        how="left",
    )
    scored = overlay.apply(scores, graph)
    scored_liquid = overlay.apply(
        scores.assign(sleeve_score=scores.sleeve_score_liquid), graph
    )
    assert len(scored) == len(scored_liquid)
    scored["final_score_liquid"] = scored_liquid.final_score.to_numpy()
    scored = scored.rename(columns={"eom": "formation_date"})
    scored["timing_policy"] = timing.PREDICTION_POLICY
    scored = scored[_PREDICTION_COLUMNS].sort_values(["return_month", "permno"])
    scored = scored.reset_index(drop=True)
    scored["return_month"] = scored.return_month.dt.to_timestamp("M")
    scored["formation_date"] = scored.formation_date.dt.to_timestamp("M")
    return scored
