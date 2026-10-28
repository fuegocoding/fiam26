"""Rank-target XGBoost forecast sleeve over stocks priced at $1 or more."""

import typing

import pandas as pd

from sleeves import base
from sleeves.xgb_rank import model


class XgbRank(base.Sleeve):
    """Seed-averaged walk-forward XGBoost rank forecast."""

    name = "xgb_rank"

    @typing.override
    def build(self, context: base.Context) -> None:
        model.build(context)

    @typing.override
    def load(self, context: base.Context) -> None:
        forecasts = pd.read_parquet(
            model.path(context.work_dir),
            columns=["permno", "eom", "target_month", "pred_xgb_rank"],
        ).rename(columns={"pred_xgb_rank": "score"})
        self.frame = base.next_month_forecasts(forecasts, context.chars)

    @typing.override
    def score(self, year: int, broad: bool = False) -> pd.DataFrame:
        frame = self.frame
        if not broad:
            frame = frame.loc[frame.prc >= 1.0]
        return base.formation_output(frame, year)
