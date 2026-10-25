"""Stocks ranked within size group and sector by their SEC filing events."""

import typing

import pandas as pd

from sleeves import base
from sleeves.filing_events import model


class FilingEvents(base.Sleeve):
    """Seed-averaged walk-forward forecast from late-filing notices, comment
    letters, offerings, 13D/13G stakes, proxy fights, 10-K/A amendments,
    spin-offs, deal registrations and the like."""

    name = "filing_events"

    @typing.override
    def build(self, context: base.Context) -> None:
        model.build(context)

    @typing.override
    def load(self, context: base.Context) -> None:
        forecasts = pd.read_parquet(
            model.path(context.work_dir),
            columns=["permno", "eom", "target_month", "pred_filing_events"],
        ).rename(columns={"pred_filing_events": "score"})
        self.frame = base.next_month_forecasts(forecasts, context.chars)

    @typing.override
    def score(self, year: int, broad: bool = False) -> pd.DataFrame:
        frame = self.frame
        if not broad:
            frame = frame.loc[frame.prc >= 1.0]
        return base.formation_output(frame, year)
