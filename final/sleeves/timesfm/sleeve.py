"""Zero-shot TimesFM forecast sleeve over stocks priced at $1 or more."""

import typing

import pandas as pd

from sleeves import base
from sleeves.timesfm import forecast

# A zero-shot forecast from a short return history is mostly noise; 48 months
# was the best threshold on the 2019-2020 validation years.
MIN_HISTORY_MONTHS = 48


class TimesFm(base.Sleeve):
    """Median next-month TimesFM forecast of excess return."""

    name = "timesfm"

    @typing.override
    def build(self, context: base.Context) -> None:
        forecast.build(context)

    @typing.override
    def load(self, context: base.Context) -> None:
        forecasts = pd.read_parquet(
            forecast.path(context.work_dir),
            columns=["permno", "eom", "target_month", "tfm_forecast",
                     "tfm_n_ctx"],
        )
        forecasts = forecasts.loc[forecasts.tfm_n_ctx >= MIN_HISTORY_MONTHS]
        forecasts = forecasts.drop(columns="tfm_n_ctx").rename(
            columns={"tfm_forecast": "score"}
        )
        self.frame = base.next_month_forecasts(forecasts, context.chars)

    @typing.override
    def score(self, year: int, broad: bool = False) -> pd.DataFrame:
        frame = self.frame
        if not broad:
            frame = frame.loc[frame.prc >= 1.0]
        return base.formation_output(frame, year)
