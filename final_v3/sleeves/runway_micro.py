"""Micro-cap cash runway for companies with negative operating cash flow."""

import typing

import pandas as pd

from sleeves import base


class RunwayMicro(base.Sleeve):
    """Cash over operating cash burn, native to tradable micro caps."""

    name = "runway_micro"

    @typing.override
    def score(self, year: int, broad: bool = False) -> pd.DataFrame:
        frame = self.frame.copy()
        frame["score"] = (frame.cash_at / -frame.ocf_at).where(frame.ocf_at < 0)
        if not broad:
            frame = frame.loc[frame.tradable & frame.size_grp.eq("micro")]
        return base.formation_output(frame, year)
