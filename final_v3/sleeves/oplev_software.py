"""Scores software stocks using sales growth minus expense growth."""

import typing

import pandas as pd

from sleeves import base

_SIZE_GROUPS = ("small", "large", "mega")
_INDUSTRIES = ("4510",)


class OplevSoftware(base.Sleeve):
    """Sales minus SG&A growth for software and IT services."""

    name = "oplev_software"

    @typing.override
    def score(self, year: int, broad: bool = False) -> pd.DataFrame:
        frame = self.frame.assign(score=self.frame.dsale_dsga)
        if not broad:
            frame = frame.loc[
                frame.tradable
                & frame.size_grp.isin(_SIZE_GROUPS)
                & frame.gics.astype("string").str[:4].isin(_INDUSTRIES)
            ]
        return base.formation_output(frame, year)
