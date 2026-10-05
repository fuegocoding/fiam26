"""Scores retail stocks using sales growth minus inventory growth."""

import typing

import pandas as pd

from sleeves import base

_SIZE_GROUPS = ("small", "large", "mega")
_INDUSTRIES = ("2550", "2520")


class InventoryRetail(base.Sleeve):
    """Sales minus inventory growth for retailers and consumer durables."""

    name = "inventory_retail"

    @typing.override
    def score(self, year: int, broad: bool = False) -> pd.DataFrame:
        frame = self.frame.assign(score=self.frame.dsale_dinv)
        if not broad:
            frame = frame.loc[
                frame.tradable
                & frame.size_grp.isin(_SIZE_GROUPS)
                & frame.gics.astype("string").str[:4].isin(_INDUSTRIES)
            ]
        return base.formation_output(frame, year)
