"""Earnings-surprise plus post-report breadth sleeves for chosen size groups."""

import typing

import pandas as pd

from sleeves import base
from sleeves.e7 import panel


class E7(base.Sleeve):
    """Sector-relative surprise and breadth ranks in one size bucket."""

    def __init__(self, name: str, size_groups: tuple[str, ...]) -> None:
        """Initializes one size-bucket variant.

        Args:
            name: Sleeve name, e.g. "e7_lm".
            size_groups: Size groups that form the native universe.
        """
        super().__init__()
        self.name = name
        self._size_groups = size_groups

    @typing.override
    def build(self, context: base.Context) -> None:
        panel.build(context)

    @typing.override
    def load(self, context: base.Context) -> None:
        self.frame = pd.read_parquet(panel.path(context.work_dir))
        self.frame["eom"] = pd.to_datetime(self.frame.eom)

    @typing.override
    def score(self, year: int, broad: bool = False) -> pd.DataFrame:
        frame = self.frame.copy()
        if not broad:
            frame = frame.loc[
                frame.tradable & frame.size_grp.isin(self._size_groups)
            ].copy()
        groups = frame.groupby(["eom", "sector"])
        frame["score"] = groups.sue.rank(pct=True) + groups.post_breadth.rank(pct=True)
        return base.formation_output(frame, year)
