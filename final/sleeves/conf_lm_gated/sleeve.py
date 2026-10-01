"""Large/mega-cap analyst revisions kept only when price momentum agrees."""

import typing

import numpy as np
import pandas as pd

from sleeves import base
from sleeves.conf_lm_gated import panel


class ConfLmGated(base.Sleeve):
    """Near-quarter revision, gated by the sign of 3-1 month momentum."""

    name = "conf_lm_gated"

    @typing.override
    def build(self, context: base.Context) -> None:
        panel.build(context)

    @typing.override
    def load(self, context: base.Context) -> None:
        revisions = pd.read_parquet(panel.path(context.work_dir))
        revisions["eom"] = pd.to_datetime(revisions.eom)
        self.frame = context.chars.merge(
            revisions, on=["permno", "eom"], how="inner"
        )

    @typing.override
    def score(self, year: int, broad: bool = False) -> pd.DataFrame:
        frame = self.frame.copy()
        agree = np.sign(frame.ret_3_1).eq(
            np.sign(frame.rev3_near)
        ) & frame.rev3_near.ne(0)
        frame["score"] = (
            frame.rev3_near if broad else frame.rev3_near.where(agree)
        )
        if not broad:
            frame = frame.loc[
                frame.tradable & frame.size_grp.isin(["large", "mega"])
            ]
        return base.formation_output(frame, year)
