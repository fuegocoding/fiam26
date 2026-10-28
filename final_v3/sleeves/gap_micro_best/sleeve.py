"""Micro-cap valuation gap from each test year's selected pricing model."""

import json
import typing

import pandas as pd

from sleeves import base
from sleeves.gap_micro_best import valuation


def _month_end(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values).dt.to_period("M").dt.to_timestamp("M")


class GapMicroBest(base.Sleeve):
    """Model-implied minus actual log value, native to tradable micro caps."""

    name = "gap_micro_best"

    def __init__(self) -> None:
        """Starts with no selected models or valuation predictions."""
        super().__init__()
        self._best: dict[int, str] = {}
        self._xs = pd.DataFrame()
        self._pooled = pd.DataFrame()

    @typing.override
    def build(self, context: base.Context) -> None:
        valuation.build(context)

    @typing.override
    def load(self, context: base.Context) -> None:
        self.frame = context.chars
        directory = valuation.directory(context.work_dir)
        selection_path = directory / valuation.SELECTION_FILE
        selection = json.loads(selection_path.read_text())
        self._best = {
            int(year): fold["best_pricing_model"]
            for year, fold in selection["folds"].items()
        }
        chosen = set(self._best.values())
        required = chosen - set(valuation.POOLED_MODELS)
        if "M9" in chosen:
            required.update(set(valuation.ENSEMBLE) - set(valuation.POOLED_MODELS))
        columns = ["permno", "eom", "y", *sorted(required)]
        self._xs = pd.read_parquet(
            directory / valuation.XS_FILE, columns=list(dict.fromkeys(columns))
        )
        self._xs["eom"] = _month_end(self._xs.eom)
        pooled = pd.read_parquet(
            directory / valuation.POOLED_FILE,
            columns=["permno", "eom", "fold_year", "M6", "M8"],
        )
        pooled["eom"] = _month_end(pooled.eom)
        self._pooled = pooled.merge(self._xs, on=["permno", "eom"], how="left")
        if "M9" in chosen:
            self._pooled["M9"] = self._pooled[valuation.ENSEMBLE].mean(
                axis=1, skipna=False
            )

    @typing.override
    def score(self, year: int, broad: bool = False) -> pd.DataFrame:
        if year not in self._best:
            return pd.DataFrame(
                {
                    "return_month": pd.Series(dtype="period[M]"),
                    "permno": pd.Series(dtype="int64"),
                    "score": pd.Series(dtype="float64"),
                }
            )
        model = self._best[year]
        if model in valuation.POOLED_MODELS:
            gap = self._pooled.loc[self._pooled.fold_year.eq(year)].copy()
        else:
            gap = self._xs.copy()
        gap["score"] = gap[model] - gap.y
        chars = self.frame
        if not broad:
            chars = chars.loc[chars.tradable & chars.size_grp.eq("micro")]
        frame = chars[["permno", "eom"]].merge(
            gap[["permno", "eom", "score"]], on=["permno", "eom"], how="inner"
        )
        return base.formation_output(frame, year)
