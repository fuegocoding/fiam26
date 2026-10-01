"""Sleeve contract plus the formation panel and score scaling shared by all."""

import abc
import dataclasses
import pathlib

import numpy as np
import pandas as pd

PANEL_FILE = "chars_final_with_names.parquet"
FACTOR_LIST_FILE = "factor_char_list.csv"
TEST_YEARS = tuple(range(2021, 2027))
LAST_TARGET = pd.Timestamp("2026-08-31")
_PANEL_COLUMNS = [
    "permno", "eom", "gics", "size_grp", "prc", "dolvol",
    "cash_at", "ocf_at", "ret_3_1", "dsale_dinv", "dsale_dsga",
]


@dataclasses.dataclass(frozen=True)
class Context:
    """The shared formation panel plus where data is read and models go."""

    chars: pd.DataFrame
    data_dir: pathlib.Path
    ibes_dir: pathlib.Path
    work_dir: pathlib.Path
    device: str = "cuda"

    @property
    def panel_path(self) -> pathlib.Path:
        """Location of the characteristics panel."""
        return self.data_dir / PANEL_FILE


class Sleeve(abc.ABC):
    """A stock signal scored each month inside its native universe.

    Sleeves backed by derived panels or fitted models override `build`, which
    writes artifacts under `context.work_dir` once, and `load`, which reads
    them before scoring. Simple sleeves score the shared formation panel.
    """

    name = ""

    def __init__(self) -> None:
        """Starts with no formation inputs; `load` fills them."""
        self.frame = pd.DataFrame()

    def build(self, context: Context) -> None:
        """Writes the features or models this sleeve needs; none by default."""
        del context  # Simple sleeves only read the shared formation panel.

    def load(self, context: Context) -> None:
        """Reads the sleeve's formation inputs once before scoring."""
        self.frame = context.chars

    @abc.abstractmethod
    def score(self, year: int, broad: bool = False) -> pd.DataFrame:
        """Scores formation months whose next month falls in `year`.

        Args:
            year: Calendar year of the return months to score.
            broad: Score every covered stock instead of the native universe.

        Returns:
            Frame with return_month (monthly period), permno and score.
        """


def load_panel(data_dir: pathlib.Path) -> pd.DataFrame:
    """Formation panel with sector and the liquid, priced tradable flag."""
    chars = pd.read_parquet(data_dir / PANEL_FILE, columns=_PANEL_COLUMNS)
    chars["eom"] = (
        pd.to_datetime(chars.eom).dt.to_period("M").dt.to_timestamp("M")
    )
    chars = chars.drop_duplicates(["permno", "eom"]).reset_index(drop=True)
    chars["sector"] = chars.gics.astype("string").str[:2]
    chars["tradable"] = (
        (chars.prc.abs() >= 5)
        & (chars.groupby("eom").dolvol.rank(pct=True) >= 0.2)
        & chars.sector.notna()
    )
    return chars


def formation_output(frame: pd.DataFrame, year: int) -> pd.DataFrame:
    """Keeps finite scores and aligns each formation row to its next month."""
    frame = frame.loc[np.isfinite(frame.score)].copy()
    frame["return_month"] = pd.to_datetime(frame.eom).dt.to_period("M") + 1
    return frame.loc[
        frame.return_month.dt.year.eq(year), ["return_month", "permno", "score"]
    ].reset_index(drop=True)


def next_month_forecasts(
    forecasts: pd.DataFrame, chars: pd.DataFrame
) -> pd.DataFrame:
    """Keeps one-month-ahead forecasts and attaches the formation price."""
    forecasts = forecasts.copy()
    forecasts["eom"] = pd.to_datetime(forecasts.eom)
    forecasts["target_month"] = pd.to_datetime(forecasts.target_month)
    forecasts = forecasts.loc[
        forecasts.target_month.dt.to_period("M").eq(
            forecasts.eom.dt.to_period("M") + 1
        )
    ]
    return forecasts.merge(
        chars[["permno", "eom", "prc"]], on=["permno", "eom"], how="left"
    )


def scale_scores(broad: pd.DataFrame, native: pd.DataFrame) -> pd.DataFrame:
    """Uniform monthly ranks; non-native stocks are ranked against natives."""
    broad = broad.copy()
    broad["broad_score"] = 0.0
    references = {
        month: np.sort(group.score.to_numpy(float))
        for month, group in native.groupby("return_month", sort=False)
    }
    by_month = broad.groupby("return_month", sort=False).indices
    for month, positions in by_month.items():
        values = broad.iloc[positions].score.to_numpy(float)
        reference = references.get(month)
        if reference is None or not len(reference):
            ranks = pd.Series(values).rank(pct=True).to_numpy(float)
        else:
            below = np.searchsorted(reference, values, side="left")
            above = np.searchsorted(reference, values, side="right")
            ranks = (below + above) / (2 * len(reference))
        broad.iloc[positions, broad.columns.get_loc("broad_score")] = (
            ranks - 0.5
        ) * np.sqrt(12.0)
    native = native.copy()
    groups = native.groupby("return_month").score
    native["native_score"] = (
        (groups.rank(pct=True) - 0.5) * np.sqrt(12.0)
    ).where(groups.transform("size") > 1, 0.0)
    return broad[["return_month", "permno", "broad_score"]].merge(
        native[["return_month", "permno", "native_score"]],
        on=["return_month", "permno"],
        how="left",
    )
