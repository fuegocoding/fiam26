"""Official data supplies no honest short-interest crowding proxy."""

import pathlib

import pandas as pd


def build(work_dir: pathlib.Path, panel_path: pathlib.Path) -> pathlib.Path:
    """Writes the unchanged screen schema with no unsupported observations."""
    output = work_dir / 'short_interest' / 'short_interest.parquet'
    if output.exists():
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        'permno': pd.Series(dtype='int64'),
        'return_month': pd.Series(dtype='datetime64[ns]'),
        'short_pct': pd.Series(dtype='float64'),
        'days_to_cover': pd.Series(dtype='float64'),
    }).to_parquet(output, index=False)
    return output
