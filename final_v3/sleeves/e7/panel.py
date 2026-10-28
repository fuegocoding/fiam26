"""Delayed earnings surprise and earnings/sales directional agreement."""

import pathlib

import numpy as np
import pandas as pd

from sleeves import base, expectations


def path(work_dir: pathlib.Path) -> pathlib.Path:
    return work_dir / 'e7' / 'delayed.parquet'


def build(context: base.Context) -> pathlib.Path:
    """Forms one month after the filing month, only after surprise availability."""
    output = path(context.work_dir)
    if output.exists():
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    events = expectations.load_surprises()
    events['eom'] = expectations.month_end(events.anndats) + pd.offsets.MonthEnd(1)
    events = events.loc[events.available_date.le(events.eom)].copy()
    events['sue'] = events.suescore
    # Agreement between two independent accounting surprises replaces analyst breadth.
    events['post_breadth'] = (np.sign(events.suescore) + np.sign(events.saleq_su)) / 2
    events = events.sort_values('anndats').drop_duplicates(['permno', 'eom'], keep='last')
    delayed = events.merge(context.chars[['permno', 'eom', 'sector', 'size_grp', 'tradable']],
                            on=['permno', 'eom'], how='inner')
    delayed[['permno', 'eom', 'sector', 'size_grp', 'tradable', 'sue',
             'post_breadth']].to_parquet(output, index=False)
    return output
