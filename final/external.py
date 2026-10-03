"""Reads the external data: one parquet file, external_data/external.parquet.

Every source table is stacked in the file with a `table` column. The file's
metadata records each table's columns and types, so a read returns the frame
the original source gave. external_data/build.py writes the file.
"""

import functools
import json

import pandas as pd
import pyarrow.parquet as pq

import config

FILE = config.EXTERNAL_DIR / "external.parquet"


@functools.cache
def _tables() -> dict:
    return json.loads(pq.read_schema(FILE).metadata[b"tables"])


def read(table: str, columns: list[str] | None = None) -> pd.DataFrame:
    """One source table, optionally only some of its columns."""
    spec = _tables()[table]
    wanted = [c for c in spec["columns"] if columns is None or c in columns]
    frame = pd.read_parquet(FILE, columns=wanted, filters=[("table", "==", table)])
    retype = {c: spec["dtypes"][c] for c in wanted}
    return frame.astype({c: t for c, t in retype.items() if str(frame[c].dtype) != t})
