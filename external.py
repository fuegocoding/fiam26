"""Reads the source tables stored in external.parquet."""

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
