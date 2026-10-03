"""Paths and run settings. Edit here; every module imports them."""

import pathlib

import pandas as pd

FINAL = pathlib.Path(__file__).resolve().parent

# Competition files go in DATA_DIR (see data/README.md).
DATA_DIR = FINAL / "data"
PANEL_FILE = "chars_final_with_names.parquet"
FACTOR_LIST_FILE = "factor_char_list.csv"
FILINGS_FILE = "8k_20150101_20260831_identified.parquet"

# I/B/E/S, FRED, FINRA and SEC EDGAR in one parquet file (see its README).
EXTERNAL_DIR = FINAL / "external_data"

# Models, predictions, the book and the report. main.py clears it first
# unless run with --reuse.
WORK_DIR = FINAL / "output"

# Fund size in dollars; every dollar limit in the optimizer derives from it.
NAV = 10_000_000.0

# TimesFM and the graph model run on this device.
DEVICE = "cuda"

# Holding months: January 2021 to August 2026. Year Y is predicted by models
# trained on target months up to Y-3 and validated on Y-2..Y-1.
TEST_YEARS = tuple(range(2021, 2027))
LAST_TARGET = pd.Timestamp("2026-08-31")
