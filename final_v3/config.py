"""Sets file locations, fund size and evaluation dates."""

import pathlib

import pandas as pd

FINAL = pathlib.Path(__file__).resolve().parent

# Competition files go in DATA_DIR (see data/README.md).
DATA_DIR = FINAL.parent / "data"
PANEL_FILE = "chars_final_with_names.parquet"
FACTOR_LIST_FILE = "factor_char_list.csv"
FILINGS_FILE = "8k_20150101_20260831_identified.parquet"

# Models, predictions, the book and the report. main.py clears it first
# unless run with --reuse.
WORK_DIR = FINAL / "output"

# Fund size in dollars; every dollar limit in the optimizer derives from it.
# The final mandate endows US$1bn at the start of the out-of-sample period and
# the fund compounds from there (Instructions 3.1, 3.3).
NAV = 1_000_000_000.0

# Fitted sleeves and the graph model run on this device.
DEVICE = "cuda"

# Holding months: January 2021 to August 2026. Year Y is predicted by models
# trained on target months up to Y-3 and validated on Y-2..Y-1.
TEST_YEARS = tuple(range(2021, 2027))
LAST_TARGET = pd.Timestamp("2026-08-31")
REPORTING_FILE = FINAL.parent / 'external_data' / 'fred_reporting.parquet'
if not REPORTING_FILE.exists():
    REPORTING_FILE = FINAL.parent / 'external_data' / 'external.parquet'

# Discovery uses expanding pre-year evidence and never purchases data.
DISCOVERY = True
DISCOVERY_ROUNDS = 2
CARBON_PURCHASE_BUDGET_USD = 0
