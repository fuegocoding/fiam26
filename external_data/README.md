# External data

This is the **historical multi-table container** specification. The final v3
strategy uses the disclosed I/B/E/S revision input for `conf_lm_gated` and FRED
for reporting; other former external inputs use the documented competition
proxies. See [the current data guide](../docs/data.md) and
[the final input map](../final_v3/docs/OFFICIAL_DATA_MAP.md).
References to the earlier standalone pipeline below now correspond to
`archive/initial_strategy/`. Source exports remain local and excluded from Git.

Every input that does not come from the competition toolkit is in one file,
`external.parquet` (zstd, 150 MB). Its tables are stacked, and the `table`
column says which one a row belongs to. The pipeline reads it only through
`external.read(table)` (`external.py`), which returns each table with
the columns and types of the original source.

| `table` | Source | Rows | Used by |
|---|---|---|---|
| `ibes_consensus` | WRDS I/B/E/S Summary Statistics, unadjusted | 4.5M | analyst sleeves (`e7`, `conf_lm_gated`, `gap_micro_best`) |
| `ibes_price` | WRDS I/B/E/S Summary Actuals, Prices and Shares | 2.7M | analyst sleeves |
| `ibes_surprise` | WRDS I/B/E/S Surprise History | 0.4M | analyst sleeves, graph EPS-surprise target |
| `ibes_link` | WRDS CRSP–I/B/E/S link table | 38K | ticker → permno |
| `fred` | FRED TB3MS and SP500 | 3.7K | cash return, benchmark, S&P 500 (`evaluate.py`) |
| `finra` | FINRA bi-monthly short interest, since 2017 | 3.9M | borrow screen (`borrow.py`) |
| `edgar` | SEC EDGAR quarterly form indexes, since 2013 | 4.2M | filing-events sleeve |

## Rebuilding

```bash
python -m external_data.build
```

`build.py` reads the original files in `raw/` and writes `external.parquet`.
Run this command from the project root. Place the raw source files in
`external_data/raw/` before rebuilding. Paths in the table below are relative
to `external_data/`.
- **Original files:** the four WRDS exports (`*.csv.gz`) and FRED's
  `TB3MS.csv` and `SP500.csv`.
- **Downloads:** FINRA and EDGAR are fetched from their public URLs and cached
  in `raw/downloads/`. Finished years and quarters are not fetched again.
- **Optional:** `raw/` is not needed to run `main.py`.

| Raw file | Origin |
|---|---|
| `raw/s365yejuuvprhila.csv.gz` | WRDS export, I/B/E/S Summary Statistics (EPS, unadjusted) |
| `raw/lhn8sdwvp01ym0q7.csv.gz` | WRDS export, I/B/E/S Summary Actuals, Prices and Shares |
| `raw/b2tkpunzdr2n0s5t.csv.gz` | WRDS export, I/B/E/S Surprise History |
| `raw/j2vz7zigsgxzab3h.csv.gz` | WRDS export, CRSP–I/B/E/S link table |
| `raw/TB3MS.csv`, `raw/SP500.csv` | `https://fred.stlouisfed.org/graph/fredgraph.csv?id=<series>` |
| `raw/downloads/finra/` | `https://cdn.finra.org/equity/otcmarket/biweekly/shrt<YYYYMMDD>.csv` |
| `raw/downloads/edgar/` | `https://www.sec.gov/Archives/edgar/full-index/<year>/QTR<q>/form.gz` |

Only the columns the pipeline reads are kept. Reading the parquet gives
exactly the frames the raw files give; this was checked table by table when
the file was built.

## How each table is used, and its time rule

The pipeline uses only information dated on or before each formation
month-end.

- **I/B/E/S:**
  - *Join:* TICKER → PERMNO through the dated link (`sdate` ≤ month-end ≤
    `edate`, best `SCORE`), then (permno, month-end) to the panel.
  - *Time rule:* a feature uses only consensus snapshots (STATPERS) and
    announcements (ANNDATS) dated on or before the month-end. Later
    announcements are used only as training labels.
  - *Licence:* I/B/E/S is licensed through WRDS. Do not publish
    `external.parquet` or `raw/`; share them only with people covered by a WRDS
    licence.
- **FRED:**
  - *Format:* long, with `date`, `series` (TB3MS, a percent rate; SP500, the
    daily index level without dividends) and `value`.
  - *Use:* evaluation only, never a predictor. The cash return is TB3MS / 100 /
    12, and S&P 500 monthly returns come from month-end closes.
- **FINRA:**
  - *Columns:* `symbol`, `short_shares`, `days_to_cover`, `settlement`.
  - *Join:* symbol → the panel's ticker, forward-filled per permno; a symbol
    shared by two permnos in a month is dropped.
  - *Time rule:* an explicit publication date is used where supplied; otherwise
    settlement is delayed ten business days, accounting for holidays. The
    latest public figure at each month-end is used.
- **EDGAR:**
  - *Columns:* `form_type`, `company_name`, `cik`, `date_filed`, `accession`.
    Only the form types the filing-events sleeve counts (offerings,
    late-filing notices, comment letters, 13D/13G, proxies, tender offers and
    similar) are kept, plus those used to learn company names.
  - *Join:* CIK → permno through the CIKs in the supplied 8-K file, then exact
    unique company-name matches.
  - *Time rule:* a filing in month m first counts for the return of month m+1.

## Not stored here

- **Competition files** are not included; see `../data/README.md`.
