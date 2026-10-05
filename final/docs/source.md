# Data sources

Everything the pipeline reads, where it comes from, and what is stored where.
Code pointers at the bottom. Input data are supplied separately: competition
files go in `data/`, external tables in `external_data/external.parquet`.

## 1. Competition toolkit (`data/`, see [data/README.md](../data/README.md))

| File | Contents | Rows × cols | Size |
|---|---|---:|---:|
| `chars_final_with_names.parquet` | Monthly characteristics panel, 01/2015–08/2026; 147 selected characteristics + 51 identifiers/auxiliary; unique on (permno, eom) | 529,082 × 198 | 413 MB |
| `factor_char_list.csv` | The 147 baseline characteristic names (`variable` column) | 147 | 2 KB |
| `8k_20150101_20260831_identified.parquet` | Linked 8-K filings, provider-cleaned text; unique on `document_id` and (permno, filing_date, text_sha256) | 373,139 × 94 | 358 MB |

Toolkit SHA-256 identifiers (from the supplied competition data dictionary): 8-K `f333cd02…0cd07`, chars `797c01eb…215ca`.
Key timing facts: characteristics row month t pairs with `ret_exc_lead1m` (next-month
excess return, the target — never a feature); filing `filing_date` is a provider date,
`filing_timestamp_utc` a midnight placeholder, `content_report_date` the reported event
(not a release date). Tickers/names are period-supported labels, null for 14,645 rows;
join on permno + dated month key, cross-checking gvkey/iid — never on bare ticker/name.

## 2. WRDS I/B/E/S (licensed; in `external.parquet` via `external_data/build.py`)

| Table | WRDS export (`raw/` exact name) | Rows | Columns kept |
|---|---|---:|---|
| `ibes_consensus` | `s365yejuuvprhila.csv.gz` — Summary Statistics, EPS unadjusted | 4,520,889 | TICKER, STATPERS, FPI, FPEDATS, NUMEST, NUMUP, NUMDOWN, MEDEST, MEANEST, STDEV, ANNDATS_ACT, USFIRM, CURCODE |
| `ibes_price` | `lhn8sdwvp01ym0q7.csv.gz` — Summary Actuals, Prices and Shares | 2,676,578 | TICKER, STATPERS, PRICE, CURR_PRICE, USFIRM |
| `ibes_surprise` | `b2tkpunzdr2n0s5t.csv.gz` — Surprise History | 434,717 | TICKER, MEASURE, FISCALP, PYEAR, PMON, USFIRM, anndats, actual, surpmean, suescore |
| `ibes_link` | `j2vz7zigsgxzab3h.csv.gz` — CRSP–I/B/E/S link table | 37,662 | TICKER, PERMNO, sdate, edate, SCORE |

Join: TICKER → PERMNO through the dated link (`sdate` ≤ month-end ≤ `edate`, best
SCORE), then (permno, month-end) to the panel. Time rule: features use only consensus
snapshots (STATPERS) and announcements (ANNDATS) dated on or before month-end; later
announcements are training labels only. Consumers: `e7`, `conf_lm_gated`,
`gap_micro_best` (forward EPS), graph EPS-surprise target. Licence: share the parquet
and `raw/` only with WRDS-licensed people; never publish them.

## 3. FRED (`fred` table)

`raw/TB3MS.csv` + `raw/SP500.csv` (`https://fred.stlouisfed.org/graph/fredgraph.csv?id=<series>`),
3,721 rows long (`date`, `series`, `value`; TB3MS a percent rate, SP500 daily index
without dividends). Evaluation only, never a predictor: cash = TB3MS/100/12, S&P 500
monthly returns from month-end closes, benchmark = T-bill + 4%/yr (`evaluate.py`).

## 4. FINRA short interest (`finra` table)

`https://cdn.finra.org/equity/otcmarket/biweekly/shrt<YYYYMMDD>.csv`, bi-monthly
settlements since 2017, 3,949,157 rows (`symbol`, `short_shares`, `days_to_cover`,
`settlement`). Join: symbol → panel ticker forward-filled per permno (shared symbols
dropped). Time rule: use an explicit publication date where supplied; otherwise
delay settlement by ten business days, accounting for holidays. The borrow screen
(`borrow.py`, cap 15% of shares outstanding) reads the latest public figure at each
month-end. Auto-downloaded and cached per year by `build.py`.

## 5. SEC EDGAR form indexes (`edgar` table)

`https://www.sec.gov/Archives/edgar/full-index/<year>/QTR<q>/form.gz`, quarterly
indexes since 2013, 4,173,166 kept rows (`form_type`, `company_name`, `cik`,
`date_filed`, `accession`). Only the filing-events sleeve's form families are kept
(offerings, late-filing notices, comment letters, 13D/13G, proxies, tender offers,
amendments, deregistrations, spin-offs, Form D/144 and similar) plus name-learning
forms. Join: CIK → permno through the 8-K file's CIKs, then exact unique name
matches. Time rule: a filing in month m first counts for month m+1.
Auto-downloaded and cached per quarter by `build.py` (polite `research-bot`
user-agent, 0.4s sleep).

## 6. Rebuilding

```bash
python -m external_data.build   # external_data/raw/ -> external_data/external.parquet
```

Run this command from the project root. Raw file paths above are relative to
`external_data/`. Place the WRDS exports and FRED CSVs in `external_data/raw/`;
the builder downloads FINRA and EDGAR. `MAIN.py` requires the completed
`external_data/external.parquet` and does not automatically acquire it.

Table-by-table equality with the raw files was checked at build time; each table's
columns/types are stored in the parquet metadata, which `external.py` enforces on
read. `raw/` is not needed to run `main.py` once the parquet exists.

## Code pointers

- [external_data/build.py](../external_data/build.py) — raw → parquet (WRDS/FRED reads, FINRA/EDGAR fetch)
- [external_data/README.md](../external_data/README.md) — full per-table reference this file summarizes
- [external.py](../external.py) — typed reads every consumer uses
- [sleeves/ibes.py](../sleeves/ibes.py) — analyst-data loaders (as-of joins, target-quarter selection)
- [sleeves/filing_events/edgar.py](../sleeves/filing_events/edgar.py) — kept form families, CIK crosswalk
- [borrow.py](../borrow.py) — short-interest table + 15% screen
- Competition toolkit data dictionary (supplied separately) — schema, join examples and integrity hashes
