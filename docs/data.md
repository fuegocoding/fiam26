# Data and input requirements

The public snapshot contains code, the team's presentation and selected
aggregate results. Raw competition inputs, licensed exports, vendor responses,
trained models and prediction checkpoints are not distributed with it.

## Inputs for the final baseline

| Default location | Purpose | Origin |
| :--- | :--- | :--- |
| `data/chars_final_with_names.parquet` | Monthly formation characteristics and labels | Competition toolkit |
| `data/factor_char_list.csv` | 147 baseline feature names | Competition toolkit |
| `data/8k_20150101_20260831_identified.parquet` | Linked 8-K corpus and event/identity data | Competition toolkit |
| `data/dsf_mini_from_2014.csv` | Daily stock returns, local beta estimation and accounting | Supplied daily-return file |
| `external_data/external.parquet` | Disclosed I/B/E/S revision input | Separately licensed container |
| `external_data/fred_reporting.parquet` | T-bill hurdle and S&P 500 reporting series | Separately prepared FRED input |
| `data/arcana/arcana_spy_beta_previous_holdings.csv` | Saved security beta observations for hybrid sizing | Supplied Arcana export |

When `fred_reporting.parquet` is absent, configuration falls back to
`external_data/external.parquet` for reporting. That container must then
include the required FRED table. A command can supply a different reporting
file with `--reporting-file`.

The monthly characteristics and filings start in 2015; the out-of-sample
holding window is January 2021-August 2026. Daily observations start before
the test window so trailing risk estimates can be formed.

The [technical input map](../final_v3/docs/OFFICIAL_DATA_MAP.md) explains
the replacements for historical external inputs. In final v3, the I/B/E/S
exception is `conf_lm_gated`; other expectation consumers use the documented
competition-data proxies. Compatibility column names do not imply that
licensed analyst observations were reconstructed.

## Point-in-time rules and remaining questions

Formation information predicts the next holding month. Filing-driven events
use an explicit availability clock, with next-trading-day handling. Training
labels are separated from formation features, and checkpoint/input hashes
record the exact saved payloads.

The final deck still flags unverified historical vintages, I/B/E/S availability
bounds/coverage, 8-K survivorship, universe bias and source publication lags.
The historical availability and identity mapping of exported Arcana betas
also require verification. These limitations remain even when a local file's
hash matches an archived input.

## Other local data

Historical Carbon Arc extracts under `data/carbon/` supported exploration;
the final deck reports no Carbon predictive asset in the final book. Their
vendor payloads, manifests and acquisition responses remain local. The
[legacy container guide](../external_data/README.md) documents the earlier
I/B/E/S, FINRA, EDGAR and FRED tables and their build process.

During cleanup, full replay directories/checkpoints were preserved under
`local/artifacts/final_v3/` and earlier upload packages under
`local/archive/arcana_upload/`. Personal CVs and feedback/reference PDFs are
under `local/private/`. All of `local/` is ignored and excluded from the
public export. Credentials were preserved in their existing ignored locations.

## Alternate experiments

`--official-only` explicitly drops the licensed revision sleeve;
`--no-arcana-beta` removes the saved Arcana-beta input and uses local
estimators. These are useful alternatives but do not reproduce the published
hybrid backtest. Missing required inputs are not silently replaced.
