# Competition and saved beta inputs

Source inputs are provided separately and excluded from Git. The final runtime
reads these four files from this directory by default:

| File | Contents |
| :--- | :--- |
| `chars_final_with_names.parquet` | Monthly company-characteristics panel, January 2015-August 2026 |
| `factor_char_list.csv` | The 147 baseline characteristics |
| `8k_20150101_20260831_identified.parquet` | Linked 8-K filings |
| `dsf_mini_from_2014.csv` | Daily stock returns used for risk estimation and accounting |

The final hybrid construction also requires the separately supplied export
`arcana/arcana_spy_beta_previous_holdings.csv`. Its original local copy was
preserved when the upload/review files were archived.

Use `python final_v3/main.py` from the repository root. Override locations with
`--data-dir`, `--arcana-beta-file`, `--external-file`, and `--reporting-file`.
See [the full data guide](../docs/data.md) for licensed and reporting inputs,
availability limits, and alternative experiments.

The historical Carbon extracts remain local in `data/carbon/`; they are not
included in the public snapshot and are not predictive assets in the final book.
