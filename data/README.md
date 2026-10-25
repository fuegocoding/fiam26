# Competition data (not included)

Put the three files of the McGill-FIAM 2026 toolkit here, with these exact
names, before running `MAIN.py` from the project root:

| File | Contents |
|---|---|
| `chars_final_with_names.parquet` | Monthly characteristics panel, 01/2015–08/2026 |
| `factor_char_list.csv` | The 147 baseline characteristics |
| `8k_20150101_20260831_identified.parquet` | Linked 8-K filings |

To use another folder, set `DATA_DIR` in `config.py`.
Place the separately supplied `external.parquet` in `external_data/` at the
project root. To override input locations for `MAIN.py`, use `--data-dir` and
`--external-dir`.
