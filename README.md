# FIAM 2026 strategy pod platform

The governing design is [POD_FRAMEWORK_IMPLEMENTATION_SPEC.md](final_v2/POD_FRAMEWORK_IMPLEMENTATION_SPEC.md).
The active platform is in [final_v2/](final_v2/README.md): four private strategy pods,
paper and alternative-data research, registered experiments, independent admission,
CIO coordination, risk, execution, accounting and reporting.

## Install and verify

```sh
python -m venv .venv
. .venv/bin/activate
pip install -e './final_v2[research,test]'
python -m pytest final_v2/tests -q -p no:cacheprovider
```

## Run

Edit `final_v2/config.json`, then start the configured workflow:

```sh
python final_v2/run.py
```

Use `application.mode=demo` for synthetic acceptance, or `research` for configured
multi-paper hypotheses, Carbon requests, native graph evaluation and eligible
portfolio stages. All runtime options live in `final_v2/config.json`.
Real source eligibility, measured candidate evidence and independent admission are
required before a real strategy can enter a book. Test certificates and replay
results do not establish investment performance.

Raw inputs, credentials and detailed private state stay under `data/`, `external_data/`,
`local/` and `final_v2/.pod_state/`; they are not supplied by a fresh checkout. Competition
inputs include `chars_final_with_names.parquet`, `factor_char_list.csv`, the linked
8-K corpus and `dsf_mini_from_2014.csv`. Keep the existing local input symlinks intact.

`MAIN.py` and `main.py` remain the original submission entrypoints. Their local
requirements and inputs are separate from the platform package.
