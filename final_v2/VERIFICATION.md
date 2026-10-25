# Verification — 2026-10-25

The final integrated source-to-book workspace passed **836 tests**, with one sandbox-dependent socket check skipped and three existing NumPy/pandas deprecation warnings (54.21 seconds). This includes the final live-clock restart/default-publication assertion.

- Actual four-pod paper/evidence -> structured extractor/lead/challenger -> reviewed adaptation -> registration before data access -> executor -> frozen independent check -> scoped publication -> coordinated CIO -> risk -> partial fills -> NAV/dossier -> feedback review is exercised in `test_autonomous_pods.py` and `test_complete_platform.py`.
- Positive fixtures measure two admissions. Valuation stays shadow after a timing-placebo failure; events fails design evidence sufficiency. No fixture certificate authorizes a competition book.
- CIO tests verify covariance-sensitive actual holdings, economic duplicates, source-only overlap diagnostics, three shared-hedge requests netted once, current holdings/pending orders, partial-fill reconciliation and strict actual post-fee risk. Fee/executable-book headroom does not weaken limits.
- Failure checks cover producer-bound evidence, tampering, private namespaces/protected partitions, durable trial and model budgets, failed-call charging, source/profile changes, incompatible horizons, callback retry and a crash between sealed challenge and publication without a second check.
- Publication loads current feature columns without labels; native target resolution and monthly return bridge units/horizons remain distinct.
- Daily accounting checks cover cash and short proceeds, fee-dollar allocation, drift, missing held returns, total losses, decay coverage and four-trading-day event windows. Existing supplied-data January 2021 smoke reconciled 19 days to independent end NAV `1000102.449274516682995535700`; this is accounting verification, not strategy evidence.
- 211 Python files parsed; eight new CLI help paths passed. Actual stdio MCP initialize/tools-list/catalogue-call passed with 31 tools; an injected rival namespace was denied. No daemon/model/network call is needed for default startup.
- Fresh `run_complete_platform.py --mode demo` completed two accounted periods, four validated publication batches, one shared hedge and two feedback reviews. Investment validation remains unproven; prospective status is not prospective.
- Fresh `run_complete_platform.py --mode research` ran the official manifest: all four real candidates rejected, zero admissions/publications/orders. The current campaign measured 12 variants. Gates were retained.

Compact reproducible receipts: [integrated acceptance](artifacts/verification_20261025/integrated-acceptance.json), [official research acceptance](artifacts/verification_20261025/official-research-acceptance.json), [real design summary](artifacts/verification_20261025/latest-real-design-summary.json), [CLI/MCP smoke](artifacts/verification_20261025/cli-mcp-smoke.json). Raw data, private authority keys, credentials and detailed runtime artifacts remain ignored under `data/`, `local/` and `.pod_state/`.

Actual model smoke evidence: Groq gpt-oss passed one chronology task in 0.930 seconds (335 input/562 completion tokens). Local Qwen8 initially failed its output contract twice, then passed with a typed schema and explicit public choices; observed inference GPU memory was 5,560,992,384 bytes on the working RTX3070Ti. Failed attempts remain recorded. This is not the complete benchmark or adoption evidence. The original thinking lead endpoint remains unconfigured. Earlier arXiv/OpenAlex and tool-less Claude draft checks succeeded. Carbon Arc read-only catalogue discovery returned actual card-spending/foot-traffic metadata; it does not establish raw-data entitlement or historical vintage.

Arcana OAuth authenticated and a fresh client discovered read-only tools. No vendor portfolio-upload schema/tool was exposed in that inspection; no live upload or vendor-compatible export is claimed.

Reproduce from `final_v2/` after `pip install -e '.[research,test]'`:

```sh
python -m pytest -q -p no:cacheprovider
python scripts/run_complete_platform.py --state-dir /tmp/complete/state --output-dir /tmp/complete/out
```

This laptop uses `local/carbonarc-venv/bin/python` with locally cached pytest/pyarrow via a temporary wrapper. No GPU training or heavy market backtest was started. See [AUTONOMOUS_PLATFORM.md](AUTONOMOUS_PLATFORM.md) for configuration and the explicit remaining operational/evidence gates.
