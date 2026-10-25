# Verification — 2026-10-25

Verified against the integrated workspace before the checkpoint commit:

- Full `final_v2/tests` offline suite: **653 passed, 1 skipped**, 20.10 seconds.
  The skipped test needs loopback sockets unavailable in this sandbox.
- Parquet loader/CSV parity exercised using the cached compatible pyarrow wheel;
  the full built-in executor suite passed all 53 tests.
- All 170 Python files under `final_v2/` parsed successfully.
- Eight CLI `--help` checks passed: paper research, pod research, paper watches,
  controlled replay, strategy campaign, reference workflow, daily reporting and Arcana export.
- Fresh integrated reference demo returned `software_status=PASS` and
  `judge_status=COMPLETE_SOFTWARE_FIXTURE`; investment validation remained `UNPROVEN`
  and prospective evaluation `INCOMPLETE`.
- Full paired books, admission/environment gates, persistent budgets, namespace isolation,
  tamper detection, normal resume and interrupted/hard-exit recovery are covered in tests.
- Staged whitespace check passed. Common credential patterns were checked in changed
  text files; local credentials and large source data are excluded from Git.

Real-input checks (saved under `artifacts/verification_20261025/`):

- Official daily source: 12,175,448 rows, 6,827 identifiers, zero duplicate identifier/date
  keys; dates 2014-01-02 through 2026-09-28. Returns retain supplied missing/outlier values.
- Four-charter design campaign: 12 variants, zero variant execution failures, all four
  candidates rejected. No strategy admitted or deployed and no protected challenge consumed.
- Supplied daily January 2021 reporting smoke: 19 trading days, zero missing selected
  returns, independently reconciled end NAV `1000102.449274516682995535700`.
  This uses an explicit small holding/fill and ex-post monthly TB3MS conversion; it is
  accounting verification, not a strategy result or forecast input.
- Arcana OAuth CLI returned `Successfully logged in` for `https://app.arcana.io/mcp`.
  Current session did not dynamically expose the new tools; live analytics/schema/upload
  remains unverified and no portfolio was uploaded.

Earlier live checks in the same build:

- arXiv and OpenAlex each returned two papers.
- Tool-less Claude structured JSON and one valid pod draft succeeded. An oversized
  response was rejected and retained as a failure.
- Four-pod offline demo completed four mock jobs; mock outputs were never model skill.

Reproduce after installing `.[research,test]`, from `final_v2/`:

```sh
python -m pytest -q -p no:cacheprovider
```

This laptop used `local/carbonarc-venv/bin/python` with locally cached pytest and pyarrow,
through temporary wrappers under `/tmp/fiam26-complete/`. No heavy training was started.
