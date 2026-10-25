# 2025 one-year trial

The full-year trading mechanics work in the labelled synthetic acceptance test. Live research reads papers and produces proposals/critiques, but it is **not yet an autonomous, validated trading strategy**: the live ideas stopped at review, and no real portfolio was authorized.

[Machine-readable results](artifacts/year_trial_2025/summary.json). Detailed local diary: `../.pod_state/year_trial_2025/report.json`. All original failures, attempts, caches and amendments are preserved in that state directory.

| Check | Observed result |
|---|---|
| Study calendar | January–December 2025, 72 explicit ticks; 53 ISO weekly cycles per pod, including boundary weeks |
| Literature discovery |16/16 quarterly searches succeeded; 8 distinct paper versions newly discovered in the search cache |
| Full-text seed papers |4 version-pinned public papers retrieved |
| Alternative data |7 catalogue candidates discovered; metadata alone did not establish raw-data entitlement or eligibility |
| Real model use |39 of the authorized maximum 40 HTTP attempts, sharing one persistent ledger across all amendments |
| Saved model output |6 proposals and 5 valid critiques; some proposals repeat an idea after an infrastructure failure |
| Registered design follow-ups |4 operator-controlled comparisons; all had negative modeled net bridge spreads |
| Real trading |0 production admissions, 0 orders; no real 2025 strategy return is claimed |
| Synthetic trading acceptance |12 actual monthly decisions, 24 publication batches, shared hedge netting, fills/accounting/risk and feedback |

The 53 weekly cycles include partial boundary weeks. The 13 matured cohort cycles per pod include warm-up months before 2025; they do not represent 13 months of calendar-year performance.

The seed papers were [Investor Emotions and Earnings Announcements](https://arxiv.org/abs/2006.13934v2), [Digital transformation: A systematic review and bibliometric analysis from the corporate finance perspective](https://arxiv.org/abs/2412.19817v1), [Value matters: Predictability of Stock Index Returns](https://arxiv.org/abs/1204.5055v2) and [Is Factor Momentum More than Stock Momentum?](https://arxiv.org/abs/2009.04824v1). The events review was not a usable issuance-mechanism recipe and remained unapproved. Quarterly metadata discoveries still need full-text and method review.

## What the experiments found

The model reviewers requested controls and further evidence rather than authorizing execution. Four already-reviewed panel comparisons were therefore run **separately as operator-controlled design follow-ups**, using registration and the actual executors. They are not autonomous pod successes, literal paper replications, a fresh holdout or 2025 investment results.

| Panel comparison | Candidate belief IC | Baseline IC | Modeled monthly net spread |
|---|---:|---:|---:|
| Fundamentals, core |0.3449|0.3061|−15.01 bp|
| Valuation, core |0.0540|−0.0178|−16.61 bp|
| Dynamics, core |0.0645|0.0000|−62.89 bp|
| Dynamics, extended |0.0642|0.0000|−57.78 bp|

These are retrospective 2015–2018 design measurements, with declared flat modeled costs and proxy targets. Better belief prediction did not yield a positive net return bridge. All comparisons stay research/shadow; none opened the protected check partition or issued an admission certificate.

The supplied 2025 daily file slice contained 929,617 rows over 250 observed trading days, with 178 missing returns retained. Those values were not sent to the model. No missing observation was used to manufacture a strategy return.

## Problems exposed and fixed

- Multiple reviewed ideas can coexist in a pod; the earlier demo allowed only one immutable review per pod.
- The simulated allocation calendar and contemporary model execution clock are separate. Current models are never backdated into a 2025 eligibility claim.
- CIO authorization uses publication time; staleness still uses the underlying information date, preventing old information being refreshed by re-publication.
- The official daily CSV's compact `YYYYMMDD` dates are normalized before aggregation.
- Groq calls share a durable 40-call cap, retain failed attempts and are paced after the observed rate limit. Strict structured output uses Groq's documented [JSON-schema interface](https://console.groq.com/docs/structured-outputs); unapproved empty parameter objects remain closed and do not grant execution.
- Automatic approval initially blocked inference until outgoing payloads were audited and restricted. Only allowlisted public paper evidence, bounded method metadata and generated proposals go to Groq. Competition rows, prices, protected outcomes and paths remain local.

The original burst hit provider rate limits. One later critique also ended in a provider error and was not counted as a valid review. Later JSON transport failures were retained and corrected. No budget was reset: the final 39 attempts include those failures, and one authorized attempt remains unused.

## Remaining work

The original live trial stopped at review. A subsequent [component-research improvement](RESEARCH_PROGRESSION.md) replayed three saved proposals/critiques through bounded real-data diagnostics with zero new API calls: earnings was negative, valuation/dynamics were noisy. Partial paper ideas and original hypotheses can now be tested before building a complete strategy. These native panel diagnostics leave unmatched sentiment/CAPE/factor mechanisms untested. Reliable paper-to-feature mapping, authentic core measurements, integrated controls and source approvals remain necessary for those original strategies. The original exact-model roster was not used: the user explicitly authorized Groq for this bounded trial. Source/model/candidate admission and prospective investment validation remain separate requirements.

## Reproduce the offline year

From `final_v2/`, after installing the research/test dependencies:

```sh
python scripts/run_complete_platform.py --fixture-year 2025 \
  --state-dir /tmp/year/state --output-dir /tmp/year/output
python -m pytest -q -p no:cacheprovider
```

This is a finite offline fixture; it starts no model or background service. For live options, see `scripts/run_year_research_trial.py --help`. The existing ledger already has 39/40 attempts; a rerun is not a new authorized budget. Use `scripts/summarize_year_trial.py` to rebuild a summary without inference.

Final regression: **847 passed, 1 skipped**, with three existing deprecation warnings (80.69 seconds). New transport-focused checks also passed.
