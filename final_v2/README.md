# Strategy research campaign

Install the runtime with `pip install -e './final_v2[research]'`, then run
`python final_v2/run.py` from the repository root.

`run.py` is the only entrypoint. It reads [config.json](config.json) and runs the
research campaign from evidence discovery and dated inputs through strategy
proposals, measured feedback, pod ownership, capital allocation, observed-price
fills, daily NAV, attribution, and reporting.

The separate [competition configuration](competition_config.json) freezes
pre-2021 agent recipes and evaluates January 2021–August 2026 through the same
entrypoint, selected with `FIAM_CONFIG`. Part 2 sector limits, the 12% realised
volatility acceptance ceiling, local exhibits and submission status are described
in [the compliance workflow](docs/PART2_COMPLIANCE.md). Arcana is currently
deferred; research completion does not imply submission readiness.

Configuration has five sections: `application`, `models`, `carbon`, `training_lab`
(the shared market and research settings), and `research_campaign`. Supplied data
and credentials remain in the repository's existing data and local directories.
The runtime does not import an earlier fitted portfolio.
The research pool starts empty. Only evaluated agent proposals can become
strategies; there are no built-in controls, seeded strategies, default feature
models, or trading fallbacks.

Research requests up to three distinct hypotheses per pod at a time, repairs
invalid proposals with their exact failures, and feeds measured returns, IC,
uncertainty, attribution and training feature diagnostics into further rounds.
The search targets at least five supported, distinct candidates per pod; it
selects up to three for ownership using the same evidence criteria as funding.
Renamed duplicates and valid but unsupported code do not satisfy the pool target.
Each pod first plans several economic topics. The host explores each before
tuning, compares refinements with their parents on common chronological windows,
and continues branches that improve measured evidence. There is no campaign
round ceiling. Exhausted branches and unresolved provider failures are recorded
separately; neither the candidate count nor a time/call allowance forces acceptance.

Every new strategy declares its target cap band, GICS sectors/industries and
optional observable filters, with an economic rationale. Fitting, diagnostics
and trading use that same formation-time cohort. Each strategy chooses its
recent model/evidence lookback and chronological evaluation window. Long-only
proposals remain long-only. Adaptive training windows are robustness checks;
final validation and test outcomes do not enter model feedback.
Fitted forecasts may use agent-specified ridge or XGBoost models. Both refit
monthly on settled labels within the declared cohort and lookback. XGBoost
requires explicit features, regularization, tree parameters and a rank/return
target; no fitted model is selected automatically.
Scope and history declarations are mandatory for every proposal. A failed tweak
returns for diagnosis and another defensible test. Empty cohorts report aggregate
raw-feature ranges so that source-unit mistakes can be corrected explicitly.

An interrupted campaign can restore its topic frontier at the same data cutoff
after its recorded agent programs are restored. Later monthly research can
continue from a completed initial plan using a recorded per-pod research decision
and concrete agent-proposed topics. Continuations retain earlier measured parents
when revisiting a topic. New topics receive their own numerical evaluation.

All model instructions live in [prompts.txt](prompts.txt). Callers read their named
`@@ section` and supply any `{{field}}` values. Prompts are included in run hashes
and saved source snapshots.

Orders fill immediately in full at the first supplied observed quote on or after
the decision date. Performance uses observed returns and posted cash interest.
Portfolio limits, source timing, missing-data
handling, and accounting reconciliation are enforced in the normal workflow.
At each observed close, gross, net and name-risk drift can reduce tradable
positions proportionally. Missing quotes are never replaced by invented fills;
any unresolved limit failure remains in the daily audit. Sleeve accounting
attributes these observed-price risk trims to the positions being reduced.
A successful run is research evidence, not a claim of admitted or investable alpha.

An explicit `research_campaign.evaluation_period` separates the portfolio return
window from the earlier observations needed for model fitting. Portfolio reviews
then use each month's first available session. `portfolio_evidence_scope` defaults
to `prior_settled`; the optional `retrospective_evaluation_period` uses the complete
declared window for selection and funding and marks the report as in-sample.
The disclosed 2021–2022 comparison and its recorded-response reproduction
configuration are in [the saved results](../results/2021_2022/RESULTS.md).

Reports and regenerable caches are written under the configured `.pod_state`
directory. API spending quotas and spending ledgers are not used. Provider rate
limits, bounded retries, and request/response size limits remain transport rules.

Development checks, tests, experiments, and diagnostic scripts belong in `/tmp`.
The repository keeps the production workflow; see [AGENTS.md](AGENTS.md) for this
working rule. Campaign details are in [docs/RESEARCH_CAMPAIGN.md](docs/RESEARCH_CAMPAIGN.md).
