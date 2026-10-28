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
selects up to three for ownership using predictive evidence and overlap checks.
Funding separately requires supported calibrated effects in the proposed targets.
Renamed duplicates and valid but unsupported code do not satisfy the pool target.
Each pod first plans several economic topics. The host explores each before
tuning, compares refinements with their parents on common chronological windows,
and continues branches that improve measured evidence. There is no campaign
round ceiling. Exhausted branches and unresolved provider failures are recorded
separately; neither the candidate count nor a time/call allowance forces acceptance.
An already-tested branch that stalls gets a separate evidence review. Further
proposals need a distinct hypothesis and resolving test; otherwise the reviewer
retains the existing result with its reasons. This adds no return gate or search quota.

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

Candidate ranking and refinement use `S = 0.40P + 0.25C + 0.20R + 0.15B`.
`P` clips the monthly rank-IC t-statistic divided by three to [0,1]; `C` is
the fraction of positive-IC months; `R` clips `0.5 + recent_mean_IC / 0.10`
to [0,1] using the latest twelve calendar months of settled evidence. `B`
is the fraction of performed, predefined robustness checks with positive
mean IC: shift the outcome start one session, shorten/lengthen the declared
horizon by 20%, and trim the outer 5% of active scores at each end. A missing
check cannot pass; no performed checks gives zero robustness credit and a flag.
The IC t-statistic uses a Newey-West error accounting for overlapping horizons.
All measurements stay within the declared dated cohort and prediction horizon;
weekly/daily reviews are grouped into complete settled formation months. The
configured minimum valid-month rule applies, without an additional t-statistic
or return threshold for ranking. Scores are heuristics, not calibrated success
probabilities. Funding uses settled peer-relative score-bin effects, shrunk for
uncertainty and applied to the actual target weights. Historical sleeve returns
estimate covariance rather than veto hypotheses. Long and short alpha are
validated separately; zero-score passive hedges receive no invented alpha.
Reports record score components, effect coverage and funding reasons.
`maximum_owned_strategies_per_pod` caps ownership at one, two, or three candidates
(default three); a pod can remain empty.

`training_lab.tree_device` accepts `auto` (default), `cpu` or `cuda`. Automatic
selection benchmarks CPU and CUDA on settled training rows and uses CUDA when
it is at least 10% faster. Device choice and timings are recorded with each
tree model. CUDA requires the process to have access to the NVIDIA device;
automatic mode uses CPU when the device or CUDA build is unavailable.

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

Fresh proposals must declare `forecast_horizon_days` and a `prediction_contract`
with observable event inputs, expected direction, peer comparator and a falsifiable
mechanism. Model labels, forecast expiry and predictive evaluation use that
horizon. Partial horizons never become training labels. Fresh signals replace
older signals; forecasts expire at an observed close when their horizon ends.

The default `robustness_suite=expanded` augments the four timing/score checks with
issuer subgroups, disjoint forward blocks, deterministic shuffled-score controls
and genuine issuer-held-out forecasts for fitted models (`issuer_validation_folds=2`).
Held-out issuers are excluded from every historical training label and label
transformation. Subgroup checks alone are not called unseen-issuer validation.
The score weights remain unchanged. `basic` and `issuer_validation_folds=0` support
controlled comparisons with earlier evidence. `align_forecast_horizons=false` and
`funding_method=legacy_returns` reproduce the former horizon and funding choices.
Optional `allocation_comparisons` replays the other funding method on identical
forecasts and targets within `run.py`; results disclose this controlled scope.

Cached forecasts keep their original vintage: native evaluation runs from the
review close to that vintage's expiry, rather than renewing the horizon at each
review. Calibration retains raw remaining-window effects and normalizes them
to per-session units before pooling. Annual expected effects therefore assume
constant peer-relative return per session; reports make that assumption explicit.
Actual executions and reported realized returns use observed prices unchanged.
At capital reviews, candidates must have supported conditional effects before
filling the one-to-three funded roster slots; ranking within that set still uses S.

Fresh fitted contracts train on peer-relative outcomes as declared. The trainer
computes leave-one-out peer benchmarks after excluding validation issuers;
held-out returns cannot enter other training labels. Issuer folds use a fixed
identity hash, so adding future issuers cannot reassign earlier folds.

A constant binary trigger can have an event-versus-peer effect while its
within-trigger Rank IC is undefined. Calibration still reports that effect,
but the specified S formula cannot rank an entirely constant forecast. Current
research therefore declares graded prediction strengths; replacing IC with an
event-effect statistic would be a separate selector change.

Interrupted campaigns may set `evaluated_trial_sources` to previous campaign
directories on the identical data prefix. The normal entrypoint verifies input
hashes, evaluator code, issuer ordering, consumed feature values and actual model
request/response provenance before restoring evaluated trials. Import receipts
record every artifact hash. Changed or invalid per-strategy artifacts require
fresh evaluation; global prefix/policy mismatches stop the import. This reuses
evaluated agent proposals and never supplies a default strategy.
