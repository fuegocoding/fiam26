# Final formation-beta bound and CPU replay

## Current construction: causal daily and monthly S&P 500 regressions

The active implementation supersedes the earlier panel-beta band described
below. It fits stock excess returns against the S&P 500 benchmark used by
reporting using only observations available before each holding month. It
enforces both the trailing-252-session daily estimate and trailing-36-month
monthly estimate within +/-0.05 in deployment and turnover optimization. The
minimum sample sizes are 126 daily pairs and 12 monthly pairs. Short-history
stocks use formation-sector, then formation-cross-sectional medians of
observed regressions; these priors are disclosed per holding and are never
represented as observed stock betas. Risk-free rates in risk fitting are
lagged two months for availability; ex-post reporting uses the mandated rates.

The replay destination is `sp500_beta_005_output/`. It saves all estimation
settings in `beta_estimation.json`, every formation's coefficients and sources
in `beta_estimates.parquet`, and both final portfolio exposures in
`optimizer_diagnostics.csv`. The bounds concern estimates available at
formation; realized full-period beta remains an out-of-sample measurement,
not a quantity that can be forced using future returns. No training or
Arcana upload is required to rebuild this construction.

The completed replay independently passed both estimated-beta bands across
all 68 months. Fifteen daily and fifteen monthly estimates were checked with
separate least-squares regressions, with maximum discrepancies below 2e-15.
All 13,600 holdings and the original forecast values were verified, and each
calendar year's counted one-way turnover remained below 400%. The final
sizing LP uses 1e-9 feasibility tolerances to retain tiny protected weights;
no capacity or membership bound was relaxed. Evidence is in
`sp500_beta_005_output/sp500_beta_validation.json`. The user-upload handoff
is `../../arcana_upload/sp500_beta_005/`; it remains a draft because six original
Coherent positions require a separate historical Arcana record.

## Preserved earlier panel-beta experiment

The user-requested +/-0.05 bound now applies to the final constructed portfolio, including the turnover-constrained solution. It uses unclipped formation-date panel `beta_60m`; missing estimates retain the disclosed beta-1 assumption and are counted in diagnostics. Initial beta sizing still uses its original clipped hedge heuristic, but that heuristic can no longer override the final band. Liquidity, gross/net, sector, short protections, name counts and annual turnover constraints are retained. An infeasible combination fails explicitly.

`neutrality.py` supplies shared beta coefficients and final validation. `deployment.py` adds beta bounds to its bounded L1 sizing projection. `turnover.py` adds the same bound to the mixed-integer selection/sizing problem and LP polishing step. `optimizer.py` validates final weights again and exports the limit and beta source. `report.py` independently checks the stored positions when a beta limit is recorded in the run's settings.

The completed 68-month CPU rerun is in `beta_neutral_005_output/`; the unchanged archived-allocation baseline under the same local reporting inputs is in `beta_neutral_005_baseline/`. Existing archives and original output were preserved. Forecast values match the checkpoint exactly. No training, signal fitting or graph preparation was needed. All 68 months pass independent formation-beta recomputation and calendar-year turnover verification. See `beta_neutral_005_output/beta_validation.json`.

The current machine lacks the original I/B/E/S training container. Explicit `--replay-with-local-reporting` supports frozen replay without reopening that training-only dependency: all archived payloads and four competition inputs still require matching hashes, while the actual local reporting file is separately recorded. This mode does not claim that the original external training/reporting container was reverified. Strict original-input replay remains the default. Replay imports training modules lazily and can run with CPU reporting/optimization dependencies only.

The new construction uses historical formation beta, not Arcana's separate predicted-beta estimator. Do not claim that this guarantees realized or Arcana beta neutrality. The user requested uploading before Arcana portfolio analysis; evaluation of the changed book in Arcana is deferred until that upload. The manual handoff at `../../arcana_upload/beta_neutral_005/` is a draft because two distinct historical Coherent securities require separate mapping. Never combine or drop them to complete an import.

Targeted verification exercised feasible beta redistribution in both solvers, preservation of leg budgets and membership, and rejection of infeasible beta books. Independent full-history verification checked formation-date betas against the original panel, names, weights, gross/net, turnover, diagnostic reconciliation, forecast-value preservation and return compounding. Existing daily held-return completeness and daily/monthly reconciliation failures remain; this change does not certify submission readiness.
