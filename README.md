# McGill FIAM Asset Management Hackathon 2026

## Strategy memo

This repository contains the starter data and reference scripts for the 2026 McGill FIAM Asset Management Hackathon. The recommendation below is a strategy design, not a reported backtest result. No performance numbers are invented here; they must be produced by the final reproducible pipeline.

## Recommended strategy: Evidence-Weighted Alpha

Build a market-neutral U.S. equity strategy whose alpha comes primarily from disciplined cross-sectional modeling of the 147 supplied characteristics, with 8-K disclosures used as a compact, point-in-time event overlay.

The standout feature is an **evidence-aware agent council**. Agents do not choose trades or write the final portfolio. They independently turn the same eligible filing evidence into structured features, challenge one another's interpretation, and expose disagreement as a measurable uncertainty signal. The statistical model and portfolio optimizer remain the decision makers.

The central principle is:

> Alpha comes from structured characteristics plus disclosure events; agents make research auditable and challenge weak trades, but out-of-sample data—not prose—determines success.

## Challenge framing

The released challenge brief defines a monthly U.S. equity panel from January 2015 through August 2026, with January 2021 through August 2026 reserved for out-of-sample evaluation. The portfolio must be market-neutral and rebalance monthly.

The design should respect these requirements from the local brief:

- 100–500 positions across both long and short legs.
- Gross exposure no greater than 200%.
- Net exposure between -50% and +50%.
- Benchmark of 3-month Treasury bills plus 4% annualized.
- Evaluation emphasis on information ratio, alpha, near-zero beta, drawdown, turnover, liquidity, and short-book quality.
- Deliverables: an eight-page maximum deck plus appendix, monthly holdings, monthly/daily returns, and one `MAIN.py` entry point.

The public website currently describes a different universe size (“50–70 global stocks”), while the dated released PDF in this repository specifies the U.S. equity panel and 100–500 positions. This memo follows the released PDF; the final submission should explicitly confirm the applicable rule set with the organizers if the discrepancy remains.

## Model architecture

### 1. Point-in-time feature layer

- Rank-normalize characteristics cross-sectionally by month.
- Median-impute within the training process and retain missingness indicators where useful.
- Add winsorization and robust scaling only inside each training fold.
- Aggregate eligible 8-Ks to one stock-month record before joining to returns or characteristics.
- Use event features such as filing category, novelty, recency, text length, TF-IDF or embeddings, and deterministic keyword/section flags.

### 2. Predictive models

Use a regularized linear model as the transparent baseline: Ridge or Elastic Net on the factor panel and event features. Add one nonlinear challenger, such as gradient-boosted trees, only if expanding-window validation shows stable incremental value.

The final score should combine expected excess return with uncertainty and implementation costs:

```text
alpha_score = predicted_return
              × confidence_adjustment
              × (1 - contradiction_penalty)
              - turnover_cost
              - liquidity_and_short_risk_penalty
```

The exact coefficients must be learned or fixed using training/validation data only. Do not hand-tune them on the final out-of-sample period.

## Agent council: useful, but bounded

An agent council is worthwhile if it is treated as a feature-generation and audit layer rather than an ensemble of autonomous portfolio managers. Multiple LLMs do not automatically provide independent evidence: they can share the same blind spots and produce false consensus.

Give every agent the same point-in-time evidence packet and require strict structured output:

```text
direction             [-1, 1]
materiality           [0, 1]
novelty               [0, 1]
expectedness          [0, 1]
confidence            [0, 1]
contradiction_score   [0, 1]
evidence_quality      [0, 1]
abstain               {0, 1}
evidence_count        integer
```

Suggested roles:

1. **Event classifier** — identifies filing type, topic, and likely economic direction.
2. **Materiality analyst** — estimates whether the event can plausibly affect future cash flows, risk, or financing conditions.
3. **Contrarian reviewer** — searches for alternative explanations, stale information, and reasons not to trade.
4. **Numeric/audit checker** — validates dates, identifiers, units, and whether the claim is actually supported by the filing.

Aggregate deterministically with medians or trimmed means, preserve disagreement, and allow abstention. Learn agent weights only on validation data. Recommended council features are median direction, median confidence, dispersion, contradiction rate, abstention rate, and evidence count. High disagreement should reduce position size or cause the event signal to be ignored.

The agent council should be evaluated by ablation, not by how persuasive its explanations sound:

- characteristics only;
- characteristics plus deterministic filing features;
- characteristics plus one agent;
- characteristics plus the full council.

Keep the council only if it improves out-of-sample risk-adjusted returns, drawdown, turnover, or robustness after accounting for inference cost and latency.

## Portfolio construction and risk

Each month, rank the final scores and form roughly 100–150 longs and 100–150 shorts, subject to the official universe and liquidity filters. Use an optimizer or constrained weighting rule to target:

- dollar neutrality and beta near zero;
- sector and industry exposure limits;
- position and issuer concentration caps;
- gross and net exposure limits;
- turnover penalties and minimum trade thresholds;
- liquidity-aware sizing and explicit short availability/borrow-risk penalties.

Do not let a high-confidence event signal override a hard risk constraint. The portfolio layer should be deterministic, logged, and reproducible from the submitted `MAIN.py`.

## Validation and leakage controls

Use expanding or rolling time-series splits. A practical initial schedule is training on 2015–2018, validation on 2019–2020, and untouched out-of-sample evaluation from 2021 through August 2026, with annual refits after the model is frozen.

Critical controls:

- `ret_exc_lead1m` is the future target and must never enter features.
- A row in month `t` predicts the next month's return; do not shift the target a second time.
- A January 2021 forecast must use information available by December 2020 under the chosen cutoff policy.
- Filing timestamps are date/midnight placeholders; do not treat them as precise publication times.
- `content_report_date` is not a public-availability timestamp.
- Same-month joins can leak when a filing arrives after the characteristic snapshot.
- Deduplicate and aggregate filings before joining, or repeated filings will duplicate observations and portfolio weights.
- Freeze prompts, model versions, feature definitions, and hyperparameters before the final out-of-sample run.

Report active-return information ratio against `TB3MS / 100 / 12 + 0.04 / 12`, plus Sharpe, alpha and beta versus the S&P 500, maximum drawdown, turnover, concentration, liquidity, and short-book diagnostics. Reconcile the metric definitions with the challenge brief; the legacy script's printed “Information Ratio” is not automatically the competition metric.

## Execution plan

1. Build and test the point-in-time monthly panel and leakage checks.
2. Establish the regularized linear factor baseline and the constrained long-short portfolio.
3. Add deterministic 8-K features and test whether they add stable incremental signal.
4. Add the bounded council, cache its structured outputs, and run the ablation matrix.
5. Freeze the research specification, run the untouched out-of-sample period, and generate holdings/returns.
6. Package the reproducible `MAIN.py`, CSV outputs, eight-page deck, appendix, and an audit log of feature/model versions.

## Repository map

- [`data/raw/readme.md`](data/raw/readme.md) — data dictionary, join guidance, and leakage warnings.
- [`data/raw/factor_char_list.csv`](data/raw/factor_char_list.csv) — supplied characteristic list.
- [`data/raw/penalized_linear_hackathon.py`](data/raw/penalized_linear_hackathon.py) — legacy regularized-model template; update its dates and target alignment before reuse.
- [`data/raw/portfolio_analysis_hackathon.py`](data/raw/portfolio_analysis_hackathon.py) — legacy decile portfolio template; reconcile its metrics and constraints with the brief before reuse.
- `data/raw/McGill-FIAM Asset Management Hackathon First Challenge 2026.pdf` — released challenge brief.

## References

Local data and rules should take precedence over generic examples. Useful external references include:

- [2026 FIAM Asset Management Hackathon](https://hackathonfiam-ia.com/hackathon/)
- [Gu, Kelly, and Xiu, “Empirical Asset Pricing via Machine Learning”](https://doi.org/10.1093/rfs/hhaa009)
- [SEC EDGAR application programming interfaces](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)
- [SEC EDGAR FAQ](https://www.sec.gov/about/webmaster-frequently-asked-questions)
- [FRED 3-Month Treasury Bill series (TB3MS)](https://fred.stlouisfed.org/series/TB3MS)
- [8-K text-based strategy research](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3910451)
