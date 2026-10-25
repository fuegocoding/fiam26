# Part 0 — Decision freeze (final_v2)

Status: frozen for build. Each row is labeled **[official]** (mandate/brief), **[spec]** (adopted as written in POD_FRAMEWORK_IMPLEMENTATION_SPEC.md v1.1), or **[choice]** (team default; change requires a new decision ID linked to this file).

Authority order (spec 1.2): official finals instructions > team decisions and frozen run config > implementation spec > earlier strategy docs > agent suggestions.

## Locked decisions

| ID | Decision | Resolution | Basis |
|---|---|---|---|
| C1 | Agent/LLM layer scope | Bounded: schema-constrained extractor, contemporary research aid, judged agent-vs-fixed-rule twin exhibit (spec 28.4.7). No era-matched historical replay. Deterministic code is always the authority (spec 20). | [choice] |
| C2 | Missing-family neutral prior in forecast integration | Zero active alpha for the missing family; missingness mask retained; uncertainty inflated. Not raw-data zero imputation; not coverage renormalization (spec 16.3 items 6-7). | [choice] |
| C3 | Breadth mechanism | Sanctioned broad baseline book: named, certified, fully evaluated liquid permitted-stock-hedge baseline (spec 16.8) counts toward the 200-name floor with honest performance labeling. No dust, no manufactured engines (spec 15.8, 3.5). | [choice] |
| C4 | Portfolio construction engine | Deterministic selection with incumbent retention band, then continuous sizing under limits (spec 16.7); feasibility/relaxation order spec 16.11. Mean-variance objective (spec 16.2) is a registered challenger only, not the primary path. | [choice] |
| C5 | Temporal policy for registered models | Policy A release-gated; cutoff-qualified backcast disabled (spec 20.14). Every registered checkpoint meets spec 20.19 checkpoint-verification before freeze. | [choice] |
| C6 | Integration horizon and combination | 20-trading-day reference horizon; family-level equal weights summing to one (spec 16.3 reference baseline); registered conversion rules; unconverted forecasts excluded from allocation (spec 15.4). | [choice] |
| C7 | Costs, borrow, decision frequency | Freeze spec 19.4 / 29.3 cost and borrow schedules as versioned config at the Part 3/Part 7 boundary. Monthly decision frequency with daily reporting layer, justified by alpha-decay evidence. | [choice] |
| C8 | Data posture | Core build uses competition inputs only (spec 7.4). Legacy artifacts import only with established lineage (spec 27.6). Daily returns, permitted price/volume, CA0032 filing text are upgrades that never block the vertical slice. | [spec] |
| C9 | 68-Month Alignment, Broad-vs-Native ECDF & $1B ADV Caps | Evaluate all 68 holding months (`2021-01` .. `2026-08`, formation `2020-12-31` .. `2026-07-31`) with `TB3MS` and `SP500` aligned to `return_month`, broad-vs-native empirical CDF scaling across liquid sleeves (`e7_lm`, `e7_small_gated`, `conf_lm_gated`, `inventory_retail`, `oplev_software`, `xgb_rank`, `filing_events`), and 5-day ADV participation caps (`5%` long, `2%` short, `prc >= $5`). | [official/choice] |
| C10 | Daily TimesFM 3.0 Trajectory & Uncertainty Sleeve | Admitted `google/timesfm-3.0-pytorch` 21-trading-day market-adjusted return trajectory (`tfm_daily_cum21`, standalone IC `+0.0282`, `t=+2.18`) and quantile uncertainty spread (`tfm_daily_q_spread`, `49.4x` short-squeeze separation) on `dsf_mini_from_2014.csv`, lifting `1.00x` net IR from `0.460` to `0.608` (`Sharpe 1.489`) and `1.75x` net IR from `0.713` to `0.841` (`0.866` with AgentTwin). | [choice] |
| C11 | Offline Arcana Attribution, §9.1 21-Row Table & AgentTwin Freeze | Full offline Arcana lens (factor vs idiosyncratic, sizing vs selection, hit rate vs win/loss, monthly/daily alpha decay, earnings vs ex-earnings window, stress regimes, §6.3 survivorship/coverage bias audit, §7 conservative flat borrow sensitivity) and 68-month `AgentTwin` counterfactual + `freeze_run` acceptance verified. | [official/spec] |

## Mandate constants [official]

Universe US public equities; initial NAV USD 1,000,000,000 compounding; evaluation 2021-01-01 through 2026-08-31; positions 200-500 combined; gross at most 200% NAV; net in [-50%, +50%] NAV; single-name at most 5% NAV absolute; new/increased shorts require formation market cap at least USD 500M; benchmark = 3M T-bill + 4%/year; execution convention: information through t, book at close of t+1, first earned return t+1 to t+2 (spec 2.1-2.2).

## Research labels [spec]

2021-01 through 2026-08 is previously-seen chronological replay, never a fresh holdout (spec 2.4, 13.2). Design 2015-2018; check 2019-2020 with controlled access; prospective only after a genuine commitment record.

## Deferred (explicitly not decided here)

- Internal risk tolerance values (net/beta/sector bands): calibrated on the design period at Part 6, then frozen as versioned config.
- Cost/borrow numeric schedules: frozen at Part 3/Part 7 boundary (C7).
- Registered model roster beyond the bounded C1 layer: decided at Part 10 if that part is activated.
