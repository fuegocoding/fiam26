# Capacity and expert-mix study — 2026-09-25

## Bottom line

- **The TimesFM + XGBoost core does not survive tight capacity limits on its own.** Under a $5 short-price floor and 5%/2%-of-ADV
  position caps, its return falls from 15.7% to about 3–5% a year, because roughly 60% of its short dollars sit in stocks under $5
  and 78% in nano/micro caps. Lowering the floor to $1 restores most of it. The floor, not fund size, is the driver.
- **Mixing the core with seven analyst/accounting sleeves is much more robust.** The eight-expert equal-weight book (capped at 500
  names) keeps **14.6%/yr (Sharpe 1.68) under the tight limits and 18.8%/yr (Sharpe 2.28) under looser, prime-broker-style limits**,
  at $100m, against 20.2% (Sharpe 2.33) with no limits. Beta stays between −0.01 and 0.13 and alpha is significant in every case
  (t 3.5–5.4).
- **A trailing-performance router adds nothing we can rely on.** At return level the best of ten variants added about 0.1–0.2
  Sharpe over equal weights (2.83–2.87 vs 2.70) while the worst lost up to 0.6 (2.07). After capacity limits the primary router is
  within about ±0.1 Sharpe of equal weights, and it tilts toward the borrow-dependent micro sleeves. We recommend fixed equal
  weights and report the router as tested and not better.
- **Everything here assumes the micro-cap shorts can be borrowed.** Nothing in the data says whether they can. The borrow fees are
  stylised, and the broker tiers are our judgment.

## Question

Which of our strategies still work when each stock's position is limited by its own liquidity and borrow, and which broker
capability would the mix need? The competition sets no fund size and does not ask for borrow costs, but it says it will read the
short book for small-cap and hard-to-borrow names.

## Experts

Every expert is a unit book (about 100 long / 100 short, gross about 2.0). Weights are point-in-time and monthly; the test window
is 2021-01 to 2026-08 (68 months).

| Expert | Mechanism | Uses analyst data | Uses an ML model |
|---|---|---|---|
| `e7_lm` | Reported earnings surprise, then next-quarter analyst revisions agreeing (large/mega) | yes | no |
| `e7_small_gated` | Same rule in small caps, switched off by a validation gate | yes | no |
| `gap_micro_best` | Price versus a fair value from fundamentals and forward analyst EPS (micro caps) | yes | gradient-boosted trees |
| `runway_micro` | Cash / cash burn, dilution risk in cash-burning micro caps | no | no |
| `conf_lm_gated` | Analyst revisions agreeing with the price trend (large/mega) | yes | no |
| `inventory_retail` | Sales growth minus inventory growth in retail | no | no |
| `oplev_software` | Sales growth minus SG&A growth in software | no | no |
| `core` | Submitted v3 book: TimesFM 3.0 forecasts 50/50 with XGBoost on the panel plus TimesFM features, 3-month staggered tranches | no | TimesFM, XGBoost |

- The seven sleeves' stock weights come from the walk-forward sleeve builder (`src.edges.combined_book`, scale 0.20, per-sleeve exit
  horizon chosen each year on the two-year validation window). They are saved in
  `docs/research/capacity_mix_20260925/sleeve_weights.parquet`; this study starts from that file.
- The core is the submitted 1.0x-gross book with weights doubled so it is a unit book like the sleeves.
- The S&P 500 sleeve of the submitted v3 configuration is not included.

## Method

1. **Return frame.** Each expert's monthly net return is its weights times next-month excess returns, minus turnover times the
   size-group cost (mega 5, large 10, small 20, micro 40, nano 60 bp per side traded). A held name with no next-month return is
   credited zero.
2. **Netting.** For a given set of expert weights, stock weights are summed across experts, so overlapping names net out. The book
   is then trimmed to the 500 largest absolute weights each month to respect the 100–500-name rule (about 4% of gross is dropped).
3. **Capacity engine** (`src/capacity_backtest.py`): target dollars are `weight × NAV`, each name is clipped to
   `participation × ADV × days`, and dollars that do not fit stay as cash. Names without ADV data cannot be traded. Costs are charged
   on executed dollars. Borrow fees, where used, are applied to short notional.
4. **Scenarios.**

| | Long cap (× ADV) | Short cap (× ADV) | Days | Short price floor | Borrow fee |
|---|---|---|---|---|---|
| U | none | none | – | none | 0 |
| A | 0.25 | 0.10 | 5 | $5 | 0 |
| B | 0.25 | 0.10 | 5 | $1 | 0 |
| C | 1.0 | 0.5 | 10 | $1 | 0 |
| D | 2.0 | 1.0 | 10 | $1 | 0 |
| E | 2.0 | 1.0 | 10 | $1 | tiered |
| F | 1.0 | 0.5 | 10 | $1 | tiered |
| G | 0.25 | 0.10 | 5 | $5 | tiered |

   Tiered borrow fee, in bp per year: mega 25, large 40, mid 100, small 300, micro 700, nano 1500. **These are stylised numbers, not data.**
5. **Metric conventions.** Sharpe is excess of TB3MS. IR is versus TB3MS + 4%, with no credit for cash on collateral (the v3 memo
   convention). Alpha and beta come from a regression on the market excess return with Newey–West errors. Turnover is dollars traded
   on both sides per year divided by NAV. P&L compounds the monthly net returns from the stated starting NAV.
6. **Router.** Weights for month *t* use only expert net returns through *t−1*, with equal weights until the lookback is filled. It
   is a small trailing-performance soft gate, not a learned neural mixture.

## Results

### Expert standalone (unit books, 68 months, net of trading cost)

| Expert | Ann. return | Sharpe | IR | Vol | Max DD | Beta |
|---|---|---|---|---|---|---|
| `e7_lm` | 6.0% | 0.30 | −0.15 | 8.9% | −12.9% | −0.17 |
| `e7_small_gated` | 1.9% | −0.12 | −0.46 | 11.3% | −32.9% | −0.02 |
| `gap_micro_best` | 44.4% | 1.77 | 1.60 | 23.2% | −17.5% | 0.01 |
| `runway_micro` | 39.1% | 1.49 | 1.32 | 24.1% | −16.8% | 0.04 |
| `conf_lm_gated` | 23.3% | 1.10 | 0.88 | 18.1% | −21.0% | −0.22 |
| `inventory_retail` | 14.1% | 0.55 | 0.34 | 19.7% | −18.1% | −0.21 |
| `oplev_software` | 22.7% | 0.80 | 0.63 | 24.5% | −26.2% | 0.48 |
| `core` (2.0x gross) | 34.1% | 1.70 | 1.48 | 18.1% | −12.5% | −0.08 |

The two micro-cap sleeves are the strongest and also the most borrow-dependent.

### Fixed mixes and leave-one-out (return level, before netting and capacity)

| Book | Ann. return | Sharpe | IR | Max DD |
|---|---|---|---|---|
| `core` alone (2.0x) | 34.1% | 1.70 | 1.48 | −12.5% |
| 7 sleeves equal | 21.6% | 2.36 | 1.85 | −7.2% |
| **8 equal (7 + core)** | **23.2%** | **2.70** | **2.16** | −7.5% |
| Core + 5 no-micro equal | 17.0% | 1.58 | 1.12 | −11.9% |
| 8 equal minus `gap_micro_best` | 20.2% | 2.08 | 1.59 | −9.4% |
| 8 equal minus `runway_micro` | 20.9% | 2.37 | 1.83 | −9.3% |
| 8 equal minus `core` | 21.6% | 2.36 | 1.85 | −7.2% |
| 8 equal minus `e7_lm` | 25.7% | 2.80 | 2.30 | −7.3% |
| 8 equal minus `e7_small_gated` | 26.2% | 2.99 | 2.47 | −7.7% |
| 8 equal minus `inventory_retail` | 24.5% | 2.88 | 2.33 | −6.0% |

Adding the core to the seven sleeves lifts Sharpe from 2.36 to 2.70 because it correlates weakly with them. Leave-one-out is a
hindsight comparison on the test window and should not be used to choose which experts to keep.

### Routing

Ten variants were run (`routing_returnlevel.csv`). The primary specification, declared before the grid was run, is a trailing
24-month performance gate blended halfway with equal weights (`perf`, L=24, λ=0.5).

| Book | Sharpe, all 68 months | Sharpe, 2023-01 onward |
|---|---|---|
| 8 equal | 2.70 | 2.20 |
| Router `perf` L=24 λ=0.5 (primary) | 2.83 | 2.39 |
| Router `posgate` L=24 | 2.87 | 2.45 |
| Router `perf`, other settings | 2.19–2.79 | 1.83–2.36 |
| Router `invvol` | 2.07–2.25 | 1.49–1.56 |

Mean primary-router weights from 2023: `gap_micro_best` 17%, `core` 16%, `runway_micro` 15%, `oplev_software` 13%, `conf_lm_gated`
11%, `inventory_retail` 10%, `e7_lm` 9%, `e7_small_gated` 8%. After capacity limits the routed book is within a few tenths of the
equal-weight book (see below), so we keep equal weights.

### Capacity grid, 8 equal-weight experts, $100m, 500-name cap

| | P&L | Ann. return | CAGR | Vol | Sharpe | IR | Max DD | Worst month | Beta | Alpha (t) | Hit rate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| U | $206.7m | 20.2% | 21.9% | 7.1% | 2.33 | 1.78 | −7.1% | −4.2% | −0.01 | 20.3% (5.4) | 79% |
| A | $124.6m | 14.6% | 15.4% | 6.6% | 1.68 | 1.08 | −6.3% | −2.7% | 0.13 | 13.2% (4.0) | 75% |
| B | $147.8m | 16.3% | 17.4% | 6.1% | 2.09 | 1.45 | −6.2% | −3.0% | 0.09 | 15.4% (4.8) | 77% |
| C | $185.0m | 18.8% | 20.3% | 6.7% | 2.28 | 1.69 | −6.5% | −3.7% | 0.03 | 18.5% (5.0) | 77% |
| D | $196.4m | 19.6% | 21.1% | 6.9% | 2.31 | 1.74 | −6.8% | −4.1% | 0.01 | 19.5% (5.1) | 79% |
| E | $150.9m | 16.6% | 17.6% | 6.9% | 1.88 | 1.32 | −7.8% | −4.4% | 0.01 | 16.5% (4.3) | 74% |
| F | $143.8m | 16.0% | 17.0% | 6.7% | 1.87 | 1.28 | −7.6% | −4.0% | 0.03 | 15.8% (4.3) | 72% |
| G | $106.1m | 13.0% | 13.6% | 6.6% | 1.45 | 0.86 | −7.1% | −2.8% | 0.13 | 11.7% (3.5) | 71% |

| | Turnover (×/yr) | Trading cost (%/yr) | Borrow fee (%/yr) | Fill | Long / short (% NAV) | Names long / short | Short $ under $5 | Nano+micro share of short $ | Short as % of mkt cap (median / p90) | Broker tier (judgment) |
|---|---|---|---|---|---|---|---|---|---|---|
| U | 11.3 | 2.78 | 0 | 99% | 85% / 86% | 236 / 259 | 10.5% | 34% | 0.1% / 1.4% | – |
| A | 8.9 | 1.84 | 0 | 82% | 77% / 64% | 239 / 208 | 2.1% | 18% | 0.0% / 0.2% | 3–4: ordinary locates |
| B | 9.3 | 2.00 | 0 | 84% | 76% / 69% | 239 / 260 | 8.3% | 22% | 0.0% / 0.5% | 2–3: sub-$5 lend list |
| C | 10.6 | 2.49 | 0 | 94% | 83% / 80% | 236 / 259 | 9.4% | 29% | 0.1% / 0.9% | 2: deep small-cap lending |
| D | 11.0 | 2.61 | 0 | 96% | 84% / 83% | 235 / 258 | 9.8% | 31% | 0.1% / 1.0% | 2: large prime broker |
| E | 11.0 | 2.63 | 3.05 | 97% | 84% / 83% | 235 / 258 | 9.8% | 32% | 0.1% / 0.9% | 2: nano/micro lent at ≤15%/yr |
| F | 10.7 | 2.51 | 2.86 | 94% | 83% / 81% | 236 / 259 | 9.5% | 30% | 0.1% / 0.8% | 2 |
| G | 9.0 | 1.86 | 1.65 | 82% | 77% / 65% | 239 / 208 | 2.2% | 18% | 0.0% / 0.2% | 3–4 |

**Broker tiers are a judgment, not a measurement.** Rule of thumb used: an ordinary prime broker's small-cap lending covers a book
whose 90th-percentile short is about 1% of market cap or less, with under about 10% of short dollars below $5. Tier 1 is a
bulge-bracket broker with a dedicated hard-to-borrow desk; Tier 4 is a basic or retail-style broker.

### Other books at $100m (same limits)

| Book | Scenario | Ann. return | Sharpe | IR | Max DD | Beta | Alpha (t) | Turnover | Fill | Short $ under $5 | Nano+micro of short $ | p90 short % of mkt cap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 routed | A | 14.3% | 1.60 | 1.02 | −6.8% | 0.15 | 12.8% (3.9) | 8.5 | 77% | 2.5% | 20% | 0.2% |
| 8 routed | C | 19.2% | 2.33 | 1.75 | −6.5% | 0.05 | 18.7% (5.0) | 10.2 | 91% | 11.5% | 34% | 1.0% |
| 8 routed | E | 17.3% | 1.96 | 1.40 | −8.0% | 0.02 | 17.0% (4.3) | 10.7 | 95% | 12.0% | 37% | 1.1% |
| Core + 5 no-micro | A | 13.3% | 1.23 | 0.74 | −9.1% | 0.10 | 12.2% (3.5) | 9.8 | 90% | 1.2% | 5% | 0.1% |
| Core + 5 no-micro | C | 16.4% | 1.55 | 1.08 | −11.3% | −0.03 | 16.7% (4.5) | 11.1 | 98% | 9.8% | 13% | 1.0% |
| Core + 5 no-micro | E | 14.0% | 1.26 | 0.79 | −12.2% | −0.03 | 14.3% (3.8) | 11.2 | 99% | 10.2% | 14% | 1.1% |
| 7 sleeves (no core) | C | 18.3% | 2.02 | 1.48 | −7.3% | 0.04 | 17.8% (4.6) | 10.7 | 94% | 3.5% | 24% | 0.4% |

The core + 5 no-micro book is the borrow-light option: only 5–14% of its short dollars are nano/micro.

### The core alone (2.0x gross, $100m)

| | U | A | C | E |
|---|---|---|---|---|
| Ann. return | 31.0% | 5.1% | 23.2% | 20.9% |
| Sharpe | 1.54 | 0.15 | 1.39 | 1.11 |
| Short $ under $5 | 64% | 10% | 56% | 59% |

Under the tight scenario the core collapses. The earlier single-book run at the submitted 1.0x gross showed the same pattern
(15.7% unconstrained, 2.6% at scenario A, 12.5% at C).

### Fund size, 8 equal

| Fund size | C: ann. return | C: Sharpe | C: IR | C: fill | E: ann. return | E: Sharpe | E: IR |
|---|---|---|---|---|---|---|---|
| $25m | 19.8% | 2.28 | 1.73 | 98% | 17.0% | 1.89 | 1.34 |
| $100m | 18.8% | 2.28 | 1.69 | 94% | 16.6% | 1.88 | 1.32 |
| $250m | 17.7% | 2.23 | 1.61 | 89% | 15.9% | 1.86 | 1.27 |
| $500m | 16.3% | 2.12 | 1.47 | 83% | 15.3% | 1.87 | 1.25 |

## Caveats

- **Selection.** The seven sleeves were chosen after looking at the same 2021–2026 test years, and the router was tuned on the
  same 68 months. Levels are inflated; the differences between scenarios are the safer reading.
- **Borrow is unverified.** The micro-cap value-gap and runway shorts carry most of the sleeves' return and are exactly the shorts
  that may not be lendable. The borrow-light book (core + 5 no-micro) earns 13–16% under the same limits.
- **Fees and tiers are assumptions.** The fee tiers are stylised and the broker tiers are judgment.
- **No redistribution.** Unfilled dollars stay as cash; a constrained re-optimization would recover part of the gap at large NAV.
- **Missing ADV.** Names without ADV data cannot be traded, so the engine's unconstrained figure for the seven sleeves (19.9%) sits
  below the sleeve builder's own net figure (21.8%).
- **Router is small.** It is a trailing-performance gate over eight experts on 68 months; it is not a learned mixture.
- **Cost line.** The sleeve builder's command-line report mis-scaled trading cost in our run. Only its holdings and gross returns
  were used here; costs were charged by the capacity engine.

## Reproduce

Inputs required: `data/raw/chars_final_with_names.parquet`, the TimesFM and XGBoost prediction files used by
`src.capacity_backtest._load_core_targets`, `output/repro/final_config_monthly.csv`, and `data/external/SP500.csv`.

```bash
python -m pytest tests/test_capacity_mix.py
python -m src.capacity_mix.routing     # fixed mixes, leave-one-out, router grid; writes CSVs and a git-ignored state file
python -m src.capacity_mix.capacity    # A-G grid and fund-size sensitivity; writes capacity_grid.csv
```

The seven sleeves' weights were produced by the sleeve builder and are shipped as `sleeve_weights.parquet`; regenerating them
needs `python -m src.edges.combined_book --scale 0.20 --sleeves e7_lm,e7_small_gated,gap_micro_best,runway_micro,conf_lm_gated,inventory_retail,oplev_software`.

## Files

- `src/capacity_mix/` — `common.py`, `routing.py`, `capacity.py`
- `tests/test_capacity_mix.py`
- `docs/research/capacity_mix_20260925/` — `sleeve_weights.parquet`, `expert_stats.csv`, `expert_correlations.csv`,
  `routing_returnlevel.csv`, `router_primary_mean_weights.csv`, `capacity_grid.csv`
- Depends on the capacity engine: `src/capacity_constraints.py`, `src/capacity_backtest.py`, `src/large_mega_hedge.py`
