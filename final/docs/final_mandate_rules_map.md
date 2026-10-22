# Final mandate: rules map and what it changes in our model

Date: 2026-10-21. Source: `data/raw/final/McGill-FIAM Asset Management Hackathon 2026
— Final Challenge Instructions.pdf` (24 pages, read in full). This memo maps every
rule in the final mandate against the first challenge and against what `final/`
does today. Run results for the mandate-compliant book are in
[RESULTS_AND_BASELINE_DELTA.md](RESULTS_AND_BASELINE_DELTA.md).

**Reading rule (Instructions §2):** the change table below the section heading is
the complete list of differences. Anything not in it keeps the first-challenge
rule.

---

## 1. What `data/raw/final` adds

The core panel is byte-identical to the first-challenge drop (verified by hash):
`chars_final_with_names.parquet`, `8k_20150101_20260831_identified.parquet`,
`factor_char_list.csv`, `readme.md`, both hackathon `.py` scripts. That `readme.md`
describes the data only and is **not** the specification for the final. New:

| File | What it is |
|---|---|
| `...Final Challenge Instructions.pdf` | The mandate. 24 pages. This is the spec. |
| `MCGILL_2026_CarbonArc_Data_Screen.xlsx` | Carbon Arc catalogue screened for this mandate: 4 sheets (Screen, Recommended set, Merge rules, Full catalog of 76 assets) with entity key, history, publication lag and survivorship verdicts. |
| `dsf_mini_from_2014.csv` | `permno, date, prc, daily_return` — CRSP daily from 2014. Enables the daily-frequency reporting metrics; daily trading is permitted, not required. |

`OneDrive_1_21-10-2026.zip` is only the container for the above.

**Not supplied yet**, though §9.2 promises it: the benchmark series
`b_t = TB3MS_t/1200 + 0.04/12`, computed identically for every team. We compute the
same formula ourselves from FRED `TB3MS`. §12 says further constraints and the
final data configuration arrive at the start of the event (28–29 October).

---

## 2. The complete rules map

| Rule | First challenge | **Final mandate** | `final/` before | **What it entails now** |
|---|---|---|---|---|
| Universe | U.S. public equities | unchanged | same | none |
| Return data | monthly | **daily for all stocks + monthly** | monthly | daily layer built from `dsf_mini` for §9.1/§9.4 metrics; trading stays monthly |
| Rebalancing | monthly or slower | **daily to monthly, your choice, justified** | monthly | unchanged; alpha-decay evidence (§6) is the justification |
| Shortable universe | no restriction | **market cap ≥ US$500m at formation** | no restriction | **`SHORT_ME_FLOOR = 500` in `optimizer._book`, short leg only** (long leg unrestricted per §3.1) |
| Long universe | no restriction | unchanged | same | none |
| News and text | 8-K corpus | **8-K + Carbon Arc library** | 8-K only | optional; 8-K survivorship defect must be disclosed (§6.4) |
| Platform | own environment | **Carbon Arc + Arcana** | own | onboarding pending; Arcana is also the review lens |
| Fund size | not specified | **US$1bn endowed at 01/2021, compounding** | `NAV` flexible (sweep) | **`NAV = 1e9`; dollar limits follow the compounding capital path** (§3.3: orders scale with the fund) |
| Position count | 100–500 | **200–500** | 100 + 100 = 200 | sits exactly on the new floor; compliance now checks `200..500` |
| Single position | ≤10% of capital | **≤5%** | max 0.76% | passes trivially; now checked explicitly |
| Agentic work | encouraged, on signal | **construction + risk, with counterfactual** | identity agent only | **still open** — see §5 below |
| Mandate | market neutral, gross ≤200%, net ±50% | unchanged | gross ≤100%, net ±50% | compliant at half the allowed leverage; justify in the deck |
| Benchmark | 3m T-bill + 4%/yr | unchanged (formula fixed) | same | `benchmark = rf + 0.04/12`; supplied series to be adopted when it arrives |
| Headline metric | Sharpe, information ratios | unchanged + **standardised reporting** | ad-hoc stats | **§9.1 table produced** (`output/report/performance.json`) |
| Reporting | prescribed statistics | **fixed §9.1 table + §9.4 exhibits + §9.3 turnover definition** | partial | filled in; a few exhibits still to be drawn (§8) |
| Evaluation | code review for look-ahead | **economic and factor review on Arcana** | — | lenses pre-computed (`python -m arcana`) |

---

## 3. What this entails in the code

### 3.1 Portfolio construction (`optimizer.py`)

- **Short floor.** `SHORT_ME_FLOOR = 500.0` (`me` is market equity in **$ millions**;
  median stock ~$676m, largest ~$5.26e6). Only the short leg is filtered — §3.1
  restricts the shortable universe and leaves the long universe unrestricted.
  Over the test window the floor blocks ~120k candidate stock-months and leaves
  ~713 short-eligible names per month before sector matching.
- **Compounding capital (§3.1, §3.3).** The endowment is US$1bn at the start of the
  out-of-sample period and the fund compounds: the capital month *t* is sized on is
  the endowment grown by every prior month's net return. Every **dollar** limit
  (the ADV/execution filter, reported position size) follows that path, so a
  successful fund tightens its own liquidity constraints. Because the capital path
  depends on the book and the book depends on the path, `main.py` iterates
  `optimize → nav_schedule` to a fixed point (settles in ~9 passes).
- **Already compliant:** 200 names (100 + 100) exactly at the new floor; largest
  position 0.57% against the 5% cap; gross ≤100% (200% allowed); net inside ±50%;
  monthly rebalance permitted by §4.

### 3.2 Evaluation and reporting (`evaluate.py`, `report.py`)

- **Daily layer** from `dsf_mini_from_2014.csv`: maximum one-day loss, daily beta to
  the S&P 500 (with standard errors), and the §9.4 position-size-in-ADV-days
  exhibit, long and short separately. Each month's book drifts on the daily returns
  exactly as the monthly evaluator drifts it, so the two frequencies are the same
  book.
- **Compliance checks** extended to the mandate: names 200–500, short book ≥ $500m
  market cap at formation (minimum observed), single position ≤5% of capital.
- **§9.1 rows** the base statistics omit: ending endowment value, annualised
  volatility, tracking error, maximum one-month loss, positions avg/min/max,
  one-way turnover per rebalance and annualised, average holding period, average
  position size in US$m, gross and net exposure ranges.
- **Turnover (§9.3)** is one-way: `½·Σ|wᵢ,τ − wᵢ,τ₋|` per rebalance, annualised by
  ×12 for monthly books. Holding period is measured directly (position-weighted
  consecutive months × 21) and cross-checked against the gross/turnover
  approximation the instructions offer.
- **Conventions (§9.2).** Monthly statistics annualise ×12 and ×√12; the daily
  layer uses 252/√252. CAGR (ties to the dollar outcome) and the arithmetic mean
  (drives Sharpe/IR) are both reported and labelled. Benchmark
  `b_t = TB3MS_t/1200 + 0.04/12` monthly, `TB3MS_t/25200 + 0.04/252` daily.
- **Signals, walk-forward and risk rules are untouched.** The mandate changes the
  feasible set and the reporting, not the models.

### 3.3 Deliberately unchanged (to justify in the deck)

- **Gross at 100%** against the 200% allowed — half the leverage is compliant and
  is what the submitted baseline ran.
- **Monthly rebalancing.** §4 makes daily trading permitted, not required, and the
  alpha-decay measurement (spread still ~26%/yr at 12 months, peaking at 2–3
  months) says a daily book would pay costs for signal that does not decay in days.
- **Micro-sleeve cutoff at $50M** and the liquid blend above it — an internal
  fund-size policy, keyed to the endowed size so the blend is fixed over the
  sample.

---

## 4. Reporting spec the deck must satisfy (§9)

| §9.1 row | Status in `output/report/` |
|---|---|
| Ending value of the endowment, cumulative return | `performance.json` |
| Annualised return, annualised volatility | `performance.json` |
| Annualised alpha vs S&P 500 (+ s.e.), monthly and daily beta (+ s.e.) | `performance.json` |
| Sharpe, IR vs T-bill + 4%, tracking error | `performance.json` |
| Max drawdown, max one-month loss, max one-day loss | `performance.json` |
| Positions avg/min/max; gross, net ranges | `performance.json` |
| One-way turnover per rebalance and annualised; avg holding period | `performance.json` |
| Average position size US$m; short book median market cap | `performance.json` |
| Trading cost, borrow cost %NAV/yr; IR net of all costs | `performance.json` (returns are net) |

| §9.4 exhibit | Status |
|---|---|
| Cumulative active return vs benchmark, S&P 500 for context | `charts/cumulative.png` |
| Rolling 252-day beta to the S&P 500 | `charts/rolling_beta.png` |
| Calendar-year returns, before and after costs, vs benchmark | `calendar_years.csv` (after-cost only so far) |
| Gross exposure, net exposure, position count over time | `monthly_detail.csv` (not yet charted) |
| Top/bottom ten holdings by average weight | `top_average_holdings.csv` |
| Ten largest positive and negative contributors | `contributors.csv`, `charts/contributors.png` |
| Short-book market cap and dollar-volume distributions (floor evidence) | `performance.json` short stats + `position_adv_days.csv` (not yet plotted) |
| Position size in days of ADV, legs separate | `position_adv_days.csv` (not yet plotted) |

---

## 5. Forward-looking bias (§7) — what applies to us

Critical rule: any forward-looking information in construction, signal generation,
model selection or optimisation is disqualification. This year verification changes
from code review to an **economic** review on Arcana: a leak shows up as a strategy
that works for no articulable reason, not as a dated line of code (§11).

| §7 case | Our exposure |
|---|---|
| Feature selection on the full sample | avoided — walk-forward refits annually, train through Y−3, validate Y−2..Y−1 |
| Regime labels from future data | not used |
| Same-day news without intraday timestamp | monthly book: features form at month-end *t*, book earns month *t+1* (the §7 convention) |
| Earning a return the signal already used | formation/target alignment enforced by `timing.validate_predictions` |
| Current-vintage alternative data | only if a Carbon Arc asset is used — then §6.3/§7.3 tests apply first |
| Survivorship through the entity map | 8-K corpus defect confirmed (§6.4); Carbon Arc assets must pass the delisting test before use |
| Delisting / missing stub return | handled: missing next-month returns exit at the last valid quote, costed, in `liquidations.csv` |
| Full-sample standardisation | scores are cross-sectional monthly ranks; no full-sample moments |
| Hyperparameters tuned on the test period | split dates are fixed in code and stated; no post-hoc tuning on 2021–2026 |
| Model memory (pretrained LLMs) | only relevant if an LLM layer is added: declare release date vs test window, run and report the re-identification rate, prefer small models, keep at least one non-pretrained signal (§7) |
| Cost and borrow assumptions | disclosure rule, not a leak: state the schedule and date, report a flat conservative alternative (e.g. 50bp large / 5% rest), never let cost parameters enter the forecast |

---

## 6. The agent ask (§5) — short version

The obvious move (agent reads a filing, scores a company, output becomes a factor)
is table stakes. §5 wants effort **downstream** of the signal (portfolio
construction and position-level risk) and **upstream** (agents exploring the Carbon
Arc library). §5.1.3 *context-conditional position sizing* is called the single
highest-value application available.

Hard design requirements (§5.3), not suggestions:

1. The agent proposes; **deterministic code enforces** every hard constraint (never a prompt).
2. **Log every decision** with its inputs and reasons.
3. Point-in-time applies to agents too (§8.2).
4. **Build the counterfactual and report it** — same strategy with the agent and
   with a fixed rule, side by side. "The single most important requirement."
5. Design the **trigger**, not the agent (per-name-per-day is ~700k calls and is
   explicitly disallowed).
6. **Cache and version** every response.

"A negative result is a real result." An agent in the architecture diagram with no
evidence it changed anything earns nothing. The identity agent does not count
toward this. Status: **not built**; see §8.

Carbon Arc half: REST API + MCP server let an agent query data directly; the whole
competition is throttled to **20k tokens/day**. Four applications (§5.2): automated
dataset triage, hypothesis generation at scale (economic rationale mandatory,
denominator of hypotheses tested must be reported), cross-dataset combinations,
documentation archaeology.

---

## 7. Carbon Arc (§6) — the parts that decide whether we use it

- **Good at:** entity resolution against one ontology; real-economy breadth (card
  spend, foot traffic, jobs, software adoption, claims, contracts); documented lags
  and in places maturity curves; record versioning (vintages reconstructible); a
  few long histories (prices 1990, EDGAR 2012, events 2011).
- **Not good at:** short histories vs a 2021 test start; heterogeneous lags
  (4 hours–90 days, metric-specific); revisions/backfill (current vintage = leak);
  a present-day entity map (survivorship); panel composition drift (work in ranks
  and growth, not levels); skewed sector coverage (consumer/tech rich, financials/
  utilities poor); some closed series; crowding.
- **Four screening questions, in writing, before any asset is used (§6.6):** when
  does history start; what is the lag for the specific metric; is the series revised
  and can the vintage be reconstructed; what fraction of our universe does it cover
  and is coverage random with respect to returns.
- **Survivorship test we will be asked about (§6.3):** count how many 2017–2020
  delistings (Sears, Bon-Ton, Pier 1, GNC, Ascena, J.C. Penney) resolve in each
  asset's entity map; then the complementary return test of covered vs uncovered
  names. Documenting a coverage bias and controlling for it scores better than not
  looking.
- **8-K corpus defect (§6.4), confirmed in the first round:** the supplied corpus
  under-represents firms that left the universe (built against a present-day filer
  list). If we use it, the limitation must be disclosed and the handling described.
  Rebuilding the text channel from EDGAR (Carbon Arc **CA0032**) avoids it — map
  through **CIK, never ticker** — and will be credited.
- **News reality check (§6.5), CA0035 Benzinga:** it is an **event-tagged table**,
  not text — no headlines, no bodies, no intraday timestamp, no sentiment. Daily
  event-count features are possible; LLMs over news text are not (use 8-K/CA0032
  text instead). An event dated day *t* is actionable **no earlier than the open of
  day t+1** — trading day *t* close on day-*t* news is the single most likely way to
  disqualify this year. Platform refresh is weekly, so no live daily stream during
  the event.

---

## 8. Still open

| Item | Section | Note |
|---|---|---|
| §5 agentic risk layer + mandatory counterfactual | 5.1, 5.3.4 | not built; the identity agent does not count |
| Carbon Arc ingestion + the four screening questions per asset | 6.6 | onboarding emails pending; 20k tokens/day cap |
| §6.3 survivorship test on any Carbon Arc asset we use | 6.3 | one hour each; judges will ask |
| 8-K survivorship disclosure (or rebuild from CA0032) | 6.4 | confirmed defect; must be acknowledged in the deck |
| Model-memory disclosure for any pretrained model | 7 | release date vs test window, re-identification rate |
| §9.4 exhibits not yet drawn | 9.4 | see the status table in §4 |
| Daily-frequency trading | 4 | permitted, not required; decay evidence supports staying monthly |
| Benchmark series adoption | 9.2 | supplied at the event; formula already matches |
| Two of six years trail the hurdle (2023, 2025) | — | disclose with the economic reason |
