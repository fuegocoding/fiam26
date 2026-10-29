# final_v3 — four research desks on official data

## Designated main strategy

The current main strategy is the **no-agent Arcana-beta hybrid** documented in
[CURRENT_STRATEGY.md](docs/CURRENT_STRATEGY.md). It retains the established
signal stack and portfolio rules, and uses Arcana stock betas as an additional
formation sizing constraint where the input has a valid observation. The
agent-enabled variant uses the same baseline strategy and adds the research
discovery implementation; it is not a separate portfolio construction recipe.
The exact replay parameters and the Arcana/local beta coverage are recorded in
that document. The designated strategy runs with `--no-discovery`; the
application's original default remains agent-enabled. Reproducing the saved
run also requires its frozen-prediction checkpoint.

A monthly long/short U.S. equity book for the McGill-FIAM 2026 finals
(US$1bn endowed at 01/2021 and compounding, 100 longs / 100 shorts, short leg
≥ US$500m market cap, test window 01/2021–08/2026).

final_v3 keeps the signal engine that produced results in `final` and gives it
the structure and discipline of `final_v2`'s research platform:

- **Four desks.** Every signal belongs to one of final_v2's research pods —
  fundamentals & expectations, corporate events, valuation, market dynamics.
- **Official data.** Every input is a supplied competition file, except one
  disclosed I/B/E/S input (below).
- **Nothing silent.** Each component reports its coverage every run, inactive
  components say so, and each component's effect on the book is measured.
- **Automated discovery.** Each year, LLM research agents propose new sleeves
  from the official sources and the Carbon Arc catalogue; only proposals that
  pass a fixed gate on data settled before that year can enter the book.

## Run

Part 2 enforcement and reporting now run through `final_v3/main.py`; see
[docs/PART2.md](docs/PART2.md). The saved results below predate this adaptation
and must be regenerated before making claims under the new mandate. Sector
breadth and net bounds are hard construction checks. Realised volatility is
checked against 12% using the full daily net-return history. Arcana is deferred.

```bash
cd final_v3 && ../.venv/bin/python main.py          # full rebuild into final_v3/output
cd final_v3 && ../.venv/bin/python main.py --reuse  # resume matching cached stages
python final_v3/main.py --no-discovery --official-only --device cpu --reuse  # no LLM calls; omit unavailable I/B/E/S
```

Inputs: `../data/` (panel, 8-K corpus, factor list, `dsf_mini_from_2014.csv`)
and `../external_data/external.parquet` (read only for the disclosed exception).
A full rebuild uses the GPU for the graph model and XGBoost. The measured
2026-10-28 no-agent A100 run peaked near 9.5 GiB of process RAM; allow sufficient
headroom and run one heavy job at a time on a 16 GB machine.

`--no-discovery` retains the established sleeves and graph training. It makes
no LLM or Carbon scouting calls. The standard workflow makes no live Arcana
calls; the designated Arcana-beta hybrid uses the saved beta export described
in [CURRENT_STRATEGY.md](docs/CURRENT_STRATEGY.md). The I/B/E/S
analyst-revision sleeve still needs its disclosed input; disabling agents does
not silently remove that investment signal. Discovery mode is fingerprinted
in the cache, so enabled and disabled runs cannot reuse each other's results.

Trained weights, forecasts and monthly portfolio allocations can now be saved
outside `output/`. `--from-weights CHECKPOINT` replays allocations and reporting;
`--from-predictions CHECKPOINT` rebuilds construction/reporting without training
or graph preparation. Input and checkpoint contents are verified by hash.
See [checkpoint commands and preservation](docs/CHECKPOINTS.md) and the
[measured attribution and research priorities](docs/RESEARCH_PRIORITIES.md).

## Result

The latest valuation-enabled rerun caps Treasury at 5% and calendar-year
one-way turnover strictly below 400%, with initial deployment exempt:
[results, exposures and turnover](docs/ANNUAL_TURNOVER_400PCT_RERUN.md).
The [monthly and pod metrics](docs/ANNUAL400_MONTHLY_POD_METRICS.md) include
all five complete books, rank IC, CAPM/factor alpha and trailing/expanding
performance statistics. There is no fixed monthly turnover ceiling.
It is stored separately in `annual_turnover_400pct_output/`. The annual budget
includes forced exits and is paced through December using settled turnover;
a formation-only minimum-trade fallback may borrow later monthly allocations.
A realized annual breach rejects the run. The preceding
[30% monthly-cap run](docs/TURNOVER_30PCT_RERUN.md) remains preserved in
`turnover_30pct_reserved_output/`. The earlier [5% Treasury run](docs/TREASURY_5PCT_RERUN.md) remains
preserved in `treasury_5pct_output/`. The earlier
[Treasury-sweep run and pod/cap breakdown](docs/VALUATION_TREASURY_RERUN.md)
remain preserved in `valuation_treasury_output/`; the tables below are older
historical results.

**Historical result:** these figures used the earlier panel-rate/proxy reporting
conventions. Current reporting uses the documented Treasury/S&P 500 input and
explicit daily/monthly reconciliation; the figures are not a Part 2 certificate.

Full rebuild, 01/2021–08/2026, net of trading and borrow costs. The benchmark
is T-bill + 4% with the panel's own risk-free rate; "market" is a cap-weighted
proxy of the 500 largest stocks.

| Metric | final_v3 |
|---|---|
| Information ratio vs T-bill + 4% | **0.25** |
| CAGR | 8.7% (benchmark 7.4%) |
| Sharpe | 1.00 |
| Alpha vs market proxy | 6.4%/yr (t 2.89; HAC 3.59) |
| Beta | −0.09 (s.e. 0.04) |
| Maximum drawdown | −2.9% |
| Long leg / short leg | +6.6%/yr / −0.2%/yr |
| Trading + borrow cost | 0.81% + 0.33% of NAV per year |
| Compliance (names, gross, net, $500m short floor, 5% position cap, labels) | all pass |

| Year | Strategy | T-bill + 4% | Active |
|---|---|---|---|
| 2021 | 9.4% | 4.1% | +5.1 pp |
| 2022 | 12.6% | 5.6% | +6.6 pp |
| 2023 | 0.7% | 9.2% | −8.0 pp |
| 2024 | 16.0% | 9.6% | +5.9 pp |
| 2025 | 6.3% | 8.4% | −1.9 pp |
| 2026 (Jan–Aug) | 4.9% | 5.1% | −0.1 pp |

## The desks

| Desk | Sleeves | What they read |
|---|---|---|
| Fundamentals & expectations | `e7_lm`, `e7_small_gated`, `conf_lm_gated`, `inventory_retail`, `oplev_software` | earnings surprise and breadth after 8-K Item 2.02 results; analyst revisions gated by momentum; sales vs inventory and vs SG&A growth |
| Corporate events | `filing_events` | 54 counts of 8-K item families over 1/3/12 months |
| Valuation | `gap_micro_best` | model-implied vs actual value (micro caps) |
| Market dynamics | `xgb_rank` | walk-forward XGBoost rank forecast on the 147 characteristics |
| Cross-desk overlay | company graph | business-text and peer graph forecast of next earnings, weight 0.05 |

The two micro-cap sleeves (`gap_micro_best`, `runway_micro`) are enabled at
the mandate size. Their signals enter the blend; actual trades still pass the
same liquidity, position-size and $500m short-market-cap checks. The optional
$50m fund-size switch is disabled. Earlier archived runs used that switch.

The current sizing rule caps unused own capital at 5% of NAV at each monthly
rebalance, deploying at least 95% to longs. The short target follows the beta
hedge as a preferred target within 200% gross and +/-50% net, with final per-name ADV and position
caps. The monthly rebalance schedule and 0.5 retention buffer are unchanged;
one-way turnover is budgeted across the calendar year under a 400% ceiling,
with initial funding exempt and forced exits included. A
constrained projection retains eligible incumbents and adjusts weights to
meet the budget using actual drifted holdings. Involuntary liquidations are
reported separately. This cash
ceiling supersedes the global ES/edge reductions to cash, while individual
peer-squeeze/run-up short limits remain in force. Exposures can drift between
rebalances; daily verification remains subject to the documented data gaps.

Unused own capital (`max(1 - long notional, 0)`) accrues the FRED TB3MS cash
proxy, lagged two calendar months to avoid using an unpublished monthly average.
Short proceeds remain separate collateral earning the panel-implied rate;
borrowing above own NAV is financed at that rate. This replaces the previous
implicit cash accrual rather than adding interest twice. Treasury income and
market-cap contributions are exported with each report. The proxy does not
model Treasury price changes, bid/ask costs or a broker margin schedule.

## Data policy

Everything comes from the supplied panel, 8-K corpus and daily returns, with
one disclosed exception: **I/B/E/S analyst revisions feed `conf_lm_gated`**.
The full map, with what each official stand-in loses, is in
[docs/OFFICIAL_DATA_MAP.md](docs/OFFICIAL_DATA_MAP.md).

Why the exception: replacing analyst revisions with official data cost about
half an IR point, and no official stand-in recovered it.

| Book (01/2021–08/2026) | IR vs T-bill+4% |
|---|---|
| Official data only | −0.25 |
| Official data + I/B/E/S revisions only (**final_v3**) | **+0.25** |
| `final` (all Stage-1 external data) | +0.51 |

Eight panel stand-ins for revisions were ranked on 2015–2020 validation only;
the best (`saleq_gr1`, validation t 1.23) is negative on the test window.
Carbon Arc's Benzinga analyst events (insight 491) are the official route, with
history from 2015, but a 2015–2026 pull was quoted at US$185.73–431.24 against a
US$13.78 balance and was not bought.

The I/B/E/S extract ends in early 2026, so `conf_lm_gated` covers only one
2026 month; the coverage report shows this every run.

## What we tested from final_v2, and kept or dropped

Each idea was applied to the same signals and run through the full book.

| final_v2 idea | Effect on a base with edge | Kept |
|---|---|---|
| Four-pod equal weighting | `final` 0.53 → 0.01; final_v3 base 0.25 → −0.09 | as structure only |
| CIO selection score S = .4P + .25C + .2R + .15B | 0.53 → −0.05; 0.25 → −0.10 | no |
| Separate long/short validation | 0.53 → −0.26; 0.25 → −0.14 | no |
| Calibrated score-bin expected returns | 0.53 → −0.69 | no |

These layers help only when the base contains dead sleeves to switch off (on the
official-only base, long/short validation lifted IR from −0.25 to −0.06). On a
base with edge they dilute it, so final_v3 keeps the equal blend.

## Component impact (leave one out)

Each component removed from the final_v3 book (IR 0.250), everything else
unchanged. Positive contribution means the component helps.

| Removed | IR without it | Contribution | Alpha without it |
|---|---|---|---|
| `oplev_software` | 0.062 | +0.188 | 5.5% |
| `filing_events` | 0.095 | +0.155 | 5.6% |
| `inventory_retail` | 0.143 | +0.107 | 6.0% |
| `conf_lm_gated` (I/B/E/S) | 0.172 | +0.078 | 5.9% |
| `e7_lm` | 0.191 | +0.059 | 6.0% |
| `e7_small_gated` | 0.282 | −0.032 | 6.6% |
| `xgb_rank` | 0.284 | −0.034 (alpha falls without it) | 5.7% |
| graph overlay | 0.391 | −0.141 | 7.1% |

Removing `conf_lm_gated` leaves a book built only from competition files at
IR +0.17. Its official stand-in (3-month change in earnings yield) had a
2015–2020 spread t of 0.36 and cost the book 0.42 IR; the discovery gate
(spread t ≥ 2 before the test year) would have rejected it.

The graph overlay helped with I/B/E/S earnings labels (+0.10) and costs 0.14
in this book with the panel's own surprise labels, yet removing it from the
fully official book lowers IR from 0.17 to 0.16. The effects interact, so each
removal is measured as its own full book; any change made on these numbers is a
test-window decision.

## Automated discovery

Each test year, one agent per desk proposes sleeves from the panel, 8-K item
counts, daily-return features and the owned Carbon Arc extracts, and a scout
per desk builds a costed Carbon Arc shopping list (free quotes only; purchase
budget US$0). A proposal enters that year's book only if, on data settled
before the year, it passes S ≥ 0.60, a shuffled-score control, spread t ≥ 2 and
a correlation cap (at most two per desk).

| Test year | Evidence through | Proposed | Admitted |
|---|---|---|---|
| 2021 | Dec 2020 | 24 | 0 |
| 2022 | Dec 2021 | 24 | 0 |
| 2023 | Dec 2022 | 24 | 0 |
| 2024 | Dec 2023 | 24 | 0 |
| 2025 | Dec 2024 | 24 | 0 |
| 2026 | Dec 2025 | 24 | 0 |

144 proposals over six years, none admitted: no discovered sleeve entered the
book. Proposals came from Groq gpt-oss models until their daily quota ran out,
then from a local qwen3:8b.

Run it on its own with `discovery.loop.run(work_dir, years)`, or set
`config.DISCOVERY = True` to fold admitted sleeves into the book; responses are
cached in `discovery_cache/`, and `FIAM_DISCOVERY_OFFLINE=1` replays them.
Every year prints `Discovery <year>: N proposed, M admitted`, including zero.

## Agents: what worked and what did not

- LLM research desk on 2015–2020 validation: 0 of 51 valid proposals admitted
  (best spread t 1.05). The gate held; the agents' main contribution so far is
  the documented search, not alpha.
- IC-based scores often rated proposals highly while their top-minus-bottom
  spread was near zero, so admission requires the spread test.
- Hosted model quotas (Groq, 200k tokens/day per model) run out in one night of
  research; discovery falls back to a local model and caches every response, so
  reruns cost nothing and are reproducible.
