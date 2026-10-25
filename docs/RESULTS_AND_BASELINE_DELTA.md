# Results and baseline delta

Run date: 2026-10-21. Test window 01/2021–08/2026 (68 holding months), monthly
rebalancing, 100 longs / 100 shorts, all risk rules on, returns on NAV net of
trading costs and borrow fees, capital earning the 3-month T-bill. Fund size is
the mandate's **US$1bn endowment, compounding** (Instructions 3.1, 3.3), and the
short leg is restricted to names with **market cap ≥ US$500m at formation**
(Instructions 3.1).

```bash
cd final && python main.py --reuse    # pipeline: predictions -> book -> report
cd final && python -m arcana          # attribution lenses (Instructions 11)
```

Rules map and mandate interpretation: [final_mandate_rules_map.md](final_mandate_rules_map.md).

---

## 0. Run provenance (read before quoting any number)

- **Code base:** the submission tooling on `origin/main` plus the mandate changes
  listed in §4. Signals, walk-forward and risk rules are untouched.
- **Fitted models reused.** The sleeve and graph artifacts in `final/output/` are
  the submitted baseline's; the mandate does not change the signals, and refitting
  them would change the baseline. Scoring, blending, construction, evaluation and
  reporting all run under the current code. `timing_manifest.json` was seeded so
  the pre-audit caches pass the reuse check on this CPU-only machine; `MAIN.py` on
  a fresh machine retrains everything from scratch.
- **Superseded draft.** An earlier draft of this memo (commit
  `backup/new-rules-65394b1`) ran on a commit that predates the final-data timing
  fixes (`76ab228`) and the submission tooling (`41bcebd`). Its numbers — IR 0.34,
  rank IC 0.130 — came from pre-fix signals and are replaced by the ones below
  (rank IC 0.122). The rule implementations it introduced are carried forward
  unchanged.

---

## 1. Standardised table (Instructions 9.1)

| Metric | Units | Strategy | Benchmark (T-bill + 4%) |
|---|---|---|---|
| Ending value of the endowment | US$bn | **1.64** | 1.51 |
| Cumulative return | % | 64.0 | 50.9 |
| Annualised return (arithmetic) | %/yr | 8.85 | 7.28 |
| Annualised return (CAGR) | %/yr | 9.12 | 7.53 |
| Annualised volatility | %/yr | 4.26 | 0.56 |
| Annualised alpha vs S&P 500 | %/yr | **6.40** (t 3.52, HAC 3.77) | n/a |
| Beta vs S&P 500 — monthly | coefficient | **-0.079** (s.e. 0.034) | n/a |
| Beta vs S&P 500 — daily | coefficient | **-0.070** (s.e. 0.008) | n/a |
| Sharpe ratio | annualised | 1.27 | — |
| Information ratio vs T-bill + 4% | annualised | **0.36** | n/a |
| Tracking error | %/yr | 4.37 | n/a |
| Maximum drawdown | % | -3.96 | 0.00 |
| Maximum one-month loss | % | -1.95 (2026-05) | 0.34 (worst month) |
| Maximum one-day loss | % | **-1.28** (2022-02-15) | n/a |
| Positions held | avg / min / max | 200 / 200 / 200 | n/a |
| Gross exposure | avg / max, % of capital | 87.9 / 100.0 | n/a |
| Net exposure | avg / min / max, % of capital | -0.2 / -6.8 / +6.8 | n/a |
| One-way turnover | % per rebalance | **33.5** | n/a |
| One-way turnover | annualised, % | **402** | n/a |
| Average holding period | trading days | **61.2** | n/a |
| Average position size | US$m | **5.91** | n/a |
| Short book median market cap | US$m | **13,693** | n/a |
| Trading cost | % of NAV per year | 0.76 | n/a |
| Borrow cost | % of NAV per year | 0.23 | n/a |
| Information ratio, net of all costs | annualised | 0.36 | n/a |

Turnover is the Instructions 9.3 formula `½·Σ|wᵢ,τ − wᵢ,τ₋|`, annualised ×12.
Holding period is measured directly (position-weighted consecutive months × 21);
the gross/turnover cross-check gives 55.2 days against the direct 61.2, as §9.3
expects for a leveraged book. Average position size is the mean over the
**compounding** capital path ($1.0bn → $1.64bn), which is why it is larger than
the endowed-weight figure. Returns are net of costs and borrow, so the headline IR
is the "net of all costs" row.

### Compliance (`output/report/compliance.json`)

| Check | Result |
|---|---|
| `covers_the_test_window` / `no_missing_month` | true / true |
| `names_between_200_and_500` | true (200 every month) |
| `gross_at_most_200pct` / `net_within_50pct` | true / true |
| `every_holding_has_ticker_and_name` / `no_duplicate_holdings` | true / true |
| `predictions_carry_no_outcomes` | true |
| `short_book_above_500m_floor` | **true** (minimum short mkt cap **$512.6m**, 0 names below) |
| `single_position_at_most_5pct` | true (largest **0.57%** of capital) |
| **`all_pass`** | **true** |

### Calendar years

| Year | Strategy | Benchmark | S&P 500 | Active |
|---|---|---|---|---|
| 2021 | 12.6% | 4.1% | 26.9% | +8.0pp |
| 2022 | 11.5% | 6.2% | -19.4% | +5.0pp |
| 2023 | 5.3% | 9.5% | 24.2% | **-3.8pp** |
| 2024 | 11.9% | 9.3% | 23.3% | +2.4pp |
| 2025 | 5.6% | 8.4% | 16.4% | **-2.6pp** |
| 2026 (to Aug) | 5.0% | 5.2% | 12.3% | -0.1pp |

Two of six years trail the hurdle. Say so, with the economic reason.

---

## 2. What the new rules cost

Same predictions, same code, one rule switched at a time (`rule_deltas`, reports
under `output/report` variants): the first column is the first-challenge rules at
the mandate fund size; the last column is the submitted book.

| Metric | No floor, constant $1bn | No floor, compounding | **Mandate: floor + compounding** |
|---|---|---|---|
| Information ratio | 0.524 | 0.377 | **0.359** |
| Sharpe ratio | 1.38 | 1.27 | **1.27** |
| CAGR | 10.07% | 9.24% | **9.12%** |
| Annualised volatility | 4.57% | 4.36% | **4.26%** |
| Maximum drawdown | -2.95% | -3.96% | **-3.96%** |
| Ending endowment | $1.72bn | $1.65bn | **$1.64bn** |
| Average position size | $4.39m | $5.89m | **$5.91m** |
| One-way turnover, annualised | 421% | 400% | **402%** |
| Short book median market cap | $10,007m | $13,652m | **$13,693m** |
| Shorts below the $500m floor | 83 | 55 | **0** |
| 4-factor alpha (t) | 3.65% (1.31) | 1.99% (0.86) | **2.13% (0.96)** |

Reading:

- **The compounding fund is the expensive rule.** As capital grows from $1bn to
  $1.64bn every order grows with it, the ADV/execution limits tighten through
  2024–2026, and weaker-liquidity names leave the book: IR 0.524 → 0.377
  (-0.15), CAGR -0.83pp. This is §3.3's point ("success tightens your own
  capacity constraint") showing up in the backtest.
- **The $500m short floor is cheap at this fund size** (-0.018 IR, -0.12pp CAGR),
  because the $1bn book was already large-cap on the short side (median short
  $13.7bn) and the liquidity limits had already excluded most micro caps. The
  floor still matters for compliance and for the borrow story: 83 short
  position-months below $500m in the unconstrained constant-NAV book are gone,
  and the new minimum is $512.6m.
- Micro/nano share of shorts: 1.6% (unconstrained) → **0.22%** (mandate).

### Against the submitted first-challenge baseline ($500M)

The submitted book ran at **$500M** with no short floor: IR 1.07, Sharpe 1.83,
CAGR 13.5% (`README.md` results table). Most of the gap to this run is **fund
size**, not rules: the same strategy at $1bn already loses roughly half its
information ratio to capacity (README's $1bn row: IR 0.46), and the mandate
removes a further ~0.17 through the compounding liquidity path and the short
floor (table above). Defend the number that matters — the 4-factor alpha of
2.1%/yr (t 0.96) is marginal, and that is the weak spot of this book at size.

---

## 3. Attribution the Arcana review will ask for (Instructions 11)

`python -m arcana` → `output/report/arcana.json`.

| Lens | Finding |
|---|---|
| **Factor vs idiosyncratic** | Factor **61.8%** of mean return / 33.4% of variance. Idiosyncratic **38.2%** of mean / 66.6% of variance, **+2.13%/yr**. Loadings: mkt -0.038, size -0.095, value +0.098, momentum +0.112. Most of the *mean* is factor exposure (a momentum tilt and a small-cap short tilt) — §11 warns that a persistent tilt is "a cheap factor exposure with an expensive process attached"; the defence is that most of the *variance* is idiosyncratic. |
| **Sizing vs selection** | Selection **+6.70%/yr** (t 3.64) = 102% of P&L. Sizing **-0.14%/yr** (t **-2.44**) = -2%. **Sizing detracts** — equal-weighting the same picks would have done better. This is the construction finding to investigate before the event. |
| **Hit rate vs win/loss** | Positions 52.4% / 1.03. Months 54.4% / 1.09. A slight-edge-often book, not a fat-tail one; it fails differently from a 48%/1.6 book. |
| **Alpha decay** | Quintile spread 35%/yr (1m), **39% (2m)**, 38% (3m), 34% (6m), **26% (12m)**, all t > 4. The edge peaks at 2–3 months and is still large at a year — the empirical case for staying monthly (§4): daily trading would pay costs for signal that does not decay in days. |

---

## 4. Code changes (against `origin/main`)

| File | Change |
|---|---|
| `config.py` | `NAV = 1_000_000_000.0` with the mandate citation |
| `optimizer.py` | `SHORT_ME_FLOOR = 500.0` on the short leg; `me` carried through positions; compounding capital (`nav` scalar or per-month Series); micro-sleeve cutoff keyed to the endowed size |
| `main.py` | `nav_schedule()` + fixed-point loop so dollar limits follow the compounding fund |
| `combiner.py` | `me` plumbed into formation and prediction columns |
| `evaluate.py` | `capital_at()`, daily layer from `dsf_mini` (max one-day loss, daily beta), `adv_days_exhibit` on current capital |
| `report.py` | compliance to the mandate (200–500 names, short floor, 5% cap); `mandate_stats()` for the §9.1 rows; `holding_period()` per §9.3 |
| `graph/train.py` | return early when every fold is on disk (`--reuse` no longer needs the training GPU) |
| `arcana.py` | new; reports only, never trades |
| `MAIN.py` | standalone submission source regenerated from the modules above |
| `submission/holdings.csv`, `submission/returns.csv` | the mandate book (68 × 200 positions) |
| `tests/test_final_report.py`, `tests/test_final_liquidation.py`, `tests/test_final_timing.py` | updated for the new compliance key and the `me` column |

Nothing in the signal stack changed: sleeve scores are produced by the same
models and blend as the submission (`rank_ic_mean` 0.122 vs 0.130 in the
superseded pre-timing-fix draft; the difference is the timing fixes on
`origin/main`, not the mandate rules).

---

## 5. Defend / disclose list for the deck

1. **4-factor alpha 2.1%/yr (t 0.96)** — marginal. The headline IR at $1bn is the
   honest number; do not oversell.
2. **Sizing detracts (-2% of P&L, t -2.4)** — equal weighting would have done
   better. Say it before a judge finds it in Arcana.
3. **Gross at 100%** against the allowed 200% — deliberate, capacity-aware.
4. **Monthly rebalancing** — justified by the alpha-decay measurement.
5. **2023 and 2025 trail the hurdle** — with the economic reason.
6. **8-K corpus survivorship defect** (Instructions 6.4) — disclose and describe
   the handling, or rebuild from EDGAR/CA0032 through CIK.
7. **Cost and borrow assumptions** (Instructions 7 disclosure rule): the schedule
   used, what it does not capture (market impact, locate failures, recalls), and a
   flat conservative alternative alongside the headline.
8. **Not modelled:** market impact beyond the ADV participation limit, borrow
   availability, short recalls, financing spreads, intraday execution.
