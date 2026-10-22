# FIAM 2026 long/short equity strategy

A monthly U.S. equity pipeline that combines earnings, valuation, accounting,
characteristics and filing signals with a company graph. It builds a portfolio
of 100 longs and 100 shorts, applies liquidity and risk constraints, and produces
holdings, returns, performance statistics and charts.

The evaluation covers **January 2021–August 2026 (68 months)**. Under the final
challenge mandate (Instructions 3.1, 3.3) the fund is endowed with **US$1bn at
the start of the test period and compounds from there**; the shortable universe
is restricted to names with **market capitalisation of at least US$500m at
formation**. Every dollar limit in the optimizer follows the compounding capital.

Data provenance and source tables: [docs/source.md](docs/source.md). The final
challenge rules, what they change and the mandate-compliant results are in
[docs/final_mandate_rules_map.md](docs/final_mandate_rules_map.md) and
[docs/RESULTS_AND_BASELINE_DELTA.md](docs/RESULTS_AND_BASELINE_DELTA.md).

## Setup

Use Python 3.13 and install the pinned dependencies from the project root:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`MAIN.py` selects CUDA when available and CPU otherwise. The modular runner
uses `DEVICE` in `config.py`. Both runners read local inputs.

Put these competition files in `data/`:

| File | Contents |
|---|---|
| `chars_final_with_names.parquet` | Monthly stock characteristics and identifiers |
| `factor_char_list.csv` | The 147 characteristic features |
| `8k_20150101_20260831_identified.parquet` | Linked 8-K filings |
| `dsf_mini_from_2014.csv` | Daily returns from 2014, for the daily-frequency metrics |

Place the external input file at **`external_data/external.parquet`**.
It contains I/B/E/S analyst records, FRED T-bill and S&P 500 data, FINRA short
interest and SEC EDGAR indexes. See [external_data/README.md](external_data/README.md)
for its source tables and rebuilding instructions. This file must be obtained
separately; `MAIN.py` does not automatically download it.

Input data and generated model/report directories are excluded from Git.
A fresh checkout needs these actual files; local symlinks to another workspace
are not supplied inputs. The required layout is:

```text
.
├── MAIN.py
├── requirements.txt
├── data/
│   ├── chars_final_with_names.parquet
│   ├── factor_char_list.csv
│   └── 8k_20150101_20260831_identified.parquet
└── external_data/
    └── external.parquet
```

## Run

After installing dependencies and placing the inputs above, run from the project root:

```bash
python MAIN.py          # rebuild all stages on the mandate rules
python MAIN.py --reuse  # resume matching cached stages
```

New results are written to **`output_1000M/report/`**, including
`holdings.csv`, `returns.csv`, performance statistics and charts. CUDA is
selected when available; otherwise the single-file runner uses CPU.

The modular entry point remains available from the project root:

```bash
python main.py           # rebuild into output/
python main.py --reuse   # reuse matching caches
```

`--reuse` requires the saved code and input manifest to match the current
pipeline. After a model or input change, use a full rebuild.

Settings are in [config.py](config.py):

| Setting | Purpose | Default |
|---|---|---|
| `NAV` | Endowed fund size in dollars (the fund compounds from it) | `1_000_000_000` |
| `DEVICE` | Model computation device | `"cuda"` |
| `DATA_DIR` | Competition input directory | `data/` |
| `WORK_DIR` | Models, forecasts and reports | `output/` |
| `TEST_YEARS`, `LAST_TARGET` | Holding-month evaluation window | 2021–August 2026 |

Model settings and portfolio rules are defined in the modules below.

## How it works

### Signals and blend

The pipeline has nine sleeves; the earnings sleeve supplies two signals:

| Sleeve | Signal | Native universe |
|---|---|---|
| `e7_lm`, `e7_small_gated` | Earnings surprise and post-report analyst revision breadth | Large/mega caps; small caps |
| `gap_micro_best` | Model-implied minus actual log valuation | Micro caps |
| `runway_micro` | Cash divided by operating cash burn | Micro caps with negative operating cash flow |
| `conf_lm_gated` | Consensus EPS revision that agrees with price momentum | Large/mega caps |
| `inventory_retail` | Sales growth minus inventory growth | Retail and consumer durables |
| `oplev_software` | Sales growth minus SG&A growth | Software and IT services |
| `xgb_rank` | XGBoost ranking the 147 characteristics | Stocks priced at $1 or more |
| `filing_events` | XGBoost on SEC event counts over 1, 3 and 12 months | Stocks priced at $1 or more |

Scores are ranked within each month and averaged equally over the sleeves that
cover a stock. Broad-universe scores use the sleeve's native scores as their
ranking reference where available.

**The fixed fund-size cutoff is $50M.** At or below $50M, all nine sleeves are
included. Above $50M, the blend excludes `gap_micro_best` and `runway_micro` and
uses `final_score_liquid`. Stock-level liquidity limits apply at every fund size.

### Company graph

Each company links to its 15 most similar peers. Similarity combines:

| Component | Weight |
|---|---:|
| Business text | 40% |
| Fundamentals | 35% |
| Industry | 20% |
| Recent filing events | 4% |
| Historical return co-movement | 1% |

Business text uses up to 60 months of available filings. A peer-attention model
forecasts the next earnings surprise, with return and risk prediction as auxiliary
tasks. Three seeds are averaged. The final score combines **95% sleeve rank and
5% graph forecast rank**; stocks without a graph forecast keep their sleeve rank.

### Portfolio construction and risk

| Rule | Setting |
|---|---|
| Holdings | 100 longs and 100 shorts at formation (mandate floor: 200–500 names) |
| Shortable universe | Market capitalisation ≥ $500m at formation (Instructions 3.1) |
| Initial gross exposure | 100% of NAV |
| Minimum share price | $5 |
| Execution allowance | Five days at 5% of ADV for longs, 2% for shorts |
| Liquidity measure | 126-day average dollar volume |
| Short-interest screen | Exclude shorts at 15% or more; unknown values pass |
| Sector matching | Equal long and short name counts within each sector |
| Retention buffer | Keep incumbents within 1.5 times their sector selection quota |
| Beta sizing | Balance estimated leg betas within bounded notional adjustments |
| Capital path | Endowed $1bn compounding; dollar limits sized on current capital (Instructions 3.3) |

Risk rules reduce crowded-short exposure, cap short run-ups and scale gross
exposure when historical expected shortfall increases. The edge throttle uses
the last 12 months' realized top-minus-bottom 5% score spread:

`edge scale = clip(0.5 + 0.25 × t-statistic, 0.5, 1.0)`

The multiplier is rounded to 25-point steps. Fewer than six usable months imply
full edge exposure. These rules use settled past returns; bounded beta sizing
does not guarantee zero realized market beta.

### Timing and evaluation

- Features at month-end t form the book held during month t+1.
- Annual learned models train through Y−3 and validate or early-stop on
  Y−2…Y−1 before predicting year Y. Earnings labels must be announced by the
  relevant training or validation cutoff.
- Analyst snapshots, company links and filing evidence are bounded by their
  availability dates. A filing in month m first contributes to month m+1.
- Short interest uses an explicit publication date where supplied; otherwise
  it is delayed ten business days, accounting for holidays.
- Previous holdings drift using the previous month's returns, cash rate and
  fees before opening turnover is calculated.
- Missing next-month returns do not remove stocks from the formation universe.
  A held stock with a missing return closes during that holding month at the
  latest positive quote dated no later than month-end. The exit price determines
  P&L; an exit fee is charged and the closed position leaves subsequent drift
  and retention. The trade is recorded in `liquidations.csv`.

Returns are on NAV, net of trading costs and borrow fees. Trading costs range
from 5 to 60 basis points per unit traded; annual borrow fees range from 25 to
1,500 basis points by size tier. Cash earns the three-month T-bill. The benchmark
is **T-bill + 4% annually**.

## Results (final mandate)

The book below is the submitted baseline rerun under the final mandate: the same
signals, walk-forward and risk rules, with the **US$500m short-side market-cap
floor** and the **compounding US$1bn endowment** in force. 68 months, 200 names
every month, all compliance checks pass (`output/report/compliance.json`).

| Metric | Mandate book |
|---|---|
| Information ratio vs T-bill + 4% | **0.36** |
| Sharpe ratio (excess of T-bill) | 1.27 |
| CAGR / cumulative return | 9.1% / 64.0% |
| Ending value of the endowment | **$1.64bn** (from $1.00bn) |
| Annualised volatility | 4.3% |
| Maximum drawdown | -4.0% |
| Maximum one-month / one-day loss | -2.0% (2026-05) / -1.3% (2022-02-15) |
| Alpha vs S&P 500 (CAPM, monthly beta) | 6.4%/yr (t 3.5); beta -0.079 (s.e. 0.034) |
| Beta vs S&P 500, daily | -0.070 (s.e. 0.008) |
| 4-factor alpha (market, size, value, momentum) | 2.1%/yr (t 1.0) |
| Trading costs / borrow fees per year | 0.8% / 0.2% |
| Gross exposure, average (range) | 88% (37%–100%) |
| Net exposure, average (range) | -0.2% (-6.8% to +6.8%) |
| One-way turnover (per rebalance / annualised) | 33.5% / 402% |
| Average holding period | 61 trading days |
| Holdings per month (long / short) | 100 / 100 |
| Position size, average / max | $5.9m / 0.57% NAV |
| Short book median market cap (minimum) | $13.7bn (**$512.6m**) |
| Micro/nano share of short positions | 0.2% |
| OOS R² / rank IC (all scored stocks) | +0.02% / 0.122 |

| Year | Strategy | Hurdle (T-bill + 4%) | S&P 500 |
|---|---|---|---|
| 2021 | 12.6% | 4.1% | 26.9% |
| 2022 | 11.5% | 6.2% | -19.4% |
| 2023 | 5.3% | 9.5% | 24.2% |
| 2024 | 11.9% | 9.3% | 23.3% |
| 2025 | 5.6% | 8.4% | 16.4% |
| 2026 (to Aug) | 5.0% | 5.2% | 12.3% |

What each new rule costs, and the attribution lenses the Arcana review asks for,
are in [docs/RESULTS_AND_BASELINE_DELTA.md](docs/RESULTS_AND_BASELINE_DELTA.md);
the rule map is [docs/final_mandate_rules_map.md](docs/final_mandate_rules_map.md).
The first-challenge submission ran at $500M with no short floor (IR 1.07, CAGR
13.5%); most of the gap to this table is fund size, not the rules.

## Outputs

The mandate report is in `output/report/`; new `MAIN.py` runs write
`output_1000M/report/`.

| Output | Contents |
|---|---|
| `holdings.csv` | Date, PERMNO, ticker, company name and weight as percent of NAV |
| `returns.csv`, `monthly_detail.csv` | Monthly returns, exposures, turnover and costs |
| `performance.json`, `comparison_metrics.csv`, `calendar_years.csv` | Aggregate statistics, strategy/benchmark/market comparison and annual returns |
| `compliance.json` | Portfolio checks |
| `liquidations.csv` | Missing-return exits, quote dates, prices and costs |
| `contributors.csv`, `top_average_holdings.csv` | Return contributions and average positions |
| `holdings_label_sources.csv` | Identifier and company-name provenance |
| `arcana.json` | Factor vs idiosyncratic split, sizing vs selection, hit rate, alpha decay (Instructions 11) |
| `position_adv_days.csv` | Position size in days of average daily volume, legs separate (Instructions 9.4) |
| `rule_deltas/` | Attribution runs: what the short floor and the compounding fund each cost |
| `charts/` | Performance, drawdown, risk and contributor charts |

`output/` also contains saved models, `predictions.parquet`,
`portfolio.parquet`, `optimizer_config.json` and `optimizer_diagnostics.csv`.

## Submission

The submission code is `MAIN.py` at the project root. The saved **mandate**
holdings and returns CSVs (US$1bn compounding, $500m short floor) are in
`submission/`.

| File | Contents |
|---|---|
| `MAIN.py` (project root) | Complete Python source in one file; rebuilds models and reports from separately supplied inputs |
| `holdings.csv` | Monthly long/short holdings, beginning-of-month dates, tickers, full company names and signed weights in percent of NAV |
| `returns.csv` | Monthly total NAV returns in decimal units |

Run the single-file version with Python 3.13 and the dependencies above:

```bash
python MAIN.py
```

It rebuilds every stage under the **mandate rules** and writes the report to
`output_1000M/report/`. Place the three competition files in
`data/` and `external.parquet` in `external_data/` beside `MAIN.py`. CUDA is
used when available; otherwise it uses CPU.

Optional path/device overrides:

```bash
python MAIN.py \
  --data-dir /path/to/competition/files \
  --external-dir /path/to/folder/containing/external.parquet \
  --work-dir /path/to/new/output --device cuda
```

A normal run rebuilds all models; `--reuse` resumes matching cached stages.
The supplied `holdings.csv` and `returns.csv` are the saved mandate results;
newly generated versions are written inside `output_1000M/report/`.

Add the eight-page slide deck and up to ten appendix pages in one PDF, plus
the team members' CVs. Submit `MAIN.py`, both CSVs in `submission/`, the slide PDF and CVs;
keep input data and fitted model caches outside the submission package.

The supporting performance pack is in `output/report/`: `performance.json`,
`comparison_metrics.csv`, `calendar_years.csv`, `top_average_holdings.csv`,
`contributors.csv`, the historical label sources and the six required charts.

## Project layout

| Path | Role |
|---|---|
| `MAIN.py` | Standalone submission entry point; full rebuild on the mandate rules |
| `main.py`, `config.py` | Entry point, paths and run settings |
| `arcana.py` | Attribution lenses the Arcana review will ask for; reports only |
| `sleeves/`, `combiner.py` | Signal models, monthly ranking and blend |
| `graph/` | Dated company graphs, attention model and score overlay |
| `identity.py`, `borrow.py` | Company labels and public short-interest data |
| `optimizer.py`, `risk.py` | Stock selection, sizing and risk rules |
| `evaluate.py`, `report.py` | Returns, costs, metrics, submission files and charts |
| `timing.py` | Prediction timing checks and cache provenance |
| `data/`, `external_data/` | Competition and external inputs |
