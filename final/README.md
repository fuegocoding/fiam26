# FIAM 2026 long/short equity strategy

A monthly U.S. equity pipeline that combines earnings, valuation, accounting,
characteristics and filing signals with a company graph. It builds a portfolio
of 100 longs and 100 shorts, applies liquidity and risk constraints, and produces
holdings, returns, performance statistics and charts.

The evaluation covers **January 2021–August 2026 (68 months)**. The default fund
size is **$500M**; results for $10M, $100M, $250M, $750M and $1B are shown below.

Data provenance and source tables: [docs/source.md](docs/source.md).

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
python MAIN.py          # rebuild all stages at $500M
python MAIN.py --reuse  # resume matching cached stages
```

New results are written to **`output_500M/report/`**, including
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
| `NAV` | Fund size in dollars | `500_000_000` |
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
| Holdings | 100 longs and 100 shorts at formation |
| Initial gross exposure | 100% of NAV |
| Minimum share price | $5 |
| Execution allowance | Five days at 5% of ADV for longs, 2% for shorts |
| Liquidity measure | 126-day average dollar volume |
| Short-interest screen | Exclude shorts at 15% or more; unknown values pass |
| Sector matching | Equal long and short name counts within each sector |
| Retention buffer | Keep incumbents within 1.5 times their sector selection quota |
| Beta sizing | Balance estimated leg betas within bounded notional adjustments |

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

## Results by fund size

Current configuration: **40% text / 35% fundamentals graph**, the **$50M sleeve
cutoff**, and the **linear edge throttle** described above. All six portfolios
cover 68 months and pass the portfolio compliance checks.

| Metric | $10M | $100M | $250M | $500M | $750M | $1B |
|---|---|---|---|---|---|---|
| Information ratio vs T-bill + 4% | **1.55** | **1.22** | **0.89** | **1.07** | **0.64** | **0.46** |
| Sharpe ratio (excess of T-bill) | 2.11 | 1.80 | 1.53 | 1.83 | 1.42 | 1.33 |
| CAGR (compounded annual return) | **19.8%** | **16.7%** | **13.3%** | **13.5%** | **11.0%** | **9.7%** |
| Cumulative return (Jan 2021 – Aug 2026) | 178.3% | 139.6% | 103.3% | 105.4% | 80.5% | 69.1% |
| Annualized active return (vs T-bill + 4%) | 11.2% | 8.5% | 5.5% | 5.6% | 3.3% | 2.1% |
| Annualized volatility | 7.2% | 6.9% | 6.2% | 5.2% | 5.1% | 4.6% |
| Maximum peak-to-trough drawdown (month-end NAV) | -2.9% (Jul 2025) | -2.3% (Jan 2023) | -2.3% (Jul 2025) | -1.8% (Jul 2025) | -3.3% (Jul 2025) | -3.5% (Aug 2025) |
| Maximum drawdown on deployed capital (gross-normalized) | -3.54% | -2.94% | -3.36% | -2.68% | -4.14% | -4.54% |
| Worst / best month | -2.9% / +8.9% | -2.3% / +9.2% | -1.5% / +7.9% | -1.7% / +5.6% | -2.1% / +4.3% | -2.0% / +4.0% |
| Months beating the hurdle | 65% | 57% | 56% | 60% | 60% | 60% |
| Alpha vs S&P 500 (CAPM) | 16.2%/yr (t 4.5) | 13.3%/yr (t 4.1) | 10.3%/yr (t 3.6) | 10.6%/yr (t 5.1) | 8.2%/yr (t 4.0) | 6.8%/yr (t 3.9) |
| Beta vs S&P 500 | -0.10 (SE 0.071) | -0.08 (SE 0.058) | -0.08 (SE 0.052) | -0.09 (SE 0.043) | -0.08 (SE 0.038) | -0.07 (SE 0.033) |
| Correlation with S&P 500 | -0.22 | -0.18 | -0.18 | -0.26 | -0.24 | -0.22 |
| 4-factor alpha (market, size, value, momentum) | 7.6%/yr (t 2.8) | 3.8%/yr (t 1.7) | 3.3%/yr (t 1.5) | 5.2%/yr (t 2.4) | 3.7%/yr (t 1.5) | 3.4%/yr (t 1.4) |
| 4-factor loadings (mkt / size / value / mom) | -0.06 / -0.30 / +0.32 / +0.13 | -0.01 / -0.28 / +0.29 / +0.20 | -0.03 / -0.26 / +0.21 / +0.14 | -0.05 / -0.18 / +0.13 / +0.12 | -0.04 / -0.17 / +0.10 / +0.11 | -0.04 / -0.14 / +0.07 / +0.09 |
| Trading costs / borrow fees per year | 2.0% / 1.6% | 1.6% / 1.3% | 1.4% / 0.9% | 1.1% / 0.6% | 1.0% / 0.4% | 0.9% / 0.3% |
| Gross exposure, average (range) | 75% (37%–100%) | 82% (37%–100%) | 85% (37%–100%) | 87% (37%–100%) | 87% (37%–100%) | 86% (37%–100%) |
| Net exposure, average (range) | +1.8% (-7.0% to +13.1%) | +2.0% (-5.4% to +8.9%) | +1.5% (-5.4% to +7.7%) | +0.9% (-8.3% to +8.4%) | +0.3% (-10.4% to +7.9%) | +0.3% (-9.0% to +6.6%) |
| Monthly turnover (of NAV) | 71% | 76% | 76% | 75% | 71% | 69% |
| Holdings per month (long / short) | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| Position size, average / max (% NAV) | 0.38% / 0.79% | 0.41% / 0.76% | 0.42% / 0.76% | 0.44% / 0.76% | 0.43% / 0.59% | 0.43% / 0.62% |
| Micro/nano share of short positions | 43% | 21% | 9% | 4% | 2% | 1% |
| OOS R² (zero benchmark, all scored stocks) | +0.05% | +0.02% | +0.02% | +0.02% | +0.02% | +0.02% |

Deployed-capital drawdown compounds monthly `(total return − T-bill return) /
gross exposure`, net of trading costs and borrow fees. Both drawdown metrics
use month-end observations and an initial index of 1.

| Year | $10M | $100M | $250M | $500M | $750M | $1B | Hurdle (T-bill + 4%) | S&P 500 |
|---|---|---|---|---|---|---|---|---|
| 2021 | 21.6% | 26.4% | 22.1% | 16.3% | 15.6% | 12.5% | 4.1% | 26.9% |
| 2022 | 36.5% | 26.4% | 15.8% | 17.0% | 13.0% | 11.4% | 6.2% | -19.4% |
| 2023 | 24.2% | 7.8% | 7.8% | 5.5% | 2.9% | 2.7% | 9.5% | 24.2% |
| 2024 | 11.8% | 10.3% | 13.0% | 16.9% | 15.6% | 13.0% | 9.3% | 23.3% |
| 2025 | 12.7% | 16.4% | 11.9% | 13.9% | 10.4% | 9.0% | 8.4% | 16.4% |
| 2026 (to Aug) | 7.2% | 8.4% | 5.4% | 7.5% | 5.1% | 6.6% | 5.2% | 12.3% |

## Outputs

The saved $500M report is in `output/report/`; new `MAIN.py` runs write
`output_500M/report/`. The tables above include results for all six fund sizes.

| Output | Contents |
|---|---|
| `holdings.csv` | Date, PERMNO, ticker, company name and weight as percent of NAV |
| `returns.csv`, `monthly_detail.csv` | Monthly returns, exposures, turnover and costs |
| `performance.json`, `comparison_metrics.csv`, `calendar_years.csv` | Aggregate statistics, strategy/benchmark/market comparison and annual returns |
| `compliance.json` | Portfolio checks |
| `liquidations.csv` | Missing-return exits, quote dates, prices and costs |
| `contributors.csv`, `top_average_holdings.csv` | Return contributions and average positions |
| `holdings_label_sources.csv` | Identifier and company-name provenance |
| `charts/` | Performance, drawdown, risk and contributor charts |

`output/` also contains saved models, `predictions.parquet`,
`portfolio.parquet`, `optimizer_config.json` and `optimizer_diagnostics.csv`.

## Submission

The submission code is `MAIN.py` at the project root. The saved **$500M**
holdings and returns CSVs are in `submission/`:

| File | Contents |
|---|---|
| `MAIN.py` (project root) | Complete Python source in one file; rebuilds models and reports from separately supplied inputs |
| `holdings.csv` | Monthly long/short holdings, beginning-of-month dates, tickers, full company names and signed weights in percent of NAV |
| `returns.csv` | Monthly total NAV returns in decimal units |

Run the single-file version with Python 3.13 and the dependencies above:

```bash
python MAIN.py
```

It rebuilds every stage at **$500M** and writes the report to
`output_500M/report/`. Place the three competition files in
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
The supplied `holdings.csv` and `returns.csv` are the saved $500M results;
newly generated versions are written inside `output_500M/report/`.

Add the eight-page slide deck and up to ten appendix pages in one PDF, plus
the team members' CVs. Submit `MAIN.py`, both CSVs in `submission/`, the slide PDF and CVs;
keep input data and fitted model caches outside the submission package.

The supporting performance pack is in `output/report/`: `performance.json`,
`comparison_metrics.csv`, `calendar_years.csv`, `top_average_holdings.csv`,
`contributors.csv`, the historical label sources and the six required charts.

## Project layout

| Path | Role |
|---|---|
| `MAIN.py` | Standalone submission entry point; full $500M rebuild |
| `main.py`, `config.py` | Entry point, paths and run settings |
| `sleeves/`, `combiner.py` | Signal models, monthly ranking and blend |
| `graph/` | Dated company graphs, attention model and score overlay |
| `identity.py`, `borrow.py` | Company labels and public short-interest data |
| `optimizer.py`, `risk.py` | Stock selection, sizing and risk rules |
| `evaluate.py`, `report.py` | Returns, costs, metrics, submission files and charts |
| `timing.py` | Prediction timing checks and cache provenance |
| `data/`, `external_data/` | Competition and external inputs |
