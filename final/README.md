# FIAM 2026 long/short equity book (`final/`)

Each month this pipeline scores every U.S. stock in the competition panel, buys
the 100 best and shorts the 100 worst inside each sector. It applies bounded
beta sizing and tail-risk rules, then writes the submission files. It covers January 2021 to August 2026 (68 holding months).
Self-contained: nothing outside this folder is imported.

**Current version: committed strategy recipes plus timing fixes and explicit
last-known-price exits, refreshed 2026-10-04.** Annual validation-selection
changes were removed at the user's request. TimesFM is excluded from sleeves
and graph inputs. The committed $50M NAV sleeve switch is restored; its
historical test-period selection bias remains.
The fund-size tables below describe this version. Later ablations and research
comparisons remain historical. Prior use of the test period during research
and missing source-vintage timestamps prevent an unbiased-holdout claim.

```
competition data ─┐
external data ────┼─> 9 sleeves ─> rank blend ─> graph tilt (5%) ─> identity labels
                  │                                                       │
                  └───────────────── borrow screen ─> optimizer ─> risk rules ─> book ─> evaluation + report
```

## Setup

| Requirement | Detail |
|---|---|
| Python | 3.13 |
| Packages | `pip install -r final/requirements.txt` (numpy, pandas, pyarrow, scipy, xgboost, torch, timesfm, matplotlib, pinned) |
| GPU | NVIDIA GPU with CUDA for the fitted sleeves and graph model; built on an RTX 3070 laptop (8 GB). `DEVICE = "cpu"` in `config.py` should also work, much slower; not tested |
| Memory | about 8 GB free; the largest step peaks at 7.2 GB |
| Disk | about 0.8 GB of competition files, 144 MB of external data, 0.6 GB of outputs |
| Network | The active pipeline uses supplied local data; no pretrained checkpoint download |
| Time | about 1 h 45 min for a full rebuild on the laptop above; seconds with `--reuse` |

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r final/requirements.txt
```

Then put the three competition files in `final/data/` with these exact names:

| File | Contents |
|---|---|
| `chars_final_with_names.parquet` | Monthly characteristics panel, 01/2015–08/2026 |
| `factor_char_list.csv` | The 147 baseline characteristics |
| `8k_20150101_20260831_identified.parquet` | Linked 8-K filings |

All other inputs ship in `final/external_data/external.parquet`: I/B/E/S
analyst data, FRED T-bill and S&P 500, FINRA short interest and SEC EDGAR form
indexes. That folder's README says where each table comes from, and its
`build.py` regenerates the file from the original CSVs.

## Run

```bash
python final/main.py           # rebuild every model and recompute everything
python final/main.py --reuse   # keep what is already in final/output/; resume a stopped run
```

By default `main.py` empties `final/output/` and rebuilds everything. Every
stage saves its output, so `--reuse` skips finished stages, including after an
interrupted run. The rebuild is deterministic: two full runs give identical
sleeve scores and identical statistics.

Settings live in `config.py`; there are no other command-line options:
- `NAV`: fund size in dollars. Every dollar and liquidity limit derives from
  it. Default $10M.
- `DEVICE`: where fitted sleeves and the graph model run.
- `DATA_DIR`, `WORK_DIR`: where inputs are read and outputs written.
- `TEST_YEARS`, `LAST_TARGET`: the test window.

Model and rule constants (hyperparameters, thresholds) are at the top of the
module that uses them.

## Outputs

`final/output/report/` holds the submission files:
- **Required CSVs:** `holdings.csv` (Date, PERMNO, TICKER, COMPANY NAME,
  WEIGHT in % of NAV) and `returns.csv` (monthly total return).
- **Performance pack:** `performance.json` (every statistic the competition
  lists), plus `calendar_years.csv`, `top_average_holdings.csv`,
  `contributors.csv`, `holdings_label_sources.csv` and `compliance.json`.
- **Charts:** `charts/`, with cumulative, underwater, rolling active return
  and IR, rolling beta, the histogram with the hurdle, and contributors.

`final/output/` also keeps each stage's models and forecasts,
`predictions.parquet` (every stock's scores each month), `portfolio.parquet`
(the book) and `optimizer_diagnostics.csv`.

## How the model works

Timing convention: at each month-end t the pipeline uses only information
available at t and forms the book held over month t+1 (`return_month`).

### 1. Nine return signals ("sleeves", `sleeves/`)

Each sleeve gives every stock a score. Its *native universe* is where it is
meant to work, and the rest of the universe is scored too where data exist.
"Tradable" means price ≥ $5, dollar volume at or above the month's 20th
percentile, and a known sector.

| Sleeve | Signal | Native universe | Fitted? |
|---|---|---|---|
| `e7_lm`, `e7_small_gated` | Earnings surprise (I/B/E/S SUE) plus post-report analyst breadth (upgrades minus downgrades over estimates), each ranked within sector. Enters the month after the report month, once the first post-report consensus is public | tradable large/mega; tradable small | no |
| `gap_micro_best` | Valuation gap: model-implied minus actual log(market equity / assets). Candidate pricing models are peer multiples, ridge, linear levels, gradient-boosted trees, pooled trees, sector-theory ridge, a mixture of experts and an ensemble. Each year uses the one with the lowest median absolute pricing error on Y−2..Y−1 | tradable micro caps | yes |
| `runway_micro` | Cash runway: cash ÷ operating cash burn, for firms with negative operating cash flow | tradable micro caps | no |
| `conf_lm_gated` | Three-month revision of the nearest open quarter's consensus EPS, scaled by price, kept only when it agrees in sign with 3-to-1-month price momentum | tradable large/mega | no |
| `inventory_retail` | Sales growth minus inventory growth | tradable small/large/mega in retailing and consumer durables (GICS 2550, 2520) | no |
| `oplev_software` | Sales growth minus SG&A growth (operating leverage) | tradable small/large/mega in software and IT services (GICS 4510) | no |
| `xgb_rank` | XGBoost on the 147 characteristics, each ranked to [−1, 1] within the month. Target: the next-month return's percentile rank. Heavily regularized, 4 seeds averaged | price ≥ $1 | yes |
| `filing_events` | XGBoost on counts of SEC EDGAR filing events over 1, 3 and 12 months: late filings, comment letters, offerings, 13D/13G, proxy fights, deal filings and similar (details below). Target ranked within size group × sector, 4 seeds | price ≥ $1 | yes |

### 2. Scaling and blend (`combiner.py`)

- **Scaling:** each month a sleeve's native scores become uniform ranks,
  centred and scaled to unit variance ((rank − 0.5)·√12). A stock outside the
  native universe is ranked against the native scores; a stock the sleeve
  does not cover scores 0 and does not count.
- **Blend:** the sleeve score is the equal-weight average over the sleeves that
  cover the stock. There is no learned weighting: the sleeves have low
  pairwise correlation, and equal weights avoid fitting the mix on the test
  years.
- **Second score:** a version without the two micro-cap sleeves
  (`final_score_liquid`) is used above the committed $50M NAV threshold.
  This switch remains a fixed, historically test-selected design choice.

### 3. Graph overlay (`graph/`)

- **Graph:** a point-in-time company graph is rebuilt every month. Each company
  links to its 15 most similar peers. The similarity weights are 42% business
  text (8-K text over the last 60 months, hashed to 512 dimensions), 33%
  fundamentals, 20% industry (GICS), 4% recent 8-K event text and 1% return
  co-movement.
- **Model:** a small attention model reads each stock's own features and its
  peers' (an own-stock encoder plus a 4-head peer-attention adapter whose
  attention is biased toward the graph weights). It forecasts the next EPS
  surprise, alongside two auxiliary targets (next-month return rank and
  absolute-return rank).
- **Training:** three seeds (7, 19, 43), retrained every year on the
  walk-forward schedule, with early stopping on the validation years.
- **Use:** only a 5% tilt, `final_score = 0.95 · rank(sleeve blend) + 0.05 ·
  rank(graph EPS forecast)`. Stocks without a graph forecast keep their sleeve
  rank.

### 4. Identity labels (`identity.py`)

Every holding must be named by ticker and company name. CRSP labels in the
panel end in 12/2025, and the identity agent carries each 2026 stock's last
verified label forward, with its source (details below).

### 5. Book construction (`optimizer.py`)

1. **Eligibility:**
   - finite score, price ≥ $5 and a verified label;
   - the position (NAV × 50% / 100) must trade within 5 days at 5% (longs) or 2%
     (shorts) of 126-day average dollar volume;
   - shorts also need short interest under 15% of shares outstanding (the
     borrow screen, below).
2. **Sector matching:** names per GICS sector follow each sector's share of
   eligible stocks (sectors with at least 4). Every sector holds as many longs
   as shorts, so the book is dollar-neutral inside each sector.
3. **Selection:** the highest scores are bought and the lowest shorted. A
   current holding is kept while it ranks within the top 1.5 × k of its sector
   (rank buffer), which cuts turnover.
4. **Sizing:** equal weight within each side. The long and short notionals are
   set so the 60-month betas of the two legs offset (betas clipped to
   [0.2, 3], each leg kept within ±25% of half the gross). Gross starts at 100%
   of NAV.
5. **Risk rules** (`risk.py`, all on by default; each is a constant at the top
   of the file):

| Rule | What it does |
|---|---|
| Peer-EVT squeeze sizing | A GPD fitted to three years of industry x size peer returns gives each stock's chance of a +50% month; shorts in the riskiest 5% get half weight, freed weight goes to the sector's other shorts (capped at 1.5x an average short) |
| Run-up cap | A short that rallied last month is cut so its exposure stays within 1.5x target. Monthly in the backtest; live it runs daily at the close |
| Expected-shortfall limit | The book is replayed on its holdings' last 36 monthly returns; if its 95% expected shortfall is above the median the same book showed over the previous 24 month-ends, gross is cut in proportion (never up, never below half) |
| Edge throttle | Gross scales from 50% to 100%, in quarter steps, with the t-statistic of the last 12 months' realized top-minus-bottom 5% score spread |
| Own-EVT diagnostic | A GEV on each short's monthly maxima of daily returns gives its 1-in-100-month jump, reported in `portfolio.parquet` |

### 6. Evaluation and report (`evaluate.py`, `report.py`)

- **Rebalancing:** the book is rebalanced at the start of each month; the
  previous book drifts with its returns first, so turnover is measured against
  what is actually held.
- **Costs:** trading costs of 5 to 60 bp per unit traded and borrow fees of 25
  to 1,500 bp a year, both by size tier. Capital earns the 3-month T-bill.
- **Missing returns:** a stock selected at month-end t with no t+1 return
  closes during holding month t+1 at the latest positive panel price dated
  no later than that holding month's end. Price P&L is exit/opening price − 1,
  converted to excess return using that month's cash rate. Exit trading costs
  are charged and the closed position is removed from subsequent drift and
  the next retention buffer. `report/liquidations.csv` records the quote and
  trade. An unchanged last price can give zero price return; missing returns
  are never silently replaced by zero excess returns.
- **Benchmark:** T-bill + 4% a year. Information ratio = √12 × mean(active) /
  sd(active).
- **Statistics:** alpha and beta come from a regression on S&P 500 excess
  returns, with classical and HAC standard errors; a 4-factor version adds
  size, value and momentum built from the panel.
- **OOS R²:** each score is mapped to an expected return using only slopes
  already realized, then compared with a zero forecast.

## Time discipline and limitations

| Fitted model | Trains on target months | Validates / early-stops on | Predicts |
|---|---|---|---|
| `xgb_rank`, `filing_events`, graph model | 2015-02 … Y−3 | Y−2 … Y−1 | year Y (refit every year, 2021 … 2026) |
| `gap_micro_best` pooled models | formation months ≤ Y−3 | model choice on Y−2 … Y−1 | year Y |
| `gap_micro_best` cross-sectional models | other firms in the same month (5 firm folds, out-of-fold) | | the same month (the target is today's valuation, not a future return) |

- **Panel:** characteristics dated month t are used only for the return of
  month t+1.
- **Analyst data:** a feature uses only consensus snapshots and announcements
  dated on or before the month-end. Later announcements serve only as training
  labels. Later revisions cannot replace the first earnings report, and
  supplied ticker-link end dates are respected.
- **SEC filings:** a filing in month m first counts for month m+1. Each
  company's CIK/name mapping uses only evidence known at formation.
- **Short interest:** use explicit publication dates when supplied; otherwise
  delay ten business days, excluding federal holidays and Good Friday.
- **Labels:** the identity agent uses nothing dated after formation.
- **Universe:** stocks without a next-month return stay in the universe (no
  survivorship filter); they exit at their last price if held.
- **Risk rules:** they use returns up to the formation month only.
- **TimesFM:** excluded from the active sleeves and graph inputs.
- **Opening turnover:** previous holdings drift using the previous month's
  realized return, cash rate, costs and borrow fees; no current-month rate.
- **Guards:** no outcome columns in predictions; exact next-month targets;
  dated identity evidence; no predictions past August 2026; stale caches rejected.
- **Source vintages:** supplied extracts do not establish historical publication
  and revision timestamps for every value.
- **Parameter selection:** committed recipes are restored, with the original
  Y−2…Y−1 early-stopping windows. Blend weights are equal, graph tilt is 5%,
  borrow cap is 15% and risk/portfolio settings are fixed. They are not all
  validation-selected in this requested version.
- **Research choices:** several design choices (the borrow-screen cap, the
  filing-events source, the micro-sleeve switch) were made looking at
  2021–2026 results, so read their gains as upper estimates (see the notes
  below).

## Layout

| Path | Role |
|---|---|
| `main.py` | The whole process in one run |
| `config.py` | Paths and run settings |
| `external.py` | Reads one table of `external_data/external.parquet` |
| `combiner.py` | Runs the ten sleeves and the graph model and blends the scores |
| `identity.py` | Identity agent: point-in-time ticker and company-name labels, with sources |
| `optimizer.py` | Picks 100 longs and 100 shorts (sector quotas, rank buffer, liquidity limits, borrow screen, beta-neutral sizing), then calls `risk.protect` each month |
| `risk.py` | Risk rules; one public function, `protect` |
| `borrow.py` | Builds the point-in-time short-interest table the borrow screen reads |
| `evaluate.py` | Monthly evaluation under the rules (costs, borrow, T-bill cash, benchmark, IR, alpha, beta) |
| `report.py` | Submission CSVs, performance pack, compliance checks, OOS R², charts |
| `sleeves/` | The ten signals (one file, or one folder, per sleeve); `base.py` holds the sleeve contract and scaling, `ibes.py` the analyst-data loaders |
| `graph/` | Point-in-time company graph (`builder.py`, `business_text.py`), tokens (`dataset.py`), model (`model.py`, `train.py`) and the 5% overlay (`overlay.py`) |
| `data/` | The competition files (not included) |
| `external_data/` | `external.parquet` (I/B/E/S, FRED, FINRA, EDGAR), its README and `build.py` |
| `requirements.txt`, `ruff.toml` | Pinned packages; formatting and lint settings |

## Results by fund size

Evaluation period January 2021 to August 2026 (68 months). Monthly
rebalancing, committed risk settings, 100 longs and 100 shorts at formation.
Returns are on NAV, net of trading and borrow costs, with T-bill cash interest.
Missing holding-month returns trigger explicit last-quote liquidations and
exit fees. All six sizes pass portfolio compliance checks.

| Metric | $10M | $100M | $250M | $500M | $750M | $1000M |
|---|---|---|---|---|---|---|
| Information ratio vs T-bill + 4% | **1.55** | **1.22** | **0.91** | **1.08** | **0.64** | **0.44** |
| Sharpe ratio (excess of T-bill) | 2.11 | 1.80 | 1.55 | 1.84 | 1.42 | 1.31 |
| CAGR (compounded annual return) | **19.7%** | **16.7%** | **13.5%** | **13.6%** | **10.9%** | **9.6%** |
| Cumulative return (Jan 2021 – Aug 2026) | 176.7% | 140.1% | 104.7% | 105.9% | 80.0% | 68.4% |
| Annualized active return (vs T-bill + 4%) | 11.1% | 8.5% | 5.6% | 5.7% | 3.3% | 2.0% |
| Annualized volatility | 7.1% | 7.0% | 6.2% | 5.2% | 5.1% | 4.6% |
| Max drawdown | -2.9% (Jul 2025) | -2.3% (Jan 2023) | -2.3% (Jul 2025) | -1.8% (Dec 2023) | -3.3% (Jul 2025) | -3.6% (Aug 2025) |
| Worst / best month | -2.9% / +8.9% | -2.3% / +9.2% | -1.5% / +7.9% | -1.7% / +5.6% | -2.1% / +4.3% | -2.0% / +4.0% |
| Months beating the hurdle | 65% | 59% | 56% | 59% | 60% | 60% |
| Alpha vs S&P 500 (CAPM) | 16.1%/yr (t 4.5) | 13.3%/yr (t 4.1) | 10.4%/yr (t 3.7) | 10.6%/yr (t 5.1) | 8.1%/yr (t 4.0) | 6.8%/yr (t 3.8) |
| Beta vs S&P 500 | -0.10 (SE 0.072) | -0.08 (SE 0.058) | -0.08 (SE 0.052) | -0.09 (SE 0.042) | -0.08 (SE 0.038) | -0.07 (SE 0.033) |
| Correlation with S&P 500 | -0.21 | -0.18 | -0.19 | -0.26 | -0.24 | -0.22 |
| 4-factor alpha (market, size, value, momentum) | 7.5%/yr (t 2.8) | 3.9%/yr (t 1.7) | 3.5%/yr (t 1.6) | 5.2%/yr (t 2.5) | 3.6%/yr (t 1.5) | 3.2%/yr (t 1.3) |
| 4-factor loadings (mkt / size / value / mom) | -0.06 / -0.30 / +0.32 / +0.13 | -0.01 / -0.29 / +0.28 / +0.20 | -0.03 / -0.26 / +0.21 / +0.14 | -0.04 / -0.18 / +0.14 / +0.13 | -0.04 / -0.17 / +0.10 / +0.11 | -0.03 / -0.14 / +0.07 / +0.09 |
| Trading costs / borrow fees per year | 2.0% / 1.6% | 1.6% / 1.3% | 1.4% / 0.9% | 1.1% / 0.6% | 1.0% / 0.4% | 0.9% / 0.3% |
| Gross exposure, average (range) | 75% (37%–100%) | 82% (37%–100%) | 85% (37%–100%) | 87% (37%–100%) | 87% (37%–100%) | 86% (37%–100%) |
| Net exposure, average (range) | +1.8% (-6.8% to +13.1%) | +2.1% (-5.4% to +8.9%) | +1.5% (-5.4% to +8.0%) | +0.9% (-8.3% to +8.4%) | +0.3% (-10.3% to +7.9%) | +0.3% (-9.2% to +6.5%) |
| Monthly turnover (of NAV) | 71% | 76% | 76% | 75% | 71% | 69% |
| Holdings per month (long / short) | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| Position size, average / max (% NAV) | 0.38% / 0.79% | 0.41% / 0.76% | 0.42% / 0.76% | 0.44% / 0.76% | 0.43% / 0.59% | 0.43% / 0.62% |
| Micro/nano share of short positions | 43% | 21% | 9% | 4% | 2% | 1% |
| OOS R² (zero benchmark, all scored stocks) | +0.05% | +0.02% | +0.02% | +0.02% | +0.02% | +0.02% |

| Year | $10M | $100M | $250M | $500M | $750M | $1000M | Hurdle (T-bill + 4%) | S&P 500 |
|---|---|---|---|---|---|---|---|---|
| 2021 | 21.2% | 26.4% | 22.1% | 16.3% | 15.5% | 12.5% | 4.1% | 26.9% |
| 2022 | 36.5% | 26.4% | 15.9% | 16.9% | 13.0% | 11.4% | 6.2% | -19.4% |
| 2023 | 23.5% | 7.8% | 7.8% | 5.5% | 2.9% | 2.7% | 9.5% | 24.2% |
| 2024 | 11.9% | 10.8% | 13.3% | 17.3% | 15.5% | 13.0% | 9.3% | 23.3% |
| 2025 | 12.9% | 16.3% | 12.2% | 14.1% | 10.3% | 8.8% | 8.4% | 16.4% |
| 2026 (to Aug) | 7.2% | 8.3% | 5.5% | 7.3% | 5.1% | 6.5% | 5.2% | 12.3% |

### Reading the table

- **Capacity:** above $50M the committed blend drops the two micro-cap
  sleeves. Formation-time liquidity limits further restrict tradable names.
- **Costs:** fixed rates by size tier include normal trading and liquidation
  fees. Additional order-size market impact is not modeled.
- **Drawdown:** measured on monthly NAV and anchored to initial NAV; it does
  not measure daily or intramonth losses.
- **Settlement:** the last available quote is a valuation convention; it does
  not establish executable proceeds or recover unavailable delisting returns.
- **Interpretation:** timing corrections do not erase test-period research
  selections or certify historical source vintages.

### Historical changes from the 2026-10-01 book

All comparisons below predate the current timing and liquidation corrections.
They are retained as research history, not current performance.

Information ratio vs T-bill + 4% / CAGR / 4-factor alpha (t), same 68 test
months, same costs, all risk rules on:

| Book | $10M | $100M | $500M |
|---|---|---|---|
| 2026-10-01 book (nine sleeves, no borrow screen) | 1.53 / 18.8% / 7.7% (3.1) | 0.87 / 14.0% / 0.9% (0.3) | 0.52 / 10.8% / 0.8% (0.3) |
| + borrow screen only | 1.62 / 19.3% / 7.4% (3.1) | 1.01 / 14.3% / 2.2% (0.9) | 0.82 / 12.3% / 3.0% (1.5) |
| + filing-events sleeve only | 1.73 / 22.5% / 10.7% (3.5) | 0.97 / 15.2% / 3.4% (1.2) | 0.54 / 11.4% / 0.5% (0.2) |
| Both, scored with all ten sleeves | 1.78 / 22.3% / 10.4% (3.8) | 1.05 / 15.4% / 2.8% (1.1) | 0.79 / 12.5% / 1.8% (0.8) |
| Both + micro-sleeve switch | 1.78 / 22.3% / 10.4% (3.8) | 1.43 / 18.9% / 6.3% (3.1) | 1.14 / 14.8% / 4.9% (3.4) |
| **+ identity agent (historical book)** | **1.93 / 23.5% / 11.8% (4.0)** | **1.47 / 19.7% / 6.6% (3.3)** | **1.18 / 15.1% / 6.0% (3.7)** |

The borrow screen is what helps at $500M (0.52 to 0.82, CAGR +1.5 points):
it removes the shorts a large book could not borrow. At $10M it adds 0.1 of
information ratio and 0.5 points of CAGR. The filing-events sleeve is what helps
at $10M and $100M and is worth little at $500M and above. The micro-sleeve
switch is what lifts $100M to $1B (next section). Compared with the 2026-10-01
book the new one is ahead in 5 of 6 calendar years at $10M (all but 2025,
-0.1 point), and in 6 of 6 at $100M and at $500M. The identity agent changes only Feb-Aug 2026
(next section). None of the three return changes was
chosen walk-forward (see the selection caveats above), so read the gains as
upper estimates. The 2021 to 2026 months were used to pick the screen's cap,
the sleeve's source and the switch.

## Borrow screen (`borrow.py`, `optimizer.py`)

A stock is never shorted when 15% or more of its shares outstanding are
already sold short (`SHORT_INTEREST_CAP`). FINRA publishes short interest
twice a month for every U.S. equity; a settlement date is treated as public
12 days later. Stocks with no short-interest figure pass the screen. The
`short_blocked` column of `optimizer_diagnostics.csv` counts the shorts
excluded each month (about 110 stocks at $10M). The cap was picked from the
ladder 5%, 7.5%, 10%, 15%, 20%, 30% by looking at the 2021 to 2026 test
months: 15% and 20% improved the book, 5% to 10% hurt it, 30% was neutral.
Treat 15% as a judgement, not a walk-forward choice.

The same ladder at larger sizes (research harness, with the micro-sleeve switch above
$50M; information ratio, none then 5% / 7.5% / 10% / 15% / 20% / 30%): $100M 1.48 /
1.36 / 1.64 / 1.55 / 1.62 / 1.59 / 1.49; $500M 1.10 / 1.02 / 1.33 / 1.42 / 1.34 / 1.26 /
1.10; $1B 0.62 / 0.34 / 0.78 / 0.74 / 0.72 / 0.66 / 0.65. Any cap from 7.5% to 20% beats
no screen at those sizes and 5% hurts everywhere, so 15% is a middle value, not a
finely tuned one.

What it does and does not do: in the research harness the screen leaves the
book's return roughly flat (-0.2%/yr at 15%) and raises the information ratio
by cutting volatility and drawdown (maximum drawdown -5.2% to -3.4%). It
raised return in only 2 of the 6 test years (+0.9 and +2.1 points) and lowered
it in the other 4. Its purpose is to keep the short leg inside stocks that can
be borrowed. The cost model charges a fixed fee by size tier and nothing extra
for scarce borrow, so the table understates that benefit. The larger gain at
$500M below is consistent with the same effect when the book is
capacity-bound, though that is an interpretation, not a separate test.

What it cannot do: it removes crowded shorts, not unborrowable ones. Little short
interest in a micro cap often means there are no shares to lend, so a low reading
does not make the stock borrowable. Example: shorting stocks with an 8-K Item
3.02 (unregistered equity sale) earns +24%/yr (t 6.0, positive every year) among
stocks under 15% short interest, but almost all of that is micro and nano caps;
without them it is +11.7% (t 2.7) and about zero in 2024 and 2025. Treat any
micro-cap short result as unproven until real locate data says otherwise.

What it removes: without the screen, 11.8% of short gross at $10M and 17.6% at
$500M sits in stocks with 15% or more of their shares already short (6.2% and
9.4% at 15-20%, 5.6% and 8.2% above 20%). With it the share-weighted mean short
interest of the short book falls from 6.9% to 4.8% at $10M and from 8.8% to
6.0% at $500M.

## Filing-events sleeve (`sleeves/filing_events/`)

Ranks stocks inside their size group and GICS sector from counts of SEC
EDGAR filing events over the last 1, 3 and 12 months: late-filing notices
(NT 10-K / 10-Q), SEC comment letters and company replies, offerings and
shelf registrations, Schedule 13D/13G stakes, proxy contests, deal proxies
and tender offers, 10-K/A amendments, deregistrations, spin-offs, Form D,
Form 144, lock-up expiries and similar. The model is the characteristics
XGBoost recipe (rank target, 4 seeds, train through Y-3, validate Y-2..Y-1)
with the target ranked within size group and sector, so it cannot earn
anything from a size or sector tilt. It adds one equal-weight score to the
blend.

Where it works: micro caps (top-minus-bottom decile spread +39%/yr, t 5.4)
and small caps (+24%/yr, t 3.3); large and mega caps are not significant
(+8%, +4%). Health care, consumer discretionary and industrials are the
strongest sectors (t 4 to 5); financials (t 1.3) and utilities (t 0.6) are
not significant. That is why it helps at $10M to $100M and not above.

Checks run before it went in (all on the 2021 to 2026 test months): a
shuffled-feature placebo makes the blend worse by 0.3 to 0.5 information
ratio, while the real sleeve adds 0.20 to 0.24 across three independent seed
sets; the gain survives an extra 1 to 3 months of publication lag, dropping
deregistration and deal filings or comment letters, and restricting to stocks
with a verified company link (details in
`docs/research/alt_sources_20261002.md`). Selection caveat: EDGAR was the
best of six sources tried (the others added nothing significant or hurt), and
that choice was made looking at the test months, so the +0.2 is an upper
estimate of what to expect live. The 25 form families were also grouped after
the event studies had been seen: the same model on all 74 form types, nothing
chosen, keeps the standalone return (13.5%/yr against 13.9%) but adds +0.12 to
the blend (paired t 2.0) instead of +0.20, so expect about +0.12 at $10M.

Sub-sample checks of the sleeve's gain to the blend (research harness, paired monthly
difference in active return): with every blank-check-named stock vetoed from the book
+0.31 IR (it does not rely on SPAC shells); excluding 2021 +3.4%/yr (t 2.4, 56
months); from 2023 onward +2.4%/yr (t 1.5, 44 months), smaller, consistent with the
fade described next.

Friction stress: with micro, nano and small-cap trading costs and borrow fees at
2x, 3x and 4x the assumed rates, the sleeve's gain to the blend is +0.23, +0.31
and +0.37 (from +0.20), so it is not an artifact of cheap micro-cap frictions.
The whole book is far more sensitive: the blend's information ratio at $10M falls
from 1.77 (1.97 with the sleeve, research harness) to 0.87 at 3x and 0.42 at 4x
(1.18 and 0.78 with the sleeve). A book that holds 45% of its shorts in micro and
nano caps is only as good as those cost and borrow assumptions.

Regime dependence (read this before trusting the +0.2): the financing-event
effects that carry the sleeve (offerings, shelf, registration effectiveness,
comment letters) were roughly zero in 2016 to 2018 and 2019 to 2020, strong in
2021 to 2022 and weaker from 2023. Shorting stocks with an offering in the last
three months earned +2.5%/yr (t 1.1) in 2016-18, -2.2% (t -0.5) in 2019-20,
+20.8% (t 3.3) in 2021-22 and +12.2% (t 3.3) in 2023-26 across all eligible
stocks, and only +2.1% (t 0.6) in 2023-26 among borrowable stocks. Late-filing
notices were the one family positive in every period. The sleeve's own gain to
the blend by test year, in points of active return, was +6.1, +7.4, +2.3, +1.7,
+0.3 for 2021 to 2025 and +6.5 for the first eight months of 2026. A fading
effect is the likeliest reason to expect less than the table shows live.

## Identity agent (`identity.py`)

The deck must name every holding by ticker and full company name, so the
optimizer only trades stocks with a verified point-in-time label. The supplied
panel's CRSP labels end in 12/2025; for 2026 it labels a stock-month only when
the company filed an 8-K in that same month. Without help the 2026 book could
only pick from "this month's 8-K filers", about 60% of the universe
(candidates fall from ~3,600 to 1,600-2,500 from February 2026). The
competition asks teams to keep stock-months without a filing (or document the
restriction) and to verify labels from an authoritative source rather than
drop the holding.

For each unlabelled stock-month after the last CRSP month, the agent takes the
issuer's last verified label dated on or before formation (at most a year
old), re-dates it to a newer 8-K cover page when that cover lists the same
trading symbol (taking the SEC issuer name on that filing), and records the
source and date (`holdings_label_sources.csv`). It uses nothing dated after
formation and never substitutes a later name. Earlier names stay with earlier
holdings: `top_average_holdings.csv` and `contributors.csv` list every label a
stock carried while held (e.g. DWAC before DJT, FB before META).

- **Labels:** agree with the SEC filing records on 99.6% of names and 99.8% of
  tickers.
- **Book:** only February-August 2026 change, and all seven months improve. At $10M
  2026 goes from +3.6% to +9.7% (IR 1.78 to 1.93); every NAV gains (table above).
- **Mechanism check:** imposing the same "same-month filers only" restriction
  on 2021-2025, where it never applied, cuts the $10M information ratio from
  1.78 to 0.83 and raises turnover from 76% to 109% a month. The 2026 weakness
  was this universe artefact, not a fading signal.

Research and checks: `docs/research/council_v2_20261002_results.md`.

## Historical micro-sleeve switch (`combiner.py`, `optimizer.py`)

The committed switch is restored in this timing-and-liquidation-only version.
The comparisons below are historical; the current results are in the fund-size
tables above.

Two sleeves, `gap_micro_best` and `runway_micro`, are built for micro caps.
Above `MICRO_SLEEVES_MAX_NAV` ($50M) the optimizer reads `final_score_liquid`,
the same blend and graph tilt computed without them; at or below $50M it reads
`final_score` (all ten sleeves), so the $10M book is unchanged. Both columns are
written to `predictions.parquet`; `final_score` is bit-identical to before.

Evidence (2021 to 2026 test months, official chain, paired monthly difference in
active return, switch on minus off): $100M +3.1%/yr (HAC t 2.7, better in 6 of 6
years), $250M +1.9% (t 2.2, 5 of 6), $500M +2.1% (t 1.6, 5 of 6), $1B +0.6%
(t 0.5), and at $10M -0.2% (t -0.2), which is why it is off there. Information
ratio: $100M 1.05 to 1.43, $250M 0.74 to 1.04, $500M 0.79 to 1.14, $1B 0.29 to
0.41.

How much to believe it: this was found by looking, not pre-registered. A
drop-one table (each of ten sleeves at five fund sizes) showed that removing
almost any single sleeve moved the information ratio by 0.1 or more, so
differences of that size are within noise; the micro pair stood out because it
was the only pair with a capacity reason and a same-sign effect at four sizes.
Selection check: ranking all 45 possible pairs of sleeves by the paired monthly
difference from dropping them (research harness, no risk rules), the micro pair
ranks 1st at $100M, 15th at $250M and 2nd at $500M, and 2nd of 45 by mean t over
the three sizes. That makes pure luck unlikely but the harness t-statistics are
small (0.9, 0.4, 1.9), and almost every top pair contains `runway_micro`, so that
sleeve is probably doing most of the work.

A warning sign: dropping either micro sleeve alone at $100M lowers the information
ratio (1.31 to 1.18 without `runway_micro`, to 1.16 without `gap_micro_best`; research
harness), and only dropping both raises it (1.48). A gain that appears only for the
pair, with opposite-signed singles, is what noise or a fragile interaction looks like.
Historically, the switch was enabled because the official-chain paired evidence is positive
at four of five sizes; for the conservative book that ignores it, set
`MICRO_SLEEVES_MAX_NAV = float("inf")` in `optimizer.py` (the by-NAV table then
reads 1.78 / 1.05 / 0.74 / 0.79 / 0.54 / 0.29 from $10M to $1B).

The mechanism is only partly supported: `runway_micro` has no large-cap signal
(top-minus-bottom spread -1.2%, t -0.2), but `gap_micro_best` does (+9.8%,
t 1.3). A general "each sleeve scores only its own native universe" rule is
clearly worse (information ratio 1.65 vs 1.98 at $10M, 0.29 vs 0.80 at $500M), so
the broad scoring stays for every other sleeve. The $50M threshold is a
judgement between two tested sizes. Treat the $100M to $500M gain as probable but
overstated, and the $10M headline as unaffected.

## Reproducibility and memory

The current fund-size tables were rebuilt on 2026-10-04 from the committed
strategy recipes with timing and liquidation corrections. Timing-safe model
artifacts were reused; reassembly reproduced all 270,149 saved timing-corrected
forecasts exactly (normalizing timestamp precision for comparison). All six
portfolios and reports were recalculated for explicit exits.

The $10M results are in `final/output/report/`; the other sizes are in
`output/final_nav/`. `final/output/timing_exit_protocol.json` records the active
protocol and reuse hashes. The preceding validation-selected outputs and source
recovery are in `output/final_timing_exit_20261004/`. Timing/report/liquidation
regression checks: 48 passed. Runtime measurements below describe the older
full pipeline, including TimesFM.

Peak memory by stage, each built in its own process on a 16 GB laptop:

| Stage | Peak RAM | Time |
|---|---|---|
| e7 analyst panel | 7.2 GB | 1 min |
| gap_micro_best valuation trees | 4.8 GB | 33 min |
| conf_lm_gated panel | 3.7 GB | 1 min |
| TimesFM forecasts (GPU) | 3.5 GB | 22 min |
| xgb_rank | 4.9 GB | 2 min |
| filing events | 4.2 GB | 2 min |
| graph text, snapshots, tokens, training (GPU) | 1.7–3.4 GB | 40 min |
| blend, identity labels, book, report | 3.9 GB | 1 min |

`main.py` runs the stages in one process and needs about 8 GB of free memory
(the e7 panel peaks at 7.2 GB). Every stage saves its output, so if a run is
interrupted, `python final/main.py --reuse` resumes where it stopped.
