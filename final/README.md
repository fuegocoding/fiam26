# FIAM 2026 market-neutral book (`final/`)

Each month this pipeline scores every U.S. stock in the competition panel, buys
the 100 best and shorts the 100 worst inside each sector, sizes the two legs so
the book is beta-neutral, applies tail-risk rules and writes the submission
files. It covers January 2021 to August 2026 (68 holding months).
Self-contained: nothing outside this folder is imported.

```
competition data ─┐
external data ────┼─> 10 sleeves ─> rank blend ─> graph tilt (5%) ─> identity labels
                  │                                                       │
                  └───────────────── borrow screen ─> optimizer ─> risk rules ─> book ─> evaluation + report
```

## Setup

| Requirement | Detail |
|---|---|
| Python | 3.13 |
| Packages | `pip install -r final/requirements.txt` (numpy, pandas, pyarrow, scipy, xgboost, torch, timesfm, matplotlib, pinned) |
| GPU | NVIDIA GPU with CUDA for TimesFM and the graph model; built on an RTX 3070 laptop (8 GB). `DEVICE = "cpu"` in `config.py` should also work, much slower; not tested |
| Memory | about 8 GB free; the largest step peaks at 7.2 GB |
| Disk | about 0.8 GB of competition files, 144 MB of external data, 0.6 GB of outputs |
| Network | first run only: the TimesFM 3 weights (`google/timesfm-3.0-pytorch`) download from Hugging Face into `HF_HOME` (default `~/.cache/huggingface`). Nothing else is downloaded |
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
- `DEVICE`: where TimesFM and the graph model run.
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

### 1. Ten return signals ("sleeves", `sleeves/`)

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
| `timesfm` | Zero-shot TimesFM 3 forecast of next-month excess return from the stock's own monthly return history (at least 48 months). The model sees anonymous return series: no ticker, name or date | price ≥ $1 | no (pretrained) |
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
  (`final_score_liquid`) is computed for books above $50M (micro-sleeve switch,
  below).

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
- **Missing returns:** a holding with no next-month return exits at its last
  price (zero return).
- **Benchmark:** T-bill + 4% a year. Information ratio = √12 × mean(active) /
  sd(active).
- **Statistics:** alpha and beta come from a regression on S&P 500 excess
  returns, with classical and HAC standard errors; a 4-factor version adds
  size, value and momentum built from the panel.
- **OOS R²:** each score is mapped to an expected return using only slopes
  already realized, then compared with a zero forecast.

## Time discipline (no look-ahead)

| Fitted model | Trains on target months | Validates / early-stops on | Predicts |
|---|---|---|---|
| `xgb_rank`, `filing_events`, graph model | 2015-02 … Y−3 | Y−2 … Y−1 | year Y (refit every year, 2021 … 2026) |
| `gap_micro_best` pooled models | formation months ≤ Y−3 | model choice on Y−2 … Y−1 | year Y |
| `gap_micro_best` cross-sectional models | other firms in the same month (5 firm folds, out-of-fold) | | the same month (the target is today's valuation, not a future return) |

- **Panel:** characteristics dated month t are used only for the return of
  month t+1.
- **Analyst data:** a feature uses only consensus snapshots and announcements
  dated on or before the month-end. Later announcements serve only as training
  labels.
- **SEC filings:** an 8-K or EDGAR filing in month m first counts for month m+1.
- **Short interest:** treated as public 12 days after settlement.
- **Labels:** the identity agent uses nothing dated after formation.
- **Universe:** stocks without a next-month return stay in the universe (no
  survivorship filter); they exit at their last price if held.
- **Risk rules:** they use returns up to the formation month only.
- **TimesFM:** a pretrained foundation model whose training data is not
  certified to end before 2021. It sees only anonymous return series. This is
  a disclosed limitation.
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

Out-of-sample test period January 2021 to August 2026 (68 months). Monthly
rebalancing, all risk rules on, 100 longs and 100 shorts every month. Returns
are on NAV and net of trading costs (5 to 60 bp per unit traded by size tier)
and borrow fees (25 to 1,500 bp a year by size tier); capital earns the
3-month T-bill. A stock with no next-month return exits at its last price.

| Metric | $10M | $100M | $250M | $500M | $750M | $1000M |
|---|---|---|---|---|---|---|
| Information ratio vs T-bill + 4% | **1.93** | **1.47** | **1.05** | **1.18** | **0.73** | **0.46** |
| Sharpe ratio (excess of T-bill) | 2.46 | 2.01 | 1.64 | 1.85 | 1.47 | 1.31 |
| CAGR (compounded annual return) | **23.5%** | **19.7%** | **15.0%** | **15.1%** | **11.7%** | **9.7%** |
| Cumulative return (Jan 2021 – Aug 2026) | 231.2% | 176.4% | 121.1% | 121.4% | 86.9% | 69.2% |
| Annualized active return (vs T-bill + 4%) | 14.3% | 11.1% | 7.0% | 7.0% | 3.9% | 2.1% |
| Annualized volatility | 7.4% | 7.5% | 6.7% | 5.9% | 5.4% | 4.7% |
| Max drawdown | -2.3% (Jul 2025) | -3.3% (Jan 2023) | -4.3% (Jan 2023) | -2.5% (Dec 2023) | -3.5% (Aug 2025) | -3.5% (Aug 2025) |
| Worst / best month | -2.2% / +8.6% | -3.3% / +8.5% | -3.2% / +7.1% | -2.0% / +6.9% | -2.3% / +4.8% | -2.7% / +4.0% |
| Months beating the hurdle | 63% | 65% | 62% | 59% | 60% | 60% |
| Alpha vs S&P 500 (CAPM) | 19.7%/yr (t 5.7) | 16.1%/yr (t 5.3) | 11.9%/yr (t 4.4) | 11.8%/yr (t 5.4) | 8.7%/yr (t 4.5) | 6.8%/yr (t 4.3) |
| Beta vs S&P 500 | -0.13 (SE 0.068) | -0.10 (SE 0.060) | -0.09 (SE 0.058) | -0.08 (SE 0.045) | -0.07 (SE 0.042) | -0.06 (SE 0.036) |
| Correlation with S&P 500 | -0.26 | -0.20 | -0.20 | -0.20 | -0.19 | -0.20 |
| 4-factor alpha (market, size, value, momentum) | 11.8%/yr (t 4.0) | 6.6%/yr (t 3.3) | 4.8%/yr (t 2.5) | 6.0%/yr (t 3.7) | 3.7%/yr (t 2.5) | 2.5%/yr (t 1.9) |
| 4-factor loadings (mkt / size / value / mom) | -0.08 / -0.34 / +0.26 / +0.13 | -0.02 / -0.31 / +0.24 / +0.22 | -0.02 / -0.25 / +0.16 / +0.17 | -0.02 / -0.20 / +0.12 / +0.15 | -0.02 / -0.15 / +0.09 / +0.14 | -0.01 / -0.10 / +0.06 / +0.13 |
| Trading costs / borrow fees per year | 2.1% / 1.9% | 1.7% / 1.5% | 1.4% / 1.0% | 1.1% / 0.7% | 1.0% / 0.5% | 0.9% / 0.4% |
| Gross exposure, average (range) | 81% (37%–100%) | 87% (38%–100%) | 87% (38%–100%) | 90% (37%–100%) | 89% (37%–100%) | 89% (37%–100%) |
| Net exposure, average (range) | +1.4% (-7.8% to +9.4%) | +1.1% (-9.4% to +8.7%) | +0.8% (-6.9% to +6.5%) | +0.0% (-13.5% to +7.2%) | +0.0% (-13.5% to +7.3%) | +0.2% (-11.5% to +6.9%) |
| Monthly turnover (of NAV) | 74% | 78% | 77% | 74% | 71% | 68% |
| Holdings per month (long / short) | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| Position size, average / max (% NAV) | 0.41% / 0.82% | 0.43% / 0.79% | 0.43% / 0.79% | 0.45% / 0.70% | 0.45% / 0.61% | 0.44% / 0.59% |
| Micro/nano share of short positions | 47% | 23% | 9% | 4% | 2% | 1% |
| OOS R² (zero benchmark, all scored stocks) | +0.06% | +0.03% | +0.03% | +0.03% | +0.03% | +0.03% |

| Year | $10M | $100M | $250M | $500M | $750M | $1000M | Hurdle (T-bill + 4%) | S&P 500 |
|---|---|---|---|---|---|---|---|---|
| 2021 | 23.3% | 26.9% | 21.1% | 16.9% | 16.5% | 11.1% | 4.1% | 26.9% |
| 2022 | 40.1% | 29.6% | 18.6% | 19.0% | 13.4% | 13.5% | 6.2% | -19.4% |
| 2023 | 21.8% | 10.2% | 11.5% | 9.9% | 7.1% | 3.8% | 9.5% | 24.2% |
| 2024 | 23.9% | 18.5% | 17.4% | 19.4% | 15.5% | 14.0% | 9.3% | 23.3% |
| 2025 | 15.9% | 18.4% | 13.0% | 14.2% | 8.8% | 7.0% | 8.4% | 16.4% |
| 2026 (to Aug) | 9.7% | 8.7% | 4.1% | 6.1% | 5.1% | 5.9% | 5.2% | 12.3% |

### Reading the table

- **Capacity.** A position must be tradable within five days at 5% (longs)
  or 2% (shorts) of daily dollar volume. As NAV grows, small and illiquid
  names drop out (micro/nano shorts fall from 47% to 1%) and the book drifts
  to large caps, where the signals are weaker. Above $50M the book is scored
  without the two micro-cap sleeves (see the switch section below).
- **Stock-specific alpha is largest at $10M and still visible to $500M.** The
  4-factor alpha is 11.8%/yr (t 4.0) at $10M, 6.6% (t 3.3) at $100M, 4.8%
  (t 2.5) at $250M, 6.0% (t 3.7) at $500M and 3.7% (t 2.5) at $750M; it is
  marginal at $1B (2.5%, t 1.9). The sizes above $10M carry more
  selection risk than the headline (the micro-sleeve switch was chosen on the
  test months), so treat them as the capacity curve, not as separate results.
- **Larger sizes are, if anything, optimistic.** Costs are a fixed rate per
  unit traded by size tier and do not grow with order size, so the extra
  market impact of trading millions per name is not charged.
- **Beta** is small and negative (-0.06 to -0.13) at every size. With HAC
  standard errors |t| is 1.6 to 1.9, so it is not significant at 5%, but it is
  close at $10M (t -1.9; -2.2 with classical standard errors). Say so on the
  neutrality slide rather than calling it zero.
- **Exposure.** Gross is at most 100% of NAV against the 200% limit; the
  risk rules hold the remainder as cash when tail risk is high or the
  signal's recent edge is weak.

### What changed from the 2026-10-01 book

Information ratio vs T-bill + 4% / CAGR / 4-factor alpha (t), same 68 test
months, same costs, all risk rules on:

| Book | $10M | $100M | $500M |
|---|---|---|---|
| 2026-10-01 book (nine sleeves, no borrow screen) | 1.53 / 18.8% / 7.7% (3.1) | 0.87 / 14.0% / 0.9% (0.3) | 0.52 / 10.8% / 0.8% (0.3) |
| + borrow screen only | 1.62 / 19.3% / 7.4% (3.1) | 1.01 / 14.3% / 2.2% (0.9) | 0.82 / 12.3% / 3.0% (1.5) |
| + filing-events sleeve only | 1.73 / 22.5% / 10.7% (3.5) | 0.97 / 15.2% / 3.4% (1.2) | 0.54 / 11.4% / 0.5% (0.2) |
| Both, scored with all ten sleeves | 1.78 / 22.3% / 10.4% (3.8) | 1.05 / 15.4% / 2.8% (1.1) | 0.79 / 12.5% / 1.8% (0.8) |
| Both + micro-sleeve switch | 1.78 / 22.3% / 10.4% (3.8) | 1.43 / 18.9% / 6.3% (3.1) | 1.14 / 14.8% / 4.9% (3.4) |
| **+ identity agent (this table)** | **1.93 / 23.5% / 11.8% (4.0)** | **1.47 / 19.7% / 6.6% (3.3)** | **1.18 / 15.1% / 6.0% (3.7)** |

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

## Micro-sleeve switch (`combiner.py`, `optimizer.py`)

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
The switch is on by default because the official-chain paired evidence is positive
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

## Turnover cap (tested, not adopted)

Question tested: does capping monthly turnover at 30% or 20% of NAV help? Turnover
here is buys plus sells as a share of NAV, the same measure as the 68% to 78% in
the results table. It does not help: the cap saves 0.5% to 1.5% a year of trading
costs but gives up 2% to 9% a year of return, and it lowers the information ratio
at every fund size except $1000M at 30%, where the gain is within noise (4-factor
alpha t-statistic unchanged at 1.9). The 20% cap removes the 4-factor alpha from
$100M up.

How the cap was applied:
- Each month the optimizer builds its book as described above. The trade is
  measured from the current holdings, which are last month's book drifted with
  its realized returns.
- If that trade exceeds the cap, the rank buffer is widened (1, 2, 4, 8 times
  the sector quota, then every eligible incumbent) until the trade fits. Any
  unused budget is spent moving toward the production book.
- If the widest buffer still trades more than the cap, the book moves only part
  of the way to its target, the same fraction for every position, so that it
  trades exactly the cap.
- The risk rules run on every candidate book.

Names being sold off stay in the book while they fade out, so a capped book holds
220 to 264 names on average (at most 362) instead of exactly 100 / 100, and gross
can reach 107% from drifted positions. Every capped book passes all the compliance
checks, and the no-cap column reproduces the results table exactly. The test used
a research script outside the submission (`output/turnover_cap/run_cap.py` in the
project repository); nothing in `final/` changes.

| Fund | IR: none / 30% / 20% | CAGR: none / 30% / 20% | 4-factor alpha (t): none / 30% / 20% | Trading costs per year: none / 30% / 20% |
|---|---|---|---|---|
| $10M | 1.93 / 1.29 / 1.09 | 23.5% / 17.4% / 14.9% | 11.8% (4.0) / 7.1% (3.3) / 2.9% (1.5) | 2.1% / 0.8% / 0.6% |
| $100M | 1.47 / 0.76 / 0.59 | 19.7% / 13.2% / 11.4% | 6.6% (3.3) / 1.3% (0.6) / -1.5% (-0.7) | 1.7% / 0.7% / 0.5% |
| $250M | 1.05 / 0.77 / 0.29 | 15.0% / 13.6% / 9.2% | 4.8% (2.5) / 2.1% (0.7) / -1.4% (-0.6) | 1.4% / 0.6% / 0.4% |
| $500M | 1.18 / 0.80 / 0.14 | 15.1% / 12.2% / 8.1% | 6.0% (3.7) / 2.7% (1.7) / -0.8% (-0.4) | 1.1% / 0.5% / 0.3% |
| $750M | 0.73 / 0.46 / -0.02 | 11.7% / 10.0% / 7.3% | 3.7% (2.5) / 2.7% (1.4) / 0.4% (0.2) | 1.0% / 0.4% / 0.3% |
| $1000M | 0.46 / 0.60 / 0.05 | 9.7% / 10.5% / 7.7% | 2.5% (1.9) / 3.2% (1.9) / 1.2% (0.8) | 0.9% / 0.4% / 0.2% |

Why it fails: the signals are monthly, so a position kept after its signal turns
earns little, and the short side loses most. At $10M, short-leg P&L falls from
14.7% a year with no cap to 9.2% at 30% and 7.2% at 20%, while long-leg P&L falls
from 7.6% to 6.6% and 5.9%. The risk rules also move gross by about 8% of NAV a
month, which uses part of the budget before any name is replaced.

Turnover is often quoted one-way (half of buys plus sells). Read that way, 20% to
30% one-way is 40% to 60% in the measure above, and the cost is much smaller. At
$10M:

| Metric | No cap | 60% cap | 40% cap |
|---|---|---|---|
| Information ratio vs T-bill + 4% | **1.93** | **1.83** | **1.63** |
| Sharpe ratio | 2.46 | 2.36 | 2.16 |
| CAGR | **23.5%** | **23.0%** | **21.1%** |
| Cumulative return | 231.2% | 223.8% | 195.9% |
| Active return per year | 14.3% | 13.9% | 12.3% |
| Volatility | 7.4% | 7.6% | 7.5% |
| Max drawdown | -2.3% (Jul 2025) | -2.5% (Jul 2025) | -2.1% (Jul 2025) |
| Worst / best month | -2.2% / +8.6% | -2.5% / +8.6% | -2.1% / +7.8% |
| Months beating the hurdle | 63% | 66% | 66% |
| Alpha vs S&P 500 (t) | 19.7% (5.7) | 19.2% (5.2) | 17.9% (4.9) |
| Beta vs S&P 500 (SE) | -0.13 (0.068) | -0.12 (0.072) | -0.15 (0.072) |
| Correlation with S&P 500 | -0.26 | -0.23 | -0.30 |
| 4-factor alpha (t) | 11.8% (4.0) | 11.3% (3.6) | 10.5% (3.8) |
| Trading costs / borrow per year | 2.1% / 1.9% | 1.6% / 1.9% | 1.1% / 1.9% |
| Gross exposure, avg (range) | 81% (37–100%) | 82% (37–100%) | 83% (37–100%) |
| Net exposure, avg (range) | +1.4% (-7.8% to +9.4%) | +1.6% (-8.0% to +8.1%) | +1.3% (-8.8% to +8.4%) |
| Monthly turnover | 74% | 56% | 39% |
| Holdings, avg long / short (max total) | 100 / 100 (200) | 108 / 107 (229) | 110 / 110 (267) |
| Position size, avg / max | 0.41% / 0.82% | 0.38% / 0.77% | 0.38% / 0.79% |
| Micro/nano share of shorts | 47% | 46% | 44% |
| Months the cap forced a partial move | – | 0% | 7% |
| Compliance checks | pass | pass | pass |
| Return 2021 (hurdle 4.1%, S&P 26.9%) | 23.3% | 22.9% | 21.5% |
| Return 2022 (hurdle 6.2%, S&P -19.4%) | 40.1% | 41.6% | 41.4% |
| Return 2023 (hurdle 9.5%, S&P 24.2%) | 21.8% | 22.2% | 19.6% |
| Return 2024 (hurdle 9.3%, S&P 23.3%) | 23.9% | 21.3% | 18.6% |
| Return 2025 (hurdle 8.4%, S&P 16.4%) | 15.9% | 15.5% | 12.5% |
| Return 2026 to Aug (hurdle 5.2%, S&P 12.3%) | 9.7% | 8.6% | 8.0% |

If lower turnover is wanted anyway, a 60% cap (30% one-way) costs about 0.1 of
information ratio at $10M. Full tables for the 30% and 20% caps, by fund size:

<details>
<summary>$10M fund: full table</summary>

| Metric | No cap | 30% cap | 20% cap |
|---|---|---|---|
| Information ratio vs T-bill + 4% | **1.93** | **1.29** | **1.09** |
| Sharpe ratio | 2.46 | 1.86 | 1.73 |
| CAGR | **23.5%** | **17.4%** | **14.9%** |
| Cumulative return | 231.2% | 148.7% | 120.0% |
| Active return per year | 14.3% | 9.1% | 6.9% |
| Volatility | 7.4% | 7.1% | 6.3% |
| Max drawdown | -2.3% (Jul 2025) | -2.9% (May 2026) | -2.9% (Dec 2024) |
| Worst / best month | -2.2% / +8.6% | -2.6% / +6.6% | -2.3% / +5.6% |
| Months beating the hurdle | 63% | 62% | 60% |
| Alpha vs S&P 500 (t) | 19.7% (5.7) | 15.0% (5.2) | 12.2% (4.6) |
| Beta vs S&P 500 (SE) | -0.13 (0.068) | -0.17 (0.062) | -0.13 (0.054) |
| Correlation with S&P 500 | -0.26 | -0.37 | -0.30 |
| 4-factor alpha (t) | 11.8% (4.0) | 7.1% (3.3) | 2.9% (1.5) |
| Trading costs / borrow per year | 2.1% / 1.9% | 0.8% / 1.7% | 0.6% / 1.6% |
| Gross exposure, avg (range) | 81% (37–100%) | 85% (37–102%) | 80% (37–106%) |
| Net exposure, avg (range) | +1.4% (-7.8% to +9.4%) | +3.2% (-7.3% to +11.1%) | +2.4% (-5.3% to +8.9%) |
| Monthly turnover | 74% | 30% | 20% |
| Holdings, avg long / short (max total) | 100 / 100 (200) | 120 / 130 (327) | 119 / 144 (348) |
| Position size, avg / max | 0.41% / 0.82% | 0.34% / 0.82% | 0.30% / 0.82% |
| Micro/nano share of shorts | 47% | 40% | 39% |
| Months the cap forced a partial move | – | 72% | 84% |
| Compliance checks | pass | pass | pass |
| Return 2021 (hurdle 4.1%, S&P 26.9%) | 23.3% | 19.7% | 16.0% |
| Return 2022 (hurdle 6.2%, S&P -19.4%) | 40.1% | 36.6% | 32.1% |
| Return 2023 (hurdle 9.5%, S&P 24.2%) | 21.8% | 14.2% | 14.1% |
| Return 2024 (hurdle 9.3%, S&P 23.3%) | 23.9% | 12.5% | 10.0% |
| Return 2025 (hurdle 8.4%, S&P 16.4%) | 15.9% | 13.3% | 6.7% |
| Return 2026 to Aug (hurdle 5.2%, S&P 12.3%) | 9.7% | 4.6% | 7.1% |

</details>

<details>
<summary>$100M fund: full table</summary>

| Metric | No cap | 30% cap | 20% cap |
|---|---|---|---|
| Information ratio vs T-bill + 4% | **1.47** | **0.76** | **0.59** |
| Sharpe ratio | 2.01 | 1.33 | 1.23 |
| CAGR | **19.7%** | **13.2%** | **11.4%** |
| Cumulative return | 176.4% | 101.7% | 84.1% |
| Active return per year | 11.1% | 5.4% | 3.7% |
| Volatility | 7.5% | 7.1% | 6.3% |
| Max drawdown | -3.3% (Jan 2023) | -4.4% (Aug 2022) | -5.8% (Aug 2022) |
| Worst / best month | -3.3% / +8.5% | -4.1% / +7.3% | -5.1% / +5.7% |
| Months beating the hurdle | 65% | 62% | 60% |
| Alpha vs S&P 500 (t) | 16.1% (5.3) | 10.8% (3.1) | 8.7% (2.8) |
| Beta vs S&P 500 (SE) | -0.10 (0.060) | -0.14 (0.070) | -0.09 (0.058) |
| Correlation with S&P 500 | -0.20 | -0.29 | -0.22 |
| 4-factor alpha (t) | 6.6% (3.3) | 1.3% (0.6) | -1.5% (-0.7) |
| Trading costs / borrow per year | 1.7% / 1.5% | 0.7% / 1.2% | 0.5% / 1.1% |
| Gross exposure, avg (range) | 87% (38–100%) | 82% (37–101%) | 84% (37–107%) |
| Net exposure, avg (range) | +1.1% (-9.4% to +8.7%) | +0.7% (-9.0% to +9.4%) | +2.3% (-5.8% to +7.6%) |
| Monthly turnover | 78% | 30% | 20% |
| Holdings, avg long / short (max total) | 100 / 100 (200) | 115 / 120 (281) | 123 / 141 (362) |
| Position size, avg / max | 0.43% / 0.79% | 0.35% / 0.82% | 0.32% / 0.87% |
| Micro/nano share of shorts | 23% | 18% | 15% |
| Months the cap forced a partial move | – | 55% | 88% |
| Compliance checks | pass | pass | pass |
| Return 2021 (hurdle 4.1%, S&P 26.9%) | 26.9% | 22.5% | 18.9% |
| Return 2022 (hurdle 6.2%, S&P -19.4%) | 29.6% | 24.3% | 19.6% |
| Return 2023 (hurdle 9.5%, S&P 24.2%) | 10.2% | 10.2% | 12.6% |
| Return 2024 (hurdle 9.3%, S&P 23.3%) | 18.5% | 11.2% | 5.1% |
| Return 2025 (hurdle 8.4%, S&P 16.4%) | 18.4% | 7.2% | 1.3% |
| Return 2026 to Aug (hurdle 5.2%, S&P 12.3%) | 8.7% | 0.8% | 8.0% |

</details>

<details>
<summary>$250M fund: full table</summary>

| Metric | No cap | 30% cap | 20% cap |
|---|---|---|---|
| Information ratio vs T-bill + 4% | **1.05** | **0.77** | **0.29** |
| Sharpe ratio | 1.64 | 1.31 | 0.97 |
| CAGR | **15.0%** | **13.6%** | **9.2%** |
| Cumulative return | 121.1% | 105.6% | 64.7% |
| Active return per year | 7.0% | 5.8% | 1.7% |
| Volatility | 6.7% | 7.5% | 5.9% |
| Max drawdown | -4.3% (Jan 2023) | -5.0% (Aug 2022) | -5.5% (Aug 2022) |
| Worst / best month | -3.2% / +7.1% | -4.8% / +6.9% | -4.0% / +5.3% |
| Months beating the hurdle | 62% | 59% | 56% |
| Alpha vs S&P 500 (t) | 11.9% (4.4) | 11.2% (3.3) | 7.0% (2.4) |
| Beta vs S&P 500 (SE) | -0.09 (0.058) | -0.14 (0.065) | -0.12 (0.050) |
| Correlation with S&P 500 | -0.20 | -0.27 | -0.32 |
| 4-factor alpha (t) | 4.8% (2.5) | 2.1% (0.7) | -1.4% (-0.6) |
| Trading costs / borrow per year | 1.4% / 1.0% | 0.6% / 0.9% | 0.4% / 0.8% |
| Gross exposure, avg (range) | 87% (38–100%) | 85% (37–102%) | 84% (37–105%) |
| Net exposure, avg (range) | +0.8% (-6.9% to +6.5%) | -0.6% (-10.0% to +4.9%) | -0.6% (-9.0% to +4.7%) |
| Monthly turnover | 77% | 30% | 20% |
| Holdings, avg long / short (max total) | 100 / 100 (200) | 112 / 114 (271) | 120 / 130 (302) |
| Position size, avg / max | 0.43% / 0.79% | 0.37% / 0.73% | 0.33% / 0.96% |
| Micro/nano share of shorts | 9% | 7% | 5% |
| Months the cap forced a partial move | – | 39% | 81% |
| Compliance checks | pass | pass | pass |
| Return 2021 (hurdle 4.1%, S&P 26.9%) | 21.1% | 20.8% | 18.0% |
| Return 2022 (hurdle 6.2%, S&P -19.4%) | 18.6% | 23.6% | 15.5% |
| Return 2023 (hurdle 9.5%, S&P 24.2%) | 11.5% | 8.1% | 8.6% |
| Return 2024 (hurdle 9.3%, S&P 23.3%) | 17.4% | 16.0% | 6.5% |
| Return 2025 (hurdle 8.4%, S&P 16.4%) | 13.0% | 8.9% | 2.5% |
| Return 2026 to Aug (hurdle 5.2%, S&P 12.3%) | 4.1% | 0.8% | 2.1% |

</details>

<details>
<summary>$500M fund: full table</summary>

| Metric | No cap | 30% cap | 20% cap |
|---|---|---|---|
| Information ratio vs T-bill + 4% | **1.18** | **0.80** | **0.14** |
| Sharpe ratio | 1.85 | 1.53 | 1.03 |
| CAGR | **15.1%** | **12.2%** | **8.1%** |
| Cumulative return | 121.4% | 92.1% | 55.4% |
| Active return per year | 7.0% | 4.4% | 0.6% |
| Volatility | 5.9% | 5.5% | 4.5% |
| Max drawdown | -2.5% (Dec 2023) | -2.5% (Jan 2023) | -5.6% (May 2026) |
| Worst / best month | -2.0% / +6.9% | -2.5% / +5.2% | -1.8% / +3.9% |
| Months beating the hurdle | 59% | 56% | 53% |
| Alpha vs S&P 500 (t) | 11.8% (5.4) | 9.5% (4.1) | 5.3% (2.6) |
| Beta vs S&P 500 (SE) | -0.08 (0.045) | -0.10 (0.045) | -0.07 (0.037) |
| Correlation with S&P 500 | -0.20 | -0.27 | -0.22 |
| 4-factor alpha (t) | 6.0% (3.7) | 2.7% (1.7) | -0.8% (-0.4) |
| Trading costs / borrow per year | 1.1% / 0.7% | 0.5% / 0.6% | 0.3% / 0.5% |
| Gross exposure, avg (range) | 90% (37–100%) | 88% (37–105%) | 85% (37–104%) |
| Net exposure, avg (range) | +0.0% (-13.5% to +7.2%) | -0.1% (-6.9% to +7.2%) | +1.6% (-5.9% to +7.2%) |
| Monthly turnover | 74% | 30% | 20% |
| Holdings, avg long / short (max total) | 100 / 100 (200) | 112 / 113 (247) | 116 / 120 (289) |
| Position size, avg / max | 0.45% / 0.70% | 0.39% / 0.80% | 0.36% / 0.70% |
| Micro/nano share of shorts | 4% | 3% | 3% |
| Months the cap forced a partial move | – | 40% | 76% |
| Compliance checks | pass | pass | pass |
| Return 2021 (hurdle 4.1%, S&P 26.9%) | 16.9% | 15.0% | 12.7% |
| Return 2022 (hurdle 6.2%, S&P -19.4%) | 19.0% | 19.5% | 15.5% |
| Return 2023 (hurdle 9.5%, S&P 24.2%) | 9.9% | 8.6% | 5.7% |
| Return 2024 (hurdle 9.3%, S&P 23.3%) | 19.4% | 15.2% | 11.2% |
| Return 2025 (hurdle 8.4%, S&P 16.4%) | 14.2% | 9.3% | 1.2% |
| Return 2026 to Aug (hurdle 5.2%, S&P 12.3%) | 6.1% | 2.2% | 0.4% |

</details>

<details>
<summary>$750M fund: full table</summary>

| Metric | No cap | 30% cap | 20% cap |
|---|---|---|---|
| Information ratio vs T-bill + 4% | **0.73** | **0.46** | **-0.02** |
| Sharpe ratio | 1.47 | 1.22 | 0.86 |
| CAGR | **11.7%** | **10.0%** | **7.3%** |
| Cumulative return | 86.9% | 71.9% | 49.4% |
| Active return per year | 3.9% | 2.4% | -0.1% |
| Volatility | 5.4% | 5.3% | 4.6% |
| Max drawdown | -3.5% (Aug 2025) | -3.6% (Aug 2025) | -5.2% (May 2026) |
| Worst / best month | -2.3% / +4.8% | -3.1% / +4.1% | -2.5% / +3.9% |
| Months beating the hurdle | 60% | 56% | 54% |
| Alpha vs S&P 500 (t) | 8.7% (4.5) | 7.5% (3.6) | 5.0% (2.5) |
| Beta vs S&P 500 (SE) | -0.07 (0.042) | -0.10 (0.040) | -0.10 (0.034) |
| Correlation with S&P 500 | -0.19 | -0.28 | -0.34 |
| 4-factor alpha (t) | 3.7% (2.5) | 2.7% (1.4) | 0.4% (0.2) |
| Trading costs / borrow per year | 1.0% / 0.5% | 0.4% / 0.4% | 0.3% / 0.4% |
| Gross exposure, avg (range) | 89% (37–100%) | 88% (37–101%) | 86% (37–103%) |
| Net exposure, avg (range) | +0.0% (-13.5% to +7.3%) | -0.7% (-8.9% to +6.1%) | -2.2% (-7.5% to +1.7%) |
| Monthly turnover | 71% | 30% | 20% |
| Holdings, avg long / short (max total) | 100 / 100 (200) | 110 / 110 (239) | 115 / 117 (263) |
| Position size, avg / max | 0.45% / 0.61% | 0.40% / 0.57% | 0.37% / 0.67% |
| Micro/nano share of shorts | 2% | 2% | 1% |
| Months the cap forced a partial move | – | 25% | 75% |
| Compliance checks | pass | pass | pass |
| Return 2021 (hurdle 4.1%, S&P 26.9%) | 16.5% | 14.4% | 12.6% |
| Return 2022 (hurdle 6.2%, S&P -19.4%) | 13.4% | 16.6% | 14.4% |
| Return 2023 (hurdle 9.5%, S&P 24.2%) | 7.1% | 6.1% | 7.1% |
| Return 2024 (hurdle 9.3%, S&P 23.3%) | 15.5% | 15.4% | 7.0% |
| Return 2025 (hurdle 8.4%, S&P 16.4%) | 8.8% | 5.0% | -0.0% |
| Return 2026 to Aug (hurdle 5.2%, S&P 12.3%) | 5.1% | 0.3% | 1.4% |

</details>

<details>
<summary>$1000M fund: full table</summary>

| Metric | No cap | 30% cap | 20% cap |
|---|---|---|---|
| Information ratio vs T-bill + 4% | **0.46** | **0.60** | **0.05** |
| Sharpe ratio | 1.31 | 1.44 | 1.16 |
| CAGR | **9.7%** | **10.5%** | **7.7%** |
| Cumulative return | 69.2% | 76.3% | 51.9% |
| Active return per year | 2.1% | 2.9% | 0.2% |
| Volatility | 4.7% | 4.8% | 3.6% |
| Max drawdown | -3.5% (Aug 2025) | -2.8% (Aug 2025) | -2.2% (Dec 2024) |
| Worst / best month | -2.7% / +4.0% | -2.5% / +3.9% | -1.4% / +4.0% |
| Months beating the hurdle | 60% | 59% | 47% |
| Alpha vs S&P 500 (t) | 6.8% (4.3) | 7.6% (4.9) | 4.9% (3.1) |
| Beta vs S&P 500 (SE) | -0.06 (0.036) | -0.07 (0.038) | -0.07 (0.031) |
| Correlation with S&P 500 | -0.20 | -0.21 | -0.30 |
| 4-factor alpha (t) | 2.5% (1.9) | 3.2% (1.9) | 1.2% (0.8) |
| Trading costs / borrow per year | 0.9% / 0.4% | 0.4% / 0.3% | 0.2% / 0.3% |
| Gross exposure, avg (range) | 89% (37–100%) | 88% (37–100%) | 87% (37–104%) |
| Net exposure, avg (range) | +0.2% (-11.5% to +6.9%) | -0.9% (-8.9% to +5.1%) | +0.8% (-7.3% to +5.1%) |
| Monthly turnover | 68% | 30% | 20% |
| Holdings, avg long / short (max total) | 100 / 100 (200) | 110 / 108 (237) | 121 / 120 (259) |
| Position size, avg / max | 0.44% / 0.59% | 0.40% / 0.61% | 0.36% / 0.67% |
| Micro/nano share of shorts | 1% | 1% | 1% |
| Months the cap forced a partial move | – | 18% | 78% |
| Compliance checks | pass | pass | pass |
| Return 2021 (hurdle 4.1%, S&P 26.9%) | 11.1% | 12.0% | 10.0% |
| Return 2022 (hurdle 6.2%, S&P -19.4%) | 13.5% | 14.3% | 14.6% |
| Return 2023 (hurdle 9.5%, S&P 24.2%) | 3.8% | 7.6% | 8.5% |
| Return 2024 (hurdle 9.3%, S&P 23.3%) | 14.0% | 13.8% | 5.1% |
| Return 2025 (hurdle 8.4%, S&P 16.4%) | 7.0% | 8.2% | 2.8% |
| Return 2026 to Aug (hurdle 5.2%, S&P 12.3%) | 5.9% | 3.9% | 2.9% |

</details>

## Reproducibility and memory

Every number above comes from a full rebuild with `python final/main.py`, run
on 2026-10-02 with no saved models: all ten sleeves, TimesFM, the graph model
and the book. The run reproduced the previous prepared run exactly: identical
sleeve scores on all 270,149 stock-months and identical statistics. Metrics
are computed by `evaluate.py`. The $10M column is `final/output/report/`; the
other sizes rerun the book and report at each NAV from the same predictions
(`output/nav_report.py`, results in `output/final_nav/`), and
`output/final_nav/readme_table.py` regenerates the two tables above.

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
