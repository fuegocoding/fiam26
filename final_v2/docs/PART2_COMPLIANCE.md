# Part 2 competition workflow

The gap analysis in `docs/final_part2_gap_analysis.md` measured the older root
submission. Those passing results are not evidence that a `final_v2` campaign
passes. This workflow measures and reports the actual executed campaign.

## Run training, then freeze recipes for evaluation

Keep `config.json` as the existing through-2020 research configuration. Run it
through `python final_v2/run.py` with the research dependencies installed.
Supply the configured daily prices and benchmark inputs; check the panel and
filings paths against their locations in your checkout.

For the full competition period, select the separate configuration:

```powershell
$env:FIAM_CONFIG = 'C:\Code\fiam26\final_v2\competition_config.json'
python final_v2/run.py
```

The competition configuration reads the completed pre-2021 campaign identified
by `research_campaign.frozen_training_campaign`. The default points to the
training state's `latest_campaign.json`. Use a fixed `campaign_report.json`
path to pin a specific training campaign instead. The resolved report hash is
retained in the evaluation report.

Only model response receipts linked to evaluated training proposals are replayed.
The source report and configuration must end before January 2021 and explicitly
declare prior-settled evidence without retrospective selection. New agent
research and remote acquisition are disabled for this replay. Numerical refits
and allocation may use observations that have settled before each decision.
Recipes are not replaced by deterministic defaults if training is missing or
unsupported. A response with both accepted and rejected siblings is replayed in
full and each sibling is evaluated again with its original provenance.

Current pretrained-model knowledge, data vintages and prior inspection of the
historical window remain limitations; freezing recipes does not erase them.

The existing numerical score worker requires POSIX resource limits. Use the
supported Linux research environment; this change does not remove its isolation
or resource safeguards to make native Windows evaluation work.

## Enforcement and status

- At every accepted fund rebalance, at least five recognised GICS sectors must
  contain positions across the combined long and short book.
- The optimiser constrains absolute sector net weight to at most 20% of NAV.
  The composed target is checked after sleeve compression, and quote-restricted
  targets are checked again. Invalid targets are cancelled, with reasons, and
  actual carried positions remain visible. The solver does not invent positions
  to satisfy cardinality or breadth.
- Observed-price risk trimming includes sector nets, retaining fixed holdings
  without available quotes. Any unresolved breach stays in the daily audit.
- The existing 200–500-name, gross, net, single-name and short-cap requirements
  remain in force. Cash is a possible research outcome and fails the competition
  breadth/cardinality requirements.
- Full-period realised volatility uses daily NAV returns, sample standard
  deviation and annualisation by sqrt(252). The conservative acceptance ceiling
  is 12%. There is no volatility targeting or new sizing policy.

`constraints_pass` includes the new sector and volatility checks, independently
of the research-completion status. `submission_ready` remains false while the
deferred verification and presentation requirements are incomplete. Never infer
submission readiness from research completion.

## Outputs from the same executed portfolio

The normal campaign output directory includes:

- `sector_exposures.csv` and `sector_exposures_rebalances.csv`: long, short
  magnitude, signed net, gross and position count for all eleven sectors plus
  unknown classifications. Unknown classifications do not count toward breadth.
- `submission_status.json`: dated breaches, sector extrema, daily volatility,
  OOS coverage, input completeness and pending deliverables.
- `performance_table.csv`: the prescribed table, with monthly arithmetic moments
  separated from CAGR, daily and monthly beta, and explicitly unmodelled costs.
- `holdings_history.csv`: canonical dated identifiers, signed fractional weights,
  market cap in millions, dollar volume and position size in days of ADV.
  This is not an asserted Arcana upload schema; conversion awaits its account
  example. `actual_holdings.parquet` retains the original executed weights.
- Local SVG exhibits, `rolling_beta.csv`, portfolio history, annual returns,
  average-weight extremes and linked stock return contributors. Local plots
  are not labelled as independent Arcana verification.
- `submission_deck.html`: a portable, printable draft with the standard table,
  local exhibits and visible unfinished requirements. Keep it with its SVG files.
  The older root `strategy_deck.pdf` is not reused for this campaign.
- `submission_manifest.json`: hashes for generated evidence and references to
  the configuration, prompts and source-input hashes in `experiment.json`.

Average holding period is measured from closed notional in trading sessions;
open positions are excluded and the convention is labelled. Position size uses
absolute weights and actual compounded reporting NAV. Contributors are linked
to initial reporting NAV, including opening carried holdings where applicable.
Costs remain unmodelled under the existing full observed-price-fill convention;
net-of-all-cost information ratio is unavailable rather than falsely certified.

## Optional reporting inputs and pending work

Set `research_campaign.submission.expected_sessions_path` to a CSV with a `date`
column containing the official exchange/benchmark session calendar. Full OOS
coverage requires all expected sessions, all 68 months, the required boundary
sessions and prior-settled selection. Missing calendar evidence is not a pass.

Set `research_campaign.submission.earnings_calendar_path` to a dated CSV with
`permno` and `announcement_date`. The local analysis uses one observed trading
session before through one after the announcement. Overlapping windows are
counted once. A date-only event on a non-session maps to the next observed
session. Earnings and ex-earnings stock contributions reconcile; cash is separate.
Event completeness remains explicitly unverified until its historical coverage
is established. An 8-K filing is not automatically an earnings announcement.

Arcana is deferred at the user's request because credentials are unavailable.
When access returns, obtain the Upload History example, upload the entire OOS
history, and use HISTORICAL rather than Backtest. Capture the summary table and
ten original, labelled exhibits in brief order. Use STYLE EXPOSURE, add the
S&P 500 to drawdown, retain risk-model/window/unit definitions, review more than
50 Arcana scenarios and document attribution disagreements. No local substitute
can complete that verification.

The draft also identifies the remaining agent counterfactual/worked decision
example, data-screening answers, cost sensitivity and final deck review. A cost
model requires separate authorisation under `final_v2/AGENTS.md`.

## Validation

One-off checks belong in the system temporary directory, never in a repository
test directory. Exercise the actual runtime modules, including narrow-sector
target rejection, fixed-position offsets, observed-price sleeve reconciliation,
static replay, volatility boundaries, OOS session gaps, receipt provenance and
opening-holdings attribution. Run the existing repository regression suite too.
