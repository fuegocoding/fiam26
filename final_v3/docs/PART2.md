# Part 2 adaptation

The active target is `final_v3/main.py`. The existing sleeves, equal blend,
graph overlay and trading/borrow cost model are retained. No new alpha or
volatility-targeting policy is introduced.

At construction, the combined nonzero long/short book must span at least five
recognised GICS sectors. Signed net exposure in each sector must remain within
20% of NAV. Unknown classifications and infeasible selections fail explicitly.
The underlying selector still matches sector name counts and applies its beta
notional tilt; this is not exact dollar neutrality. This definition is exported.

Reporting exports `sector_exposures.csv`, `sector_exposures.png`, the
`standardised_performance.csv` table and a `part2_review.html` draft. Daily
returns, 252-session beta and daily/monthly reconciliation are also exported.
Annualised realised volatility uses the entire daily net series, sample standard
deviation and sqrt(252), with the agreed conservative 12% ceiling. Missing
daily inputs or held returns fail the audit. S&P coverage does not filter
portfolio returns out of the volatility calculation.

Treasury/S&P reporting comes from `config.REPORTING_FILE`: the FRED-only
`external_data/fred_reporting.parquet` when present, otherwise the `fred` rows
in the existing external container. The file needs `table,date,series,value`
and TB3MS/SP500 observations. Source provenance for a rebuilt FRED subset is
kept alongside its raw downloads. Panel-implied risk-free rates remain the
accounting conversion for supplied excess-return labels; the reporting hurdle
uses same-month TB3MS/1200 + .04/12. Market regressions use the actual S&P 500
price index, not the previous large-cap proxy.

Daily stock returns are total returns. Daily accounting deducts the monthly
trading cost on the first session and spreads the existing monthly borrow cost
across sessions. A separate reconciliation against the monthly ledger is
mandatory: mismatch is reported as a failure, not hidden by rescaling returns.
This adaptation does not certify daily/monthly equivalence or invent missing
quotes. Existing missing-monthly-return liquidation rules remain unchanged.

Carbon discovery now uses the existing project MCP/OAuth helper and team-account
credentials. It does not call the old REST SDK. The worker uses the current
Python (override with FIAM_CARBON_PYTHON if needed); dependencies include the
v2 helper's MCP, httpx and pydantic requirements. Old cache entries are preserved,
and the versioned keys prevent reusing REST results as new MCP receipts. Legacy
dollar quotes are explicitly unavailable through this adapter. Scouts still
cannot silently acquire new data; owned dated extracts remain usable.

`part2_status.json` and `compliance.json` expose individual checks.
`submission_ready` stays false while external and editorial deliverables are
pending. Arcana is deferred at the user's request: its HISTORICAL full-history
upload, original summary plus ten exhibits, STYLE EXPOSURE, S&P drawdown chart,
more than fifty scenarios and disagreement discussion remain required. Earnings
versus ex-earnings coverage, final deck/data screening and the agent
counterfactual remain pending too. Local plots are not Arcana substitutes.

The prior README metrics are historical and have not been regenerated under
these rules. A full rebuild needs the supplied inputs, the disclosed I/B/E/S
source and the configured training hardware. Source/input cache fingerprints
change with this adaptation; stale `--reuse` artifacts must not be relabelled
as new results. Development checks belong in the system temporary directory.
