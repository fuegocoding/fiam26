# Publication policy

This repository publishes the team's strategy source, final presentation,
selected aggregate backtest evidence and an earlier implementation for comparison.
The development history is retained with private files selectively removed
from historical trees. Original author identities, dates, messages and commit
relationships are preserved; affected commit hashes change.

## Excluded from the current tree

- Competition and licensed source data, raw vendor exports and per-security label histories.
- Credentials, personal documents, internal agent instructions and research handoffs.
- Editable/draft presentations, private review material and superseded portfolio exports.
- Trained models, forecast checkpoints, build caches and local run workspaces.

Ignore rules protect these local files from new commits. Commits that only
changed excluded files are retained even when they become empty. GitHub's
contributors graph excludes empty and merge commits, so displayed contribution
counts and line-change totals can differ after filtering even though the Git
author records remain. Do not merge unfiltered private history back into the
published branches.

## Evidence and checks

The published monthly returns reproduce the saved CAGR, cumulative return,
annual returns, month-end drawdown, Sharpe and information ratio. Result-file
hashes are recorded in `results/final_strategy/source.json`, and Git preserves
those evidence files byte for byte. The earlier implementation's 48 regression
tests passed during cleanup; the final strategy's open daily-accounting and
input-timing questions are documented in the results guide.

No new model training, independent holdout evaluation, or daily compliance
certification was performed during publication cleanup. Authorized input files
are required for full reproduction.

No new license has been selected on behalf of the team. Third-party source
data are not included in the repository.
