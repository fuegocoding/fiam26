# Publication policy

This repository publishes the team's strategy source, final presentation,
selected aggregate backtest evidence and an earlier implementation for comparison.
The original project history is retained alongside the current cleaned source tree.

## Excluded from the current tree

- Competition and licensed source data, raw vendor exports and per-security label histories.
- Credentials, personal documents, internal agent instructions and research handoffs.
- Editable/draft presentations, private review material and superseded portfolio exports.
- Trained models, forecast checkpoints, build caches and local run workspaces.

Ignore rules protect these local files from new commits. Earlier commits still
contain personal documents and vendor exports; the current file exclusions do
not remove historical objects. Removing those historical files while preserving
the development history would require a separate, selective history rewrite.

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
