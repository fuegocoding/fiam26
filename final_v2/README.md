# Strategy pod platform

The governing design is [POD_FRAMEWORK_IMPLEMENTATION_SPEC.md](POD_FRAMEWORK_IMPLEMENTATION_SPEC.md).
Settings live in [config.json](config.json). Run `python final_v2/run.py` from the
repository root after installing `pip install -e './final_v2[research,test]'`.
Set `application.mode` to `demo` for synthetic source-to-book acceptance,
or `research` for configured data requests, multi-paper research and experiments.

Four private pods combine paper excerpts, original hypotheses and alternative
measurements through a reviewed feature graph. Carbon entity/metric discovery
lets a pod select its own stocks; recorded selections become bounded extraction
requests. Data receipts preserve missingness, source identity and receipt time.
Today's vendor response does not establish historical vintages.

Configure `models.roles` and enable `models.online` for actual inference. Roles
share persistent budgets and cached typed responses. Source metadata and public
quotes enter prompts; raw licensed observations, outcomes and credentials stay
local. Configure `research.measurement_grants`, `proposals` or `compositions`, and
hash-pinned `observation_sources` to run the actual registered graph evaluator.
Missing core measurements wait for data; missing optional controls are recorded.

An explicit `financial_reviews` mapping freezes the exact graph, targets, costs
and independent data snapshots. The existing integrity service decides admission.
Only admitted graphs can publish forecasts; actual market/risk inputs in
`portfolio` use the existing covariance, duplicate detection, shared hedge,
execution and accounting services. No real strategy is currently admitted.

`python -m pod_platform.services.app` starts the fixed pod configured in `mcp`.
Tools include paper discovery, multi-paper investigation/composition, alternative
data discovery and stock-specific requests. Tool arguments cannot grant financial
authority or choose a caller namespace. Utility scripts use matching config
sections and accept settings mappings when embedded; they do not parse flags.

Verify with `python -m pytest final_v2/tests -q -p no:cacheprovider` and
`ruff check final_v2/pod_platform final_v2/scripts final_v2/run.py`.
Formatting uses 80 columns, Google docstring conventions and module-level imports.
Private state is inside `final_v2/.pod_state`; the old root location is a
compatibility link for immutable historical receipt paths. Shared competition
inputs and credentials remain in the original repository data/local locations.
