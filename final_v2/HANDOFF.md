# Teammate handoff — 2026-10-25

Start here: [BUILD_STATUS.md](BUILD_STATUS.md), [SPEC_COVERAGE.md](SPEC_COVERAGE.md),
[VERIFICATION.md](VERIFICATION.md), then [README.md](README.md).
The governing `../POD_FRAMEWORK_IMPLEMENTATION_SPEC.md` remains unchanged.

## Continue from the integrated application

Read [AUTONOMOUS_PLATFORM.md](AUTONOMOUS_PLATFORM.md), then run the complete offline CLI below. Research, sealed check, publication, covariance/shared-hedge sizing, actual fills/NAV and diagnostic scheduling now share the same backend services. The latest official design campaign still rejects all four real candidates; no production strategy is admitted.

Model profiles are in [LLM_TRIAL.md](LLM_TRIAL.md). The Groq key is private local state. One live task passed on Groq gpt-oss and local Qwen8 with typed output; two earlier local output-contract failures are retained. The RTX3070Ti GPU works. Original Qwen thinking lead service, full benchmark, checkpoint/adoption evidence and production source evidence are still required. No automatic replacement or model adoption occurs.

## Run immediately

From `final_v2/`, using Python 3.11+:

```sh
python -m venv .venv
. .venv/bin/activate
pip install -e '.[research,test]'
python -m pytest -q -p no:cacheprovider
python scripts/run_complete_platform.py --state-dir /tmp/complete/state --output-dir /tmp/complete/out
python scripts/run_reference.py --mode demo --state-dir /tmp/reference/state --output-dir /tmp/reference/out
python scripts/run_strategy_research.py --fixture-demo /tmp/campaign/data --state-dir /tmp/campaign/state --summary
```

The reference workflow should report `software_status=PASS`,
`judge_status=COMPLETE_SOFTWARE_FIXTURE`, `investment_validation=UNPROVEN`, and
`prospective_status=INCOMPLETE`. Test certificates do not authorize a competition book.
Mock provider results do not establish LLM skill.

## Current real-data result

`artifacts/verification_20261025/real-campaign.json` contains the completed design-only
campaign: all four charters, 12 variants, no variant execution failures, all four rejected.
Do not promote these strategies by relaxing the gates. Improve the registered hypotheses
within a declared trial budget, then use independent protected challenge/admission.

The official daily CSV was imported locally as `../data/dsf_mini_from_2014.csv`.
It has 12,175,448 unique identifier/date observations and includes September 2026;
the evaluation cutoff stays August 31. The saved January 2021 daily check covers 19
trading days and an explicit toy holding/fill, with independently reconciled NAV. It
checks reporting only. Production fills, borrow and liquidity inputs remain necessary.

## Inputs and private state

Competition datasets, archive ZIP, external parquet, local environments, API keys and
`.pod_state/` are excluded from Git. Obtain the input files separately using the root
README layout. Local campaign/operator manifests are in `.pod_state/datasets/` on the
original laptop; use `configs/campaigns/competition_panel_manifest.example.json` to make
your own source/hash/schema-bound manifest. Approve only inputs whose evidence you can attest.

The current root layout preserves the submitted strategy alongside `final_v2/`; earlier
research was archived locally under ignored `stuff_tried/`. Git history retains earlier
tracked sources. Rebuild artifacts and machine-specific symlinks are not portable inputs.

## Arcana and models

Arcana OAuth successfully authenticated on the original laptop using:

```sh
codex mcp add arcana --url https://app.arcana.io/mcp
```

Each teammate authenticates separately. A new client session/reload may be needed to
load the server tools. Granted scopes were read-only notes, portfolio, metrics and assets.
No live portfolio upload or vendor-specific schema validation was performed. The local
CSV/JSON exporter remains review-only until a real vendor schema is supplied.
Official connection guide: https://help.arcana.io/en/articles/14998904-arcana-mcp-for-claude-and-chatgpt

Claude Code did most parallel implementation until its local quota was exhausted.
Existing live smoke evidence covers one valid draft and paper queries; the default demo
is offline. Real exception-agent usage requires exact checkpoint release/cutoff/adoption
records, an admitted policy and persistent call budget. Keep tools/code execution unavailable
to the model provider and never treat descriptive historical output as causal skill evidence.

## Next useful work

1. Verify Arcana tools in a new authenticated client and obtain its actual portfolio schema.
2. Resolve production source/vintage/identity cards and cost/borrow/liquidity evidence.
3. Research improved candidates with bounded declared variants; retain failed results.
4. Run independent protected challenges before any formation admission or deployment.
5. Freeze a genuinely admitted policy and collect future outcomes; historical fixtures
   cannot supply prospective investment or real-agent-skill evidence.

Completed-stage resume verifies hashes. Interrupted stages retain immutable attempts and
quarantined incomplete outputs; call/trial charges survive restart. Changed inputs/code require
a new run identity. Read the failure record before retrying rather than resetting private state.
