# Source-to-book application

The offline application now connects the actual services, rather than independent demos:

```text
approved paper/source evidence -> scheduler -> extractor -> lead -> challenger
 -> reviewed adaptation -> immutable registration -> bounded design executor
 -> frozen model -> sealed independent check -> scoped certificate/publication
 -> CIO covariance/overlap/shared hedge coordination -> independent risk
 -> orders/partial fills -> post-fee actual-book risk -> accounting/dossier
 -> matured diagnostic -> bounded research review
```

Run from `final_v2/` with the research dependencies installed:

```sh
python scripts/run_complete_platform.py --mode demo --state-dir /tmp/pods/state --output-dir /tmp/pods/out
python scripts/run_autonomous_pods.py --fixture --state-dir /tmp/research-only
```

Both commands perform finite explicit ticks. They start no daemon and make no model/network calls by default. Reuse the same paths to resume; inputs, source/code hashes, role profiles, registrations, budgets, check accesses, deliveries and completed outputs are bound. Changed inputs require a new state/run, rather than erasing failed attempts. Consumers must honor callback delivery keys: delivery can repeat if a process exits before committing its receipt.

The planted fixture researches all four charters. Fundamentals and dynamics actually pass measured independent criteria and publish; valuation remains shadow after failing the timing placebo; events fails for insufficient evaluable evidence. No acceptance label or forecast is substituted for an executor result. Two exact published books feed the CIO, including three requests (two pods and CIO) for one shared hedge. Thin hedge liquidity causes partial fills; the residual is recomputed, stale remainders cancelled and the actual post-fee book reviewed again. The dossier retains upstream paper/method/hypothesis/experiment/check/publication hashes and the order/fill/NAV chain. Losses queue diagnostic reviews, not automatic retraining.

The monthly return bridge uses out-of-training design predictions, declared turnover costs and measured residual dispersion. Native belief units/resolution remain separate. Current publication snapshots read only observable feature columns; no label column is loaded. A trusted host grants the feature window against the same immutable approved dataset; this grant is never an MCP model argument. Active forecasts with incompatible return horizons fail before allocation.

## Real input research

```sh
python scripts/run_complete_platform.py --mode research --run-id official_design_v1 \
  --manifest /path/operator-manifest.json --dataset-id comp_chars_design \
  --research-asof 2026-10-25T12:00:00+00:00 \
  --state-dir /tmp/official/state --output-dir /tmp/official/out
```

This runs the registered four-charter campaign and source classification. It deliberately stops before orders/admission. The latest official-panel campaign measured 12 variants, with all four candidates rejected and negative net bridge spreads. Improve registered hypotheses without weakening criteria. Real proxy-target recipes remain shadow even if design results improve; authentic target/source evidence and independent challenge are required.

## Models and source discovery

A live research tick requires `--online --manifest PATH --serving-profiles PATH` on `run_autonomous_pods.py`. Profiles must explicitly configure the exact lead, challenger and extractor roles; there is no fallback. Inject reviewed method/evidence events through `AutonomousResearchLoop.review_method` and `add_evidence` before the finite tick. Models can propose only reviewed executors/scalar parameters. Unknown methods, incomplete papers and unresolved sources require review. They cannot approve sources, change risk or read protected outcomes.

`discover_alt_sources.py QUERY --online` makes only read-only Carbon Arc `data_library` discovery calls through the existing authenticated MCP transport, reserving the persistent query budget before network access. Metadata discovery establishes neither raw-data entitlement nor historical vintages. No paid analytical tool is invoked. Register other trusted `SourceConnector` objects with `DiscoveryService`; callers cannot supply URLs, import paths or credentials.

`build_server(..., discovery=loop.discovery, research_runtime=loop)` exposes the same source service, own scheduler status and own research status under a host-fixed pod identity. Existing research submissions use the same queue and private memory. A model cannot invoke the four-pod tick, integrity authority, allocation or shell. Default MCP startup remains offline with no inference.

## What remains external

The framework software and fixture acceptance do not establish live investment readiness. The original Qwen thinking lead endpoint is still unconfigured. Groq gpt-oss and local Qwen8 each passed one bounded live task, not the full quality/adoption benchmark. Real source/vintage/identity and cost/borrow/liquidity evidence, successful protected real-candidate admission, live Arcana vendor compatibility and prospective outcomes remain necessary. Local GPU inference was verified on the RTX3070Ti; the full proposed model roster has not been served or admitted.
