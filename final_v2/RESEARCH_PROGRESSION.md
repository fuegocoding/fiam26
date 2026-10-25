# Component research before full strategy implementation

Research can start with **one mechanism, feature, method or diagnostic from a paper**, or a falsifiable idea from the pod's own reasoning. Full-paper replication and a complete trading dataset are not prerequisites. A useful negative or inconclusive result ends an experiment successfully without becoming a trading candidate.

| Stage | Evidence and work required | Authority |
|---|---|---|
| Idea / excerpt | One cached exact quote, including a pinned abstract, or an explicitly unverified original hypothesis | No performance or replication claim |
| Component probe | Reviewed executable transforms, the intrinsic feature/target/baseline data, one frozen DESIGN window and bounded trial | Descriptive component evidence only |
| Integrated design | Complete registered model/baseline comparison, authentic target, financial bridge and declared controls/cost assumptions | Existing design gates |
| Independent check / shadow | Frozen model, independent authority and existing integrity rules | Existing scoped admission gates |
| Trading | Eligible sources/models, admission, CIO/risk and executable data | Existing trading gates |

The challenger no longer vetoes every testable idea merely by saying `review` or `reject`. A host-reviewed comparison can run a bounded **falsification diagnostic** in those cases. Rejected ideas cannot schedule a full integrated design. Unsafe inputs, unknown methods and unsupported evidence still require review. A proposal with unmatched feature definitions can only test the explicitly recorded canonical component: its original mechanism remains **UNTESTED**. A positive proxy test never validates a missing core measurement.

The default envelope is at most three features, 12 months, 100 names and 5,000 loaded rows per probe, three probes per pod, four integrated designs per pod and 28 cases overall. Registration and counted reservations commit before rows are parsed. Policy/code/data/source bindings persist; failures consume budget. Identical executable measurements reuse their result across rewordings and different papers. Confidently negative evidence stops the same selected component, rather than unrelated features in the charter. Failed infrastructure and noisy results do not establish a negative mechanism result. These small-sample uncertainty labels are exploratory; they are not corrected investment significance tests.

## Minimal data

For the reviewed sales-growth/earnings diagnostic, a physical file needs only `permno`, `eom`, `sale_gr1` and `niq_su`. No stock-return label, sector, size, borrow, exposure or transaction-cost inputs are needed. Its research-only manifest can omit `label`, financial `evaluation` and unused column declarations entirely. Return-target recipes still intrinsically need their return label; sector transforms still need sector data. Optional controls are listed as omitted, never silently fabricated.

Only `ComponentProbeService` opts into research-only manifests. Dataset access must still be operator-authorized, access-confirmed, snapshot-bound and hash-pinned; rejected/quarantined sources stay unavailable. Ordinary built-in execution still requires complete financial metadata and an approved, formation-eligible source. The component report cannot create a fitted model, forecast, admission or order.

From `final_v2/`, with installed research dependencies:

```sh
python scripts/run_component_probe.py --manifest /path/research-manifest.json \
  --state-dir /path/private-probes --spec /path/probe.json
python scripts/run_component_probe.py --manifest /path/research-manifest.json \
  --state-dir /path/private-probes --spec /path/probe.json --run
```

The first command validates metadata without reading dataset rows. The second registers and executes. Example spec:

```json
{
  "namespace": "pod_fundamentals", "probe_name": "sales_diagnostic",
  "dataset_id": "earnings_sample", "recipe_id": "fund_earnings_innovation_v1",
  "feature_names": ["sale_gr1_rank"], "start": "2015-01-01", "end": "2015-12-31",
  "research_asof": "2026-10-25T20:00:00Z",
  "evidence_refs": ["hypothesis:unverified_sales_growth"]
}
```

The target is future native earnings and the cheap baseline is earnings persistence. `baseline_increment` compares their rank ICs; it does not fit an incremental model or establish alpha. `feature_target` tests the selected feature/target association against zero. No protected check/replay dates are accessible.

## Runtime and fixed-pod MCP

The trusted application host registers a reusable diagnostic capability with `loop.review_component(namespace, grant)`: `grant_id`, namespace, dataset id, reviewed recipe id/digest, one to three reviewed feature names, reviewer/time and `code_digest=loop.profile['code']`. This reviews the executable measurement once, independently of any particular paper. A new feature transform still needs implementation/review; models cannot upload arbitrary code.

`AutonomousResearchLoop(..., component_manifest_path=...)` optionally supplies a separate research-only dataset manifest; the ordinary `manifest_path` retains the application's financial execution inputs. Both are hash-bound to the run. A research dataset therefore need not acquire trading metadata merely to be available to a pod. `ProgressionPolicy(probe_only=True)` can additionally restrict the entire run to diagnostics; the captured-output replay always uses that policy.

With that same runtime injected into `build_server`, a pod can:

1. Read `research.component_capabilities` to see its grants and consumed budget.
2. Read a cached paper and select an exact excerpt with `papers.component_quote` (paper id/version, source, claim kind and verbatim quote). No full-paper claim checklist is needed.
3. Call `research.component_submit` with a grant id and the selected component's name/kind, paper id/version, claim ids, source and unavailable controls. Alternatively call `research.hypothesis_submit` with a grant id, hypothesis, falsification criterion, competing explanation and unavailable controls, without a paper.
4. Inspect `research.cycle_status` for immutable diagnostic results, exact tested subset and unresolved core dependencies.

The server injects the namespace and paper as-of time. MCP shares the runtime's paper cache and cannot register its own grant, choose a dataset/path/partition, change budgets or approve admission. Component submissions and original hypotheses retain their separate provenance even when they reuse a measurement. Standalone MCP without an injected runtime does not expose these execution tools.

## Actual saved-research replay

The previous 2025 study's captured Groq proposals and critics were replayed through the new policy on the real competition panel, using 2015 DESIGN observations. **Zero new model API calls, zero admissions, zero protected checks and zero orders.** Each probe loaded 1,195 rows with at most 100 names and 12 months.

| Canonical diagnostic | Evaluated months | Feature IC minus baseline IC | Exploratory result |
|---|---:|---:|---|
| Earnings innovation, core | 9 | −0.25248 | Negative |
| Valuation, core | 10 | +0.01631 | Noisy |
| Dynamics, core | 12 | −0.00485 | Noisy |

These are equal-weighted, unfitted component comparisons over a small first-encountered sample, not the earlier trained full-design results and not a 2025 performance backtest. Missing StockTwits sentiment, index CAPE and factor measurements remain untested. The saved events paper did not supply a reviewed issuance mechanism and is not converted into one.

To reproduce from the repository root, using a fresh state directory:

```sh
python final_v2/scripts/replay_component_research.py \
  --from-research .pod_state/year_trial_2025/live_v3/research \
  --state-dir .pod_state/component-replay-new \
  --manifest .pod_state/datasets/competition_design.json \
  --papers-db .pod_state/year_trial_2025/papers.sqlite
```

Saved inputs and full lineage remain private local state; summary evidence is in `artifacts/component_research/summary.json`. This reuses captured responses through an explicitly labelled replay provider, without calling Groq. It establishes policy/data plumbing, not live LLM quality. The original year-study history is preserved.
