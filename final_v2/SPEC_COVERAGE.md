The new integrated acceptance path and explicit external gates are in [AUTONOMOUS_PLATFORM.md](AUTONOMOUS_PLATFORM.md). `tests/test_autonomous_pods.py` and `tests/test_complete_platform.py` exercise actual registered services, admitted/rejected outcomes, shared hedges, fills/NAV, feedback and restart/tamper denial. These extend the bounded component map below; no full investment-readiness claim is made.

# Specification implementation coverage

Local software acceptance map; `tested_bounded` means executable positive/negative assertions, never investment validation or file-presence completeness. Exact code/test paths are in [the machine ledger](configs/spec_coverage.json).

Evaluation cutoff: **2026-08-31**. Official `data/dsf_mini_from_2014.csv` is available; post-cutoff rows are excluded, missing held returns never zero-filled. Parent independently validated a real 19-day supplied-data accounting smoke; this is not strategy evidence.

| Package | Deliverable | Software status | Acceptance and limits |
|---|---|---|---|
| W01 | Typed contracts and manifest | tested_bounded | Valid/invalid contracts and sealed manifest invariants; external official rule attestation still required. |
| W02 | Source importer/catalogue | tested_bounded | Archive and catalogue fixture hashes/counts are checked. Unresolved source cards remain ineligible. |
| W03 | Security master/availability | tested_bounded | Dated mappings, revision selection and availability checks; production entity coverage remains data-dependent. |
| W04 | Broker/access boundaries | tested_bounded | Future and protected-partition access denied in adversarial fixtures. |
| W05 | Registry/jobs | tested_bounded | Immutable registrations/jobs and completed-stage hash verification; immutable attempt IDs bind inputs before execution. Interrupted owned outputs are quarantined without deleting evidence. Crash/retry and hard process-exit tests preserve trial reservations and budget charges. |
| W06 | Baseline/features | tested_bounded | Reviewed operators and registered bounded experiments; no generated code execution. |
| W07 | Accounting/fills | tested_bounded | Hand-calculated cash, short proceeds, borrow, fees, NAV compounding; missing held returns fail closed. |
| W08 | Risk/constraints | tested_bounded | Feasible/infeasible books and measured post-fill reviews. |
| W09 | Integration/construction | tested_bounded | Only certified published forecasts trade; rejected candidates and stale publications excluded. |
| W10 | Execution/costs | tested_bounded | Participation, partial fills, pending orders and accounting integration; actual market impact requires real calibration. |
| W11 | Reporting/exports | tested_bounded | Exact accounting identities, factor/idiosyncratic and selection/sizing split, event/decay prerequisites. Daily fixture book is separate from monthly replay; no inferred daily history. |
| W12 | Paper/method service | tested_bounded | Full-text claims -> reviewed adaptation -> preregistered executor -> frozen model -> sealed challenge/publication; incomplete evidence fails closed. |
| W13 | Diagnostics/private memory | tested_bounded | Persisted failure taxonomy, queue and prior-attempt retrieval APIs; weekly/matured-monthly scheduling and feedback reviews are now connected (see `test_research_scheduler.py` and `test_complete_platform.py`). |
| W14 | Four charters | tested_bounded | Four registered measured candidates; demo contains actual rejection. Real campaign design results never fabricate admission. |
| W15 | Challenge/promotion | tested_bounded | Baseline/placebo/check/stability/multiplicity criteria and enforceable environment/source/expiry restrictions. Synthetic evidence cannot issue formation certificates. |
| W16 | Agent exception/twin | tested_bounded | Two full self-financing ControlledReplay arms on identical bound fixture inputs include all periods, path-dependent fills, fees, borrow, cash and risk. Injectable model decisions require sealed admitted checkpoint/adoption, temporal/context validation, durable cache and persistent budget. Default and mock results never establish LLM skill. |
| W17 | Dossier/Arcana | tested_bounded | Actual position/date dossier and governing decision chain. Local Arcana review export, explicit schema mapping and injected transport/receipt checks exist. OAuth is configured by operator; live vendor schema and upload are not verified in this session. |
| W18 | Freeze/prospective | tested_bounded | Actual timestamp, output/code/model hashes, source-bound stage resume and change detection; actual strategy-removal controls and labelled cost/stress sensitivity. Future scoring requires later supplied observations; no prospective investment result is claimed. |
| W19 | Model registry/router/adoption | tested_bounded | Checkpoint temporal denial, adoption and durable call/variant budgets tested. Actual exact-checkpoint evidence and fixed-task adoption comparison require operator/provider evidence. |

| Core section | Software status | Assessment |
|---|---|---|
| 1 | tested_bounded | Scope enforced by fixture/formation environments and evaluation cutoff. |
| 2 | tested_bounded | Software constraints/statistics implemented; official eligibility remains governed by operator mandate. |
| 3 | tested_bounded | Fixed startup roles and private namespace guards; MCP cannot issue admission. |
| 4 | tested_bounded | One measured registered candidate for every charter; unsupported ideas remain rejected/shadow. |
| 5 | tested_bounded | Independent protected challenge distinct from strategy research. |
| 6 | tested_bounded | Bounded read/status/method/publication/admission/dossier tools; fixed service identity. |
| 7 | tested_bounded | Source cards and discovery interfaces; access never implies eligibility. |
| 8 | external_evidence_required | Catalogue evidence is local; unresolved Carbon Arc history/vintage remains unresolved. |
| 9 | tested_bounded | Point-in-time identity/availability and supplied daily ingestion; no daily interpolation. |
| 10 | tested_bounded | Paper evidence and real implementation method registry. |
| 11 | tested_bounded | Registered hypotheses and explicit conventional bridge; fixture signal is planted. |
| 12 | tested_bounded | Persistent diagnostics/queue and stopping-rule APIs; bounded recurring scheduling and actual fund-feedback review are exercised; learned research ranking remains an optional extension. |
| 13 | tested_bounded | Registration before outcome access, distinct partitions and trial accounting; statistics state dependence limitations. |
| 14 | tested_bounded | Measured criteria and scoped certificates; real campaigns cannot self-admit. |
| 15 | tested_bounded | Explicit return unit/horizon/bridge and uncertainty references. |
| 16 | tested_bounded | Deterministic constrained book from eligible published forecasts. |
| 17 | tested_bounded | Independent pre/post-fill measured risk gate. |
| 18 | tested_bounded | Execution constraints, decay prerequisites and cost sensitivity; real capacity calibration unproven. |
| 19 | tested_bounded | Hand-accounted daily/monthly ledger, benchmark and attribution; earnings split requires explicit event calendar. |
| 20 | tested_bounded | Isolation, durable caches/budgets and fully costed self-financing paired replay are connected; actual adopted checkpoint evidence remains external. |
| 21 | tested_bounded | Typed contracts, source-bound publication and machine certificate scopes. |
| 22 | tested_bounded | Registry/private stores, content hashes and recorded workflow stages. |
| 23 | tested_bounded | Demo/research/replay stages recover identical bound attempts using declared-output quarantine and immutable failure records; research reservations survive retries. |
| 24 | tested_bounded | Fixture demonstrates admitted/rejected candidates, losing trade and dossier; examples are explicitly synthetic and hand-selected. |
| 25 | tested_bounded | Hard process-exit and exception fault tests verify immutable attempts, preserved incomplete evidence, no trial reset and fail-closed input/source changes. |
| 26 | tested_bounded | Functional positive/negative tests and hash-resolved acceptance; no presence-only completeness claim. |
| 27 | tested_bounded | Local four-charter fixture vertical slice, full paired replay, interruption recovery and local Arcana export work. Investment and live vendor validation remain external. |
| 28 | tested_bounded | Hash-resolved judge examples, failures, rejection, actual removal controls, full paired NAV and local dossier/Arcana export; actual LLM skill/prospective performance and live vendor schema are unproven. |
| 29 | external_evidence_required | Official daily source is now available. Other source/vintage/identity/capacity/checkpoint restrictions remain data/evidence dependent. |
| 30 | external_evidence_required | Local handoff/artifacts audited; no network verification of changing official web sources was requested or performed. |

External evidence still required:

- Formation-grade source, vintage and identity attestations for all used production inputs.
- Actual exact-checkpoint release/cutoff/adoption evidence for any proposed LLM use.
- Dated daily fills/borrow/liquidity and event calendar for production reporting realism.
- Genuine post-commitment prospective outcomes; synthetic demo cannot establish investment performance or LLM skill.
- Arcana live vendor schema/API and receipt verification require a refreshed authenticated client session; local review mapping is not vendor approval.

`python scripts/run_reference.py --mode demo --state-dir /tmp/ref/state --output-dir /tmp/ref/out` runs the offline fixture reference. `ReferenceWorkflow(..., exception_agent=...)` optionally injects a registered provider or explicitly labelled fixture stand-in. The main twin artifact includes two full self-financing, fully costed replays. Its first-order position contribution rows are a separate diagnostic. Daily fixture book observations are independently generated, never interpolated from monthly returns.

Interrupted stages preserve immutable attempts and quarantine only declared incomplete outputs. Completed stages resume with artifact hash verification. Deterministic research retries settle the same reserved trial; call budgets and caches persist. Input/source changes require a new run ID. See fault-injection tests in `tests/test_reference_workflow.py`.

Live Arcana schema/upload verification remains external. Operator configured OAuth successfully, but current session tools require reload; no live upload or vendor-compatibility claim is made.
