# Strict Pseudonymized Agent Council Design

> **Archived 2026-09-22 — superseded by the Jev scorer** (`src/agentic/run_jev.py`,
> questions frozen as `jev-v1`, implementation in `src/agentic/jev_*.py`). The code below
> is frozen at `council-v3` / `redaction-v2` under `src/agentic/archive_llm_council/`
> and retained for reference and LLM-vs-Jev ablations. The redaction firewall,
> availability convention, and aggregation semantics described here carry over unchanged.

## Goal

Add an opt-in agentic feature factory beside the existing 8-K scorer, TimesFM branch, and XGBoost baseline. The factory converts each eligible filing into auditable, numeric stock-month features while keeping issuer identity outside the language model.

## Boundaries

- The existing TimesFM, FinBERT, and XGBoost scripts remain usable without the council.
- The council never selects holdings, changes risk constraints, or replaces XGBoost.
- The council scores one filing at a time and aggregates only after all filing-level verdicts are available.
- The language-model transport uses the standard OpenAI-compatible Chat Completions interface. Endpoint, API key, and model are configuration values, not provider-specific code paths.

## Data flow

```text
8-K row
  -> strict redaction firewall
  -> stable local cache key and identity map (never sent to LLM)
  -> four independent role calls on the same redacted filing
  -> validated filing-level verdicts
  -> deterministic permno/eom stock-month aggregation
  -> optional merge into XGBoost feature table
```

### Cost-controlled pre-ranking

The council runner supports an optional cheap pre-ranking stage using the
already-generated local FinBERT scores. It ranks filings by normalized
three-class entropy plus a small chunk-disagreement term, then passes only the
top N document IDs to the redaction/council stage:

```bash
python -m src.agentic.run_council --top-uncertain 5000
```

This stage uses no future-return labels and makes no LLM calls. It is a budget
control and coverage decision, not a learned trading signal. The full filing
universe remains available for later coverage and ablation checks.

## Strict identity firewall

The system uses reversible pseudonymization rather than irreversible anonymity. The local identity map contains the original `document_id`, `permno`, `permco`, filing date, and join metadata. It is never included in a prompt.

The prompt excludes issuer names, tickers, CIK/CUSIP/ISIN/FIGI/GVKEY/IID/PERMNO identifiers, URLs, accession identifiers, exchange/country/region, sector/industry, provider metadata, product-name metadata, and exact calendar-date metadata. The filing body is scrubbed for known identifier values, URLs, accession identifiers, and date expressions. Economically relevant language and quantities remain visible.

Redaction is stable and versioned. A keyed HMAC generates opaque local document identifiers when an identifier is needed for an audit artifact; the model itself does not need that identifier for single-filing scoring. Cache keys include the document ID, redaction version, prompt version, role, and model configuration. Changing redaction or prompts intentionally invalidates affected cache entries.

## Council roles

All roles receive the same redacted packet and do not see other roles' answers.

1. Event classifier: classify the primary event and its signed economic direction.
2. Materiality analyst: assess plausible cash-flow, risk, financing, or valuation impact and surprise.
3. Devil's Advocate: find routine/expected explanations, contradictory evidence, unsupported causal claims, and reasons to abstain.
4. Evidence and audit checker: assess whether the verdict is supported by the filing, including internal dates, units, and evidence sufficiency.

Role prompts are explicit task instructions, not personality prompts. Every role returns the same machine-validated numeric contract so the aggregator can compare them.

## Filing-level contract

Each role returns:

```text
event_category       closed enum
direction            [-1, 1]
materiality          [0, 1]
surprise             [0, 1]
forward_looking      [0, 1]
distress_signal      [0, 1]
cash_flow_impact     [-1, 1]
confidence           [0, 1]
contradiction_score  [0, 1]
evidence_quality     [0, 1]
abstain               {0, 1}
evidence_count       non-negative integer
summary              short redacted explanation
```

The numeric contract is the feature interface. Summaries are retained only for audit and are never used as model features.

## Stock-month aggregation and availability

Aggregate by `(permno, eom)` where `eom` is the calendar month-end of the provider `filing_date`. Under the repository's monthly target convention, evidence in month `t` predicts the return target in month `t+1`; this is an end-of-month availability convention, not an intramonth execution claim. Preserve filing counts and missingness. Use deterministic medians or trimmed means for role scores, median absolute deviation for disagreement, and rates for contradiction and abstention. Include filing-level majority category counts and materiality-weighted direction. Do not hand-code a final trading score; XGBoost learns whether the features add value.

## Provider portability

Configuration uses generic values:

```text
LLM_BASE_URL
LLM_API_KEY
LLM_MODEL
LLM_TIMEOUT_SECONDS
LLM_MAX_CONCURRENCY
LLM_TEMPERATURE
LLM_RESPONSE_FORMAT
LLM_TOOL_CALLING
```

`LLM_RESPONSE_FORMAT=none` must work everywhere. `json_object` is optional and falls back to plain prompting if the endpoint rejects the parameter. Provider-specific request bodies are not required. Usage, request IDs, errors, and model configuration are logged locally when returned by the endpoint.

## Failure handling

- Invalid JSON is repaired once using the same role prompt and then marked failed if still invalid.
- A failed role does not discard the other role outputs; aggregation records role coverage and missingness.
- Failed roles are cached and skipped on reruns unless `--retry-failed` is explicitly supplied.
- Network/rate-limit errors use bounded retries with backoff.
- SQLite caching makes reruns idempotent and avoids repeated calls for unchanged inputs.
- Tool calling is off by default. When enabled with `LLM_TOOL_CALLING=1`, roles can
  use only the bounded local `search_filing` and `get_filing_excerpt` tools over
  the same already-redacted filing; web, browsing, filesystem, and external-data
  tools remain unavailable.

## Evaluation

Run the council only as an ablation:

1. characteristics only;
2. characteristics plus deterministic filing features;
3. characteristics plus one council role;
4. characteristics plus the full council.

Use the existing expanding time splits and untouched evaluation protocol. Report inference cost, coverage, disagreement, and failure rates alongside return metrics. The council is retained only if it adds out-of-sample value after those costs and the result survives a frozen-specification rerun.
