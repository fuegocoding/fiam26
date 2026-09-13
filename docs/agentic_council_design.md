# Strict Pseudonymized Agent Council Design

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

## Stock-month aggregation

Aggregate by `(permno, eom)` where `eom` is the calendar month-end of the provider `filing_date`. Preserve filing counts and missingness. Use deterministic medians or trimmed means for role scores, median absolute deviation for disagreement, and rates for contradiction and abstention. Include category counts and materiality-weighted direction. Do not hand-code a final trading score; XGBoost learns whether the features add value.

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
```

`LLM_RESPONSE_FORMAT=none` must work everywhere. `json_object` is optional and falls back to plain prompting if the endpoint rejects the parameter. Provider-specific request bodies are not required. Usage, request IDs, errors, and model configuration are logged locally when returned by the endpoint.

## Failure handling

- Invalid JSON is repaired once using the same role prompt and then marked failed if still invalid.
- A failed role does not discard the other role outputs; aggregation records role coverage and missingness.
- Network/rate-limit errors use bounded retries with backoff.
- SQLite caching makes reruns idempotent and avoids repeated calls for unchanged inputs.
- No web, file-search, browsing, or external data tools are available to council roles.

## Evaluation

Run the council only as an ablation:

1. characteristics only;
2. characteristics plus deterministic filing features;
3. characteristics plus one council role;
4. characteristics plus the full council.

Use the existing expanding time splits and untouched evaluation protocol. Report inference cost, coverage, disagreement, and failure rates alongside return metrics. The council is retained only if it adds out-of-sample value after those costs and the result survives a frozen-specification rerun.
