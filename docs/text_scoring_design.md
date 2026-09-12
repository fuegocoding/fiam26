# Text Scoring Engine — Design (8-K filings → LLM scores → features)

Status: **design + code ready, NOT yet run.** Model: `z-ai/glm-5.3-flash` via OpenRouter
(OpenAI-compatible endpoint, reasoning excluded for cost).

## 1. What the audit of the corpus told us

| Fact | Value | Design consequence |
|---|---|---|
| Filings | 373,139 (373,054 8-K + 85 8-K/A) | amendments handled as separate rows |
| Text length | median 3,851 chars; p90 8.4k; p99 20.5k; max 1.16M | cap input at 12,000 chars (covers ~98% fully; tail truncated) |
| Cover-page boilerplate | first ~900 chars are SEC header | instruct model to ignore it; do not pre-trim (cheap enough) |
| Detected items | 9.01 in 296k (wrapper), 2.02 earnings 120k, 7.01 88k, 8.01 84k, 5.02 71k, 1.01 48k, 4.01 2,190, 4.02 698, 2.05 1,802, 2.06 461 | item codes are HINTS only (readme: annotations to verify); prompt requires classification from body text |
| No items detected | 1,038 filings | prompt handles empty item list |
| Multiple events per filing | common | model scores the single most material event + up to 2 secondary categories |

## 2. Per-filing score schema (JSON contract)

Fixed keys, machine-validated ranges:

| Key | Range | Meaning |
|---|---|---|
| `event_category` | enum | Most material event, classified from body text |
| `secondary_categories` | 0–2 enums | Other events in same filing |
| `sentiment` | [-1, +1] | Tone toward firm value (+1 clearly positive … -1 clearly negative) |
| `materiality` | [0, 1] | How much the event plausibly moves the stock (routine boilerplate ≈ 0) |
| `surprise` | [0, 1] | How unexpected vs anticipated (routine dividend = low; auditor resignation = high) |
| `forward_looking` | [0, 1] | Share of content that is guidance/outlook rather than backward-looking |
| `distress_signal` | [0, 1] | Going-concern, auditor change/resignation, non-reliance (Item 4.02), impairment, covenant stress |
| `cash_flow_impact` | [-1, +1] | Stated/inferrable impact on near-term cash flows |
| `conviction` | [-1, +1] | Signed conviction: direction × confidence (the PDF "analyst agent" verdict) |
| `summary` | ≤ 25 words | One-line event statement (auditability deliverable) |

`event_category` enum (closed list): `earnings_results`, `guidance_update`, `material_agreement`,
`financing_credit`, `capital_structure_debt`, `share_issuance_buyback`, `dividend`, `mna`,
`asset_disposal`, `restructuring_impairment`, `management_change`, `auditor_change`,
`accounting_nonreliance`, `litigation_regulatory`, `delisting_compliance`, `shareholder_vote`,
`product_operations`, `contract_partnership`, `other`, `unreadable`.
(Rules: `unreadable` forces neutral scores; `accounting_nonreliance`/`auditor_change` must be used when
Item 4.02/4.01 content is present — the corpus's strongest distress signals.)

## 3. From filing scores to stock-month features

Aggregation (separate step; join key `permno` + month of `filing_date`):

- `n_filings` (count), `n_material` (materiality ≥ 0.5)
- `sentiment_mean`, `sentiment_maxabs` (sentiment of the highest-materiality filing)
- `sentiment_matweighted` (Σ sentiment·materiality / Σ materiality)
- `surprise_matweighted` (same weighting on surprise)
- `conviction_matweighted`, `distress_any` (any filing ≥ 0.5), `distress_max`
- `n_earnings`, `n_guidance`, `n_mgmt_change`, `n_auditor`, `n_nonreliance`, `n_agreement`, `n_mna`
- category mix flags as needed; filings with no same-month chars row are kept out of the panel (deliberate, documented)

## 4. Availability convention (look-ahead guard — absolute)

- A filing dated day d **enters the feature set of month t iff d ∈ month t**; features of month t are used to
  trade at the start of t+1 (PDF recipe: "accumulate the news and extract their sentiment within the current
  month t to make forecasts for month t+1").
- `filing_date` (provider date) is the only availability clock used; `filing_timestamp_utc` is a placeholder,
  `sec_accepted_at_utc` is null, `content_report_date` is an event date — none used for timing.
- The model sees only the filing text + its own metadata; no other stocks, no future info. Deck must state this
  constraint (agent-look-ahead rule).

## 5. Cost / throughput model

- Input: median ~1k tokens/filing (≈3.9k chars); cap 12k chars. Est. total input ≈ 370–450M tokens; output
  ≈ 373k × ~150 tokens ≈ 55M tokens (reasoning excluded).
- Engine: batched month slices, ThreadPoolExecutor (default 8 workers), exponential backoff on 429/5xx,
  idempotent sqlite cache keyed by `document_id` → resumable, no double spend.
- Per-failure rows logged with error class for audit.

## 6. Files

- `src/prompts.py` — system prompt + user-message builder + JSON contract (enum/range validation)
- `src/text_scoring.py` — engine: iter_batches over the parquet, cache-aware, concurrent scoring,
  writes `data/scoring/text_scores.db` (sqlite) + export helper to parquet
- Run only when you say so: `python -m src.text_scoring --month 2021-01` etc.
