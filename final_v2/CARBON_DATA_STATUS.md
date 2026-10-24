# Carbon Arc data status: what we can use and what we can't

Date: 2026-10-24. Evidence base: `carbon/DATA_MAP.md` (live API survey, 2026-10-22),
field dictionaries in `carbon/out/dictionaries/` (34 datasets, 5,937 columns
inspected), `carbon/out/block_pricing.csv`, and spec `POD_FRAMEWORK_IMPLEMENTATION_SPEC.md`
§8 (catalogue snapshot). Source cards live in `final_v2/configs/source_cards/` —
this document is the readable summary; the cards are the machine-checked record.

**The test** (spec 9.2, 8.4): a source can feed formation only when four screens
are established — history, metric lag, vintage (do historical values stay
historical?), and coverage — plus dated identity. Any screen unresolved = the
data broker refuses formation use. Research/descriptive use is always allowed.

---

## Summary table

| Dataset | What it is | History | Clocks | Vintage | Verdict |
|---|---|---|---|---|---|
| **Competition inputs** (chars panel, 8-K corpus, prices) | The mandate data | 2015– | documented | stable | **USE — core** |
| **CA0032 EDGAR filings** | Filing facts (acceptance times, concepts) | 2012– | `acceptance_date_time` (SEC, immutable) | immutable record; amendments are new filings | **NEARLY — verify extract** |
| **CA0092 New Constructs ratings** | Ratings, ROIC, FCF yield, valuation gap | 2016– | `vendor_published_at`, `ca_receipt_date` | **explicit** (`ca_drop_partition` marks full-backfill drops) | **NEARLY — run coverage tests** |
| **CA0035 Benzinga events** | 25 event categories (M&A, IPO, legal…) | 2011– | source publication, T+1 weekly | tagging may be retagged; events themselves don't move | **LIKELY — pin tagging rules** |
| **CA0097J job postings** | Row-level US job postings | **2010–** | `ca_receipt_date`, `posted/expired`, partitions | rows vendor-edited/soft-deleted | **MAYBE — lag past edit window** |
| **CA0056 card spend** | Card transactions by brand/ticker | 2019– | ambiguous `date` field | **NO — vendor reweights monthly** | **BLOCKED — ask vendor** |
| **CA0060 foot traffic** | Modelled store visits | 2019– | none beyond measurement window | **NO — model re-estimated** | **BLOCKED** |
| **CA0058 health spend** | Pharmacy/HSA transactions | 2017– | publication years after transactions | backfill partitions | **BLOCKED** |
| **CA0043C software customers** | B2B customer relationships | 2016– | detection dates inconsistent | unresolved | **BLOCKED** |
| **CA0028/CA0031 card (historic)** | Longer card history | 2015– | closed series | closed 2024–25 | **RESEARCH ONLY** |
| **549 single-company panels** | Deep dives on one company each | varies | varies | varies | **NO — no cross-section** |

---

## What we can use

### 1. Competition inputs — use now, no gate
The mandate data the first-round book ran on: 147 characteristics panel,
returns, the 8-K corpus, factor lists. Timing convention is documented and
already enforced by the new platform's timing guards. This is the core of the
submission.

### 2. CA0032 EDGAR filings — use after one verification
**Why it's safe:** `acceptance_date_time` is the SEC's own immutable clock.
EDGAR submissions never change — amendments arrive as new filings. The
dictionary also carries `former_names` with from/to dates (dated identity).

**Why it matters beyond safety:** it fixes the *confirmed* survivorship defect
in our current 8-K corpus (spec 9.8) and replaces the weak date handling in
the filing-events channel. This is both the cleanest and the most useful asset.

**Remaining gate:** the inspected sample workbook was mismatched (it contained
"Company Reported Actuals" sheets — spec 8.3). Pull one bulk extract and check
it matches the dictionary. Also: no filing-text field was found — if business
text is needed, the text endpoint must be located.

### 3. CA0092 New Constructs ratings — use after coverage tests
**Why it's safe:** the only asset with *explicit* vintage identity —
`ca_drop_partition` literally marks which vendor drop each row arrived in,
including `Full-backfill-2022-12-31`-style markers. Pair with
`vendor_published_at`/`ca_receipt_date` and you know exactly what was knowable
when. Identity is point-in-time by design (`ticker_on_event_date`,
`cik_on_event_date`, even exchange and MIC on event date).

**Caveat:** the rating composites (rating_gap, ROIC, FCF yield) are vendor
baselines, not assumed novel alpha — they must beat cheap public style
controls (spec 4.3 admission standard).

### 4. CA0035 Benzinga events — likely, with pinning
**Why it's probably safe:** events (an acquisition announcement, an SEC
filing) are published once and never rewritten. The spec fixes the convention:
historical baseline follows source publication (§9.2).

**Remaining pin:** the per-table `date` is described ambiguously ("collection
or reporting"), and the `ticker_exclusively_mentioned` materiality flag is a
vendor transform whose computation is undocumented. Event *counts* with a
date-only convention are safe; materiality weighting waits for documentation.

---

## What we can't use (yet)

### Blocked on vendor-side rewrites
- **CA0056 card spend** — the vendor states it reweights the panel monthly
  ("panel is reweighted monthly when the vendor restates its panel and removes
  inactive cards"). Today's historical values are revised values. Binding
  unless: (a) Carbon Arc sells historical cuts (the `cut`+`lag` pricing keys
  suggest this may exist — one question to support), or (b) we lag past the
  revision window and accept slower signals.
- **CA0060 foot traffic** — visits are modelled estimates from device panels,
  normalized/adjusted by the vendor. Historical months can be re-estimated.
  No delivery clock exists in the dictionary at all.
- **CA0058 health spend** — publication timestamps *years* after the
  transaction dates, plus explicit backfill partitions (spec 8.3.3).

### Blocked on semantics
- **CA0043C software customers** — a sample row had a date preceding its own
  detection date (spec 8.3.4); period summaries can't be used before their
  constituents were knowable.
- **CA0069G government contracts** — usable later, but new obligation vs
  cumulative ceiling vs final value must not be interchanged (spec 8.3.5);
  "latest row" flags incorporate later lifecycle info.

### Not usable at all
- **549 single-company panels** (`is_webcontent`) — no cross-section to rank.
- **REJECT set** (CA0086 intraday cards, CA0083 Amazon, CA0054 apps, TikTok,
  job movements, menus, maritime, vehicles, music, POS, culinary…) — histories
  start 2021+ or unusable lags; none survive the pre-2021 training-history
  test (DATA_MAP 2.6).
- **CA0099 13F adviser profiles** — the good fields start January 2024;
  filing dates can be the literal string "Latest Filing" (spec 8.3.6).
- **CA0065S stock prices** — not found in the catalogue snapshot (spec 8.3.7);
  price/volume access remains an open dependency.

---

## Identity and coverage: the cross-cutting gate

Every asset above still passes through two shared tests before formation use,
regardless of its clock quality (spec 9.5–9.8, DATA_MAP §4):

1. **Dated identity** — no joining a 2019–2026 history on today's ticker map.
   All panels use vendor ticker/company IDs that must resolve through dated
   links. The delisting test (Sears, Bon-Ton, Pier 1, GNC, Ascena, J.C. Penney
   must resolve) is required for each.
2. **Coverage bias** — the panels skew consumer/tech/healthcare. A signal that
   fires only there is a sector bet in disguise; rank within sector and run the
   covered-vs-uncovered return comparison.

## Practical reading

- **Safe now:** competition inputs.
- **Next:** CA0032 (one extract verification) and CA0092 (coverage tests) —
  these give Charter 2 (filing events) and Charter 3 (valuation overlay) real
  new inputs with clean clocks.
- **Cheap probe:** ask Carbon Arc support whether historical cuts exist for
  CA0056/CA0060. If yes, the Charter-1 demand story reopens. If no, the demand
  panels stay descriptive and Charter 1 runs on competition inputs.
- **Rule of thumb from the evidence:** events, filings, and reported facts have
  clocks that don't move; modelled/aggregate panels get restated. Prefer the
  former for anything that trades.

Until a card's four screens are all established, the broker rejects it with
`SOURCE_NOT_FORMATION_ELIGIBLE` — mechanically, not by discipline.
