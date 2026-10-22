# Carbon Arc data map: what we can use and for what

Date: 2026-10-22. Source: live survey of `api.carbonarc.co` (catalogue, block
pricing, entity map, 34 data dictionaries) joined to the competition's screening
workbook. Reproduce with `cd carbon && python survey.py`; raw pulls and parsed
tables land in `carbon/out/`.

Mandate context: [final/docs/final_mandate_rules_map.md](../final/docs/final_mandate_rules_map.md)
(§6.6 four screening questions, §6.3 survivorship, §7 point-in-time).

---

## 1. What the library actually is

The public catalogue holds **625 rows**. They are not 625 cross-sectional
datasets:

| Scope | Rows | What it is | Usable for us? |
|---|---|---|---|
| **Cross-sectional** | **75** | Panels keyed to tickers/brands/companies across many names — the assets the screening workbook covers | **Yes, selectively** |
| Single-company panels | 549 | Deep-dive data about one named company each (e.g. "1800 Contacts", "AMN Healthcare", "AT&T Service Coverage") | No — no cross-section to rank |
| Other | 1 | **U.S. Job Postings (CA0097J)**, 2010–present, block-only — a long-history labour panel the workbook never screened | **Yes — the one real find** |

Flags across the 625: `api/builder/mcp` available on 616, `block` (S3 bulk) on
55 — and `is_webcontent` marks exactly the 549 single-company panels. Commercial list prices run $10k–$600k per cut (block pricing in
`out/block_pricing.csv`); our access is provisioned, so cost is our 500-credit
balance and the review burden, not list price.

## 2. Mapping: dataset → fields → our model

Our channels (`final/README.md`): `e7` (earnings + revisions), `gap_micro_best`
(valuation gap), `runway_micro` (cash burn), `conf_lm_gated` (consensus),
`inventory_retail` (sales−inventory), `oplev_software` (sales−SG&A), `xgb_rank`
(147 characteristics), `filing_events` (SEC events), graph overlay (peer text).

### 2.1 Text and events — priority: fixes a known defect

| Asset | History / lag | Key fields (from dictionaries) | Use |
|---|---|---|---|
| **CA0032 Financial Filing Forms (EDGAR)** | 2012–, T+1, CIK | `cik`, `acceptance_date_time`, `former_names`, `concept`, `filing_fiscal_year` + a ticker↔company mapping block | **Rebuild the text channel survivorship-clean** (`filing_events`, graph business text). `acceptance_date_time` is the §7-safe availability clock. **Bulk/MCP only** (`api=False`) |
| **CA0035 Financial News & Data (Benzinga)** | 2011–, T+1 weekly, ticker | 25 event-category tables, each `ticker, event, date, ticker_exclusively_mentioned`; SEC Filings table adds `event_id`/`parent_event_name` | Daily event-count features, materiality filter via the exclusive-mention flag; earnings/M&A/IPO/analyst calendars. **Event tags only — no text** |

### 2.2 Fundamentals and valuation

| Asset | History / lag | Key fields | Use |
|---|---|---|---|
| **CA0065C Company Reported Actuals** | 2015–, T+1 after release, ticker | Income statement incl. `revenue_value`, `sg_a_value`, `net_income_value`; balance sheet; cash flow; **KPI & Segments** (55 cols) | Direct inputs for `inventory_retail`/`oplev_software` and segment-level surprises; quarterly, so a slower overlay |
| **CA0092 Securities Ratings (New Constructs)** | 2016–, T+2, ticker | `rating_overall`, `rating_price_to_ebv_ratio`, `rating_fcf_yield`, `rating_roic`, `rating_gap`, `rating_income_net_economic_profit`; **`ticker_on_event_date`, `ca_receipt_date`** | Valuation/quality overlay for `gap_micro_best`, `conf_lm_gated`, `xgb_rank`. Point-in-time labels built in |
| CA0038 Financial Reporting | 2008–, T+90, ticker | thin (10 cols) | Slow fundamentals history; mostly superseded by CA0065C |

### 2.3 Consumer demand — the densest panel

| Asset | History / lag | Key fields | Use |
|---|---|---|---|
| **CA0056 Credit Card – US Complete (Facteus)** | 2019–, T+3 | 13 tables: `Core Panel` (`company_id, ticker_id, brand_id, avg_card_spend, category_id, date`), share-of-wallet, retention cohorts, `Overlapping Shopper` cross-shopping, geo (CBSA/zip/DMA) | Consumer revenue growth and momentum; cross-shopping/retention as slow signals |
| CA0028 / CA0031 card panels | 2019 / 2015–, **historic** | merchant + ticker spend | Longer card history but **closed series** — research only (§6.2.7) |
| CA0058 Health Spend / CA0058G Private Label | 2017 / 2015– | ticker + merchant | Pharmacy/HSA spend; private-label has no stated ticker key |
| CA0029 POS Convenience, CA0047 POS Supermarket (2007–!), CA0034 POS Online/Instore, CA006 Receipts | 2007–2019 starts | brand/category sell-through | Retail sell-through nowcasts; CA0047's 2007 history is the longest consumer series |
| **CA0060 Foot Traffic (PassBy)** | 2019–, T+5 | `foottraffic`, store→POI→company, visitor income/destinations | Same-store-sales nowcast for retail/restaurants |
| CA0052 Brand Reviews | 2013–, T+1 | 70 cols incl. review text fields, `review_count` | Sentiment/demand for product brands |
| CA0016 Movie Box Office | **1972–**, T+1 | `tbo_revenue_usd`, franchise/studio | Media revenue; the longest history in the library |

### 2.4 Labour, software and tech demand

| Asset | History / lag | Key fields | Use |
|---|---|---|---|
| **CA0097J U.S. Job Postings (the find)** | **2010–**, block-only | 87 cols: `salary`, `soc_5_name` occupation, `cip6` education, NAICS6, county, `body` | Long-history hiring demand — growth/hiring signal before 2017, where CA0043A is silent |
| CA0043A Job Openings (PredictLeads) | 2017–, bi-monthly | `ticker, company_id, brand_id, category_name, contract_type` | Same, current and API-served |
| CA0043B/C/D Relationships / Software Customers / Product Launches | 2015–2017– | company + brand links | B2B demand and product-cycle signals (`oplev_software` extension) |
| **CA0098 Open-Source Dev Activity** | **2008–**, T+1 | 16 tables: daily `commit_count, pull_request_*, issues_closed, developer_count`; **AI Model Demand** (`total_downloads`, trending); **AI Tool Dependencies** (`langchain_files, cursor_files, anthropic_files, copilot_files`); CDI; dev cohorts | Software demand and AI adoption — long history, genuinely under-explored |
| CA0093 SaaS Market Dynamics | 2011–, T+1 | supplier/app level | Software spend trends |

### 2.5 Healthcare, industrial, macro

| Asset | History / lag | Key fields | Use |
|---|---|---|---|
| **CA0049 Medical & Pharmacy Open Claims** | 2015– (PBM 2017), T+5 **with a published completion curve** (45% wk1, 80% wk4) | `claim_count, total_charge_amount` by drug/segment/payer/provider, `ticker_id` | Healthcare revenue nowcast — the best-documented PIT asset in the library |
| CA0081 Healthcare Closed Claims | 2016–, ~90% mature by month 4–6 | segment/procedure/provider | Slower, cleaner version of the same |
| CA0069B Global Trade Flows | 2015–, T+4–8 **weeks** | buyer/supplier → brand | Supply-chain revenue; long lag is the binding constraint |
| CA0069G Government Contracts | 2001–, T+1–30 | awardee × parent company | Backlog/revenue shocks, defence/IT names |
| CA0103 Facility-Level Risk | storm 1950–, emissions 2011– | issuer × physical asset (2,271 cols) | Physical-risk exposure overlay; block-only |
| CA0015 FRED, CA0077 commodities, CA0037 weather, CA002 housing permits, CA0055 SMB workforce | long | macro/geography keys | Conditioning variables, not cross-sectional alpha (workbook MACRO) |
| CA0065S Stock Prices | 1990–, T+1 | — | Infrastructure: liquidity/participation measures for the §9.4 exhibits (no dictionary) |

### 2.6 SLEEVE / REJECT — not for the core book

`CA0051 FFIEC` (banks only), `CA0099` adviser/13F profiles, `CA0082` private
companies, tickets/music/sports assets (CA0011/003/004/007/008/0050), EU/EM card
(CA0042/0044), and the REJECT set (histories starting 2021+ or unusable lags:
CA0086 intraday cards, CA0083 Amazon, CA0054 apps, CA0045B TikTok, CA0053 job
movements, CA0057 menus, CA0080 maritime, CA005 vehicles, CA0046 music, CA0048
POS, CA0078 culinary, CA0036/0040/0041/0094). Several are useful as one-off
event studies; none survive the pre-2021 training-history test.

## 3. What feature, where in the code

How a Carbon asset becomes a signal in `final/`, end to end:

```
carbon pull (client.py / survey.py / MCP)
  -> dated parquet in final/external_data/  (tables registry, like external.parquet)
  -> external.read("carbon_ca00XX")          final/external.py
  -> sleeve build()/load()                   final/sleeves/<name>/
  -> score() monthly ranks                   sleeves/base.formation_output
  -> blend                                   final/combiner.py (make_sleeves)
  -> graph overlay (5%) / optimizer / report  unchanged stages
```

Every candidate feature gets the same treatment as the existing ones: monthly
cross-sectional ranks inside its native universe, point-in-time availability
checked in `final/timing.py`, one hypothesis at a time with the denominator
reported (Instructions 5.2).

### 3.1 Priority assets: candidate features and their landing spots

| Asset | What to build (first candidates) | Where it plugs in | Stage / horizon |
|---|---|---|---|
| **CA0032 EDGAR** | (1) Item-type event counts over 1/3/12 months keyed on **CIK** — the same feature family `filing_events` already computes; (2) filing text for the company graph's business description | Input swap in `final/sleeves/filing_events/edgar.py` (EDGAR index → CA0032 tables) and `final/graph/business_text.py` (8-K text → CA0032 text). Model code unchanged | Signal + graph; monthly formation, text dated ≤ formation |
| **CA0035 events** | Event counts by category over 1/5/21 days with the `ticker_exclusively_mentioned` share as a materiality weight; earnings/M&A/IPO calendar dummies | New sleeve `final/sleeves/news_events/` (mirror `filing_events`: counts → rank or xgb), registered in `combiner.make_sleeves()`; calendar dummies feed `final/arcana.py`'s earnings-vs-ex-earnings lens and `final/risk.py` event sizing | Signal; monthly book, counts computed daily then aggregated |
| **CA0056 card core panel** | Ticker-level card spend 3m/12m growth, share-of-wallet change, retention-cohort decay | Extend `final/sleeves/inventory_retail.py` (its score is one column, `dsale_dinv`, today) or a new `consumer_demand` sleeve that loads the carbon table in `load()` | Signal; monthly, spend aggregated to month and lagged T+3 |
| **CA0060 foot traffic / CA0047 supermarket** | Foot-traffic growth (DMA-weighted); sell-through growth | Same consumer sleeve as corroborating features; keep one combined score rather than three sleeves | Signal; monthly |
| **CA0065C actuals** | Revenue and `sg_a_value` surprise vs 4-quarter trailing; segment-revenue breadth | Extend `final/sleeves/e7/panel.py` (post-report revision breadth) or `final/sleeves/oplev_software.py` (it reads `sale_gr1`/`dsale_dsga` from the panel today) | Slow overlay; quarterly, from the release date only |
| **CA0092 ratings** | Rating change 1/3m, `rating_fcf_yield`, `rating_roic`, `rating_gap` | Extra predictor columns in `final/sleeves/xgb_rank/model.py` beside the 147 characteristics, or a standalone low-turnover overlay sleeve | Overlay; monthly, `ca_receipt_date` governs availability |
| **CA0043A + CA0097J job postings** | Posting-count growth 3/6m, salary inflation, occupation-mix shift (CA0097J reaches back to 2010) | New `hiring_demand` sleeve, or extend `oplev_software.py` (sales−SG&A is a cost-side cousin of hiring) | Signal; monthly |
| **CA0098 open source** | Commit/PR velocity growth, developer-count growth, AI-tool dependency share (`langchain_files`, `cursor_files`, `copilot_files`) | Tech-demand features in `final/sleeves/oplev_software.py` or a new sleeve; AI adoption as a slow theme score | Signal; monthly from daily/weekly tables |
| **CA0049 medical claims** | Claim-count growth by drug/manufacturer (`ticker_id`), lagged beyond the published completion curve | Sector sleeve for healthcare names only; rank within sector to avoid the §6.2.6 sector-bet trap | Signal; monthly |
| **CA0065S stock prices** | Dollar ADV from 1990 daily prices — replaces/augments the panel's `dolvol_126d` | `final/combiner.py` `_FORMATION_COLUMNS` and the liquidity filter in `final/optimizer.py` (`dolvol_126d * participation * days`); also `final/report.py`'s §9.4 ADV-days exhibit | Infrastructure (not alpha); improves tradability and capacity honesty |

### 3.2 Secondary uses (no new sleeve)

| Need | Asset | Where |
|---|---|---|
| Earnings-window tagging for the §11 lens | CA0035 Earnings table / CA0065C release dates | `final/arcana.py` |
| Sector/industry corroboration for `xgb_rank` | CA0043 firmographics (NAICS) | feature columns in `final/sleeves/xgb_rank/model.py` |
| Macro conditioning (regime labels, never retroactive) | CA0015 FRED (already used), CA0077 commodities | `final/external.py` tables; risk layer only |
| Borrow/crowding context for shorts | CA0069G government contracts, CA0069B trade flows (large backlogs = capacity story) | discussion only; `final/risk.py` stays rule-based |

### 3.3 Registration checklist for anything new

1. Land the pull as a dated parquet with an as-of column; register it like the
   existing `external.parquet` tables (`final/external_data/build.py` pattern).
2. Implement `build()` (cached artifacts under `output/`) and `load()` on a
   `sleeves.base.Sleeve`, then add it to `combiner.make_sleeves()`.
3. Availability: every feature column dated ≤ formation month; add the join
   assertion `final/timing.py` expects (max feature date ≤ formation date).
4. Walk-forward only: train through Y−3, validate Y−2..Y−1, predict Y — no
   feature selection on the full sample (Instructions 7).
5. A test in `tests/` proving the timing rule (mirror `test_final_timing.py`).
6. Write the §6.6 four answers and the hypothesis denominator down first.

## 4. Point-in-time and survivorship (what the dictionaries revealed)

- **Point-in-time clocks exist**: CA0032 `acceptance_date_time`, CA0092
  `ticker_on_event_date` + `ca_receipt_date`, CA0049's completion curve, CA0032
  `former_names`. Use them; do not substitute today's labels (§7).
- **Vintages**: `cut` + `lag` are part of every dataset key (grants and pricing
  are per `(dataset_id, lag, cut)`). Prefer historical cuts over the current
  vintage; if revisions cannot be reconstructed, lag past the revision window
  with evidence or use descriptively (§7 case 3).
- **Closed series** (CA0028, CA0031, CA006, CA007, CA0059, CA0041, CA0036,
  CA0040, CA0044, CA0011) end during 2024–2026 — fine for research, useless
  forward (§6.2.7).
- **Coverage is the open question** (§6.6 iv): the entity map is present-day.
  Before any asset enters the book: run the §6.3 delisting test (Sears, Bon-Ton,
  Pier 1, GNC, Ascena, J.C. Penney must resolve) and the covered-vs-uncovered
  return test. Nothing here is used until that is written down.
- **Sector skew**: the panels are consumer/tech/healthcare rich; a signal that
  fires only there is a sector bet in disguise (§6.2.6) — rank within sector.

## 5. What to pull first (ordered)

1. **CA0032 EDGAR via bulk/MCP** — rebuild `filing_events` + graph text on CIK,
   survivorship-clean. Fixes the confirmed 8-K corpus defect (§6.4) and is
   explicitly credited.
2. **CA0035 event tables** — event-count features with the exclusive-mention
   materiality filter; earnings/M&A calendars for the §11 earnings-vs-ex-earnings
   lens. Text-free by construction, so no LLM claims to defend.
3. **CA0056 Core Panel** — consumer revenue growth (monthly ranks), the single
   best-sampled cross-section; add CA0060 foot traffic and CA0047 supermarket for
   retail corroboration.
4. **CA0043A + CA0097J** — hiring demand; 2010 history from CA0097J fills the
   pre-2017 gap.
5. **CA0065C + CA0092** — reported actuals and ratings as slow overlays on the
   valuation/consensus sleeves.
6. **CA0098** — software/AI demand for the tech sleeve; hypothesis-driven, one
   or two candidates at a time (report the denominator, §5.2).

Each needs the four answers (history start, metric lag, revision treatment,
coverage) written down **before** feature work starts.

## 6. Files produced by `survey.py`

| File | Contents |
|---|---|
| `out/catalog.csv` | All 625 rows: metadata, flags, workbook verdicts, scope classification |
| `out/block_pricing.csv` | 58 priced cuts with lags |
| `out/entity_map.csv` | 178 entity/domain/representation triples |
| `out/candidates.csv` | The 34 dictionary targets (workbook CORE/USABLE + the find) |
| `out/dictionaries/CA*.csv` | Field-level dictionaries (5,937 columns) |
| `out/library_raw.json` | Raw catalogue pull |
