# Momentum around 8-K filings

## Recommendation after checking the text

**Do not add a general 8-K momentum or sentiment-confirmation rule to the algorithm on this evidence.** Large-cap filing tone is the most plausible narrow follow-up, but it is not validated after risk adjustment, multiple testing and costs. The conclusion applies to these monthly text signals, not every possible semantic model of an 8-K.

The full-body scanner processed **all 373,139 filings and 1,898,845,656 characters**, without truncating bodies, in about 7.8 minutes. It cached pattern flags, not full text. The independently cached FinBERT scores cover every filing; 15,597 bodies (4.18%) exceed that scorer's 12,000-character limit. Main filing bodies are not necessarily complete submission packages with exhibits.

### Does filing tone predict what happens next?

In 2021–August 2026, across screened filing stocks:

- Buying relatively positive-tone filings and shorting relatively negative-tone filings earns **0.0019% per month gross**: effectively zero (HAC t 0.01).
- Filing-month momentum earns **0.389% gross/month**. Requiring tone to agree with the momentum produces **0.355%**; it does not establish an improvement.
- Prior-three-month momentum earns **0.0385% gross/month** within the filing-stock ranking. Requiring agreeing tone produces **−0.203%**.
- Tone has small positive rank correlations with prior-three-month returns (**0.0189**) and filing-month returns (**0.0227**). Its next-month unadjusted rank correlation is **−0.0027**, and its cap/sector-adjusted correlation is **−0.0015**. Neither forward correlation is significant. These are associations, not evidence that the filing caused or anticipated the earlier move.
- No next-month content candidate has an adjusted-spread q-value below 0.05 in the explored size/sector/topic/signal family.

The momentum numbers here differ slightly from the event/control table because these cutoffs rank only screened filing stocks. The earlier table uses common breakpoints across filing and control stocks.

### Where the text looks more or less useful

| Segment | Tone-ranked next-month gross spread | Interpretation |
|---|---:|---|
| Large caps, $10–200bn | **+0.367%/month** | Best narrow research lead. Selected in all six annual folds. Raw t 2.05 falls to 1.53 after cap/sector adjustment; fails multiple testing. |
| Mid caps, $2–10bn | +0.072% | No reliable benefit |
| Small caps, $0.3–2bn | −0.154% | No reliable positive-tone benefit |
| Micro caps, $50–300m | −1.276% | Positive-tone stocks underperform in this sample; not a validated reversal rule, and implementation is difficult |
| Mega caps, >$200bn | Only 7 eligible portfolio months | Insufficient portfolio breadth; 67 months of rank correlations also show no reliable positive prediction |
| Sectors | No robust winner | Communication services and technology look positive before uncertainty adjustments, but no sector passes the relevant multiple-testing check |
| Buyback mentions | +0.355%, 46 spread months | No reliable effect; t approximately 0.57 |
| Acquisition mentions | +0.043%, 28 spread months | No reliable effect |
| Financing mentions | Too few spread months | Rank correlation offers no reliable next-month prediction |
| Guidance changes, dividend cuts, distress, non-reliance, restructuring | Sparse | Insufficient monthly breadth for a conclusion; absence of evidence is not evidence these events never matter |

The large-cap tone spread is below a 0.40%-per-month **full-replacement** cost scenario at 10 bp one-way. Lower-turnover implementation could cost less, but no realized-cost claim is made. It also still needs an incremental test against the existing characteristics model; sector adjustment alone does not answer that question.

**Content-quality finding:** even strict text matches can describe somebody else's bankruptcy, a historical buyback or hypothetical dividend risk. The audited excerpts show these cases. Use these outputs as research flags, not confirmed issuer-event labels. A semantic extraction model that resolves actor, timing, novelty and missing exhibits is a separate experiment and has not been validated here.

## Filing occurrence: results

The timing/category analysis used 529,082 panel rows and 369,967 eligible filings, aggregated to 239,122 event stock-months. It checked 522,431 next-month target pairs with zero mismatches. 2,938 filings had no same-month panel row and 234 fell after the recorded monthly snapshot; they were excluded. The test holding period is January 2021 through August 2026.

For the screened universe, average monthly raw long-short spreads were:

| Momentum measured at rebalance | Filing stocks | Non-filing stocks | Cap/sector-adjusted incremental spread | HAC t for increment |
|---|---:|---:|---:|---:|
| Previous three complete months | +0.057% | +0.664% | −0.522% | −2.33 |
| Filing month itself | +0.385% | +0.333% | −0.073% | −0.41 |

The first result points toward *weaker* momentum among filing stocks, but its adjusted multiple-testing q-value is 0.177. It also reverses sign between training and validation. It does not justify adding either a momentum confirmation rule or a reversal rule.

### Company size

All figures below are screened, gross monthly filing-stock momentum spreads in 2021–2026; they are not estimated net strategy returns.

| Company market cap | Prior-three-month momentum | Filing-month momentum | Assessment |
|---|---:|---:|---|
| Micro, $50–300m | −0.09% | −0.42% | No reliable continuation; difficult implementation universe |
| Small, $0.3–2bn | +0.34% | +0.30% | No statistically reliable incremental filing benefit |
| Mid, $2–10bn | +0.01% | +0.18% | Little evidence of useful continuation |
| Large, $10–200bn | −0.22% | +0.42% | No statistically reliable incremental filing benefit |
| Mega, >$200bn | Insufficient breadth | Only 8 eligible spread months | No robust portfolio conclusion; incremental rank correlations also insignificant |
| Nano, <$50m | Insufficient breadth | Insufficient breadth | Do not infer a tradeable effect from the raw event chart |

### Sectors and item codes

- Communication services has a positive exploratory prior-momentum increment, but coverage is only 37 months and it does not survive multiple testing.
- Health care, consumer staples and industrials show negative exploratory prior-momentum increments. These are possible reversal hypotheses, not validated implementations. None survives the 5% multiple-testing threshold.
- Earnings (2.02), management/compensation (5.02), Reg FD (7.01), material agreements (1.01), and debt obligations (2.03) show no reliable positive momentum improvement.
- Other events (8.01) show a negative prior-momentum increment of approximately 0.79% per month (t −2.58), but q is about 0.15 and the group is economically heterogeneous.
- Restructuring, financial non-reliance and listing notices are too sparse under the breadth screen to support these momentum portfolios. Their value as risk flags remains a different question.
- No candidate with annual walk-forward selection produced a pooled adjusted-increment t-statistic of 2. Most size groups were never selected. Selection is a diagnostic of incremental association, not an implementable four-leg strategy.

### Cost sensitivity

A fully replaced 100%-long/100%-short portfolio trades 400% of NAV each monthly round trip. At illustrative one-way costs of 10 or 25 basis points, this costs 0.40% or 1.00% per month. The screened filing-stock spreads above become:

| Signal | Gross | After 10 bp/full replacement | After 25 bp/full replacement |
|---|---:|---:|---:|
| Prior-three-month momentum | +0.057% | −0.343% | −0.943% |
| Filing-month momentum | +0.385% | −0.015% | −0.615% |

These are **sensitivity assumptions**, not measured trading costs. Persistent holdings would reduce turnover; borrow fees, spread widening, market impact and availability are unmeasured. No net-performance or capacity claim is made.

## Scope and method

This study tests monthly momentum around the filing month using the supplied local panel. It does not estimate returns immediately before or after an individual announcement. Daily returns and exact SEC acceptance timestamps are absent. An 8-K may also follow the underlying event: the SEC says companies generally have four business days to file ([SEC investor guidance](https://www.investor.gov/introduction-investing/investing-basics/glossary/form-8-k)).

- One observation per security-month, irrespective of the number of filings. Filing item groups overlap.
- “Non-filing” means **no eligible filing in the supplied subset**, not proof that no 8-K was filed. The archive covers fewer securities than the returns panel. Filing-versus-control differences can reflect coverage and issuer selection.
- Filings must be dated on or before the panel's recorded trading snapshot date. Later calendar-month filings are excluded from that month's event flag. The date-only data still cannot resolve a release after the close on the snapshot date.
- Pre-filing-month momentum compounds raw returns in months −3, −2 and −1. Filing-month momentum is month 0's raw return; it combines time before and after the filing and is only usable at the monthly rebalance.
- The outcome is the supplied next-month excess return, checked against the next calendar month's `ret_exc`. Calendar-key joins prevent lags from crossing gaps. No future returns enter momentum or group breakpoints.
- Momentum spreads buy the top 30% and short the bottom 30%, equal weight within each leg, with at least ten stocks per leg. Event and control stocks share the same month/group momentum cutoffs. Each spread has 200% gross exposure. Missing future outcomes are excluded, so these are research spreads, not complete executable portfolios.
- Incremental value means the event-stock momentum spread minus the control-stock spread. The adjusted version first subtracts the equal-weight return of contemporaneous company-cap/GICS-sector peers. This is a diagnostic adjustment, not full factor neutralization or a causal estimate. For item groups, controls are stocks without that item, including stocks with other filings.
- Screened results require absolute price ≥ $5 and dollar volume above the month's 20th percentile. These screens do not establish short availability or realistic capacity.
- Company market equity, in USD millions, defines nano <50, micro 50–300, small 300–2,000, mid 2,000–10,000, large 10,000–200,000 and mega >200,000. These differ from the dataset's NYSE-relative `size_grp`. GICS comes from monthly panel rows; historical classification accuracy is not independently audited.
- Sample splits use **holding month**: train through 2018, validation 2019–2020, descriptive test 2021–2026. The latter period has already been explored in prior repository research and is not an untouched holdout.
- Annual selection follows `src/graph/walk_forward.py`: expanding training, preceding two years for validation, training |HAC t| ≥ 2 with ≥24 observations and a same-sign validation mean with ≥12 observations. Selection uses adjusted incremental value. It diagnoses stability; it does not constitute a portfolio implementation.
- Reported t-statistics use six-lag Newey–West errors on monthly spreads, not independent-filing errors. Benjamini–Hochberg q-values adjust across the candidate groups/signals within each period and metric. Sparse groups and missing months weaken inference; q-values do not account for all earlier repository research.
- Event-path plots average each month's event-stock returns, then average those monthly means. Adjacent events overlap; stocks can contribute in multiple months. Month 0 is descriptive, not tradeable post-filing profit. Path samples can differ by offset where returns are unavailable.

**Important chart limitation:** size, price and liquidity are measured in month 0. Conditioning on these values can select stocks that already appreciated before the filing, especially tiny stocks that crossed the $5 price screen. The apparent pre/post drop in raw event paths must not be interpreted as a filing-induced reversal. The forward predictive tests are the basis for the recommendation.

## Reproduce

Run `python -m src.filing_momentum_study` from the repository root with pandas, NumPy, PyArrow and Matplotlib available. Outputs are in `output/filing_momentum/`: audit, coverage, monthly spread observations, summary statistics, event paths, chart and annual selection decisions.

No production model, score, or portfolio setting is changed.

## Content analysis extension

`python -m src.filing_content_study` scans the complete supplied main text of every filing in batches of 512, caching compact flags by document ID. It first ran a 5,000-filing pilot, and matched excerpts were inspected before the full run. The pilot revealed false dividend-cut matches inside “executive” and generic credit-default clauses; word boundaries and explicit-action patterns address these cases. Five regression checks cover these distinctions and missing-calendar-month handling.

The body-derived flags identify **mentions** of guidance changes, buybacks, dividend changes, distress, non-reliance, restructuring, acquisitions and capital financing. They are transparent regex screens, not exhaustive semantic classifications. They can include stale announcements, conditional plans or third-party events. Capital financing includes both debt and equity. No matched flag by itself establishes a new material surprise. Full semantic evaluation of every document has not been performed.

The independent content-direction signal reuses the existing FinBERT sentiment scores, joined by unique document ID, with equal-weight aggregation within a stock-month. The existing scorer processes at most the first 12,000 characters per document and averages chunk positive-minus-negative probabilities. This limitation is audited in `content_audit.json`; it is not represented as full-document LLM understanding. The deeper LLM cache contains only 2,309 successfully parsed filings and is too sparse for the broad study.

Content tests compare next-month returns across positive/negative tone ranks and momentum ranks within each size, sector or body-flag group. Additional tests retain momentum winners with positive tone and losers with negative tone, or the opposite combination. Different breadth and membership can affect these comparisons. Content-adjusted returns subtract cap/sector peer means among screened **filing stocks**; they are not the same benchmark as the event-versus-control study. Rank correlations require ten observations; spread legs require ten each. All output records retain sample counts.

The study measures predictive association, not whether the disclosure caused the move. Exhibits such as the earnings press release may be absent from the supplied main text. A more sophisticated content model could differ from these tested signals.
