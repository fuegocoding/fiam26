# Connected research campaign

Run `python final_v2/run.py` in the project environment. The root configuration selects the end-to-end research campaign. One-off tests, checks and verification scripts belong in `/tmp`; the runtime independently replays its saved target instructions before reporting.

The campaign connects dated paper discovery, approved provider receipts, all supplied characteristics, local 8-K text measurements, agent-written Python, protected evaluation, measured feedback, permanent pod ownership, continuous capital allocation, full observed-price fills, daily NAV and competition reporting. Theo's saved portfolio and fitted models are not imported.

The candidate pool starts empty. All strategy code comes from actual model
responses with recorded provenance. No built-in economic controls, deterministic
strategy factories, fixed-rule trading counterfactuals or default feature models
enter the workflow. If agent research produces no supported candidate, the
result reports an incomplete pool and makes no substitute allocation.

1. Read only authorized development observations. Preserve missingness and distinguish the full research universe from the fund's 200–500 position constraint.
2. Search paper versions dated by the development cutoff, retain their actual text and identifiers, and discover provider metadata within durable request limits. A provider receipt or a publication timestamp alone does not certify historical vintages or issuer mappings.
3. Compute dated filing counts, transparent literal text/event measurements, and any agent-proposed literal phrase measurements locally. Raw filing text is not transmitted automatically. Unseen issuer coverage remains unknown. The supplied filing corpus's survivorship limitation remains explicit.
4. Ask the configured model for executable numerical Python, a mechanism, falsification and timing rules, one pod at a time. There is no requirement to reuse a previous proposal. Request and response hashes, actual provider identity and all failures are retained. Calls respect provider rate limits and retry transient failures. There are no API spending quotas or spending ledgers.
5. Execute each strategy using only current/past ranked features, up to twelve historical monthly snapshots and up to 252 past daily returns. Imports, IO, credentials, arbitrary calls, new array dimensions and evaluation outcomes are excluded by the numerical language. The model cannot edit its evaluator or approve its own performance.
6. Measure the full concentration curve before capital. Choose broad supported concentration regions using prior settled observations; use the fund's maximum name count as a declared construction ceiling when no region has sufficient evidence. A default research construction does not establish an edge.
7. First request several independent economic research topics per pod. Explore each initial topic before tuning. Each proposal declares a cap/sector/observable cohort and recent model/evidence windows; fit, score and trade on that same cohort. Return rejected source and exact failures for bounded immediate repair. Feed measured uncertainty, IC, side attribution, coverage and chronological training windows into refinements. Compare children and parents on common settled dates. Continue without a campaign round ceiling; send stalled refinements back for diagnosis, further tests or an explicit exhaustion decision. Record unresolved provider failures separately. New topics may follow measured failures. Five supported candidates per pod is a research target, not a forced acceptance or stopping rule. Adaptive training diagnostics are not untouched historical validation.
8. Select up to three supported, diversified strategies per pod. Ownership and funding share the same settled-history and uncertainty-adjusted edge criteria. A 95% lower bound ranks candidates; it is not an additional research-acceptance threshold. Correlation below 0.85 diversifies ownership, while correlation of 0.98 or more identifies pool duplicates. Empty books remain explicit when evidence is insufficient. Fund and pod limits, covariance and portfolio risk determine capital; the pool target does not force investment.
9. Preserve native daily, weekly or slower signal reviews. Monthly CIO reviews do not force every sleeve to refresh its forecast. Each scheduled target fills immediately in full at the first supplied observed close on or after its decision date and earns returns subsequently. There is no execution model.
10. Post cash interest inside the execution NAV, save actual holdings and sparse target instructions, and replay the saved instructions before publishing a research report. Performance uses supplied observed returns and posted cash interest.

Artifacts live under the configured private state directory. `latest_campaign.json` points to the immutable-input campaign directory. This contains source snapshots, input digests, evidence, generated `.py` files, every attempt, feedback, pod books, capital allocations, daily/monthly ledgers, actual holdings and a frozen research book. The independent verifier checks input hashes, timing, ownership, capacity, holdings, daily P&L and final NAV.

Headline reporting separates arithmetic annualized return from CAGR, Sharpe against cash from information ratio against TB3MS+4%, and monthly from daily statistics. S&P excess-return regressions report observed coverage and standard errors; missing licensed history is not fabricated. The explicitly configured FRED container is projected and filtered to reporting series only. Other legacy external rows are never alpha inputs.

A completed run means the connected experiment executed. Financial admission is a separate conclusion: constraints, stability, data provenance, attempt multiplicity and untouched evaluation must support it. Nonempty research rosters and a successful replay do not certify alpha.

## Audit remediation and current experiment discipline

The independent engineering and institutional-PM reviews led to these changes:

- Carry observed signal membership through sign-preserving construction. Missing signals never become implicit shorts. An optional liquid sector hedge is explicitly a separate construction choice; it does not certify beta neutrality.
- Measure every deployable concentration using the constructor and native retention that actually trade. These execution-to-execution labels differ from calendar-month reporting; an exact daily-replay regression checks the distinction. Whole-universe rank-IC curves remain forecast diagnostics.
- Keep each sleeve's actual dollar positions and native reviews. An unfunded daily idea cannot refresh a monthly book, including when they share the same stocks.
- Allocate continuously using aggregate incremental stock orders, portfolio-risk limits, and prior return covariance. Report expected trading dollars and the TB3MS+4% hurdle separately. Compression to the fund's holding ceiling is an explicit overlay whose results must be assessed in the actual final ledger.
- Run numerical strategies in a fresh subprocess with hard address-space, CPU and packet limits. Inputs are read-only; local array computations and masked assignments are permitted. There is no inherited provider client or credential context.
- Key daily/feature caches by exact source digest, universe, dates and representation. Validate content hashes, create unique run directories, and retain evidence/model identities. Cache successes for reproducibility; bounded failed-discovery retries retain their charges and history.
- Publish actual acquisition and agent-stage health separately from implemented capability availability. A run with rejected model proposals is explicitly degraded.
- Record long, short/hedge and cash contributions. The independent verifier checks each component against replay, recomputes headline metrics and constraints, and rejects duplicate holdings, altered sources/evidence or exceeded pod limits.

Live model feedback contains feature definitions, mandates, public paper excerpts,
aggregate development metrics, generated strategy code and exact failure reasons.
Raw filing text, stock-level rows and credentials stay outside that feedback.
Research stops when the topic frontier converges or an unresolved failure
prevents further work. Research-progress receipts distinguish those outcomes.
Neither a round count nor spending counters control research.

Data-heavy jobs are serialized. The loader streams characteristics, retained arrays use disk backing, and a worker guard stops excessive resident memory or low host/cgroup headroom. No test or successful replay establishes profitable alpha by itself.
