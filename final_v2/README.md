# final_v2 — strategy pod research platform

Auditable research pods (fundamentals, events, valuation, dynamics) that read papers,
queue research jobs, draft proposals and run only *registered* experiments. A complete
reference workflow connects challenge, scoped admission, portfolio replay, accounting,
dossiers and freeze evidence. Synthetic certificates authorize test books only. See
`BUILD_STATUS.md`, `SPEC_COVERAGE.md` and `HANDOFF.md` for verified behavior and remaining evidence.

## One-year trial

[2025 trial results](YEAR_TRIAL_2025.md): full-year synthetic execution passed; live paper research produced proposals, but the real ideas remain unadmitted. See the report for provider failures, registered design follow-ups and remaining automation/data gaps.

[Component research](RESEARCH_PROGRESSION.md) now permits partial paper ideas and original hypotheses to run small registered diagnostics before a complete strategy/data build. Three saved research ideas were replayed on real data without new model API calls; trading gates remain separate.

## Complete application

[Source-to-book commands and remaining gates](AUTONOMOUS_PLATFORM.md). The complete offline run uses the actual four-pod scheduler, registered paper adaptations, frozen independent checks, publications, covariance/shared-hedge CIO, fills/NAV and feedback:

```sh
python scripts/run_complete_platform.py --state-dir /tmp/pods/state --output-dir /tmp/pods/out
```

Two planted candidates are admitted, another remains shadow, and one fails design execution; all certificates are fixture-scoped. Real candidates are never admitted to make the demo succeed.

## Install and test

```bash
cd final_v2
pip install -e '.[research,test]'     # pydantic core; research = numpy/pandas/pyarrow/pypdf
python -m pytest                      # no network, no model, no GPU
```

Examples below run from `final_v2/`. State lives in one `--state-dir` (`queue.sqlite`,
`cache.sqlite`, `memory/<namespace>.sqlite`, `shared/papers.sqlite`, `watches.sqlite`).
Pod ids: `fundamentals`, `events`, `valuation`, `dynamics` (namespaces `pod_<id>`).

## Papers (`scripts/research_papers.py`)

Offline by default (SQLite cache only). `--online` is required for HTTP to arXiv/OpenAlex.
`--as-of` switches to historical replay (version dates must be on or before the cutoff).
Use `--db <state-dir>/shared/papers.sqlite` so the MCP server and watches see the same cache.

```bash
S=/tmp/pods
PAPER_ID="arxiv:1706.03762v1"   # use an id printed by search/ingest
python scripts/research_papers.py --db "$S/shared/papers.sqlite" --online search "post earnings announcement drift" --limit 5
python scripts/research_papers.py --db "$S/shared/papers.sqlite" --online ingest "$PAPER_ID"
python scripts/research_papers.py --db "$S/shared/papers.sqlite" ingest-local notes.md \
    --title "My paper" --first-public 2020-01-01T00:00:00+00:00
python scripts/research_papers.py --db "$S/shared/papers.sqlite" claim "$PAPER_ID" --kind universe --quote "exact sentence"
python scripts/research_papers.py --db "$S/shared/papers.sqlite" replicate "$PAPER_ID" --label INCOMPLETE
```

Replication specs fall back to `INCOMPLETE` unless full text and verbatim-quoted claims exist.

## Four-pod demo (mock, no model)

`--demo` uses a deterministic responder labelled `[DEMO MOCK]` (`provider_is_mock=true`).
It exercises queue, cache, validation and memory; it is **not** a language model.

```bash
cat > "$S-cards.json" <<'EOF'
[{"paper_id": "demo:paper1", "title": "Demo card", "abstract": "Synthetic card for plumbing tests.",
  "available_at": "2020-01-01T00:00:00+00:00"}]
EOF
for pod in fundamentals events valuation dynamics; do
  python scripts/run_pod_research.py submit --state-dir "$S" --pod "$pod" \
      --question "Demo question for $pod" --paper-cards "$S-cards.json" --executor-id demo_executor
done
python scripts/run_pod_research.py run --state-dir "$S"                     # dry run: claims nothing
python scripts/run_pod_research.py run --state-dir "$S" --demo --max-jobs 4
python scripts/run_pod_research.py list --state-dir "$S"
python scripts/run_pod_research.py memory --state-dir "$S" --pod valuation
```

A worker claims jobs only for the pods it has runners for; jobs of any other namespace stay
queued and untouched (no claim, failure or lease recovery).

Drafts are stored `evaluated=false, admission=not_admitted`. Only an executor registered
through `--executors-module your.module` (exposing `register_executors(registry)`) or the
shipped `--dataset-manifest` built-ins can create an `evaluated_result`, only on the design partition.

## Real model (explicit opt-in, may bill)

The strategy's exact lead/challenger/extractor trial is prepared separately in
[LLM_TRIAL.md](LLM_TRIAL.md). Run `python scripts/try_strategy_models.py plan` for
an offline plan. One-task live smoke checks passed for Groq gpt-oss and local Qwen8. The exact thinking lead endpoint and full quality/adoption benchmark remain pending.

```bash
python scripts/run_pod_research.py run --state-dir "$S" --online --provider claude-code --model sonnet
python scripts/run_pod_research.py run --state-dir "$S" --online --provider openai \
    --base-url https://HOST/v1 --model NAME --api-key-env MY_KEY_ENV_VAR
```

`--api-key-env` names an environment variable; keys are never passed on the command line,
stored, or exposed by MCP. Historical mode needs `--checkpoint-id` at submit and
`--registry-file` at run (release-gated model checkpoints). Budgets: `--max-calls`, `--max-tokens`
persist across restarts in a named SQLite account. Use `--budget-id`, `--budget-cycle`
and `--budget-scope` deliberately; reopening an account cannot reset usage. Failed calls
remain charged. Unknown Claude CLI usage retains the reservation rather than inventing
precise token counts.

## MCP stdio (one process per pod)

```bash
python -m pod_platform.services.app --state-dir "$S" --pod valuation            # offline papers
python -m pod_platform.services.app --state-dir "$S" --pod valuation --online   # permit paper HTTP
```

Register that command as a stdio MCP server in your client. The pod is fixed at startup;
no tool takes a namespace. Tools: `platform.status`, `sources.list`, `papers.search|get|read|
ingest|replication`, `research.submit|list|show`, `memory.list|note`, `experiments.register|
evaluate`, `features.materialize`. There is no shell, key, or admission tool, and no model
call: `research.submit` only queues `propose` (evidence by `paper_ids`/`source_ids`, resolved
server-side) or `experiment` (a stored `draft_entry_id`) jobs in the same `queue.sqlite` the CLI
worker consumes. Jobs and memory are visible only to the owning pod.

## Recurring paper watches (explicit ticks, no daemon)

```bash
python scripts/research_watch.py add --state-dir "$S" --pod valuation --watch-id spread \
    --topic "value spread predictability" --cadence-hours 168 --executor-id value_spread_ic
python scripts/research_watch.py tick --state-dir "$S"            # offline: cache only
python scripts/research_watch.py tick --state-dir "$S" --online   # network search
python scripts/research_watch.py list --state-dir "$S"
```

A tick queues bounded `propose` jobs with the papers actually returned, once per paper
version per pod and watch. It calls no model; run the CLI worker to process the jobs.
Failed searches are recorded as `failed` and retried within an hour.

## Replay and daily-market realism

`pod_platform/orchestration/replay.py` holds the source-eligibility gate, monthly accounting
replay and `ControlledReplay`; `scripts/run_controlled_replay.py` is its CLI wrapper.
`ControlledReplay` applies the same independent risk policy to the target book and again to the
actual post-fill weights (held-name union, measured net/gross/beta); a BLOCK/INFEASIBLE or
unproducible actual-fill review stops the run and the evidence is kept in
`risk_reviews_actual_fill[.partial].json`. This is monthly, not daily: no margin or daily
execution realism is modeled. `scripts/run_first_book.py`, `run_ported_book.py` and
`run_seven_book.py` run books from the competition characteristics panel (`--help` or file
docstring for inputs). Those are controlled fixtures, not a daily-market simulation.


## Reference workflow and four-charter campaign

```bash
python scripts/run_reference.py --mode demo --state-dir /tmp/fiam-reference/state --output-dir /tmp/fiam-reference/out
python scripts/run_strategy_research.py --fixture-demo /tmp/fiam-campaign/data --state-dir /tmp/fiam-campaign/state --summary
python scripts/run_strategy_research.py --manifest /path/operator-manifest.json --dataset-id comp_chars_design --research-asof 2026-10-25T00:00:00+00:00 --state-dir /tmp/fiam-research --summary
```

The reference writes measured fixture admissions/rejections, target/post-fill risk, fills,
self-financing accounting, complete paired counterfactual books, ablations, attribution,
dated dossiers, judge facts and a timestamped commitment. Repeating the same run verifies
completed stages; interrupted attempts retain evidence and quarantine declared incomplete
outputs. Changed input/code bindings require a new run identity. Fixture success is not
investment validation. The provider hook requires checkpoint/adoption evidence and a
persistent budget; its default stand-in is explicitly not evidence of LLM skill.

The design campaign pre-registers all four charters before reading labels, runs bounded
ridge/rank recipes, records conventional baselines and net return bridges, and never
self-admits. Operator manifests specify paths, source cards, snapshots, schema and clocks;
see `configs/campaigns/competition_panel_manifest.example.json`. Protected partitions are
not available to the research executors. A positive belief metric alone cannot admit a strategy.

## Daily source and Arcana review export

```bash
python scripts/check_daily_reporting.py --file ../data/dsf_mini_from_2014.csv --security-id 10026 --start 2021-01-01 --end 2021-01-29 --out /tmp/daily-check.json
python scripts/check_daily_reporting.py --help
python scripts/export_arcana.py --help
```

Daily reconciliation requires explicit cash, positions, dated fills and risk-free rates;
monthly performance is never interpolated into daily history. The source-wide calendar
prevents missing held observations from disappearing. The default evaluation cutoff is
2026-08-31; later dates require descriptive mode. `prc` is only a reference price and
`daily_return` supplies total returns. Missing values are not clipped or zero-filled.

Arcana exports are locally reviewed CSV/JSON with explicit field mapping, units and hashes.
Without a supplied vendor schema they remain `canonical_review_only`. Arcana's OAuth MCP
endpoint is `https://app.arcana.io/mcp`; credentials belong to the client, outside the repository.
The local Codex connection authenticated successfully, with read-only scopes. A new client
session may be needed to discover the new tools. No portfolio upload was performed.
