# Archived: LLM chat-completions council

Status: **archived 2026-09-22, superseded by the Jev scorer** (`src/agentic/run_jev.py`).

Contents (frozen at `council-v3` / `redaction-v2`):

- `council.py` — parallel 4-role chat-completions execution with JSON repair loop.
- `roles.py` — persona prompts (event classifier, materiality analyst, Devil's Advocate, evidence auditor).
- `tools.py` — bounded local `search_filing` / `get_filing_excerpt` tools over the redacted filing.
- `run_council.py` — CLI (`python -m src.agentic.archive_llm_council.run_council ...`).

Why archived: 4 LLM calls per filing plus parse-repair retries cost ~10–50x more
than one Jev System One request carrying all questions in parallel, with no type
errors by construction and calibrated confidence out of the box. The active
pipeline (`run_jev.py` → `output/jev_features.parquet` → `src/xgb_jev.py`)
reuses the same redaction firewall, month-`t` → target-`t+1` availability rule,
and deterministic `(permno, eom)` aggregation.

Reactivate only for a controlled LLM-vs-Jev ablation; keep the prompt version
pinned so cached `data/scoring/council_scores.db` rows stay comparable.
