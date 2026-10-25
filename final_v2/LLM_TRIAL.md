# Trial of the strategy's proposed LLM team

Prepared from spec sections 20.15 and 20.18–20.20. One-task live smoke checks now passed on Groq gpt-oss and local Qwen8; the exact thinking lead endpoint and full quality/adoption benchmark remain pending. Existing local weights were used; no new weights were downloaded.

| Role | Exact repository | Trial behavior |
|---|---|---|
| Lead | `Qwen/Qwen3-30B-A3B-Thinking-2507` | Deliberate research, experiment, baseline and rejection rule |
| Challenger | `openai/gpt-oss-20b` | Medium reasoning; concrete disconfirming tests; cannot admit |
| Extractor | `Qwen/Qwen3-8B` | Explicit non-thinking; bounded verbatim extraction |
| Optional upgrade | `Qwen/Qwen3.6-35B-A3B` | Separate benchmark candidate; never an automatic replacement |

`configs/models/strategy_roster.proposed.json` also records the complete proposed
historical table. It is not a frozen adoption roster. Release-gated historical
use still requires checkpoint evidence and recorded adoption; the early NeoX
Policy B proposal remains disabled. The trial CLI performs contemporary research
only and cannot enable historical backcasts or deploy a portfolio.

## Prepared files

- `configs/models/serving.example.json`: the three explicit endpoint profiles.
- `configs/models/serving.upgrade.example.json`: adds the optional Qwen3.6 benchmark.
- `configs/models/upstream_metadata.json`: public upstream commit IDs, published
  Safetensors LFS hashes and retrieval timestamps. These are actual metadata,
  not invented hashes, release dates, or evidence of remote serving identity.
- `configs/models/benchmark_v1.json`: 50 versioned synthetic fixed-choice cases:
  15 source/timestamp, 15 paper/method, 10 economic/experiment, 10 tool/recovery.
- `configs/models/trial_packet.example.json`: synthetic evidence for the full workflow.
- `scripts/try_strategy_models.py`: offline plan, staged workflow and fixed-task benchmark.

A follow-up check outside the restricted tool sandbox confirmed a working NVIDIA
GeForce RTX 3070 Ti Laptop GPU: 8192 MiB VRAM, 7850 MiB free at inspection, with
driver 615.78.08. Earlier `nvidia-smi` failures came from the sandbox hiding device
nodes, not a broken laptop driver. System RAM is 16 GB, with approximately 5 GB
available. A quantized `qwen3:8b` Ollama model is already installed. Smaller
quantized local candidates can be benchmarked separately; the original large
lead cannot reside entirely in this GPU's memory at the prepared precision.
The installed model's observed manifest does not verify original weight lineage.
Choose compute before attempting the complete team. An Ollama extractor profile
must explicitly pin its observed tag digest, architecture, alias and quantization;
GGUF conversion lineage remains separate from original checkpoint verification.

## Inspect the plan now

From `final_v2/`:

```sh
python scripts/try_strategy_models.py plan
python scripts/try_strategy_models.py --help
```

`plan` makes no network or model calls. It prints revision-pinned candidate vLLM
commands bound to localhost with an 8192-token context and one sequence. They are
not runtime-verified until the engine build, chat template, parsers and selected
hardware have been tested. Use SSH forwards for a remote self-hosted server, or
configure explicit HTTPS API endpoints and named key environment variables.
Do not put keys in JSON or source control. Hosted APIs receive no server-launch command.

Do not launch all example servers on one GPU simultaneously. Benchmark one model
at a time, or use separate endpoints. The workflow can resume at each role boundary,
allowing one GPU to serve the roles sequentially. Inference requires `--online`.
Nothing invokes a paid service or launches a GPU just by loading the configuration.

## Run after choosing compute and verifying serving profiles

Copy the endpoint example into an ignored private state directory and configure
the actual endpoint aliases, revisions, quantization and credentials there.
An alias attestation is an explicit operator mapping, not checkpoint verification.
The provider rejects a different echoed model and never silently substitutes another.

Example staged synthetic workflow, using the same inputs, trial ID and budget cycle:

```sh
python scripts/try_strategy_models.py workflow --endpoints /path/endpoints.json --trial-id workflow-v1 --cycle smoke-v1 --stage extractor --online
# Release extractor server; start the configured lead server.
python scripts/try_strategy_models.py workflow --endpoints /path/endpoints.json --trial-id workflow-v1 --cycle smoke-v1 --stage lead --online
# Release lead server; start the configured challenger server.
python scripts/try_strategy_models.py workflow --endpoints /path/endpoints.json --trial-id workflow-v1 --cycle smoke-v1 --stage challenger --online
```

Each later stage reuses the cached prior outputs. The same packet flows from
extractor to lead to challenger. Quotes must exist in the original evidence;
extracted dates/measures must be verbatim. The challenger must abstain from
admission. Reports remain research drafts, including when all three calls succeed.

Default budget: **three calls and 40,000 reserved tokens per persistent cycle**.
Failed calls stay charged; restarting cannot reset usage. Changed inputs, profiles,
code or task versions require a new trial ID, while the named cycle retains its
budget. Provider failures and exhausted budgets stop the benchmark. There is no
automatic retry, provider fallback, model upgrade, or tool execution.

A small benchmark defaults to three cases. A full repeated benchmark is 50 tasks
twice per role (100 calls per role). Set and review the shared cycle budget before
running it. For example, the following cycle permits at most 300 calls total across
the three roles; it does not spend them unless inference commands are invoked:

```sh
python scripts/try_strategy_models.py benchmark --endpoints /path/endpoints.json --role extractor --limit 50 --repeats 2 --trial-id extractor-bench-v1 --cycle benchmark-v1 --max-calls 300 --max-tokens 3000000 --online
# Repeat with --role lead / challenger and distinct trial IDs in the same cycle.
```

Use `--endpoints configs/models/serving.upgrade.example.json --role lead_upgrade`
only when deliberately benchmarking the Qwen3.6 candidate. Keep the incumbent
results and compare both on identical tasks before a separately controlled shadow run.

## Interpretation and evidence limits

Evaluator gold labels are excluded from prompts. Reports include factual/choice
accuracy, quote grounding, schema validity, critical chronology/tool errors,
repeated-answer and exact-output stability, usage, latency, cache/call hashes,
mock flags and preserved attempt receipts. The initial screen requires accuracy
at least 85%, validity at least 95%, zero critical errors and stable repeated answers
on the complete repeated benchmark. Code and thresholds are bound before calls.
Incomplete or mock runs cannot pass this screen.

These parameterized synthetic fixed-choice cases are an initial integration screen,
not a complete assessment of open-ended research. Add independently reviewed
open-ended tasks, re-identification/memorized-outcome probes and incumbent comparisons
before adoption. A passing screen still reports `adoption_allowed=false`: original
weight/training lineage, real serving identity, context/hardware verification,
shadow comparison and a scheduled adoption record remain required. No choice uses
portfolio IR as the language-model selection criterion.

Official serving references: [Qwen thinking lead](https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507),
[Qwen non-thinking extractor](https://huggingface.co/Qwen/Qwen3-8B),
[gpt-oss native format and reasoning](https://huggingface.co/openai/gpt-oss-20b),
[Ollama thinking controls](https://docs.ollama.com/capabilities/thinking).


## Subsequent live challenger check

Groq `openai/gpt-oss-20b` is now configured privately and passed one real grounded
chronology test. The private profile is `.pod_state/strategy_model_trials/endpoints.groq.json`;
credentials are read from `local/groq-api.env`. This is a challenger-only smoke,
not a completed three-model trial or adoption benchmark. See VERIFICATION.md.

## Current runtime connection and live evidence

The role providers now connect to `AutonomousResearchLoop` in the source-to-book application; see [AUTONOMOUS_PLATFORM.md](AUTONOMOUS_PLATFORM.md). No model is silently substituted for the exact thinking lead. Default demos explicitly inject mocks.

Groq gpt-oss returned the correct grounded chronology rejection in 0.930 seconds (335 input/562 completion tokens). Local Qwen8 initially failed the output contract twice; typed JSON schema, explicit public choice instructions and enum constraints then passed the same task. GPU memory observed during inference was 5,560,992,384 bytes. Failed attempts remain charged and recorded. These are connectivity/task smoke checks, not full benchmark results, adoption evidence, historical eligibility or investment skill. Private keys and receipts remain ignored local state.
