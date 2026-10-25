# Carbon Arc historical downloader

## Current scope: the entire accessible catalog

The active task is **all accessible Carbon Arc datasets, across all companies and available dates**. The NVIDIA-specific runner has been stopped; it submitted zero paid analytical requests.

The full-catalog collector is running against 581 dataset entries currently visible through MCP:

```bash
local/carbonarc-venv/bin/python scripts/carbonarc/download.py catalog-prepare
local/carbonarc-venv/bin/python scripts/carbonarc/download.py catalog-run
local/carbonarc-venv/bin/python scripts/carbonarc/download.py catalog-status
```

These commands collect free dataset provenance and field dictionaries under `data/carbonarc/catalog/`. They have no company filter and cannot submit paid analytical queries. Prior request/cooldown history was imported from the stopped NVIDIA queue so changing directories does not reset the account's rate accounting. `datasets/*.json` preserves each retrieved dataset definition, and `responses/` preserves every exact MCP response. Catalog visibility does not prove a full-data entitlement.

The user's downloaded `carbonarc_library_2026-10-24.zip` was also copied into `data/carbonarc/catalog/library_archive/`: 625 catalog entries, 621 field-dictionary CSVs, and 299 sample workbooks. Its `manifest.json` records source and per-file hashes. These are documentation and samples, not full historical extracts.

### Verify bulk access

Complete row-level data delivery is documented through **Block**, using Polaris/Iceberg or S3. Block access is separate from MCP access. No API token, Polaris credentials, or AWS credentials were configured in the inspected locations. The browser connector could not inspect the existing account because its browser failed to start without an X server.

A private, ignored file is prepared at `local/carbonarc-api.env`. Put the API token from [User Portal → Profile](https://app.carbonarc.ai/my/profile) in its empty field:

```dotenv
CARBONARC_API_TOKEN=your_api_token
```

Then run the read-only access check:

```bash
local/carbonarc-venv/bin/python scripts/carbonarc/check_access.py
```

It checks API authentication, calls the official SDK's `client.block.list_datasets()`, and saves `bulk_access_status.json` plus `block_catalog.json` when available. It identifies existing approved/evaluation/contracted dataset variants, excludes explicitly blocked sources, and never submits access requests, purchases, or ARN changes. It does not read Codex's OAuth credentials or print API tokens. Actual bulk transfer can be configured once these entitlements and delivery credentials are established. The earlier 3–6 hour estimate covered a NVIDIA metric pass and is not an estimate for downloading the entire catalog.

Official sources: [API authentication](https://docs.carbonarc.ai/developers/getting-started/api-auth), [Block API and access requirements](https://docs.carbonarc.ai/developers/block-for-devs/block-api), [Block data delivery](https://docs.carbonarc.ai/platform/block/block-data-delivery), and [Library API](https://docs.carbonarc.ai/developers/library-for-devs/library-api).

## Earlier NVIDIA analytical prototype

This downloader uses the official Python MCP client, saves a SQLite queue, and resumes after interruption. It is preconfigured for the NVIDIA inventory in `data/carbonarc/nvidia/discovery.json`.

## Run

The local environment has already been installed. From the repository root:

```bash
local/carbonarc-venv/bin/python scripts/carbonarc/download.py start --discovery-only
```

On the first run, finish Carbon Arc's OAuth authorization in the browser. This command gathers free metadata and prepares historical queries. After discovery completes, `run --max-requests 10` runs a capped analytical batch. The standalone client needs its own login; Codex's protected OAuth session is not exported or modified. Subsequent runs reuse and refresh the standalone login.

For a fresh environment:

```bash
python -m venv local/carbonarc-venv
local/carbonarc-venv/bin/pip install -r scripts/carbonarc/requirements.txt
```

Commands can also be run separately:

```bash
local/carbonarc-venv/bin/python scripts/carbonarc/download.py prepare
local/carbonarc-venv/bin/python scripts/carbonarc/download.py login
local/carbonarc-venv/bin/python scripts/carbonarc/download.py run
local/carbonarc-venv/bin/python scripts/carbonarc/download.py status
```

`inspect` authenticates and saves the remote server's actual tool schemas to `server_tools.json`. `--max-requests 5` limits a run to five queued requests, including rejected attempts. `--max-paid-requests 10` imposes a **persistent lifetime ceiling of 10 analytical attempts in this output directory**, including rate-limit rejections and attempts from earlier runs. It does not reset on restart. The runner finishes free discovery first, then submits at most the remaining allowance. Without either cap, the process works until the queue is exhausted or an account/connection error needs attention. Analytical calls incur Carbon Arc usage charges. There are no automated dataset purchases.

For free discovery alone, run `run --discovery-only`. That mode can prepare historical query jobs but never submits them. `run --wait-for-login --discovery-only` can wait for a separate browser login and then gather free metadata. To limit a later paid batch, use `run --max-requests 10` after free discovery finishes.

`Ctrl+C` stops the process. Rerunning `start` resumes pending work. Free requests interrupted in flight can be retried automatically. Paid requests interrupted in flight become `uncertain`, since the service might already have executed and billed them. Such requests are preserved for review and are not automatically duplicated.

## Rate limits and recovery

Carbon Arc's documentation search reported:

| Tools | Requests/minute | Requests/hour |
| --- | ---: | ---: |
| `text_to_insight`, `framework_to_insight` | 8 each | 90 each |
| `search_insights` | 45 | 500 |
| `get_insights_from_entity`, `get_entities_from_insight` | 30 each | 350 each |

The downloader conservatively combines both analytical tools into one 8/minute, 90/hour bucket. Other discovery tools default to a conservative 30/minute, 350/hour local policy; those defaults are not vendor-confirmed quotas. The authoritative server response always takes precedence.

Every attempt is persisted before sending, including rejected calls. The process checks both rolling windows, adds a one-second boundary margin, and persists server cooldowns. A rate-limit response honors `metadata.retry_after` or HTTP `Retry-After`, with increasing backoff for repeated rejections. HTTP 429 reconnects after waiting and preserves the rejected job. If a transport interruption makes a paid request's outcome uncertain, automatic replay requires a matching rejected HTTP POST; otherwise the runner stops for review. It retries the same task/tool; it does not rotate tools, users, endpoints, or credentials to evade a limit.

The ledger accounts for this output directory's downloader. Codex, other clients, or another output directory can consume the same account quota; server cooldowns handle that contention. An exclusive process lock prevents two runners from using this directory at once. Account balance/daily-quota errors stop execution.

## Dates and completeness

`prepare` seeds 201 numerical insight discovery jobs and two entity-event discovery jobs. Each numerical insight's `data_library` response supplies the advertised dataset history. The first numerical request starts at the beginning of its earliest advertised year and ends at today's UTC date. When history is missing, the default lower bound is `1900-01-01`; this is a search bound, not a claim that NVIDIA observations exist that far back.

Use `--start YYYY-MM-DD` to override the lower bound or `--fallback-start YYYY-MM-DD` to change only the unknown-history fallback. Supply those options to `run`/`start`, since history jobs are generated during discovery. `--end YYYY-MM-DD` freezes the upper bound for reproducibility. Existing queued jobs retain their original date bounds.

The MCP `framework_to_insight` input schema exposed in this session lacks an explicit date-range field. Live query metadata showed an undocumented internal `filters.date_range`, and an all-history natural-language request silently defaulted to two years. The downloader therefore uses **explicit inclusive dates in metric-specific `text_to_insight` requests**, without inventing input-schema keys. It records server-applied defaults as incomplete and splits them into smaller ranges.

Oversized results split into nonoverlapping inclusive halves and queue the older half first. A single-day result that is still too large is marked `needs_bulk_export`. Errors such as `ERROR 101` or `KeyError: '__import__'` are recorded as errors, not empty datasets, and are not retried endlessly. Server-side error detection cannot guarantee that every narrative error is recognized; raw responses remain available for inspection.

Responses with data are labeled `downloaded_unverified`. An MCP analytical table is not proof of a complete raw-data dump, accurate native resolution, all dimensions, or historical-vintage availability. The runner does not falsely label such output complete. Event/relationship insights are in the saved discovery inventory; event discovery is saved separately and is not passed through a numerical metric query.

## Saved files

Everything generated goes under `data/carbonarc/nvidia/`, which the repository already ignores:

- `downloads.sqlite`: jobs, attempt ledger, and cooldowns.
- `plan.json`: planned scope and limits.
- `server_tools.json`: actual remote MCP schemas after login.
- `exposed_tools.json`: all 29 analytical/research MCP tools exposed to this Codex session.
- `responses/*.attempt-NNNN.json`: exact request, full MCP response, instructions, metadata and receipt time for each attempt.
- `responses/*.md`: returned analysis, including its original metadata header.
- `responses/*.table-N.csv`: tables extracted verbatim, without guessed numeric conversion.
- `downloader_probes.json`: the live validation failure and internal-error/default-date-window response observed while developing this downloader.

OAuth client information and refresh tokens live under ignored `local/carbonarc-auth/`, with mode `0600`. They are not copied from Codex or written to data output. `CARBONARC_TOKEN` can supply a bearer token for the configured MCP endpoint instead. Configured static/environment HTTP headers are also supported. Never use a REST API token on an MCP endpoint unless Carbon Arc confirms that audience is supported.

## Full raw-data export limitation

All 29 exposed analytical/research tools were inspected. Discovery tools return entity/insight/ontology/schema information. The two numerical analytical tools return Markdown analysis. The research tools return strategic reports. No exposed tool provides bulk-export or pagination parameters.

Carbon Arc's documentation search separately identified `GET https://api.carbonarc.co/v2/framework/data`, a bearer-authenticated REST endpoint for purchased frameworks. Its SDK example uses `get_framework_data(framework_id=..., fetch_all=False, page=1, size=7000, data_type='dataframe')`. The response did not document the REST parameter/response schema; those parameters were inferred from the SDK example, not verified against the endpoint. No suitable REST credential or purchased framework ID was available in this session, so the script does not invent a bulk export call or automatically invoke framework purchase endpoints. Complete underlying data frames require that API or a platform export. The MCP downloader still saves every returned analytical result it can retrieve.

References for the client implementation: [official MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk/tree/v1.x) and [OAuth client example](https://github.com/modelcontextprotocol/python-sdk/tree/v1.x/examples/clients/simple-auth-client). SDK version is pinned in `requirements.txt`.

## Verify

```bash
local/carbonarc-venv/bin/python -m unittest discover -s scripts/carbonarc -p 'test_*.py' -v
```

The tests use a local fake clock and fake responses. They cover rolling quotas across restarts, rejected-attempt accounting, retry intervals, date splitting, idempotency, in-flight crash state, response/CSV preservation, and OAuth-state permissions/expiry. They send no live paid analytical queries.
