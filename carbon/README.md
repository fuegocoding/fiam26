# Carbon Arc: how to connect and parse

The method for talking to the Carbon Arc data platform (`api.carbonarc.co`) from
Python: authentication, which endpoint implements which access route, what the
payloads look like, and how this folder's code turns them into DataFrames.

| File | Role |
|---|---|
| `client.py` | `CarbonClient` — Bearer auth, GET/POST JSON, pagination, slow-query job polling |
| `parse.py` | Payload parsers: catalogue, block pricing, entity map, data dictionary, query results |
| `survey.py` | Library survey: joins the screening workbook, scopes the assets, pulls data dictionaries into `out/` |
| `DATA_MAP.md` | What we can use from the library and for which of our signals |
| `smoke.py` | End-to-end check: connect, fetch, parse, print a masked summary |

```bash
cd carbon && python smoke.py
```

Competition context: the mandate's §5.2/§6 (what the platform is for, the four
screening questions, survivorship discipline) is mapped in
`../docs/final_mandate_rules_map.md`. This folder is only the plumbing.

---

## 1. Authentication

```
Authorization: Bearer <key>
```

The key is a JWT, issued per team member at Carbon Arc onboarding. It lives in
the repo-root `.env` as `CARBON_API_KEY` (gitignored; `client.load_key()` also
accepts `CARBONARC_API_KEY` or a real environment variable). Never echo it, never
commit it, never paste it in chat — `smoke.py` prints only length and first/last
two characters. Unauthenticated calls return `401 www-authenticate: Bearer`.

## 2. The four access routes and where they live

The instructions list four delivery routes. In the API they map to:

| Route | Endpoint group | Typical use |
|---|---|---|
| **REST API** | `/v2/library/*`, `/v2/ontology/*` | Discovery: catalogue, tear sheets, data dictionaries, entity resolution, insight listings |
| **Query / "builder"** | `/v2/framework/*` (`filters` → `buy` → `data`/`export`) | Pulling actual feature data: filter a dataset, order a cut, fetch rows (`page`/`size`/`fetch_all`), export, code, panel-debias |
| **Bulk extracts** | `/v2/bucket-access/datasets/{id}/grant`, `/v2/polaris/datasets/{id}/grant` | S3 bucket / warehouse-table access per dataset SKU (`dataset_id` + `lag` + `cut`) |
| **MCP** | `non_native_mcp_available` flag on the catalogue | Agent-driven exploration; REST covers everything it covers for scripted work |

Slow framework queries use a poll contract: call with `poll=true`, get a
`job_id`, watch `GET /v2/query-jobs/{job_id}` until `state == "done"`, then
re-call the original route with `job_id=<id>` to fetch. `CarbonClient.slow()`
implements this.

## 3. Endpoint map (the parts we need)

| Method + path | Returns | Use it for |
|---|---|---|
| `GET /v2/library/data` | `{total, datasources[]}` (498 rows) | The catalogue. Each row: provider, name, `dataset_id` list, history, frequency, flags, start/end dates |
| `GET /v2/library/data/{id}` | Tear sheet | Asset overview |
| `GET /v2/library/data/{id}/data-dictionary` | `[{entity_topic_id, label, notes, columns}]` | Field-level dictionary — §5.2 documentation archaeology, §6.6 question (ii) |
| `GET /v2/library/data/{id}/tearsheet-pdf` | PDF | Deck material |
| `GET /v2/library/schedule/{id}` | Refresh schedule | Publication cadence |
| `GET /v2/ontology/entity-map` | `[{entity, domain, representation}]` (178) | Ontology shape: brands, companies, stores, repos… |
| `GET /v2/ontology/entities/{id}` | Entity details | A resolved entity node |
| `GET /v2/ontology/entity/{id}/insights` | Insight listing | All data series attached to an entity |
| `GET /v2/ontology/insight-search-ids?search=…` | `{insight_ids, total, truncated}` | Semantic search over the insight corpus |
| `GET /v2/ontology/insights?dataset_ids=…` | `{items[], total, pages}` | Browse series by dataset, subject, topic, type |
| `GET /v2/ontology/ontology-versions` | Version list | Vintages for point-in-time work |
| `GET /v2/block/datasets` | `{datasets[]}` with per-cut `annual_price`, `lags` | What a cut costs and which lags exist |
| `GET /v2/clients/payments/balance-summary` | `{total, promotional, purchased, expiry_schedule}` | The account meter |
| `GET /v2/catalog/access`, `/subscriptions` | Access flags | What this key may reach |
| `POST /v2/framework/filters`, `/buy`, `GET /v2/framework/{id}/data|export` | Filter sets, orders, row data | The actual data pull |
| `POST /v2/bucket-access/.../grant|verify` | S3 grant state | Bulk extract provisioning |
| `POST /v2/explore-use-cases/match` | Matched datasets | Quick triage by use case |

## 4. Payload shapes and the parsers

| Payload | Shape | Parser |
|---|---|---|
| `/v2/library/data` | `datasources[]` with a `data` metadata dict (Subjects, Granularity, History, Frequency, Key Fields, Start/End Date, `api_available`, `builder_available`, `block_available`, `non_native_mcp_available`, `lenses_available`, `public`) | `parse.catalog_frame` |
| `/v2/block/datasets` | `datasets[]`, each `cuts[]` with `cut`, `lags[]`, `annual_price`, `bulk_row_status` | `parse.block_pricing_frame` |
| `/v2/ontology/entity-map` | flat `[{entity, domain, representation}]` | `parse.entity_map_frame` |
| `.../data-dictionary` | `tables[]`, each `columns` a dict keyed by field name with `data_type`, `description`, `label`, `sample_value` | `parse.data_dictionary_frame` |
| Framework/`query-jobs` results | `{columns:[{name,type}], rows:[[…]], row_count}` | `parse.query_result_frame` |
| `/v2/ontology/insights` | `{items[], total, page, size, pages}` | `parse.insights_frame`, `CarbonClient.paged` |
| `balance-summary` | totals + `expiry_schedule[]` | `parse.balance_frame` |

`parse.resolve(catalog, "CA0032")` finds the catalogue row for a workbook CA id
(the API `dataset_id` looks like `CA0084_168`; the workbook's `CA0032` matches by
prefix).

## 5. The method: connect → discover → screen → pull → parse

```python
from client import CarbonClient
import parse

client = CarbonClient()                      # key from .env

catalog = parse.catalog_frame(client.get("/v2/library/data"))
pricing = parse.block_pricing_frame(client.get("/v2/block/datasets"))

core = catalog[catalog.dataset_ids.str.contains("CA0056|CA0058|CA0029")]
for _, asset in core.iterrows():
    fields = parse.data_dictionary_frame(
        client.get(f"/v2/library/data/{asset.dataset_ids.split(',')[0]}/data-dictionary"),
        asset.dataset_ids.split(",")[0],
    )
    print(asset["name"], asset.frequency, asset.history, "->", len(fields), "columns")
```

The mandate's §6.6 four-question screen maps onto it like this:

| Question | Source |
|---|---|
| (i) when does history start? | catalogue `start_date`, `history` |
| (ii) publication lag of my metric? | catalogue `frequency` + `block` `lags` + dictionary notes + `library/schedule/{id}` |
| (iii) is it revised; can I get the vintage? | `ontology-versions`, dataset `cut` parameter, versioning columns inside the data dictionary |
| (iv) coverage of my universe, random w.r.t. returns? | entity resolution (`entity/{id}/insights`) + the §6.3 delisting test |

## 6. Money and quotas

- **Balance**: `balance-summary` shows `500.0` promotional credits (no expiry) for
  this account as of 2026-10-22. The instructions' "20K tokens per day" is not
  explained by the public spec — confirm at onboarding what it meters (agent
  channel vs API). Treat calls as cheap but not free; cache discovery responses.
- **Block pricing** is the commercial list: 58 cuts from **$10k/yr (CA0032
  EDGAR) to $600k/yr (CA0086 intraday cards)**. Access for the competition is
  provisioned, not bought through `/v2/framework/buy`; check `catalog/access`
  and `block` `request_statuses` before assuming an asset is reachable.
- **Cost discipline**: discovery endpoints are small; framework data pulls are
  the expensive ones. Do discovery once, cache it under a version stamp, and
  pull data only for assets that passed the screen.

## 7. Point-in-time discipline (Instructions §7)

The API gives the knobs a leak-free backtest needs; using them is on us:

- **`cut` + `lag` are part of the dataset key** (grants and pricing are per
  `(dataset_id, lag, cut)`). A "cut" is a snapshot/vintage — pull historical
  cuts where available instead of the current one.
- **`ontology-versions`** and the version/`delay status` metadata on insights
  tell you what was knowable when.
- **Never use the current vintage of a revisable series as a historical
  feature.** If a series cannot be reconstructed, lag beyond the revision
  window with evidence, or use it descriptively only (Instructions 7, case 3).
- **Entity map is present-day** — run the §6.3 delisting test (2017–2020 exits:
  Sears, Bon-Ton, Pier 1, GNC, Ascena, J.C. Penney) against whatever we use.
- **Publication lag is metric-specific** within one asset; a blanket monthly lag
  is wrong in both directions.

## 8. MCP through opencode

The MCP server is live at `https://mcp.carbonarc.co/` (same Bearer key as the
REST API; Streamable HTTP, MCP protocol `2025-03-26`) and exposes 15 tools:
`text_to_insight`, `framework_to_insight`, `data_library`, `ontology`,
`search_docs`, `search_entities`, `search_insights`, `search_events`,
`get_insights_from_entity`, `get_entities_from_insight`, `get_filter_options`
and the event/entity/insight cross-links.

It is wired into this workspace via `../opencode.json`:

```json
"mcp": {
  "carbonarc": {
    "type": "remote",
    "url": "https://mcp.carbonarc.co/",
    "headers": { "Authorization": "Bearer {file:carbon/.carbontoken}" },
    "oauth": false,
    "timeout": 30000
  }
}
```

`carbon/.carbontoken` holds the key alone (gitignored, no trailing newline);
`{file:}` is used because `{env:CARBON_API_KEY}` resolves to an empty string
when the variable is not exported. Restart opencode after editing the config —
it is not hot-reloaded. This is also the route to **CA0032 (EDGAR)**, which the
REST API does not serve (`api_available=False, mcp_available=True`).

## 9. Known quirks

- **CA0032 (SEC EDGAR filings) is `api_available=False, builder_available=False`
  but `mcp_available=True`** — the survivorship-clean text channel is reachable
  through the MCP route, not the plain data API. Verify what that means for bulk
  (`bucket-access`) before planning the text rebuild.
- `dataset_id` in the library is a list (SKU variants like `CA0084_168`); the
  workbook's `CAxxxx` ids match by prefix.
- `GET /v2/query-jobs/{job_id}` is referenced by the poll contract but is not in
  the published OpenAPI spec; it works with the same Bearer header.
- The spec's "20KB" mentions are AWS bucket-policy size limits, not the token
  quota.
- `catalog_build: false` on this key just means the catalog-builder UI feature is
  off; API access is unaffected.
