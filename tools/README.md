# Data exploration helpers

`carbon/` retains the team's historical Carbon Arc REST exploration and
framework acquisition helpers. They are separate from the final v3 strategy,
whose optional discovery path uses the shared MCP/OAuth transport in
`final_v2/`.

| Helper | Purpose |
| :--- | :--- |
| `carbon/client.py` | Authenticated REST transport, pagination and polling |
| `carbon/parse.py` | Response parsing |
| `carbon/survey.py` | Catalogue/data-screen exploration |
| `carbon/pull_framework.py` | Quote and optional explicitly requested purchase/export |
| `carbon/smoke.py` | Historical connectivity example |

These tools need their service-specific dependencies, credentials and access.
Purchasing/exporting data is not part of the final baseline reproduction.
`pull_framework.py` defaults to quoting; `--buy` requests a purchase and can
spend account credits. Its external `carbonarc` SDK is not part of the final
runtime requirements.

Raw responses and acquired extracts stay under ignored local paths. The
final presentation reports **no Carbon Arc predictive asset in the final
book**. Their role in earlier research is described in
[the process guide](../docs/research-process.md).
