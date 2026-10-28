# FIAM 2026 — final_v3

The active strategy is [final_v3](final_v3/README.md), with one entrypoint:
`final_v3/main.py`. It combines the established signal sleeves and company-graph
overlay with four research desks and walk-forward discovery.

```bash
python final_v3/main.py          # rebuild final_v3/output
python final_v3/main.py --reuse  # resume only matching source/input caches
```

Install the runtime dependencies from `final_v3/requirements.txt`. Training uses
the configured device (`cuda` by default); inputs and disclosed licensed data
remain separate from Git. See [the data map](final_v3/docs/OFFICIAL_DATA_MAP.md).

The [Part 2 adaptation](final_v3/docs/PART2.md) enforces sector breadth and sector
net limits, audits the chosen 12% daily realised-volatility ceiling, and exports
compliance, exposure histories and draft review materials. Historical README
performance predates this adaptation and requires a fresh full run. Arcana,
earnings attribution and final submission review remain explicitly pending.

`final_v2/` is retained as the previous research platform and supplies the shared
Carbon Arc MCP/OAuth transport. The earlier root sleeves, optimiser and
`submission/` remain historical reference material. New strategy work targets v3.

Keep one-off checks and exploratory scripts in the system temporary directory.
Never commit supplied data, credentials or private research state.
