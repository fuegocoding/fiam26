"""End-to-end connectivity check: connect, fetch, parse, print a masked summary."""

import sys

import parse
from client import CarbonClient, load_key


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    key = load_key()
    print(f"key: loaded ({len(key)} chars, {key[:4]}...{key[-2:]} masked)")
    client = CarbonClient(key)

    access = client.get("/v2/catalog/access")
    print(f"catalog access: {access}")

    balance, schedule = parse.balance_frame(
        client.get("/v2/clients/payments/balance-summary")
    )
    print(f"balance: {balance.to_dict('records')[0]} | expiry rows: {len(schedule)}")

    catalog = parse.catalog_frame(client.get("/v2/library/data"))
    print(f"catalog: {len(catalog)} datasources | flags:"
          f" api={int(catalog.api_available.sum())}"
          f" builder={int(catalog.builder_available.sum())}"
          f" block={int(catalog.block_available.sum())}"
          f" mcp={int(catalog.mcp_available.sum())}")

    pricing = parse.block_pricing_frame(client.get("/v2/block/datasets"))
    priced = pricing.dropna(subset=["annual_price"])
    print(f"block cuts: {len(pricing)} | priced: {len(priced)}"
          f" | ${priced.annual_price.astype(float).min():,.0f}"
          f"–${priced.annual_price.astype(float).max():,.0f}")

    entities = parse.entity_map_frame(client.get("/v2/ontology/entity-map"))
    print(f"entity map: {len(entities)} representations"
          f" | entities: {sorted(entities.entity.unique())[:8]} ...")

    dictionary = parse.data_dictionary_frame(
        client.get("/v2/library/data/CA0052/data-dictionary"), "CA0052"
    )
    print(f"data dictionary CA0052: {dictionary.table.nunique()} tables,"
          f" {len(dictionary)} columns")
    print(dictionary[["table", "column", "data_type"]].head(5).to_string(index=False))

    row = parse.resolve(catalog, "CA0032")
    if row is not None:
        print(f"CA0032 -> {row['name']} | api={row.api_available}"
              f" mcp={row.mcp_available} | history={row.history}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
