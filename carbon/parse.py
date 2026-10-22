"""Parsers turning Carbon Arc payloads into tidy DataFrames."""

import pandas as pd


def _listify(value) -> str:
    """Comma-joined label for a field the API returns as list or scalar."""
    if value is None:
        return ""
    if isinstance(value, list):
        return ",".join(str(v) for v in value)
    return str(value)


def catalog_frame(payload: dict) -> pd.DataFrame:
    """One row per datasource from GET /v2/library/data."""
    rows = []
    for source in payload.get("datasources", []):
        meta = source.get("data") or {}
        rows.append(
            {
                "dataset_ids": ",".join(source.get("dataset_id") or []),
                "name": source.get("dataset_name"),
                "label": source.get("label"),
                "provider": source.get("provider_name"),
                "description": source.get("description"),
                "blocked": source.get("blocked"),
                "is_current": source.get("is_current"),
                "last_updated": source.get("last_updated_timestamp"),
                "subjects": _listify(meta.get("Subjects")),
                "granularity": _listify(meta.get("Granularity")),
                "type": _listify(meta.get("Type")),
                "history": _listify(meta.get("History")),
                "frequency": _listify(meta.get("Frequency")),
                "region": _listify(meta.get("Region")),
                "sectors": _listify(meta.get("Sectors")),
                "is_webcontent": meta.get("is_webcontent"),
                "key_fields": _listify(meta.get("Key Fields")),
                "start_date": meta.get("Start Date"),
                "end_date": meta.get("End Date"),
                "use_case": _listify(meta.get("Use Case")),
                "api_available": meta.get("api_available"),
                "builder_available": meta.get("builder_available"),
                "block_available": meta.get("block_available"),
                "mcp_available": meta.get("non_native_mcp_available"),
                "lenses_available": meta.get("lenses_available"),
                "public": meta.get("public"),
            }
        )
    return pd.DataFrame(rows)


def block_pricing_frame(payload: dict) -> pd.DataFrame:
    """One row per dataset cut from GET /v2/block/datasets."""
    rows = []
    for entry in payload.get("datasets", []):
        base = {
            "dataset_id": entry.get("dataset_id"),
            "label": entry.get("label"),
            "vendor": entry.get("vendor"),
            "access_type": entry.get("offered_access_type"),
            "frequency": entry.get("frequency"),
            "history": entry.get("history"),
            "description": entry.get("description"),
        }
        for cut in entry.get("cuts") or []:
            rows.append(
                {
                    **base,
                    "cut": cut.get("cut") or "default",
                    "lags": _listify(cut.get("lags")),
                    "annual_price": cut.get("annual_price"),
                    "bulk_row_status": cut.get("bulk_row_status"),
                    "request_statuses": cut.get("request_statuses"),
                }
            )
    return pd.DataFrame(rows)


def entity_map_frame(payload: list) -> pd.DataFrame:
    """Entity/domain/representation triples from GET /v2/ontology/entity-map."""
    return pd.DataFrame(payload, columns=["entity", "domain", "representation"])


def data_dictionary_frame(payload: list, dataset_id: str = "") -> pd.DataFrame:
    """One row per column from a GET .../data-dictionary response."""
    rows = []
    for table in payload or []:
        columns = table.get("columns") or {}
        items = columns.values() if isinstance(columns, dict) else columns
        for column in items:
            if not isinstance(column, dict):
                continue
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "table": table.get("label"),
                    "entity_topic_id": table.get("entity_topic_id"),
                    "notes": table.get("notes"),
                    "column": column.get("name"),
                    "label": column.get("label"),
                    "data_type": column.get("data_type"),
                    "description": column.get("description"),
                    "sample_value": column.get("sample_value"),
                }
            )
    return pd.DataFrame(rows)


def query_result_frame(payload: dict) -> pd.DataFrame:
    """QueryJobResult {columns, rows} into a labelled frame."""
    names = [column["name"] for column in payload.get("columns", [])]
    return pd.DataFrame(payload.get("rows", []), columns=names)


def insights_frame(payload: dict) -> pd.DataFrame:
    """Insight listing rows from GET /v2/ontology/insights."""
    return pd.DataFrame(payload.get("items", []))


def balance_frame(payload: dict) -> pd.DataFrame:
    """Account balance and expiry schedule from GET /v2/clients/payments/balance-summary."""
    summary = pd.DataFrame(
        [
            {
                "total_balance": payload.get("total_balance"),
                "promotional_balance": payload.get("promotional_balance"),
                "purchased_balance": payload.get("purchased_balance"),
            }
        ]
    )
    schedule = pd.DataFrame(payload.get("expiry_schedule") or [])
    return summary, schedule


def resolve(catalog: pd.DataFrame, ca_id: str) -> pd.Series | None:
    """The catalogue row holding a workbook CA identifier."""
    hits = catalog[catalog.dataset_ids.str.contains(ca_id, na=False)]
    return hits.iloc[0] if len(hits) else None
