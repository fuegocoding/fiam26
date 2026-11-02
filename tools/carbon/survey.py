"""Survey the Carbon Arc library and map what is usable for the mandate.

Pulls the catalogue, pricing and entity map, joins the competition's screening
workbook, separates cross-sectional assets from single-company panels, and
fetches data dictionaries for the assets worth a feature. Writes everything
under ``carbon/out/``.

    python survey.py            # carbon/ directory
"""

import json
import sys
from pathlib import Path

import pandas as pd

import parse
from client import CarbonClient

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
WORKBOOK = (
    HERE.parents[1]
    / "data"
    / "raw"
    / "final"
    / "MCGILL_2026_CarbonArc_Data_Screen.xlsx"
)

FLAG_COLUMNS = (
    "api_available",
    "builder_available",
    "block_available",
    "mcp_available",
    "is_webcontent",
)


def load_workbook() -> pd.DataFrame:
    """The competition's screening verdicts, keyed by CA identifier."""
    sheet = pd.ExcelFile(WORKBOOK).parse("1. Screen", header=None)
    rows = sheet.iloc[4:, [0, 1, 2, 3, 4, 6, 7, 10]].dropna(subset=[1])
    rows.columns = ["verdict", "dataset", "provider", "entity", "history", "freq", "lag", "id"]
    rows = rows[rows.id.notna() & rows.id.astype(str).str.startswith("CA")]
    return rows.drop_duplicates("id").reset_index(drop=True)


def sku_bases(ids: str) -> set[str]:
    """Base CA identifiers from a comma-joined SKU variant list."""
    return {token.split("_")[0] for token in str(ids).split(",") if token}


def booleans(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalise the availability flags to real booleans."""
    for column in FLAG_COLUMNS:
        frame[column] = frame[column].map(
            lambda value: str(value).lower() == "true"
            if not isinstance(value, bool)
            else value
        ).fillna(False)
    return frame


def join_workbook(catalog: pd.DataFrame, workbook: pd.DataFrame) -> pd.DataFrame:
    """Workbook verdicts attached to every catalogue row that carries a SKU."""
    verdicts = dict(zip(workbook.id, workbook.verdict))
    history = dict(zip(workbook.id, workbook.history))
    lag = dict(zip(workbook.id, workbook.lag))
    entity = dict(zip(workbook.id, workbook.entity))

    def lookup(ids: str, table: dict) -> str:
        for base in sku_bases(ids):
            if base in table:
                return str(table[base])
        return ""

    catalog = catalog.copy()
    catalog["ca_id"] = catalog.dataset_ids.map(
        lambda ids: next((b for b in sorted(sku_bases(ids)) if b in verdicts), "")
    )
    catalog["workbook_verdict"] = catalog.ca_id.map(lambda i: verdicts.get(i, ""))
    catalog["workbook_history"] = catalog.ca_id.map(lambda i: history.get(i, ""))
    catalog["workbook_lag"] = catalog.ca_id.map(lambda i: lag.get(i, ""))
    catalog["workbook_entity"] = catalog.ca_id.map(lambda i: entity.get(i, ""))
    return catalog


def classify(catalog: pd.DataFrame) -> pd.DataFrame:
    """Split cross-sectional assets, single-company panels and leftovers."""
    catalog = catalog.copy()
    catalog["scope"] = "single_company_panel"
    cross = catalog.workbook_verdict.ne("")
    catalog.loc[cross, "scope"] = "cross_sectional"
    untagged = catalog.company_tag.isna() & ~cross
    catalog.loc[untagged, "scope"] = "other"
    return catalog


def dictionary_targets(catalog: pd.DataFrame) -> pd.DataFrame:
    """Assets worth reading a data dictionary for."""
    screened = catalog.workbook_verdict.isin(["CORE", "USABLE"])
    find = catalog["name"].eq("U.S. Job Postings")
    return catalog[screened | find].sort_values(["workbook_verdict", "ca_id"])


def fetch_dictionaries(client: CarbonClient, targets: pd.DataFrame) -> pd.DataFrame:
    """Data dictionaries for every target asset into out/dictionaries/."""
    folder = OUT / "dictionaries"
    folder.mkdir(parents=True, exist_ok=True)
    frames = []
    for _, row in targets.iterrows():
        ca_id = row.ca_id or row.dataset_ids.split(",")[0].split("_")[0]
        try:
            payload = client.get(f"/v2/library/data/{ca_id}/data-dictionary")
        except Exception as exc:
            print(f"  dictionary {ca_id}: {exc}")
            continue
        frame = parse.data_dictionary_frame(payload, ca_id)
        if frame.empty:
            print(f"  dictionary {ca_id}: empty")
            continue
        frame.to_csv(folder / f"{ca_id}.csv", index=False)
        frames.append(frame)
        print(f"  dictionary {ca_id}: {frame.table.nunique()} tables, {len(frame)} columns")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    OUT.mkdir(exist_ok=True)
    client = CarbonClient()

    library = client.get("/v2/library/data")
    (OUT / "library_raw.json").write_text(json.dumps(library), encoding="utf-8")
    catalog = booleans(parse.catalog_frame(library))
    catalog["company_tag"] = catalog["name"].map(
        {d["dataset_name"]: d["data"].get("Company") for d in library["datasources"]}
    )
    catalog["start_year"] = catalog.start_date.fillna("").astype(str).str[:4]

    workbook = load_workbook()
    catalog = classify(join_workbook(catalog, workbook))
    catalog.to_csv(OUT / "catalog.csv", index=False)
    print(f"catalogue: {len(catalog)} rows | " +
          catalog.scope.value_counts().to_dict().__str__())

    pricing = parse.block_pricing_frame(client.get("/v2/block/datasets"))
    pricing.to_csv(OUT / "block_pricing.csv", index=False)
    print(f"block pricing: {len(pricing)} cuts")

    entities = parse.entity_map_frame(client.get("/v2/ontology/entity-map"))
    entities.to_csv(OUT / "entity_map.csv", index=False)
    print(f"entity map: {len(entities)} representations")

    targets = dictionary_targets(catalog)
    targets.to_csv(OUT / "candidates.csv", index=False)
    print(f"dictionary targets: {len(targets)}")
    fetch_dictionaries(client, targets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
