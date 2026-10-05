"""Combines source files and public downloads into external.parquet."""

import gzip
import io
import json
import pathlib
import re
import time
import urllib.error
import urllib.request

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import external
from sleeves.filing_events import edgar

RAW_DIR = pathlib.Path(__file__).resolve().parent / "raw"
CACHE_DIR = RAW_DIR / "downloads"
USER_AGENT = "fiam26-research-project research-bot"

IBES_EXPORTS = {
    "ibes_consensus": (
        "s365yejuuvprhila.csv.gz",
        ["TICKER", "STATPERS", "FPI", "FPEDATS", "NUMEST", "NUMUP", "NUMDOWN"]
        + ["MEDEST", "MEANEST", "STDEV", "ANNDATS_ACT", "USFIRM", "CURCODE"],
    ),
    "ibes_price": (
        "lhn8sdwvp01ym0q7.csv.gz",
        ["TICKER", "STATPERS", "PRICE", "CURR_PRICE", "USFIRM"],
    ),
    "ibes_surprise": (
        "b2tkpunzdr2n0s5t.csv.gz",
        ["TICKER", "MEASURE", "FISCALP", "PYEAR", "PMON", "USFIRM", "anndats"]
        + ["actual", "surpmean", "suescore"],
    ),
    "ibes_link": (
        "j2vz7zigsgxzab3h.csv.gz",
        ["TICKER", "PERMNO", "sdate", "edate", "SCORE"],
    ),
}
FRED_SERIES = ("TB3MS", "SP500")

FINRA_URL = "https://cdn.finra.org/equity/otcmarket/biweekly/shrt{date}.csv"
FINRA_FIRST_YEAR = 2017
FINRA_COLUMNS = {
    "symbolCode": "symbol",
    "currentShortPositionQuantity": "short_shares",
    "daysToCoverQuantity": "days_to_cover",
    "settlementDate": "settlement",
}

EDGAR_URL = "https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{quarter}/form.gz"
EDGAR_FIRST_YEAR = 2013
EDGAR_ROW = re.compile(r"^(.*?)\s+(\d{1,10})\s+(\d{4}-\d{2}-\d{2})\s+(edgar/\S+)\s*$")


def ibes(table: str) -> pd.DataFrame:
    name, columns = IBES_EXPORTS[table]
    return pd.read_csv(RAW_DIR / name, usecols=columns, dtype={"TICKER": str})


def fred() -> pd.DataFrame:
    series = []
    for name in FRED_SERIES:
        raw = pd.read_csv(RAW_DIR / f"{name}.csv")
        series.append(
            pd.DataFrame(
                {
                    "date": pd.to_datetime(raw.observation_date),
                    "series": name,
                    "value": pd.to_numeric(raw[name], errors="coerce"),
                }
            )
        )
    return pd.concat(series, ignore_index=True)


def _fetch(url: str, missing: tuple[int, ...]) -> bytes | None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code in missing:
                return None
            time.sleep(3 * (attempt + 1))
        except OSError:
            time.sleep(3 * (attempt + 1))
    return None


def _finra_year(year: int) -> pd.DataFrame:
    """One year of settlements: the first file found in each half-month."""
    frames = []
    for month in range(1, 13):
        if pd.Timestamp(year=year, month=month, day=1) > pd.Timestamp.now():
            break
        for days in (range(12, 19), range(26, 32)):
            for day in days:
                try:
                    stamp = pd.Timestamp(year=year, month=month, day=day)
                except ValueError:
                    continue
                url = FINRA_URL.format(date=stamp.strftime("%Y%m%d"))
                body = _fetch(url, missing=(403, 404))
                if body is None:
                    continue
                frame = pd.read_csv(
                    io.BytesIO(body), sep="|", dtype=str, quoting=3, on_bad_lines="skip"
                )
                if not set(FINRA_COLUMNS) <= set(frame.columns):
                    continue
                frame = frame[list(FINRA_COLUMNS)].rename(columns=FINRA_COLUMNS)
                for column in ("short_shares", "days_to_cover"):
                    frame[column] = pd.to_numeric(frame[column], errors="coerce")
                frame["settlement"] = pd.to_datetime(frame.settlement)
                frames.append(frame)
                break
    if not frames:
        return pd.DataFrame(columns=list(FINRA_COLUMNS.values()))
    return pd.concat(frames, ignore_index=True)


def finra() -> pd.DataFrame:
    """Every settlement since 2017; finished years are cached."""
    years = []
    for year in range(FINRA_FIRST_YEAR, pd.Timestamp.now().year + 1):
        cache = CACHE_DIR / "finra" / f"raw_{year}.parquet"
        if cache.exists() and year < pd.Timestamp.now().year:
            years.append(pd.read_parquet(cache))
            continue
        frame = _finra_year(year)
        cache.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(cache, index=False)
        years.append(frame)
    return pd.concat(years, ignore_index=True)


def _edgar_quarter(raw: bytes) -> pd.DataFrame:
    """Rows are parsed from the right: the header is not aligned with them."""
    kept = edgar.kept_forms()
    rows = []
    for line in gzip.decompress(raw).decode("latin-1").splitlines():
        match = EDGAR_ROW.match(line)
        if not match:
            continue
        form = match.group(1)[:12].strip()
        if form in kept:
            company = match.group(1)[12:].strip()
            rows.append((form, company, *match.group(2, 3, 4)))
    frame = pd.DataFrame(
        rows, columns=["form_type", "company_name", "cik", "date_filed", "filename"]
    )
    frame["cik"] = pd.to_numeric(frame.cik, errors="coerce").astype("Int64")
    frame["date_filed"] = pd.to_datetime(frame.date_filed, errors="coerce")
    frame["accession"] = frame.filename.str.extract(r"/([^/]+)\.txt$")[0]
    frame = frame.drop(columns="filename")
    return frame.dropna(subset=["cik", "date_filed"]).reset_index(drop=True)


def edgar_forms() -> pd.DataFrame:
    """Kept rows of every quarterly index since 2013; past quarters cached."""
    now = pd.Timestamp.now()
    quarters = []
    for year in range(EDGAR_FIRST_YEAR, now.year + 1):
        for quarter in range(1, 5):
            if pd.Timestamp(year=year, month=3 * quarter - 2, day=1) > now:
                continue
            cache = CACHE_DIR / "edgar" / f"forms_{year}Q{quarter}.parquet"
            current = (year, quarter) == (now.year, (now.month - 1) // 3 + 1)
            if cache.exists() and not current:
                quarters.append(pd.read_parquet(cache))
                continue
            raw = _fetch(EDGAR_URL.format(year=year, quarter=quarter), missing=(404,))
            if raw is None:
                continue
            frame = _edgar_quarter(raw)
            cache.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(cache, index=False)
            quarters.append(frame)
            time.sleep(0.4)
    return pd.concat(quarters, ignore_index=True)


def tables() -> dict[str, pd.DataFrame]:
    out = {table: ibes(table) for table in IBES_EXPORTS}
    out["fred"] = fred()
    out["finra"] = finra()
    out["edgar"] = edgar_forms()
    return out


def write(frames: dict[str, pd.DataFrame]) -> None:
    """Stacks the tables and records each one's columns and types."""
    spec = {
        name: {
            "columns": list(frame.columns),
            "dtypes": {c: str(t) for c, t in frame.dtypes.items()},
        }
        for name, frame in frames.items()
    }
    # Integers become nullable: each table's columns are empty in other rows.
    stacked = pd.concat(
        [
            frame.astype({c: "Int64" for c in frame.select_dtypes("int64")}).assign(
                table=name
            )
            for name, frame in frames.items()
        ],
        ignore_index=True,
    )
    arrow = pa.Table.from_pandas(stacked, preserve_index=False)
    metadata = {**arrow.schema.metadata, b"tables": json.dumps(spec).encode()}
    pq.write_table(
        arrow.replace_schema_metadata(metadata), external.FILE, compression="zstd"
    )


def main() -> None:
    frames = tables()
    write(frames)
    for name, frame in frames.items():
        print(f"{name}: {len(frame):,} rows")


if __name__ == "__main__":
    main()
