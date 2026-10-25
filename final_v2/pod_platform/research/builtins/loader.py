"""Strict dataset loader: CSV (stdlib) and Parquet (lazy ``pyarrow``).

Only the declared design window is materialized. Rows after the window end (check/replay/prospective time) are
counted but never parsed beyond their date, so mutating them cannot change a result. Every violation of the
declared schema, identity, finiteness, month-end, bounds or feature-availability rules raises
``DataIntegrityError``; nothing is silently repaired.
"""

from __future__ import annotations

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None


import csv
import datetime as dt
import hashlib
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

from ...contracts.base import require_aware
from .errors import (
    BuiltinConfigError,
    DataIntegrityError,
    DependencyUnavailable,
    EvidenceInsufficient,
    PartitionDenied,
)
from .manifest import DESIGN, ApprovedDataset, ColumnSpec
from .timeutil import end_of_day, is_month_end, month_index

MISSING_TOKENS = frozenset({"", "nan", "na", "n/a", "null", "none"})
_DATE_RE = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})(?:[ T]00:00:00(?:\.0+)?(?:Z|\+00:00)?)?$"
)
_HASH_CACHE: dict[tuple[str, int, int], str] = {}


@dataclass
class Panel:
    dataset_id: str
    security: list[str] = field(default_factory=list)
    month: list[int] = field(default_factory=list)
    date: list[dt.date] = field(default_factory=list)
    num: dict[str, list[Optional[float]]] = field(default_factory=dict)
    text: dict[str, list[Optional[str]]] = field(default_factory=dict)
    rows_by_month: dict[int, list[int]] = field(default_factory=dict)
    row_index: dict[tuple[str, int], int] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.security)


def file_sha256(path: Path) -> str:
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    cached = _HASH_CACHE.get(key)
    if cached is None:
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        cached = _HASH_CACHE[key] = digest.hexdigest()
    return cached


def parse_date(value: Any, where: str) -> dt.date:
    if isinstance(value, dt.datetime):
        if value.tzinfo is not None:
            value = value.astimezone(dt.timezone.utc)
        return value.date()
    if isinstance(value, dt.date):
        return value
    if isinstance(value, str):
        match = _DATE_RE.match(value.strip())
        if match:
            try:
                return dt.date(
                    int(match.group(1)),
                    int(match.group(2)),
                    int(match.group(3)),
                )
            except ValueError:
                pass
    raise DataIntegrityError(f"{where}: unparseable or missing date {value!r}")


def _normalise_security(value: Any, where: str) -> str:
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        value = int(value)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        raise DataIntegrityError(f"{where}: missing security identifier")
    text = str(value).strip()
    if not text:
        raise DataIntegrityError(f"{where}: missing security identifier")
    return text


def _parse_float(
    value: Any, name: str, spec: ColumnSpec, where: str
) -> Optional[float]:
    if value is None:
        out: Optional[float] = None
    elif isinstance(value, bool):
        raise DataIntegrityError(
            f"{where}: column {name} holds a boolean, not a number"
        )
    elif isinstance(value, (int, float)):
        out = float(value)
    elif isinstance(value, str):
        token = value.strip()
        if token.lower() in MISSING_TOKENS:
            out = None
        else:
            try:
                out = float(token)
            except ValueError:
                raise DataIntegrityError(
                    f"{where}: column {name} has non-numeric value {token[:30]!r}"
                ) from None
    else:
        try:
            out = float(value)
        except (TypeError, ValueError):
            raise DataIntegrityError(
                f"{where}: column {name} has unsupported value type {type(value).__name__}"
            ) from None
    if out is not None:
        if math.isnan(out):
            out = None
        elif math.isinf(out):
            raise DataIntegrityError(f"{where}: column {name} is not finite")
    if out is None:
        if not spec.allow_missing:
            raise DataIntegrityError(
                f"{where}: column {name} is missing but declared allow_missing=false"
            )
        return None
    if spec.min_value is not None and out < spec.min_value:
        raise DataIntegrityError(
            f"{where}: column {name}={out} below declared minimum {spec.min_value}"
        )
    if spec.max_value is not None and out > spec.max_value:
        raise DataIntegrityError(
            f"{where}: column {name}={out} above declared maximum {spec.max_value}"
        )
    return out


def _parse_text(
    value: Any, name: str, spec: ColumnSpec, where: str
) -> Optional[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        out = None
    else:
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        out = str(value).strip() or None
    if out is None and not spec.allow_missing:
        raise DataIntegrityError(
            f"{where}: column {name} is missing but declared allow_missing=false"
        )
    return out


def _iter_csv(
    path: Path, names: list[str]
) -> Iterator[tuple[int, dict[str, Any]]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = [h.strip() for h in next(reader)]
        except StopIteration:
            raise DataIntegrityError("csv file is empty") from None
        if len(set(header)) != len(header):
            raise DataIntegrityError(
                "csv header contains duplicate column names"
            )
        missing = [n for n in names if n not in header]
        if missing:
            raise DataIntegrityError(
                f"required columns missing from file: {missing}"
            )
        pos = {n: header.index(n) for n in names}
        for line_no, row in enumerate(reader, start=2):
            if not row:
                continue
            if len(row) != len(header):
                raise DataIntegrityError(
                    f"row {line_no} has {len(row)} fields but the header has {len(header)}"
                )
            yield line_no, {n: row[i] for n, i in pos.items()}


def _time_filter(
    pa_type: Any, column: str, end: dt.date
) -> Optional[list[tuple[str, str, Any]]]:
    text = str(pa_type)
    if text.startswith("timestamp"):
        if "tz=" in text:
            return [(column, "<=", end_of_day(end))]
        return [
            (
                column,
                "<=",
                dt.datetime.combine(end, dt.time(23, 59, 59, 999999)),
            )
        ]
    if text.startswith("date"):
        return [(column, "<=", end)]
    if text in ("string", "large_string"):
        return [
            (column, "<=", end.isoformat() + "~")
        ]  # '~' sorts after any time suffix
    return None


def _read_parquet(
    path: Path, names: list[str], time_column: str, end: dt.date
) -> dict[str, list]:
    if pq is None:
        raise DependencyUnavailable(
            "parquet datasets require pyarrow, which is not installed"
        )
    schema = pq.read_schema(str(path))
    missing = [n for n in names if n not in schema.names]
    if missing:
        raise DataIntegrityError(
            f"required columns missing from file: {missing}"
        )
    filters = _time_filter(schema.field(time_column).type, time_column, end)
    table = None
    if filters is not None:
        try:
            table = pq.read_table(str(path), columns=names, filters=filters)
        except (ValueError, TypeError, NotImplementedError):
            table = None  # fall back to an unfiltered read; rows after the window are still discarded below
    if table is None:
        table = pq.read_table(str(path), columns=names)
    return table.to_pydict()


def _iter_parquet(
    path: Path, names: list[str], time_column: str, end: dt.date
) -> Iterator[tuple[int, dict[str, Any]]]:
    cols = _read_parquet(path, names, time_column, end)
    count = len(cols[names[0]])
    for i in range(count):
        yield i + 1, {n: cols[n][i] for n in names}


def load_panel(
    dataset: ApprovedDataset,
    columns: "set[str] | frozenset[str]",
    *,
    partition: str,
    research_asof: dt.datetime,
) -> Panel:
    """Materialize the requested declared columns for the design window of an approved dataset."""
    if partition != DESIGN:
        raise PartitionDenied(
            f"PARTITION_ACCESS_DENIED: built-in executors may only read the design partition, not {partition!r}"
        )
    research_asof = require_aware(research_asof, "research_asof")
    spec = dataset.spec
    start, end = dataset.design_window
    if end_of_day(end) > research_asof:
        raise EvidenceInsufficient(
            f"LABEL_NOT_MATURE: design window ends {end.isoformat()} but the research clock is {research_asof.isoformat()}"
        )
    undeclared = sorted(c for c in columns if c not in spec.columns)
    if undeclared:
        raise BuiltinConfigError(
            f"columns not declared by the operator manifest (no silent substitution): {undeclared}"
        )
    wanted = sorted(set(columns))
    if spec.availability.column and spec.availability.column not in wanted:
        wanted.append(spec.availability.column)
    names = [spec.security_column, spec.time_column] + wanted
    sha = file_sha256(dataset.path)
    if spec.sha256 is not None and spec.sha256 != sha:
        raise BuiltinConfigError(
            "dataset file hash does not match the sha256 declared in the operator manifest"
        )

    iterator = (
        _iter_csv(dataset.path, names)
        if spec.format == "csv"
        else _iter_parquet(dataset.path, names, spec.time_column, end)
    )
    panel = Panel(dataset_id=dataset.dataset_id)
    for name in wanted:
        if spec.columns[name].dtype == "float":
            panel.num[name] = []
        else:
            panel.text[
                name
            ] = []  # str columns, and date columns kept only as ISO text when requested
    seen: dict[tuple[str, int], int] = {}
    rows_read = before = after = excluded = 0
    for line_no, rec in iterator:
        rows_read += 1
        where = f"row {line_no}"
        day = parse_date(rec[spec.time_column], f"{where}: {spec.time_column}")
        if day > end:
            after += 1
            continue
        if day < start:
            before += 1
            continue
        if not is_month_end(day):
            raise DataIntegrityError(
                f"{where}: {spec.time_column}={day.isoformat()} is not a month-end (frequency is monthly)"
            )
        security = _normalise_security(rec[spec.security_column], where)
        idx = month_index(day)
        key = (security, idx)
        if key in seen:
            raise DataIntegrityError(
                f"duplicate identity {security} @ {day.isoformat()} at rows {seen[key]} and {line_no}"
            )
        seen[key] = line_no
        parsed_num: dict[str, Optional[float]] = {}
        parsed_text: dict[str, Optional[str]] = {}
        for name in wanted:
            cspec = spec.columns[name]
            if cspec.dtype == "float":
                parsed_num[name] = _parse_float(rec[name], name, cspec, where)
            elif cspec.dtype == "date":
                raw = rec[name]
                if raw is None or (
                    isinstance(raw, str)
                    and raw.strip().lower() in MISSING_TOKENS
                ):
                    if not cspec.allow_missing:
                        raise DataIntegrityError(
                            f"{where}: column {name} is missing but declared allow_missing=false"
                        )
                    parsed_text[name] = None
                else:
                    parsed_text[name] = parse_date(
                        raw, f"{where}: {name}"
                    ).isoformat()
            else:
                parsed_text[name] = _parse_text(rec[name], name, cspec, where)
        if spec.availability.column:
            avail = parsed_text.get(spec.availability.column)
            known_late = avail is None or dt.date.fromisoformat(avail) > day
            if known_late:
                if spec.availability.on_violation == "error":
                    raise DataIntegrityError(
                        f"{where}: features of {security} @ {day.isoformat()} are not available by the formation date "
                        f"(availability={avail!r}); refusing to use future information"
                    )
                excluded += 1
                continue
        panel.security.append(security)
        panel.month.append(idx)
        panel.date.append(day)
        for name, value in parsed_num.items():
            panel.num[name].append(value)
        for name, value in parsed_text.items():
            panel.text[name].append(value)
        if len(panel.security) > spec.evaluation.max_rows:
            raise DataIntegrityError(
                f"more than max_rows={spec.evaluation.max_rows} rows inside the design window"
            )
    if (
        panel.date and max(panel.date) > end
    ):  # defence in depth: must be unreachable
        raise PartitionDenied(
            "PARTITION_ACCESS_DENIED: a row beyond the design window reached the panel"
        )
    for row, (security, idx) in enumerate(zip(panel.security, panel.month)):
        panel.rows_by_month.setdefault(idx, []).append(row)
        panel.row_index[(security, idx)] = row
    panel.meta = {
        "file_sha256": sha,
        "format": spec.format,
        "rows_read": rows_read,
        "rows_in_window": len(panel),
        "rows_before_window_discarded": before,
        "rows_after_window_discarded_unparsed": after,
        "rows_excluded_for_availability": excluded,
        "months": len(panel.rows_by_month),
        "securities": len({s for s in panel.security}),
        "window": [start.isoformat(), end.isoformat()],
    }
    if not len(panel):
        raise EvidenceInsufficient("no rows inside the declared design window")
    return panel
