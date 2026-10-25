"""Streaming adapter for operator supplied daily security CSVs."""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Sequence

from .daily import (
    DailyCashEntry,
    DailyFill,
    DailyInputs,
    DailyPolicy,
    DailyRate,
    DailyReturn,
)

COLUMNS = ("permno", "date", "prc", "daily_return")
DEFAULT_CUTOFF = dt.date(2026, 8, 31)


@dataclass
class SuppliedDaily:
    source: str
    sha256: str
    start: dt.date
    end: dt.date
    security_ids: list[str]
    trading_days: list[dt.date]
    returns: list[DailyReturn]
    reference_prices: dict[tuple[str, dt.date], Decimal]
    observations: int
    missing_return_count: int
    source_rows: int

    def metadata(self) -> dict[str, Any]:
        return {
            "status": "data_adapter_check",
            "source": self.source,
            "sha256": self.sha256,
            "selected_range": {
                "start": self.start.isoformat(),
                "end": self.end.isoformat(),
            },
            "security_ids": self.security_ids,
            "trading_days": len(self.trading_days),
            "observations": self.observations,
            "missing_return_count": self.missing_return_count,
            "source_rows_scanned": self.source_rows,
            "calendar_scope": "dates observed in selected source rows; not an exchange calendar",
            "return_field": "daily_return (source total return)",
            "price_field": "prc retained as unadjusted reference only; never treated as adjusted close",
        }


def read_supplied_daily(
    path: str | Path,
    security_ids: Sequence[str],
    start: dt.date | str,
    end: dt.date | str,
    *,
    descriptive: bool = False,
) -> SuppliedDaily:
    p = Path(path)
    begin = dt.date.fromisoformat(start) if isinstance(start, str) else start
    finish = dt.date.fromisoformat(end) if isinstance(end, str) else end
    ids = sorted(set(map(str, security_ids)))
    if not ids or any(not x for x in ids):
        raise ValueError("at least one explicit security ID is required")
    if begin > finish:
        raise ValueError("start must be on or before end")
    if finish > DEFAULT_CUTOFF and not descriptive:
        raise ValueError(
            f"end date after {DEFAULT_CUTOFF} requires descriptive=True"
        )
    sha = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(block)
    obs = {}
    days = set()
    scanned = missing = 0
    wanted = set(ids)
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ValueError(
                f"schema mismatch: expected columns {COLUMNS}, got {reader.fieldnames}"
            )
        for row in reader:
            scanned += 1
            if None in row:
                raise ValueError(f"malformed CSV row {scanned}: extra fields")
            sid = (row.get("permno") or "").strip()
            raw_day = (row.get("date") or "").strip()
            try:
                day = (
                    dt.datetime.strptime(raw_day, "%Y%m%d").date()
                    if len(raw_day) == 8 and raw_day.isdigit()
                    else dt.date.fromisoformat(raw_day)
                )
            except ValueError as e:
                raise ValueError(
                    f"invalid date for selected ID {sid}: {row.get('date')!r}"
                ) from e
            if not begin <= day <= finish:
                continue
            days.add(day)
            if sid not in wanted:
                continue
            key = (sid, day)
            if key in obs:
                raise ValueError(f"duplicate observation for {sid} on {day}")
            try:
                price = Decimal((row.get("prc") or "").strip())
                if not price.is_finite() or price < 0:
                    raise ValueError
            except Exception as e:
                raise ValueError(f"invalid prc for {sid} on {day}") from e
            raw = (row.get("daily_return") or "").strip()
            ret = None
            if raw:
                try:
                    ret = Decimal(raw)
                except Exception as e:
                    raise ValueError(
                        f"invalid daily_return for {sid} on {day}"
                    ) from e
                if not ret.is_finite() or ret < -1:
                    raise ValueError(f"invalid daily_return for {sid} on {day}")
            else:
                missing += 1
            if price == 0 and ret != -1:
                raise ValueError(
                    f"zero reference prc without total-loss evidence for {sid} on {day}"
                )
            obs[key] = (price, ret)
    if not obs:
        raise ValueError("no observations for selected IDs and date range")
    absent = sorted(wanted - {sid for sid, _ in obs})
    if absent:
        raise ValueError(
            f"no observations for selected security ID(s): {', '.join(absent)}"
        )
    bad = [k for k, v in obs.items() if v[1] is None]
    if bad:
        raise ValueError(
            f"missing daily_return for {len(bad)} selected observation(s), first {bad[0][0]} {bad[0][1]}"
        )
    return SuppliedDaily(
        str(p.resolve()),
        sha.hexdigest(),
        begin,
        finish,
        ids,
        sorted(days),
        [
            DailyReturn(security_id=s, on=d, total_return=v[1])
            for (s, d), v in sorted(obs.items())
        ],
        {k: v[0] for k, v in obs.items()},
        len(obs),
        missing,
        scanned,
    )


def inputs_from_json(
    data: Mapping[str, Any], source: SuppliedDaily
) -> tuple[DailyInputs, DailyPolicy]:
    required = {"start_cash_usd", "initial_positions", "risk_free", "fills"}
    if not required <= set(data):
        raise ValueError(f"inputs JSON requires: {', '.join(sorted(required))}")
    pos = {
        str(s): (Decimal(str(v[0])), Decimal(str(v[1])))
        for s, v in data["initial_positions"].items()
    }
    rates = [
        DailyRate(
            on=dt.date.fromisoformat(x["on"]),
            tb3ms_annual_percent=float(x["tb3ms_annual_percent"]),
        )
        for x in data["risk_free"]
    ]
    fills = [
        DailyFill(
            security_id=x["security_id"],
            on=dt.date.fromisoformat(x["on"]),
            quantity=Decimal(str(x["quantity"])),
            price=Decimal(str(x["price"])),
            fee_usd=Decimal(str(x["fee_usd"])),
        )
        for x in data["fills"]
    ]
    entries = [
        DailyCashEntry(
            on=dt.date.fromisoformat(x["on"]),
            kind=x["kind"],
            debit_usd=Decimal(str(x.get("debit_usd", 0))),
            credit_usd=Decimal(str(x.get("credit_usd", 0))),
            evidence_id=x["evidence_id"],
        )
        for x in data.get("cash_entries", [])
    ]
    return DailyInputs(
        start_cash_usd=Decimal(str(data["start_cash_usd"])),
        initial_positions=pos,
        returns=source.returns,
        fills=fills,
        cash_entries=entries,
        risk_free=rates,
    ), DailyPolicy(trading_days=source.trading_days)


def write_json(
    path: str | Path, value: Mapping[str, Any], *, overwrite: bool = False
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite {target}; pass --overwrite"
        )
    fd, tmp = tempfile.mkstemp(
        prefix=f".{target.name}.", dir=target.parent, text=True
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True, default=str)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        if overwrite:
            os.replace(tmp, target)
        else:
            try:
                os.link(tmp, target)
            except FileExistsError as exc:
                raise FileExistsError(
                    f"refusing to overwrite {target}; pass --overwrite"
                ) from exc
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
