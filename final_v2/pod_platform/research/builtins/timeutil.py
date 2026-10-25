"""Month arithmetic and timezone-aware boundaries. Date-only values are known at the END of their UTC day."""

from __future__ import annotations

import datetime as dt

UTC = dt.timezone.utc


def month_index(day: dt.date) -> int:
    return day.year * 12 + day.month - 1


def month_end_date(index: int) -> dt.date:
    year, month0 = divmod(index, 12)
    nxt_year, nxt_month0 = (year, month0 + 1) if month0 < 11 else (year + 1, 0)
    return dt.date(nxt_year, nxt_month0 + 1, 1) - dt.timedelta(days=1)


def is_month_end(day: dt.date) -> bool:
    return (day + dt.timedelta(days=1)).day == 1


def end_of_day(day: dt.date) -> dt.datetime:
    return dt.datetime.combine(day, dt.time(23, 59, 59, 999999), tzinfo=UTC)


def month_end_known_at(index: int, lag_days: int = 0) -> dt.datetime:
    return end_of_day(month_end_date(index) + dt.timedelta(days=lag_days))
