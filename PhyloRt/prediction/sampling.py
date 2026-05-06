"""Date parsing and sampling-proportion helpers."""

from __future__ import annotations

import bisect
from datetime import datetime, timedelta
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

def parse_latest_date(latest: str | float | int) -> datetime:
    """Parse a calendar date (YYYY-MM-DD) or decimal year."""
    latest_str = str(latest)
    if "-" in latest_str:
        return datetime.strptime(latest_str, "%Y-%m-%d")

    if "." not in latest_str:
        latest_str = latest_str + ".0"

    year_str, frac_str = latest_str.split(".", 1)
    year = int(year_str)
    frac = float("0." + frac_str)
    start = datetime(year, 1, 1)
    next_start = datetime(year + 1, 1, 1)
    return start + timedelta(seconds=frac * (next_start - start).total_seconds())

def datetime_to_decimal_year(dt: datetime) -> float:
    year_start = datetime(dt.year, 1, 1)
    next_year_start = datetime(dt.year + 1, 1, 1)
    year_length = (next_year_start - year_start).total_seconds()
    seconds_into_year = (dt - year_start).total_seconds()
    return dt.year + seconds_into_year / year_length

def get_prop_for_date(
    date: datetime, sampling_times: Sequence[datetime | str | float | int], sampling_props: Sequence[float]
) -> float:
    """Return the active sampling proportion for a date."""
    norm_times = []
    for item in sampling_times:
        if isinstance(item, datetime):
            norm_times.append(item)
        elif isinstance(item, str):
            norm_times.append(parse_latest_date(item))
        elif isinstance(item, (int, float)):
            norm_times.append(parse_latest_date(item))
        else:
            raise ValueError(f"Unrecognized sampling time: {item!r}")

    norm_times.sort()
    idx = bisect.bisect_right(norm_times, date)
    return float(sampling_props[idx])

def _parse_ints(values: str | Iterable[int]) -> list[int]:
    if isinstance(values, str):
        return [int(item) for item in values.split(",") if item]
    return [int(item) for item in values]

def _parse_floats(values: str | Iterable[float]) -> list[float]:
    if isinstance(values, str):
        return [float(item) for item in values.split(",") if item]
    return [float(item) for item in values]

def _parse_sampling_times(values: str | Iterable[str | float | int] | None) -> list[str | float | int]:
    if values is None:
        return []
    if isinstance(values, str):
        return [item for item in values.split(",") if item]
    return list(values)

def _np_datetime(dt: datetime) -> np.datetime64:
    return np.datetime64(pd.Timestamp(dt).to_datetime64())
