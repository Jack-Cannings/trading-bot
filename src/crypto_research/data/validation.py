"""Validate without fabricating, smoothing or filling market observations."""

from datetime import datetime
from typing import Literal

import numpy as np

from crypto_research.contracts import Period
from crypto_research.data.models import (
    Issue,
    Observation,
    QualityReport,
    Request,
    ValidatedData,
    aligned,
    duration,
)


def values(row: Observation) -> tuple[float, ...]:
    return (row.open, row.high, row.low, row.close, row.volume)


def validate(
    request: Request,
    observations: tuple[Observation, ...],
    issues: tuple[Issue, ...] = (),
) -> ValidatedData:
    found = list(issues)
    accepted: dict[datetime, Observation] = {}
    conflicts: set[datetime] = set()
    previous = None
    step = duration(request.market)
    for row in observations:
        timestamp = row.open_time
        if previous is not None and timestamp < previous:
            found.append(Issue(kind="order", message="out-of-order input", open_time=timestamp))
        previous = timestamp
        reason = None
        if not request.period.start <= timestamp < request.period.end:
            reason = "candle outside requested interval"
        elif not aligned(timestamp, request.market):
            reason = "incorrect UTC candle alignment"
        elif not np.isfinite(values(row)).all():
            reason = "non-finite OHLCV"
        elif min(values(row)[:4]) <= 0 or row.volume < 0:
            reason = "nonpositive price or negative volume"
        elif row.high < max(row.open, row.close, row.low) or row.low > min(row.open, row.close):
            reason = "inconsistent OHLC bounds"
        if reason:
            found.append(Issue(kind="invalid", message=reason, open_time=timestamp))
            continue
        if timestamp + step > row.observed_at:
            found.append(
                Issue(kind="incomplete", message="not closed at observation", open_time=timestamp)
            )
            continue
        if timestamp in accepted or timestamp in conflicts:
            found.append(
                Issue(kind="duplicate", message="duplicate candle open", open_time=timestamp)
            )
            if timestamp in accepted and values(accepted[timestamp]) != values(row):
                found.append(
                    Issue(
                        kind="invalid", message="conflicting duplicate values", open_time=timestamp
                    )
                )
                del accepted[timestamp]
                conflicts.add(timestamp)
            continue
        if row.volume == 0:
            found.append(
                Issue(
                    kind="zero_volume", message="zero volume requires review", open_time=timestamp
                )
            )
        accepted[timestamp] = row
    ordered = tuple(sorted(accepted.values(), key=lambda r: r.open_time))
    missing: list[Period] = []
    cursor = request.period.start
    for row in ordered:
        if row.open_time > cursor:
            missing.append(Period(start=cursor, end=row.open_time))
        cursor = row.open_time + step
    if cursor < request.period.end:
        missing.append(Period(start=cursor, end=request.period.end))
    invalid = sum(i.kind == "invalid" for i in found)
    unresolved = any(i.kind in {"order", "pagination", "duplicate", "revision"} for i in found)
    status: Literal["clean", "incomplete", "invalid"] = (
        "invalid" if invalid else "incomplete" if missing or unresolved else "clean"
    )
    report = QualityReport(
        request=request,
        retrieved=Period(start=ordered[0].open_time, end=ordered[-1].open_time + step)
        if ordered
        else None,
        candle_count=len(ordered),
        expected_intervals=int((request.period.end - request.period.start) / step),
        missing_intervals=tuple(missing),
        duplicates_detected=sum(i.kind == "duplicate" for i in found),
        invalid_records=invalid,
        incomplete_records=sum(i.kind == "incomplete" for i in found),
        zero_volume_observations=sum(i.kind == "zero_volume" for i in found),
        anomalies=tuple(found),
        status=status,
    )
    return ValidatedData(observations=ordered, report=report)
