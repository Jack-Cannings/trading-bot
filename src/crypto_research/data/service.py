"""Acquisition and incremental updates with immutable per-page checkpoints."""

import time
from collections.abc import Callable
from datetime import datetime

from crypto_research.contracts import Period
from crypto_research.data.downloader import pages
from crypto_research.data.models import (
    Issue,
    Manifest,
    Observation,
    QualityReport,
    Request,
    duration,
)
from crypto_research.data.provider import Provider, ProviderError
from crypto_research.data.storage import Store, content_hash
from crypto_research.data.validation import validate, values


class NoDataError(ProviderError):
    def __init__(self, report: QualityReport) -> None:
        super().__init__(
            "no valid completed candles retrieved; requested history may be unavailable"
        )
        self.report = report


def _ranges(periods: list[Period]) -> tuple[Period, ...]:
    merged: list[Period] = []
    for period in sorted(periods, key=lambda p: p.start):
        if merged and period.start <= merged[-1].end:
            prior = merged.pop()
            merged.append(Period(start=prior.start, end=max(prior.end, period.end)))
        else:
            merged.append(period)
    return tuple(merged)


def acquire(
    provider: Provider,
    store: Store,
    request: Request,
    *,
    parent_version: str | None = None,
    overlap: int = 2,
    checkpoint: Callable[[Manifest], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Manifest:
    if overlap < 0:
        raise ValueError("overlap must be nonnegative")
    parent: Manifest | None = None
    original: tuple[Observation, ...] = ()
    previous_report: QualityReport | None = None
    if parent_version:
        parent, original, previous_report = store.read(parent_version)
        if (
            parent.request.market != request.market
            or request.period.start != parent.request.period.start
        ):
            raise ValueError("update must preserve exchange, market, timeframe and start")
        if request.period.end < parent.request.period.end:
            raise ValueError("update cannot shrink the requested period")
    known = {r.open_time: r for r in original}
    initial = validate(request, original)
    ranges = list(initial.report.missing_intervals)
    if original and overlap:
        ranges.append(
            Period(
                start=max(
                    request.period.start,
                    original[-1].open_time - (overlap - 1) * duration(request.market),
                ),
                end=request.period.end,
            )
        )
    inherited = (
        [i for i in previous_report.anomalies if i.kind != "zero_volume"] if previous_report else []
    )
    issues: list[Issue] = []
    revised: set[datetime] = set()
    current = parent
    acquired = parent.acquired_at if parent else None
    # Rebuild quality for the entire requested range, including old gaps.
    for period in _ranges(ranges):
        partial = Request.model_validate(request.model_dump() | {"period": period})
        for page in pages(provider, partial, sleep):
            acquired = page.observed_at
            if parent and acquired < parent.acquired_at:
                raise ProviderError("exchange clock precedes prior acquisition timestamp")
            issues.extend(page.issues)
            validated = validate(partial, page.observations)
            issues.extend(i for i in validated.report.anomalies if i.kind != "zero_volume")
            for row in validated.observations:
                inherited = [i for i in inherited if i.open_time != row.open_time]
                old = known.get(row.open_time)
                if old and values(old) == values(row):
                    continue  # retain first observed availability for unchanged values
                if old:
                    revised.add(row.open_time)
                known[row.open_time] = row
            all_rows = tuple(sorted(known.values(), key=lambda r: r.open_time))
            result = validate(request, all_rows, tuple(inherited + issues))
            if not all_rows:
                continue
            if (
                parent
                and content_hash(request.market, all_rows) == parent.dataset_id
                and request.period == parent.request.period
                and result.report == previous_report
            ):
                current = parent
            else:
                current = store.save(
                    result, provider.identity, acquired, parent, tuple(sorted(revised))
                )
                if checkpoint:
                    checkpoint(current)
    if current is None:
        raise NoDataError(validate(request, (), tuple(inherited + issues)).report)
    return current
