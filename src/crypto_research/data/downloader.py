"""Bounded, causal OHLCV pagination with explicit anomaly reporting."""

import time
from collections.abc import Callable, Iterator
from datetime import datetime
from functools import partial

from pydantic import ValidationError

from crypto_research.contracts import Record
from crypto_research.data.models import Issue, Observation, Request, duration
from crypto_research.data.provider import Provider, ProviderError, from_milliseconds, retry


class Page(Record):
    observations: tuple[Observation, ...]
    issues: tuple[Issue, ...]
    observed_at: datetime


def pages(
    provider: Provider, request: Request, sleep: Callable[[float], None] = time.sleep
) -> Iterator[Page]:
    def call_time() -> datetime:
        return retry(provider.server_time, request.retries, request.backoff_seconds, sleep)

    markets, timeframes = retry(
        provider.capabilities, request.retries, request.backoff_seconds, sleep
    )
    if (
        provider.exchange != request.market.exchange
        or request.market.pair not in markets
        or request.market.timeframe not in timeframes
    ):
        raise ProviderError("unsupported exchange, market or timeframe; no substitution performed")
    server_time = call_time()
    cursor = int(request.period.start.timestamp() * 1000)
    end = int(request.period.end.timestamp() * 1000)
    step = int(duration(request.market).total_seconds() * 1000)
    seen: dict[datetime, Observation] = {}
    # One new candle per page is the slowest valid progress; this also bounds bad APIs.
    maximum = (end - cursor) // step + 1
    for _ in range(maximum):
        if cursor >= end:
            return
        raw = retry(
            partial(provider.fetch, request.market, cursor, request.batch_size, end),
            request.retries,
            request.backoff_seconds,
            sleep,
        )
        if not isinstance(raw, list):
            raise ProviderError("malformed exchange OHLCV page; expected a list")
        observed_at = call_time()
        if observed_at < server_time:
            raise ProviderError("exchange clock moved backwards during acquisition")
        server_time = observed_at
        observations: list[Observation] = []
        issues: list[Issue] = []
        opens: list[int] = []
        previous: datetime | None = None
        for item in raw:
            try:
                if len(item) != 6:
                    raise ValueError("OHLCV response must contain six fields")
                timestamp = from_milliseconds(item[0])
                row = Observation.model_validate(
                    dict(
                        zip(
                            ("open_time", "open", "high", "low", "close", "volume", "observed_at"),
                            (timestamp, *item[1:], observed_at),
                            strict=True,
                        )
                    )
                )
                if previous is not None and timestamp < previous:
                    issues.append(
                        Issue(
                            kind="order",
                            message="exchange page is out of order",
                            open_time=timestamp,
                        )
                    )
                previous = timestamp
                opening = int(timestamp.timestamp() * 1000)
                if timestamp in seen:
                    issues.append(
                        Issue(
                            kind="duplicate", message="duplicate page boundary", open_time=timestamp
                        )
                    )
                    # Preserve conflicts for validation; never silently pick one.
                    if row.model_dump(exclude={"observed_at"}) != seen[timestamp].model_dump(
                        exclude={"observed_at"}
                    ):
                        observations.extend((seen[timestamp], row))
                    continue
                if opening < cursor or opening >= end:
                    issues.append(
                        Issue(
                            kind="invalid",
                            message="unexpected candle outside page interval",
                            open_time=timestamp,
                        )
                    )
                    continue
                opens.append(opening)
                seen[timestamp] = row
                observations.append(row)
            except (ValueError, TypeError, OverflowError, OSError, ValidationError) as exc:
                issues.append(Issue(kind="invalid", message=f"malformed OHLCV record: {exc}"))
        advancing = [opening for opening in opens if opening >= cursor]
        if raw and not advancing:
            issues.append(
                Issue(kind="pagination", message="repeated or non-advancing page; stopped")
            )
        yield Page(observations=tuple(observations), issues=tuple(issues), observed_at=observed_at)
        if not raw or not advancing:
            return
        cursor = max(advancing) + step
        # Short pages do not establish exhaustion. The next empty page does.
    raise ProviderError("pagination exceeded bounded request count")
