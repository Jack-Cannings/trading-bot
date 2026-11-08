"""Explicit synthetic test campaign; never attributed to Binance or real returns."""

import math
from datetime import UTC, datetime, timedelta

from crypto_research.contracts import Market, Period
from crypto_research.data.models import Observation, Request
from crypto_research.data.storage import Store
from crypto_research.data.validation import validate
from crypto_research.experiments.momentum_plan import PAIRS


def fixture_versions(source: Store) -> dict[str, str]:
    start, day = datetime(2020, 1, 1, tzinfo=UTC), timedelta(days=1)
    versions = {}
    for index, pair in enumerate(PAIRS):
        market = Market(exchange="synthetic", pair=pair, timeframe="1d")
        count = 1200 - index * 100
        prices = [
            100
            * math.exp(
                0.0002 * i + (0.25 + index * 0.05) * math.sin(i / 55) + 0.03 * math.sin(i / 3)
            )
            for i in range(count + 1)
        ]
        rows = tuple(
            Observation(
                open_time=start + i * day,
                observed_at=start + (i + 1) * day,
                open=prices[i],
                high=max(prices[i], prices[i + 1]),
                low=min(prices[i], prices[i + 1]),
                close=prices[i + 1],
                volume=1000.0 + i % 30,
            )
            for i in range(count)
        )
        request = Request(market=market, period=Period(start=start, end=start + count * day))
        manifest = source.save(
            validate(request, rows), "synthetic-momentum-fixture-v1", request.period.end
        )
        versions[pair] = manifest.version
    return versions
