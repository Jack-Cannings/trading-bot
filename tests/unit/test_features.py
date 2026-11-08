import math
import statistics
from datetime import timedelta

import pytest
from pydantic import ValidationError
from research_helpers import DAY, M, T

from crypto_research.contracts import Candle, DatasetReference, Market, MarketData, MarketFeature
from crypto_research.features.snapshots import (
    FEATURES,
    FeatureCache,
    FeatureSnapshot,
    FeatureValue,
    calculate,
)

D = DatasetReference(source="fixture", version="v1", sha256="a" * 64)


def history(days: int = 220, *, four_hour: bool = False, skip: int = -1) -> MarketData:
    market = Market(exchange="fixture", pair="BTC/USDT", timeframe="4h") if four_hour else M
    step = timedelta(hours=4) if four_hour else DAY
    count = days * 6 if four_hour else days
    candles = tuple(
        Candle(
            market=market,
            open_time=T + i * step,
            close_time=T + (i + 1) * step,
            available_at=T + (i + 1) * step,
            open=float(i + 1),
            high=float(i + 1),
            low=float(i + 1),
            close=float(i + 1),
            volume=10.0,
            complete=True,
            dataset=D,
        )
        for i in range(count)
        if i != skip
    )
    return MarketData(market=market, as_of=T + days * DAY, candles=candles)


def test_independent_trailing_feature_values() -> None:
    h = history()
    snapshot = calculate(h, h.as_of, D)
    values = {v.name: v.observation.value for v in snapshot.values if v.observation}
    assert tuple(values) == FEATURES
    assert values["return_7d"] == pytest.approx(220 / 213 - 1)
    assert values["return_30d"] == pytest.approx(220 / 190 - 1)
    assert values["return_90d"] == pytest.approx(220 / 130 - 1)
    assert values["relative_sma_50d"] == pytest.approx(220 / 195.5 - 1)
    assert values["relative_sma_200d"] == pytest.approx(220 / 120.5 - 1)
    assert values["volatility_30d"] == pytest.approx(
        statistics.stdev([math.log(n / (n - 1)) for n in range(191, 221)]) * math.sqrt(365.25)
    )
    assert values["drawdown_30d"] == 0 and values["volume_30d"] == 10


def test_four_hour_windows_and_missing_history() -> None:
    h = history(four_hour=True)
    snapshot = calculate(h, h.as_of, D)
    assert snapshot.values[0].observation is not None
    assert snapshot.values[0].observation.value == pytest.approx(1320 / 1278 - 1)
    short = history(days=3)
    assert all(
        v.observation is None and v.missing_reason == "insufficient_history"
        for v in calculate(short, short.as_of, D).values
    )
    missing = history(skip=215)
    assert all(
        v.missing_reason == "missing_intervals" for v in calculate(missing, missing.as_of, D).values
    )


def test_future_cutoff_and_cache_identity_protection() -> None:
    h = history()
    with pytest.raises(ValueError, match="exact decision cutoff"):
        calculate(h, T + 100 * DAY, D)
    cache = FeatureCache()
    cache.register(h, D)
    first = cache.at(D, T + 100 * DAY)
    assert first is cache.at(D, T + 100 * DAY)
    assert all(
        v.observation is None or v.observation.data_through <= T + 100 * DAY for v in first.values
    )
    wrong = DatasetReference(source="fixture", version="v1", sha256="b" * 64)
    with pytest.raises(ValueError, match="checksum/source"):
        cache.at(wrong, T + 100 * DAY)
    with pytest.raises(ValueError, match="timezone-aware"):
        cache.at(D, T.replace(tzinfo=None))
    with pytest.raises(ValueError, match="no completed"):
        cache.history_at(D, T)


def test_stale_data_and_future_feature_injection() -> None:
    h = history()
    stale = MarketData(market=M, as_of=h.as_of + 2 * DAY, candles=h.candles)
    assert all(v.missing_reason == "stale_data" for v in calculate(stale, stale.as_of, D).values)
    snapshot = calculate(h, h.as_of, D)
    future = MarketFeature(
        market=M,
        name="return_7d",
        version="1",
        value=1.0,
        data_through=h.as_of + DAY,
        available_at=h.as_of + DAY,
    )
    with pytest.raises(ValidationError, match="future"):
        FeatureSnapshot.model_validate(
            snapshot.model_dump()
            | {"values": (FeatureValue(name="return_7d", observation=future), *snapshot.values[1:])}
        )
