"""Versioned, native-timeframe trailing features with explicit missingness."""

from bisect import bisect_right
from datetime import datetime, timedelta
from typing import Literal, Self

import numpy as np
from pydantic import Field, model_validator

from crypto_research.contracts import DatasetReference, Market, MarketData, MarketFeature, Record
from crypto_research.data.models import duration

FEATURES = (
    "return_7d",
    "return_30d",
    "return_90d",
    "volatility_30d",
    "relative_sma_50d",
    "relative_sma_200d",
    "drawdown_30d",
    "volume_30d",
)


class FeatureValue(Record):
    name: str
    observation: MarketFeature | None
    missing_reason: Literal["insufficient_history", "missing_intervals", "stale_data"] | None = None

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if (self.observation is None) != (self.missing_reason is not None):
            raise ValueError("feature requires an observation or an explicit missing reason")
        if self.observation and not isinstance(self.observation.value, float):
            raise ValueError("initial feature schema requires numeric observations")
        if self.observation and self.observation.name != self.name:
            raise ValueError("feature names differ")
        return self


class FeatureSnapshot(Record):
    schema_version: Literal[1] = 1
    market: Market
    dataset: DatasetReference
    as_of: datetime
    values: tuple[FeatureValue, ...]
    volume_days: Literal[30] = 30
    year_days: float = Field(default=365.25, ge=365.25, le=365.25)

    @model_validator(mode="after")
    def causal(self) -> Self:
        if tuple(v.name for v in self.values) != FEATURES:
            raise ValueError("feature schema names/order mismatch")
        for value in self.values:
            observation = value.observation
            if observation and (
                observation.market != self.market or observation.available_at > self.as_of
            ):
                raise ValueError("future or mismatched feature observation")
        return self


def calculate(history: MarketData, cutoff: datetime, dataset: DatasetReference) -> FeatureSnapshot:
    if history.as_of != cutoff or any(c.available_at > cutoff for c in history.candles):
        raise ValueError("feature history must preserve the exact decision cutoff")
    if any(c.dataset != dataset for c in history.candles):
        raise ValueError("feature dataset identity differs from historical vintage")
    bars = 1 if history.market.timeframe == "1d" else 6
    step = duration(history.market)
    anchor = history.candles[-1].close_time
    values: list[FeatureValue] = []
    for name in FEATURES:
        days = int(name.split("_")[-1][:-1]) if name != "volume_30d" else 30
        count = days * bars + (
            1 if name.startswith("return") or name.startswith("volatility") else 0
        )
        window = history.candles[-count:]
        missing: Literal["insufficient_history", "missing_intervals", "stale_data"] | None = None
        if cutoff - anchor > step:
            missing = "stale_data"
        elif len(window) < count:
            missing = (
                "missing_intervals"
                if history.candles[0].close_time <= anchor - (count - 1) * step
                else "insufficient_history"
            )
        elif any(
            b.close_time - a.close_time != step for a, b in zip(window, window[1:], strict=False)
        ):
            missing = "missing_intervals"
        if missing:
            values.append(FeatureValue(name=name, observation=None, missing_reason=missing))
            continue
        prices = np.asarray([c.close for c in window], dtype=np.float64)
        if name.startswith("return"):
            value = float(prices[-1] / prices[0] - 1)
        elif name.startswith("volatility"):
            value = float(np.std(np.diff(np.log(prices)), ddof=1) * np.sqrt(365.25 * bars))
        elif name.startswith("relative_sma"):
            value = float(prices[-1] / np.mean(prices) - 1)
        elif name.startswith("drawdown"):
            value = float(1 - prices[-1] / np.max(prices))
        else:
            value = float(np.mean([c.volume for c in window]))
        values.append(
            FeatureValue(
                name=name,
                observation=MarketFeature(
                    market=history.market,
                    name=name,
                    version="1",
                    value=value,
                    data_through=anchor,
                    available_at=max(c.available_at for c in window),
                ),
            )
        )
    return FeatureSnapshot(
        market=history.market, dataset=dataset, as_of=cutoff, values=tuple(values)
    )


class FeatureCache:
    """Bounded 201-day scans, independent of total history length; exact-version cache."""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, datetime], FeatureSnapshot] = {}
        self._indexes: dict[str, tuple[MarketData, tuple[datetime, ...]]] = {}

    def register(self, history: MarketData, dataset: DatasetReference) -> None:
        existing = self._indexes.get(dataset.version)
        if existing and existing[0] != history:
            raise ValueError("same dataset version cannot identify different observations")
        if any(c.dataset != dataset for c in history.candles):
            raise ValueError("dataset reference mismatch")
        self._indexes[dataset.version] = (history, tuple(c.close_time for c in history.candles))

    def history_at(self, dataset: DatasetReference, cutoff: datetime) -> MarketData:
        if cutoff.tzinfo is None or cutoff.utcoffset() is None:
            raise ValueError("decision cutoff must be timezone-aware")
        history, closes = self._indexes[dataset.version]
        if history.candles[0].dataset != dataset:
            raise ValueError("dataset checksum/source differs from registered version")
        lo = bisect_right(closes, cutoff - timedelta(days=201))
        hi = bisect_right(closes, cutoff)
        candles = tuple(c for c in history.candles[lo:hi] if c.available_at <= cutoff)
        if not candles:
            raise ValueError("no completed and available feature/strategy history at cutoff")
        return MarketData(market=history.market, as_of=cutoff, candles=candles)

    def at(self, dataset: DatasetReference, cutoff: datetime) -> FeatureSnapshot:
        if cutoff.tzinfo is None or cutoff.utcoffset() is None:
            raise ValueError("decision cutoff must be timezone-aware")
        if self._indexes[dataset.version][0].candles[0].dataset != dataset:
            raise ValueError("dataset checksum/source differs from registered version")
        key = (dataset.version, cutoff)
        if key not in self._cache:
            self._cache[key] = calculate(self.history_at(dataset, cutoff), cutoff, dataset)
        return self._cache[key]
