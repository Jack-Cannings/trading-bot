"""One deterministic, binary, long-only daily momentum family."""

from datetime import timedelta
from decimal import Context, Decimal, localcontext
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from crypto_research.contracts import (
    Market,
    MarketData,
    Parameter,
    Record,
    SignalBatch,
    StrategyConfig,
    TargetExposure,
)

LOOKBACKS = (30, 60, 90, 120, 180)
DAY = timedelta(days=1)


class MomentumParameters(Record):
    lookback: Annotated[int, Field(strict=True)]

    @model_validator(mode="after")
    def predefined(self) -> Self:
        if self.lookback not in LOOKBACKS:
            raise ValueError(f"lookback must be one of {LOOKBACKS}")
        return self


class Assessment(Record):
    status: Literal["ready", "insufficient_history", "missing_intervals", "stale_data"]
    momentum: float | None = None


def configuration(market: Market, lookback: int) -> StrategyConfig:
    parameters = MomentumParameters(lookback=lookback)
    return StrategyConfig(
        strategy_id="time_series_momentum",
        version="1",
        market=market,
        parameters=(Parameter(name="lookback", value=parameters.lookback),),
    )


def daily_window(history: MarketData, count: int) -> str | None:
    if history.as_of - history.candles[-1].close_time > DAY:
        return "stale_data"
    window = history.candles[-count:]
    if any(b.open_time != a.close_time for a, b in zip(window, window[1:], strict=False)):
        return "missing_intervals"
    if len(window) < count:
        return "insufficient_history"
    return None


class TimeSeriesMomentum:
    def __init__(self, config: StrategyConfig) -> None:
        if config.strategy_id != "time_series_momentum" or config.version != "1":
            raise ValueError("unsupported momentum implementation identity/version")
        if config.market.timeframe != "1d":
            raise ValueError("momentum requires daily candles")
        self.parameters = MomentumParameters.model_validate(
            {p.name: p.value for p in config.parameters}
        )
        self.config = config

    def assess(self, history: MarketData) -> Assessment:
        if history.market != self.config.market:
            raise ValueError("strategy and history market differ")
        count = self.parameters.lookback + 1
        missing = daily_window(history, count)
        if missing:
            return Assessment.model_validate(dict(status=missing))
        with localcontext(Context(prec=60)):
            momentum = (
                Decimal(str(history.candles[-1].close))
                / Decimal(str(history.candles[-count].close))
                - 1
            )
        return Assessment(status="ready", momentum=float(momentum))

    def generate(self, history: MarketData) -> SignalBatch:
        assessment = self.assess(history)
        signals: tuple[TargetExposure, ...] = ()
        if assessment.momentum is not None:
            signals = (
                TargetExposure(
                    strategy_id=self.config.strategy_id,
                    market=self.config.market,
                    data_through=history.candles[-1].close_time,
                    generated_at=history.as_of,
                    effective_at=history.as_of,
                    strength=assessment.momentum,
                    weight=1.0 if assessment.momentum > 0 else 0.0,
                ),
            )
        return SignalBatch(config=self.config, history=history, signals=signals)
