"""Immutable validated records; convert with model_dump for tabular tools."""

from datetime import UTC, datetime, timedelta
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

Positive = Annotated[float, Field(gt=0, allow_inf_nan=False, strict=True)]
NonNegative = Annotated[float, Field(ge=0, allow_inf_nan=False, strict=True)]
Identifier = Annotated[str, Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")]
Timeframe = Literal["4h", "1d"]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    @field_validator("*", mode="after")
    @classmethod
    def normalize_utc(cls, value: object) -> object:
        if isinstance(value, datetime):
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError("timestamps must be timezone-aware")
            return value.astimezone(UTC)
        return value


class Market(Record):
    exchange: Identifier
    pair: Annotated[str, Field(pattern=r"^[A-Z0-9]+/[A-Z0-9]+$")]
    timeframe: Timeframe


class Candle(Record):
    market: Market
    open_time: datetime
    close_time: datetime
    open: Positive
    high: Positive
    low: Positive
    close: Positive
    volume: NonNegative
    complete: bool = Field(strict=True)

    @model_validator(mode="after")
    def validate_candle(self) -> Self:
        duration = timedelta(hours=4) if self.market.timeframe == "4h" else timedelta(days=1)
        if self.close_time - self.open_time != duration:
            raise ValueError("candle duration must match timeframe; close time is exclusive")
        if self.low > min(self.open, self.close) or self.high < max(self.open, self.close):
            raise ValueError("OHLC prices are inconsistent")
        return self


class MarketData(Record):
    """Single market, strictly ordered, complete candles available as of a cutoff."""

    market: Market
    as_of: datetime
    candles: tuple[Candle, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_history(self) -> Self:
        previous: datetime | None = None
        for candle in self.candles:
            if candle.market != self.market or not candle.complete:
                raise ValueError("history requires completed candles for the specified market")
            if candle.close_time > self.as_of:
                raise ValueError("candle was not available at the history cutoff")
            if previous is not None and candle.open_time < previous:
                raise ValueError("candles must be chronological, unique and non-overlapping")
            previous = candle.close_time
        return self


class Parameter(Record):
    name: Identifier
    value: JsonValue

    @field_validator("value")
    @classmethod
    def scalar_parameter(cls, value: JsonValue) -> JsonValue:
        # Immutable scalar parameters keep strategy configurations reproducible.
        import math

        if isinstance(value, (dict, list)) or value is None:
            raise ValueError("parameters must be non-null JSON scalars")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("parameters must be finite")
        return value


class StrategyConfig(Record):
    strategy_id: Identifier
    parameters: tuple[Parameter, ...] = ()

    @model_validator(mode="after")
    def unique_parameters(self) -> Self:
        names = [parameter.name for parameter in self.parameters]
        if len(names) != len(set(names)):
            raise ValueError("parameter names must be unique")
        return self


class TargetExposure(Record):
    strategy_id: Identifier
    market: Market
    data_through: datetime
    generated_at: datetime
    effective_at: datetime
    weight: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False, strict=True)]

    @model_validator(mode="after")
    def causal_times(self) -> Self:
        if not self.data_through <= self.generated_at <= self.effective_at:
            raise ValueError("require data_through <= generated_at <= effective_at")
        return self


class SignalBatch(Record):
    config: StrategyConfig
    history: MarketData
    signals: tuple[TargetExposure, ...]

    @model_validator(mode="after")
    def align_signals(self) -> Self:
        closes = {candle.close_time for candle in self.history.candles}
        previous: datetime | None = None
        for signal in self.signals:
            if (
                signal.strategy_id != self.config.strategy_id
                or signal.market != self.history.market
            ):
                raise ValueError("signal identity must match input")
            if signal.data_through not in closes or signal.generated_at != self.history.as_of:
                raise ValueError(
                    "signal must reference an input close and the deterministic cutoff"
                )
            if previous is not None and signal.effective_at <= previous:
                raise ValueError("signals must have unique, chronological effective timestamps")
            previous = signal.effective_at
        return self


class ExperimentMetadata(Record):
    experiment_id: Identifier
    created_at: datetime
    code_revision: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    config_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    dataset_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    seed: Annotated[int, Field(ge=0, strict=True)]


class ResultReference(Record):
    experiment_id: Identifier
    artifact_uri: Annotated[str, Field(min_length=1)]
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
