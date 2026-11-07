"""Immutable validated records; convert with model_dump for tabular tools."""

import hashlib
import json
from datetime import UTC, datetime, timedelta
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

Positive = Annotated[float, Field(gt=0, allow_inf_nan=False, strict=True)]
NonNegative = Annotated[float, Field(ge=0, allow_inf_nan=False, strict=True)]
Identifier = Annotated[str, Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")]
Timeframe = Literal["4h", "1d"]
Checksum = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Finite = Annotated[float, Field(allow_inf_nan=False, strict=True)]


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

    def canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), allow_nan=False
        )

    @property
    def identity(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()


class DatasetReference(Record):
    source: Identifier
    version: Identifier
    sha256: Checksum


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
    available_at: datetime
    dataset: DatasetReference

    @model_validator(mode="after")
    def validate_candle(self) -> Self:
        duration = timedelta(hours=4) if self.market.timeframe == "4h" else timedelta(days=1)
        if self.close_time - self.open_time != duration:
            raise ValueError("candle duration must match timeframe; close time is exclusive")
        if self.low > min(self.open, self.close) or self.high < max(self.open, self.close):
            raise ValueError("OHLC prices are inconsistent")
        if self.available_at < self.open_time:
            raise ValueError("availability cannot precede candle opening")
        if self.complete and self.available_at < self.close_time:
            raise ValueError("completed candle cannot be available before close")
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
            if candle.available_at > self.as_of:
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
    version: Identifier
    market: Market
    parameters: tuple[Parameter, ...] = ()

    @model_validator(mode="after")
    def unique_parameters(self) -> Self:
        names = [parameter.name for parameter in self.parameters]
        if len(names) != len(set(names)):
            raise ValueError("parameter names must be unique")
        object.__setattr__(self, "parameters", tuple(sorted(self.parameters, key=lambda p: p.name)))
        return self


class TargetExposure(Record):
    strategy_id: Identifier
    market: Market
    data_through: datetime
    generated_at: datetime
    effective_at: datetime
    strength: Finite | None = None
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
        if self.config.market != self.history.market:
            raise ValueError("configuration market must match history")
        closes = {candle.close_time: candle for candle in self.history.candles}
        previous: datetime | None = None
        for signal in self.signals:
            if (
                signal.strategy_id != self.config.strategy_id
                or signal.market != self.history.market
            ):
                raise ValueError("signal identity must match input")
            if (
                signal.data_through not in closes
                or signal.generated_at > self.history.as_of
                or closes[signal.data_through].available_at > signal.generated_at
            ):
                raise ValueError(
                    "signal must reference data available at its decision time within the cutoff"
                )
            if previous is not None and signal.effective_at <= previous:
                raise ValueError("signals must have unique, chronological effective timestamps")
            previous = signal.effective_at
        return self


class MarketFeature(Record):
    """Observable input, never a forward return or retrospective regime label."""

    market: Market
    name: Identifier
    version: Identifier
    value: float | str | bool
    data_through: datetime
    available_at: datetime

    @model_validator(mode="after")
    def causal_feature(self) -> Self:
        if self.data_through > self.available_at:
            raise ValueError("feature availability must follow its source data")
        if isinstance(self.value, float):
            import math

            if not math.isfinite(self.value):
                raise ValueError("feature value must be finite")
        return self


class TradingCosts(Record):
    fee_bps: Annotated[float, Field(ge=0, lt=10000, allow_inf_nan=False, strict=True)]
    slippage_bps: Annotated[float, Field(ge=0, lt=10000, allow_inf_nan=False, strict=True)]


class Period(Record):
    start: datetime
    end: datetime

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if self.start >= self.end:
            raise ValueError("period must have start < end (half-open interval)")
        return self


class StrategyPerformance(Record):
    """Observed accounting record; no calculation is performed here."""

    config: StrategyConfig
    period: Period
    available_at: datetime
    position: Finite
    target_exposure: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    cash_allocation: NonNegative
    gross_return: Finite
    net_return: Finite
    trading_costs: NonNegative
    drawdown: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    shadow: bool

    @model_validator(mode="after")
    def observed(self) -> Self:
        if self.available_at < self.period.end:
            raise ValueError("performance cannot be available before period end")
        return self


class DecisionInputs(Record):
    """Only information available at history.as_of can enter selection."""

    history: MarketData
    features: tuple[MarketFeature, ...] = ()
    past_performance: tuple[StrategyPerformance, ...] = ()

    @model_validator(mode="after")
    def point_in_time(self) -> Self:
        for feature in self.features:
            if feature.market != self.history.market or feature.available_at > self.history.as_of:
                raise ValueError("feature must match market and be available at decision time")
        for performance in self.past_performance:
            if performance.available_at > self.history.as_of:
                raise ValueError("performance was not yet available")
        return self


class FutureOutcome(Record):
    """Research-only label, attached after the forward horizon has elapsed."""

    decision: TargetExposure
    horizon: Period
    available_at: datetime
    net_return: Finite
    drawdown: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]

    @model_validator(mode="after")
    def future_only(self) -> Self:
        if self.horizon.start < self.decision.effective_at or self.available_at < self.horizon.end:
            raise ValueError("outcome must follow execution eligibility and horizon completion")
        return self


class ResultReference(Record):
    experiment_id: Identifier
    artifact_uri: Annotated[str, Field(min_length=1)]
    sha256: Checksum


class ExperimentMetadata(Record):
    experiment_id: Identifier
    created_at: datetime
    code_revision: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    config_sha256: Checksum
    strategies: tuple[StrategyConfig, ...] = Field(min_length=1)
    datasets: tuple[DatasetReference, ...] = Field(min_length=1)
    training: Period
    validation: Period
    test: Period
    seed: Annotated[int, Field(ge=0, strict=True)] | None = None
    costs: TradingCosts
    status: Literal["succeeded", "failed"]
    failure_reason: str | None = None
    artifacts: tuple[ResultReference, ...] = ()

    @model_validator(mode="after")
    def reproducible_run(self) -> Self:
        if (
            not self.training.end <= self.validation.start
            or not self.validation.end <= self.test.start
        ):
            raise ValueError("training, validation and test periods must be ordered and disjoint")
        if self.status == "failed" and not self.failure_reason:
            raise ValueError("failed experiments require a failure reason")
        if self.status == "succeeded" and self.failure_reason is not None:
            raise ValueError("successful experiments cannot have a failure reason")
        if any(a.experiment_id != self.experiment_id for a in self.artifacts):
            raise ValueError("artifact experiment identity must match")
        return self
