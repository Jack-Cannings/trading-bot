"""Acquisition records; timestamps are UTC and intervals refer to candle opens."""

from datetime import UTC, datetime, timedelta
from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

from crypto_research.contracts import Checksum, Identifier, Market, Period, Record


def duration(market: Market) -> timedelta:
    return timedelta(days=1) if market.timeframe == "1d" else timedelta(hours=4)


def aligned(time: datetime, market: Market) -> bool:
    return time.timestamp() % duration(market).total_seconds() == 0


class Request(Record):
    market: Market
    period: Period
    batch_size: Annotated[int, Field(ge=1, le=1000, strict=True)] = 1000
    retries: Annotated[int, Field(ge=0, le=10, strict=True)] = 3
    backoff_seconds: Annotated[float, Field(ge=0, le=60, allow_inf_nan=False)] = 1.0

    @model_validator(mode="after")
    def boundaries(self) -> Self:
        if not aligned(self.period.start, self.market) or not aligned(self.period.end, self.market):
            raise ValueError("request boundaries must align to UTC candle opens")
        if self.period.start.timestamp() < 0:
            raise ValueError("request cannot precede Unix epoch")
        return self


class Observation(Record):
    """Raw numeric OHLCV, including invalid values pending pipeline validation."""

    open_time: datetime
    open: float = Field(strict=True)
    high: float = Field(strict=True)
    low: float = Field(strict=True)
    close: float = Field(strict=True)
    volume: float = Field(strict=True)
    observed_at: datetime


class Issue(Record):
    kind: Literal[
        "invalid", "duplicate", "incomplete", "zero_volume", "order", "pagination", "revision"
    ]
    message: str
    open_time: datetime | None = None


class QualityReport(Record):
    request: Request
    retrieved: Period | None
    candle_count: int
    expected_intervals: int
    missing_intervals: tuple[Period, ...]
    duplicates_detected: int
    invalid_records: int
    incomplete_records: int
    zero_volume_observations: int
    anomalies: tuple[Issue, ...]
    status: Literal["clean", "incomplete", "invalid"]


class ValidatedData(Record):
    observations: tuple[Observation, ...]
    report: QualityReport


class Manifest(Record):
    dataset_id: Checksum
    version: Checksum
    schema_version: Literal[1] = 1
    provider: Identifier
    request: Request
    actual_coverage: Period | None
    observation_count: int
    acquired_at: datetime
    artifact: Literal["observations.parquet"] = "observations.parquet"
    quality_report: Literal["quality.json"] = "quality.json"
    artifact_sha256: Checksum
    quality_sha256: Checksum
    parent_version: Checksum | None = None
    change: Literal["original", "updated", "corrected", "unchanged"]
    revised_opens: tuple[datetime, ...] = ()
    availability_policy: Literal["exchange-time-after-observation"] = (
        "exchange-time-after-observation"
    )

    @field_validator("revised_opens")
    @classmethod
    def revision_timestamps(cls, values: tuple[datetime, ...]) -> tuple[datetime, ...]:
        if any(v.tzinfo is None or v.utcoffset() is None for v in values):
            raise ValueError("revision timestamps must be timezone-aware")
        return tuple(sorted({v.astimezone(UTC) for v in values}))
