"""Predeclare the campaign from coverage metadata, before computing any returns."""

from datetime import timedelta
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from crypto_research.config.models import ResearchConfig
from crypto_research.contracts import DatasetReference, Market, Period, Record
from crypto_research.data.storage import Store

PAIRS = ("BTC/USDT", "ETH/USDT", "SOL/USDT")
DAY = timedelta(days=1)


class Window(Record):
    name: str
    period: Period
    split: Literal["training", "validation"]


class Selection(Record):
    market: Market
    dataset: DatasetReference | None = None
    coverage: Period | None = None
    quality: str | None = None
    candle_count: int = 0
    acquired_at: str | None = None
    artifact_sha256: str | None = None
    missing_intervals: tuple[Period, ...] = ()
    unavailable_reason: str | None = None
    windows: tuple[Window, ...] = ()
    holdout: Period | None = None

    @model_validator(mode="after")
    def protect(self) -> Self:
        if self.windows and (self.holdout is None or self.dataset is None):
            raise ValueError("scheduled evaluation requires source and protected holdout")
        if self.holdout and any(w.period.end > self.holdout.start for w in self.windows):
            raise ValueError("evaluation overlaps protected holdout")
        if self.windows:
            if self.coverage is None or self.holdout is None:
                raise ValueError("scheduled evaluation requires verified coverage")
            if (
                self.windows[0].period.start < self.coverage.start + 201 * DAY
                or self.holdout.end > self.coverage.end
                or any(
                    a.period.end > b.period.start
                    for a, b in zip(self.windows, self.windows[1:], strict=False)
                )
            ):
                raise ValueError("invalid chronological coverage or warmup boundaries")
        return self


class CampaignPlan(Record):
    schema_version: Literal[1] = 1
    code_revision: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    research: ResearchConfig
    replay_availability: Literal["observed", "archival-close-plus-60s"] = "observed"
    selections: tuple[Selection, ...]
    common: Period | None = None
    fixture: bool = False
    lookbacks: tuple[Literal[30, 60, 90, 120, 180], ...] = (30, 60, 90, 120, 180)
    cost_multipliers: tuple[Literal[1, 2], ...] = (1, 2)
    direction_threshold: Annotated[float, Field(ge=0.0, le=0.0)] = 0.0
    volatility_threshold: Annotated[float, Field(ge=0.60, le=0.60)] = 0.60
    risk_target_annual: Annotated[float, Field(ge=0.20, le=0.20)] = 0.20
    risk_window_days: Literal[30] = 30

    @model_validator(mode="after")
    def complete_grid(self) -> Self:
        if tuple(s.market.pair for s in self.selections) != PAIRS:
            raise ValueError("campaign requires exactly BTC, ETH and SOL in canonical order")
        if self.lookbacks != (30, 60, 90, 120, 180) or self.cost_multipliers != (1, 2):
            raise ValueError("the predefined research grid cannot be tuned")
        if any(
            s.market.timeframe != "1d"
            or s.market.exchange != ("synthetic" if self.fixture else "binance")
            for s in self.selections
        ):
            raise ValueError("campaign market must match daily real/fixture source identity")
        if self.research.starting_capital != 10000.0 or self.research.base_currency != "USDT":
            raise ValueError("momentum campaign requires 10000 USDT starting capital")
        if self.common and any(
            s.holdout is None
            or not s.windows
            or self.common.start < s.windows[0].period.start
            or self.common.end > s.holdout.start
            for s in self.selections
        ):
            raise ValueError("common evaluation must fit every market's non-holdout history")
        return self


def chronological_windows(coverage: Period) -> tuple[tuple[Window, ...], Period | None]:
    """201-day feature warmup; reserve 365 days, or 20% (min 90) for short histories.

    Require at least 180 development + 90 validation days. Prefer two subsequent
    validation windows (>=90 each); no fitted thresholds or lookback selection.
    """
    eligible = coverage.start + 201 * DAY
    usable = (coverage.end - eligible).days
    reserve = 365 if usable >= 180 + 180 + 365 else max(90, usable // 5)
    if usable - reserve < 270:
        return (), None
    holdout = Period(start=coverage.end - reserve * DAY, end=coverage.end)
    available = (holdout.start - eligible).days
    development = min(365, max(180, available // 3))
    boundary = eligible + development * DAY
    windows = [
        Window(name="development", period=Period(start=eligible, end=boundary), split="training")
    ]
    remaining = (holdout.start - boundary).days
    count = 2 if remaining >= 180 else 1
    for index in range(count):
        end = holdout.start if index == count - 1 else boundary + (remaining // count) * DAY
        windows.append(
            Window(
                name=f"validation-{index + 1}",
                period=Period(start=boundary, end=end),
                split="validation",
            )
        )
        boundary = end
    return tuple(windows), holdout


def make_plan(
    source: Store,
    versions: dict[str, str],
    revision: str,
    research: ResearchConfig,
    *,
    archival: bool = False,
    fixture: bool = False,
) -> CampaignPlan:
    selections = []
    for pair in PAIRS:
        market = Market(exchange="synthetic" if fixture else "binance", pair=pair, timeframe="1d")
        fields: dict[str, object] = {"market": market}
        try:
            if pair not in versions:
                raise ValueError(
                    "no explicit immutable dataset selected; no download attempted by runner"
                )
            manifest, rows, report = source.read(versions[pair])
            fields.update(
                dataset=DatasetReference(
                    source=manifest.provider, version=manifest.version, sha256=manifest.dataset_id
                ),
                coverage=manifest.actual_coverage,
                quality=report.status,
                candle_count=len(rows),
                acquired_at=manifest.acquired_at.isoformat(),
                artifact_sha256=manifest.artifact_sha256,
                missing_intervals=report.missing_intervals,
            )
            if manifest.request.market != market:
                raise ValueError("source market/timeframe does not match requested campaign")
            if not fixture and "binance-public-spot" not in manifest.provider:
                raise ValueError("real campaign requires the Binance public spot provider")
            if manifest.actual_coverage is None:
                raise ValueError("no actual source coverage")
            if (
                report.invalid_records
                or report.duplicates_detected
                or report.incomplete_records
                or any(gap.end > manifest.actual_coverage.start for gap in report.missing_intervals)
                or any(i.kind != "zero_volume" for i in report.anomalies)
            ):
                raise ValueError("quality anomalies or missing intervals inside actual coverage")
            windows, holdout = chronological_windows(manifest.actual_coverage)
            if not windows or holdout is None:
                raise ValueError(
                    "insufficient coverage for warmup, development, validation and holdout"
                )
            fields.update(windows=windows, holdout=holdout)
            if not archival and not any(
                windows[0].period.start <= r.observed_at < holdout.start for r in rows
            ):
                raise ValueError(
                    "historical publication vintages unavailable under observed-time policy"
                )
        except (ValueError, OSError) as exc:
            fields.update(
                unavailable_reason=f"{type(exc).__name__}: {exc}", windows=(), holdout=None
            )
        selections.append(Selection.model_validate(fields))
    common = None
    if all(s.windows and s.holdout for s in selections):
        start = max(s.windows[0].period.start for s in selections)
        end = min(s.holdout.start for s in selections if s.holdout)
        if end - start >= 90 * DAY:
            common = Period(start=start, end=end)
    return CampaignPlan(
        code_revision=revision,
        research=research,
        selections=tuple(selections),
        common=common,
        fixture=fixture,
        replay_availability="archival-close-plus-60s" if archival else "observed",
    )
