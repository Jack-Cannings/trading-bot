"""Typed causal decisions and separate execution-based, half-open outcome labels."""

from bisect import bisect_left
from datetime import datetime, timedelta
from decimal import Context, Decimal, localcontext
from typing import Literal, Self

from pydantic import Field, model_validator

from crypto_research.backtesting.models import BacktestResult
from crypto_research.contracts import (
    Checksum,
    DatasetReference,
    Identifier,
    Record,
    StrategyConfig,
    TargetExposure,
)
from crypto_research.experiments.models import ExperimentSpec
from crypto_research.features.snapshots import FeatureCache, FeatureSnapshot


class DecisionRecord(Record):
    schema_version: Literal[1] = 1
    experiment_id: Identifier
    input_hash: Checksum
    run_id: Identifier
    config: StrategyConfig
    signal: TargetExposure
    dataset: DatasetReference
    data_available_at: datetime
    preceding_exposure: float = Field(ge=0, le=1, allow_inf_nan=False)
    features: FeatureSnapshot

    @model_validator(mode="after")
    def causal(self) -> Self:
        if (
            self.config.market != self.signal.market
            or self.config.strategy_id != self.signal.strategy_id
            or self.features.market != self.config.market
            or self.features.dataset != self.dataset
            or self.features.as_of != self.signal.generated_at
            or not self.signal.data_through <= self.data_available_at <= self.signal.generated_at
        ):
            raise ValueError("decision identity, dataset or information cutoff mismatch")
        return self

    @property
    def decision_id(self) -> str:
        import hashlib

        return hashlib.sha256(
            (
                self.experiment_id + self.input_hash + self.config.identity + self.signal.identity
            ).encode()
        ).hexdigest()


class OutcomeRecord(Record):
    simulated: Literal[True] = True
    schema_version: Literal[1] = 1
    experiment_id: Identifier
    run_id: Identifier
    decision_id: Checksum
    strategy_id: Identifier
    market_id: Checksum
    horizon_days: int = Field(gt=0)
    start: datetime
    end: datetime
    available_at: datetime | None
    status: Literal["complete", "incomplete", "zero_equity"]
    scope: Literal["standalone_shadow", "standalone_simulated"]
    gross_return: float | None = Field(default=None, allow_inf_nan=False)
    net_return: float | None = Field(default=None, allow_inf_nan=False)
    fees: Decimal | None = Field(default=None, ge=0, allow_inf_nan=False)
    slippage: Decimal | None = Field(default=None, ge=0, allow_inf_nan=False)
    average_exposure: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    target_exposure: float = Field(ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def horizon(self) -> Self:
        if self.end - self.start != timedelta(days=self.horizon_days):
            raise ValueError("outcome interval differs from its horizon")
        if self.status == "complete":
            if (
                self.available_at is None
                or self.available_at < self.end
                or any(
                    v is None
                    for v in (
                        self.gross_return,
                        self.net_return,
                        self.fees,
                        self.slippage,
                        self.average_exposure,
                    )
                )
            ):
                raise ValueError("complete outcome requires fully observed simulated performance")
        elif any(
            v is not None
            for v in (
                self.gross_return,
                self.net_return,
                self.fees,
                self.slippage,
                self.average_exposure,
                self.available_at,
            )
        ):
            raise ValueError("incomplete outcome cannot carry completed performance values")
        return self


def decisions(
    spec: ExperimentSpec, result: BacktestResult, run_id: str, features: FeatureCache
) -> tuple[DecisionRecord, ...]:
    records = []
    times = tuple(p.timestamp for p in result.equity)
    for target in result.decisions:
        for instruction in target.instructions:
            if instruction.config != spec.strategy:
                raise ValueError("isolated experiment contains another strategy configuration")
            signal = instruction.signal
            history = features.history_at(spec.dataset, signal.generated_at)
            source = next((c for c in history.candles if c.close_time == signal.data_through), None)
            if source is None:
                raise ValueError("decision source absent from bounded point-in-time history")
            prior = bisect_left(times, signal.generated_at) - 1
            exposure = 0.0
            if prior >= 0:
                exposure = float(
                    next(
                        (
                            p.actual_weight
                            for p in result.equity[prior].positions
                            if p.market == signal.market
                        ),
                        Decimal(0),
                    )
                )
            records.append(
                DecisionRecord(
                    experiment_id=spec.experiment_id,
                    input_hash=spec.input_hash,
                    run_id=run_id,
                    config=instruction.config,
                    signal=signal,
                    dataset=spec.dataset,
                    data_available_at=source.available_at,
                    preceding_exposure=exposure,
                    features=features.at(spec.dataset, signal.generated_at),
                )
            )
    return tuple(records)


def outcomes(
    spec: ExperimentSpec, result: BacktestResult, records: tuple[DecisionRecord, ...]
) -> tuple[OutcomeRecord, ...]:
    if (
        result.settings != spec.settings
        or len(result.data_identities) != 1
        or any(i.config != spec.strategy for t in result.decisions for i in t.instructions)
        or any(p.market != spec.strategy.market for point in result.equity for p in point.positions)
    ):
        raise ValueError(
            "outcomes require matching standalone strategy performance, not shared allocations"
        )
    allowed = {
        (i.config.identity, i.signal.identity) for t in result.decisions for i in t.instructions
    }
    points = {p.timestamp: p for p in result.equity}
    times = tuple(points)
    if times != tuple(sorted(set(times))) or len(times) != len(result.equity):
        raise ValueError("simulated performance must be unique and chronological")
    merged: list[tuple[datetime, datetime]] = []
    for start, end in sorted((gap.period.start, gap.period.end) for gap in result.data_gaps):
        if merged and start <= merged[-1][1]:
            old_start, old_end = merged.pop()
            merged.append((old_start, max(old_end, end)))
        else:
            merged.append((start, end))
    gap_starts = tuple(start for start, _ in merged)
    gap_ends = tuple(end for _, end in merged)
    fills = tuple(sorted(result.fills, key=lambda f: f.timestamp))
    fill_times = tuple(f.timestamp for f in fills)
    labels = []
    with localcontext(Context(prec=60)):
        fee_prefix, slip_prefix = [Decimal(0)], [Decimal(0)]
        for fill in fills:
            fee_prefix.append(fee_prefix[-1] + fill.fee)
            slip_prefix.append(slip_prefix[-1] + fill.slippage)
        exposure_prefix, gap_prefix = [Decimal(0)], [0]
        cadence = 86400 if spec.strategy.market.timeframe == "1d" else 14400
        for left, right in zip(result.equity, result.equity[1:], strict=False):
            seconds = (right.timestamp - left.timestamp).total_seconds()
            exposure = sum((p.actual_weight for p in left.positions), Decimal(0))
            exposure_prefix.append(exposure_prefix[-1] + Decimal(str(seconds)) * exposure)
            gap_prefix.append(gap_prefix[-1] + int(seconds > cadence))
        for record in records:
            if (
                record.experiment_id != spec.experiment_id
                or record.input_hash != spec.input_hash
                or record.dataset != spec.dataset
                or record.config != spec.strategy
                or (record.config.identity, record.signal.identity) not in allowed
            ):
                raise ValueError("decision does not belong to the experiment performance")
            for horizon in spec.outcome_days:
                start = record.signal.effective_at
                end = start + timedelta(days=horizon)
                begin, finish = points.get(start), points.get(end)
                status: Literal["complete", "incomplete", "zero_equity"] = "incomplete"
                values: dict[str, object] = {}
                if begin and finish and end <= spec.settings.period.end:
                    net0, net1 = begin.pre_execution_equity, finish.pre_execution_equity
                    gross0, gross1 = (
                        begin.pre_execution_gross_equity,
                        finish.pre_execution_gross_equity,
                    )
                    lo, hi = bisect_left(times, start), bisect_left(times, end)
                    gap_index = bisect_left(gap_starts, end) - 1
                    source_complete = gap_index < 0 or gap_ends[gap_index] <= start
                    contiguous = gap_prefix[hi] == gap_prefix[lo] and source_complete
                    if all(v is not None for v in (net0, net1, gross0, gross1)) and contiguous:
                        if net0 and gross0 and net1 is not None and gross1 is not None:
                            status = "complete"
                            fill_lo, fill_hi = (
                                bisect_left(fill_times, start),
                                bisect_left(fill_times, end),
                            )
                            values = dict(
                                gross_return=float(gross1 / gross0 - 1),
                                net_return=float(net1 / net0 - 1),
                                fees=fee_prefix[fill_hi] - fee_prefix[fill_lo],
                                slippage=slip_prefix[fill_hi] - slip_prefix[fill_lo],
                                average_exposure=float(
                                    (exposure_prefix[hi] - exposure_prefix[lo])
                                    / Decimal(str((end - start).total_seconds()))
                                ),
                                available_at=end,
                            )
                        else:
                            status = "zero_equity"
                labels.append(
                    OutcomeRecord.model_validate(
                        dict(
                            experiment_id=spec.experiment_id,
                            run_id=record.run_id,
                            decision_id=record.decision_id,
                            strategy_id=record.config.strategy_id,
                            market_id=record.config.market.identity,
                            horizon_days=horizon,
                            start=start,
                            end=end,
                            status=status,
                            available_at=None,
                            scope="standalone_shadow"
                            if spec.settings.shadow
                            else "standalone_simulated",
                            target_exposure=record.signal.weight,
                        )
                        | values
                    )
                )
    return tuple(labels)


def overlapping_intervals(
    left: tuple[datetime, datetime], right: tuple[datetime, datetime]
) -> bool:
    return max(left[0], right[0]) < min(left[1], right[1])


def validate_label_split(
    training: tuple[OutcomeRecord, ...], evaluation: tuple[OutcomeRecord, ...]
) -> None:
    """Reject overlapping future-label windows across proposed research splits."""
    events = sorted(
        [(o.start, o.end, 0) for o in training] + [(o.start, o.end, 1) for o in evaluation]
    )
    maximum: list[datetime | None] = [None, None]
    for start, end, side in events:
        other = maximum[1 - side]
        if other is not None and start < other:
            raise ValueError("overlapping future-outcome intervals leak across evaluation splits")
        previous = maximum[side]
        maximum[side] = max(previous, end) if previous is not None else end
