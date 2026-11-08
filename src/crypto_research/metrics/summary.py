"""Explicit fixed-grid portfolio analytics. Undefined metrics are never zero-filled."""

import math
from bisect import bisect_right
from datetime import datetime, timedelta
from decimal import Context, localcontext
from typing import Annotated, Literal, Self

import numpy as np
from pydantic import Field, model_validator

from crypto_research.backtesting.accounting import amount
from crypto_research.backtesting.models import BacktestResult
from crypto_research.contracts import Identifier, Period, Record

Finite = Annotated[float, Field(allow_inf_nan=False)]
YEAR_SECONDS = 365.25 * 86400


class MetricSettings(Record):
    frequency: Literal["1d", "4h"] = "1d"
    risk_free_annual: Annotated[float, Field(gt=-1, allow_inf_nan=False)] = 0.0
    max_valuation_age_seconds: Annotated[int, Field(ge=0, le=31557600, strict=True)] = 86400

    @property
    def seconds(self) -> int:
        return 86400 if self.frequency == "1d" else 14400


class EquityObservation(Record):
    timestamp: datetime
    equity: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    gross_equity: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None
    exposure: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] | None = None
    invested: bool | None = None

    @model_validator(mode="after")
    def invested_consistency(self) -> Self:
        if self.invested is None and self.exposure is not None:
            object.__setattr__(self, "invested", self.exposure > 0)
        elif self.exposure is not None and self.exposure > 0 and self.invested is False:
            raise ValueError("positive asset exposure must be invested")
        return self


class EquitySeries(Record):
    source_run_id: Identifier | None = None
    period: Period
    initial_equity: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    observations: tuple[EquityObservation, ...]

    @model_validator(mode="after")
    def chronology(self) -> Self:
        previous = None
        for point in self.observations:
            if not self.period.start <= point.timestamp <= self.period.end:
                raise ValueError("equity observation outside period")
            if previous is not None and point.timestamp <= previous:
                raise ValueError("equity observations must be unique and strictly increasing")
            previous = point.timestamp
        return self


class UndefinedMetric(Record):
    metric: str
    reason: str


class CumulativeReturn(Record):
    timestamp: datetime
    net_return: Finite
    gross_return: Finite | None = None


class PerformanceSummary(Record):
    schema_version: Literal[1] = 1
    settings: MetricSettings
    period: Period
    initial_equity: Finite
    source_run_id: Identifier | None = None
    periodic_observations: int
    total_net_return: Finite | None
    cagr: Finite | None
    average_periodic_return: Finite | None
    geometric_periodic_return: Finite | None
    annualized_volatility: Finite | None
    sharpe_ratio: Finite | None
    sortino_ratio: Finite | None
    calmar_ratio: Finite | None
    downside_deviation: Finite | None
    maximum_drawdown: Finite | None
    drawdown_duration_seconds: Finite | None
    var_95: Finite | None
    cvar_95: Finite | None
    total_fees: Finite | None
    total_slippage: Finite | None
    turnover: Finite | None
    number_of_trades: int | None
    time_in_market: Finite | None
    average_exposure: Finite | None
    cost_drag: Finite | None
    cumulative: tuple[CumulativeReturn, ...] | None = None
    undefined: tuple[UndefinedMetric, ...]


def from_backtest(result: BacktestResult) -> EquitySeries:
    with localcontext(Context(prec=60)):
        initial = float(amount(result.settings.starting_capital))
        observations = tuple(
            EquityObservation(
                timestamp=p.timestamp,
                equity=float(p.equity),
                gross_equity=float(p.gross_equity),
                exposure=float(sum((v.actual_weight for v in p.positions), start=0)),
                invested=any(v.quantity > 0 for v in p.positions),
            )
            for p in result.equity
        )
    return EquitySeries(
        source_run_id=result.run_id,
        period=result.settings.period,
        initial_equity=initial,
        observations=observations,
    )


class PeriodicReturn(Record):
    timestamp: datetime
    net_return: Annotated[float, Field(ge=-1, allow_inf_nan=False)]
    gross_return: Annotated[float, Field(ge=-1, allow_inf_nan=False)] | None = None
    exposure: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] | None = None
    invested: bool | None = None


def from_returns(
    period: Period, initial_equity: float, observations: tuple[PeriodicReturn, ...]
) -> EquitySeries:
    validated = EquitySeries(period=period, initial_equity=initial_equity, observations=())
    gross_supplied = [point.gross_return is not None for point in observations]
    if any(gross_supplied) and not all(gross_supplied):
        raise ValueError("gross return observations must be complete or entirely absent")
    net, gross = validated.initial_equity, validated.initial_equity
    points = []
    for point in observations:
        if net == 0 or (all(gross_supplied) and gross == 0):
            raise ValueError("periodic return after zero equity is undefined")
        net *= 1 + point.net_return
        if point.gross_return is not None:
            gross *= 1 + point.gross_return
        points.append(
            EquityObservation(
                timestamp=point.timestamp,
                equity=net,
                gross_equity=gross if point.gross_return is not None else None,
                exposure=point.exposure,
                invested=point.invested,
            )
        )
    return EquitySeries(period=period, initial_equity=initial_equity, observations=tuple(points))


def summarize(
    series: EquitySeries, settings: MetricSettings, backtest: BacktestResult | None = None
) -> PerformanceSummary:
    if backtest is not None and from_backtest(backtest) != series:
        raise ValueError("backtest ledger and equity series do not match")
    issues: list[UndefinedMetric] = []
    result: dict[str, float | int | None] = {}

    def put(name: str, value: float | int | None, reason: str = "undefined") -> None:
        if value is not None and not math.isfinite(value):
            value, reason = None, "non-finite numerical result"
        result[name] = value
        if value is None:
            issues.append(UndefinedMetric(metric=name, reason=reason))

    times = tuple(p.timestamp for p in series.observations)
    elapsed = (series.period.end - series.period.start).total_seconds()

    def at(timestamp: datetime) -> EquityObservation | None:
        index = bisect_right(times, timestamp) - 1
        if index < 0:
            return None
        point = series.observations[index]
        return (
            point
            if (timestamp - point.timestamp).total_seconds() <= settings.max_valuation_age_seconds
            else None
        )

    end = at(series.period.end)
    total = end.equity / series.initial_equity - 1 if end else None
    put("total_net_return", total, "no fresh final valuation")
    cagr = None
    if end:
        try:
            cagr = (
                math.expm1(
                    (math.log(end.equity) - math.log(series.initial_equity))
                    * YEAR_SECONDS
                    / elapsed
                )
                if end.equity
                else -1.0
            )
        except OverflowError:
            pass
    put("cagr", cagr, "missing final valuation or compounded annualization overflow")
    grid: list[float] = [series.initial_equity]
    grid_ok = True
    timestamp = series.period.start + timedelta(seconds=settings.seconds)
    while timestamp <= series.period.end:
        point = at(timestamp)
        if point is None:
            grid_ok = False
            break
        grid.append(point.equity)
        timestamp += timedelta(seconds=settings.seconds)
    returns = np.array([], dtype=np.float64)
    reason = "no complete fixed-frequency intervals or missing/stale valuations"
    if grid_ok and len(grid) > 1:
        if any(value == 0 for value in grid[:-1]):
            reason = "return after zero equity is undefined"
        else:
            returns = np.asarray(grid[1:]) / np.asarray(grid[:-1]) - 1
    count = len(returns)
    periods = YEAR_SECONDS / settings.seconds
    rf = math.expm1(math.log1p(settings.risk_free_annual) / periods)
    excess = returns - rf
    mean = float(np.mean(returns)) if count else None
    put("average_periodic_return", mean, reason)
    geometric = None
    if count:
        geometric = -1.0 if np.any(returns == -1) else math.expm1(float(np.mean(np.log1p(returns))))
    put("geometric_periodic_return", geometric, reason)
    std = float(np.std(returns, ddof=1)) if count >= 2 else None
    put(
        "annualized_volatility",
        std * math.sqrt(periods) if std is not None else None,
        "at least two periodic returns required",
    )
    epsilon = np.finfo(np.float64).eps * max(1.0, abs(mean or 0.0))
    put(
        "sharpe_ratio",
        float(np.mean(excess)) / std * math.sqrt(periods)
        if std is not None and std > epsilon
        else None,
        "insufficient observations or zero volatility",
    )
    downside = float(np.sqrt(np.mean(np.minimum(excess, 0) ** 2))) if count else None
    put(
        "downside_deviation",
        downside * math.sqrt(periods) if downside is not None else None,
        reason,
    )
    put(
        "sortino_ratio",
        float(np.mean(excess)) / downside * math.sqrt(periods)
        if downside is not None and downside > epsilon
        else None,
        "insufficient observations or zero downside deviation",
    )
    peak, peak_time = series.initial_equity, series.period.start
    max_dd, max_duration = 0.0, 0.0
    underwater = False
    for point in series.observations:
        if point.equity >= peak:
            if underwater:
                max_duration = max(max_duration, (point.timestamp - peak_time).total_seconds())
            peak, peak_time = point.equity, point.timestamp
            underwater = False
        else:
            underwater = True
            max_dd = max(max_dd, 1 - point.equity / peak)
            max_duration = max(max_duration, (point.timestamp - peak_time).total_seconds())
    if end and end.equity < peak:
        max_duration = max(max_duration, (series.period.end - peak_time).total_seconds())
    put("maximum_drawdown", max_dd if series.observations else None, "empty equity series")
    put(
        "drawdown_duration_seconds",
        max_duration if series.observations else None,
        "empty equity series",
    )
    put(
        "calmar_ratio",
        cagr / max_dd if cagr is not None and max_dd > 0 else None,
        "missing CAGR or zero drawdown",
    )
    quantile = float(np.quantile(returns, 0.05, method="linear")) if count else None
    put("var_95", -quantile if quantile is not None else None, reason)
    expected_shortfall = None
    if count:
        ordered = np.sort(returns)
        mass = 0.05 * count
        full = int(math.floor(mass))
        tail_sum = float(np.sum(ordered[:full])) + (mass - full) * float(ordered[full])
        expected_shortfall = -tail_sum / mass
    put("cvar_95", expected_shortfall, reason)
    active_seconds, exposure_seconds = 0.0, 0.0
    current_time = series.period.start
    current_exposure: float | None = None
    invested: bool | None = None
    exposure_ok = market_time_ok = bool(series.observations)
    for point in (
        *series.observations,
        EquityObservation(timestamp=series.period.end, equity=series.initial_equity),
    ):
        seconds = (point.timestamp - current_time).total_seconds()
        if seconds > 0:
            if seconds > settings.max_valuation_age_seconds or current_exposure is None:
                exposure_ok = False
            else:
                exposure_seconds += seconds * current_exposure
            if seconds > settings.max_valuation_age_seconds or invested is None:
                market_time_ok = False
            else:
                active_seconds += seconds * invested
        current_time, current_exposure, invested = point.timestamp, point.exposure, point.invested
    put(
        "time_in_market",
        active_seconds / elapsed if market_time_ok else None,
        "missing/stale position history",
    )
    put(
        "average_exposure",
        exposure_seconds / elapsed if exposure_ok else None,
        "missing/stale exposure history",
    )
    for name, value in (
        (
            "total_fees",
            float(backtest.equity[-1].fees_paid) if backtest and backtest.equity else None,
        ),
        (
            "total_slippage",
            float(backtest.equity[-1].slippage_paid) if backtest and backtest.equity else None,
        ),
        (
            "turnover",
            float(backtest.equity[-1].turnover) / series.initial_equity
            if backtest and backtest.equity
            else None,
        ),
        ("number_of_trades", len(backtest.fills) if backtest else None),
    ):
        put(name, value, "backtesting ledger not supplied")
    put(
        "cost_drag",
        end.gross_equity / series.initial_equity - 1 - total
        if end and end.gross_equity is not None and total is not None
        else None,
        "comparable gross series unavailable",
    )
    cumulative = tuple(
        CumulativeReturn(
            timestamp=p.timestamp,
            net_return=p.equity / series.initial_equity - 1,
            gross_return=p.gross_equity / series.initial_equity - 1
            if p.gross_equity is not None
            else None,
        )
        for p in series.observations
    )
    return PerformanceSummary.model_validate(
        dict(
            settings=settings,
            period=series.period,
            initial_equity=series.initial_equity,
            source_run_id=series.source_run_id,
            periodic_observations=count,
            cumulative=cumulative,
            undefined=tuple(issues),
        )
        | result
    )
