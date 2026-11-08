import math
import statistics
from datetime import UTC, datetime, timedelta
from math import sqrt

import pytest

from crypto_research.contracts import Period
from crypto_research.metrics.summary import (
    EquityObservation,
    EquitySeries,
    MetricSettings,
    summarize,
)

T = datetime(2024, 1, 1, tzinfo=UTC)
DAY = timedelta(days=1)


def series(values: tuple[float, ...]) -> EquitySeries:
    return EquitySeries(
        period=Period(start=T, end=T + len(values) * DAY),
        initial_equity=100.0,
        observations=tuple(
            EquityObservation(timestamp=T + (i + 1) * DAY, equity=v, exposure=0.5)
            for i, v in enumerate(values)
        ),
    )


def test_mixed_return_reference() -> None:
    result = summarize(series((110.0, 88.0, 110.0)), MetricSettings())
    assert result.total_net_return == pytest.approx(0.1)
    assert result.average_periodic_return == pytest.approx(0.05)
    assert result.annualized_volatility == pytest.approx(sqrt(0.0525) * sqrt(365.25))
    assert result.maximum_drawdown == pytest.approx(0.2)
    assert result.drawdown_duration_seconds == 2 * 86400


def test_cash_has_undefined_ratios() -> None:
    result = summarize(series((100.0, 100.0, 100.0)), MetricSettings())
    assert result.total_net_return == 0 and result.maximum_drawdown == 0
    assert result.sharpe_ratio is None and result.calmar_ratio is None
    assert result.sortino_ratio is None and result.undefined


def test_independent_risk_reference_and_risk_free() -> None:
    cfg = MetricSettings(risk_free_annual=0.1)
    result = summarize(series((110.0, 88.0, 110.0)), cfg)
    returns = [0.1, -0.2, 0.25]
    rf = 1.1 ** (1 / 365.25) - 1
    excess = [r - rf for r in returns]
    downside = sqrt(sum(min(0, r) ** 2 for r in excess) / 3)
    assert result.sharpe_ratio == pytest.approx(
        statistics.mean(excess) / statistics.stdev(returns) * sqrt(365.25)
    )
    assert result.sortino_ratio == pytest.approx(statistics.mean(excess) / downside * sqrt(365.25))
    assert result.downside_deviation == pytest.approx(downside * sqrt(365.25))
    assert result.geometric_periodic_return == pytest.approx(1.1 ** (1 / 3) - 1)
    assert result.cagr == pytest.approx(1.1 ** (365.25 / 3) - 1)
    assert result.cagr is not None
    assert result.calmar_ratio == pytest.approx(result.cagr / 0.2)
    assert result.var_95 == pytest.approx(0.17)
    assert result.cvar_95 == pytest.approx(0.2)


@pytest.mark.parametrize(
    "values,dd,duration",
    [
        ((110.0, 120.0, 130.0), 0.0, 0),
        ((90.0, 80.0, 70.0), 0.3, 3 * 86400),
        ((90.0, 100.0, 110.0), 0.1, 2 * 86400),
        ((90.0, 80.0, 100.0), 0.2, 3 * 86400),
    ],
)
def test_drawdown_recovery_and_duration(
    values: tuple[float, ...], dd: float, duration: int
) -> None:
    result = summarize(series(values), MetricSettings())
    assert result.maximum_drawdown == pytest.approx(dd)
    assert result.drawdown_duration_seconds == duration


def test_empty_single_observation_and_zero_equity() -> None:
    empty = EquitySeries(period=Period(start=T, end=T + DAY), initial_equity=100, observations=())
    result = summarize(empty, MetricSettings())
    assert result.total_net_return is None and result.maximum_drawdown is None
    one = summarize(series((110.0,)), MetricSettings())
    assert one.average_periodic_return == pytest.approx(0.1)
    assert one.annualized_volatility is None and one.sharpe_ratio is None
    ruined = summarize(series((0.0, 0.0)), MetricSettings())
    assert ruined.total_net_return == -1 and ruined.cagr == -1
    assert ruined.average_periodic_return is None
    assert any("zero equity" in issue.reason for issue in ruined.undefined)


def test_four_hour_scaling_and_irregular_sampling() -> None:
    step = timedelta(hours=4)
    data = EquitySeries(
        period=Period(start=T, end=T + 3 * step),
        initial_equity=100,
        observations=tuple(
            EquityObservation(timestamp=T + (i + 1) * step, equity=e)
            for i, e in enumerate((110, 88, 110))
        ),
    )
    result = summarize(data, MetricSettings(frequency="4h", max_valuation_age_seconds=0))
    daily = summarize(series((110.0, 88.0, 110.0)), MetricSettings())
    assert daily.annualized_volatility is not None
    assert result.annualized_volatility == pytest.approx(daily.annualized_volatility * sqrt(6))
    irregular = EquitySeries(
        period=Period(start=T, end=T + 2 * DAY),
        initial_equity=100,
        observations=(
            EquityObservation(timestamp=T + timedelta(hours=20), equity=105),
            EquityObservation(timestamp=T + DAY + timedelta(hours=1), equity=999),
            EquityObservation(timestamp=T + 2 * DAY, equity=110),
        ),
    )
    result = summarize(irregular, MetricSettings(max_valuation_age_seconds=4 * 3600))
    assert result.average_periodic_return == pytest.approx(((105 / 100 - 1) + (110 / 105 - 1)) / 2)
    assert result.cagr == pytest.approx(1.1 ** (365.25 / 2) - 1)
    strict = summarize(irregular, MetricSettings(max_valuation_age_seconds=0))
    assert strict.average_periodic_return is None


def test_partial_period_not_annualized_as_full_bar_and_exposure_weighting() -> None:
    data = EquitySeries(
        period=Period(start=T, end=T + timedelta(hours=36)),
        initial_equity=100,
        observations=(
            EquityObservation(timestamp=T, equity=100, exposure=0.0),
            EquityObservation(
                timestamp=T + timedelta(hours=12), equity=105, exposure=0.5, invested=True
            ),
            EquityObservation(timestamp=T + DAY, equity=110, exposure=1.0, invested=True),
            EquityObservation(
                timestamp=T + timedelta(hours=36), equity=120, exposure=1.0, invested=True
            ),
        ),
    )
    result = summarize(data, MetricSettings())
    assert result.periodic_observations == 1 and result.average_periodic_return == pytest.approx(
        0.1
    )
    assert result.total_net_return == pytest.approx(0.2)
    assert result.time_in_market == pytest.approx(2 / 3)
    assert result.average_exposure == pytest.approx(0.5)


def test_metric_chronology_and_undefined_overflow() -> None:
    from pydantic import ValidationError

    point = EquityObservation(timestamp=T + DAY, equity=100)
    with pytest.raises(ValidationError, match="strictly increasing"):
        EquitySeries(
            period=Period(start=T, end=T + 2 * DAY), initial_equity=100, observations=(point, point)
        )
    with pytest.raises(ValidationError):
        EquityObservation(timestamp=T, equity=float("nan"))
    tiny_period = EquitySeries(
        period=Period(start=T, end=T + timedelta(seconds=1)),
        initial_equity=1,
        observations=(EquityObservation(timestamp=T + timedelta(seconds=1), equity=2),),
    )
    result = summarize(tiny_period, MetricSettings())
    assert result.cagr is None and result.total_net_return == 1
    assert math.isfinite(result.total_net_return)


def test_expected_shortfall_uses_fractional_tail_mass_in_cash_periods() -> None:
    result = summarize(series((80.0,) * 40), MetricSettings())
    # Forty returns: one -20% loss and 39 zeros. The worst 5% is two observations.
    assert result.var_95 == 0
    assert result.cvar_95 == pytest.approx(0.1)
    assert result.cumulative is not None
    assert result.cumulative[-1].net_return == pytest.approx(-0.2)


def test_return_input_adapter_and_unknown_exposure_are_explicit() -> None:
    from crypto_research.metrics.summary import PeriodicReturn, from_returns

    period = Period(start=T, end=T + 3 * DAY)
    points = tuple(
        PeriodicReturn(timestamp=T + (i + 1) * DAY, net_return=r)
        for i, r in enumerate((0.1, -0.2, 0.25))
    )
    data = from_returns(period, 100.0, points)
    summary = summarize(data, MetricSettings())
    assert summary.total_net_return == pytest.approx(0.1)
    assert summary.annualized_volatility == pytest.approx(sqrt(0.0525) * sqrt(365.25))
    assert summary.average_exposure is None and summary.time_in_market is None
    with pytest.raises(ValueError, match="zero equity"):
        from_returns(period, 100.0, (PeriodicReturn(timestamp=T + DAY, net_return=-1), points[1]))
    with pytest.raises(ValueError, match="entirely absent"):
        from_returns(
            period,
            100.0,
            (PeriodicReturn(timestamp=T + DAY, net_return=0, gross_return=0), points[1]),
        )
