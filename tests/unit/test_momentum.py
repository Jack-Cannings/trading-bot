from datetime import timedelta

import pytest
from pydantic import ValidationError
from test_features import D, history

from crypto_research.contracts import MarketData, Parameter, StrategyConfig
from crypto_research.strategies.time_series_momentum import LOOKBACKS, TimeSeriesMomentum


def strategy(lookback: object = 30) -> TimeSeriesMomentum:
    return TimeSeriesMomentum(
        StrategyConfig.model_validate(
            dict(
                strategy_id="time_series_momentum",
                version="1",
                market=history().market,
                parameters=[dict(name="lookback", value=lookback)],
            )
        )
    )


@pytest.mark.parametrize("lookback", LOOKBACKS)
@pytest.mark.parametrize("slope,weight", [(0, 0.0), (1, 1.0), (-1, 0.0)])
def test_manual_momentum(lookback: int, slope: int, weight: float) -> None:
    h = history(days=lookback + 1)
    prices = [float(1000 + slope * i) for i in range(lookback + 1)]
    candles = tuple(
        type(c).model_validate(c.model_dump() | dict(open=p, high=p, low=p, close=p))
        for c, p in zip(h.candles, prices, strict=True)
    )
    h = MarketData(market=h.market, as_of=h.as_of, candles=candles)
    model = strategy(lookback)
    batch = model.generate(h)
    signal = batch.signals[0]
    assert signal.weight == weight
    assert signal.strength == pytest.approx(prices[-1] / prices[0] - 1)
    assert signal.data_through == signal.generated_at == signal.effective_at == h.as_of
    assert batch == model.generate(h)


@pytest.mark.parametrize("value", [0, -1, 1, 31, 30.0, 30.5, True, "30"])
def test_invalid_lookback(value: object) -> None:
    with pytest.raises(ValueError):
        strategy(value)


def test_warmup_gaps_staleness_and_market_contract() -> None:
    model = strategy()
    assert model.assess(history(days=30)).status == "insufficient_history"
    assert model.generate(history(days=30)).signals == ()
    assert model.assess(history(days=40, skip=20)).status == "missing_intervals"
    assert not model.generate(history(days=40, skip=20)).signals
    h = history(days=40)
    stale = MarketData(market=h.market, as_of=h.as_of + timedelta(days=2), candles=h.candles)
    assert model.assess(stale).status == "stale_data"
    with pytest.raises(ValueError, match="daily"):
        TimeSeriesMomentum(
            StrategyConfig(
                strategy_id="time_series_momentum",
                version="1",
                market=history(four_hour=True).market,
                parameters=(Parameter(name="lookback", value=30),),
            )
        )
    with pytest.raises(ValueError, match="market"):
        model.generate(history(four_hour=True))
    with pytest.raises(ValidationError):
        MarketData(market=h.market, as_of=h.as_of - timedelta(days=1), candles=h.candles)
    with pytest.raises(ValidationError):
        MarketData(
            market=h.market,
            as_of=h.as_of,
            candles=(
                type(h.candles[-1]).model_validate(
                    h.candles[-1].model_dump() | {"complete": False}
                ),
            ),
        )
    assert all(c.dataset == D for c in h.candles)


def test_alternating_equal_endpoints_and_configuration_identity() -> None:
    h = history(days=31)
    candles = tuple(
        type(c).model_validate(
            c.model_dump()
            | dict(
                open=100.0 + i % 2,
                high=100.0 + i % 2,
                low=100.0 + i % 2,
                close=100.0 + i % 2,
            )
        )
        for i, c in enumerate(h.candles)
    )
    signal = (
        strategy().generate(MarketData(market=h.market, as_of=h.as_of, candles=candles)).signals[0]
    )
    assert signal.strength == 0 and signal.weight == 0
    assert len({strategy(n).config.identity for n in LOOKBACKS}) == 5
