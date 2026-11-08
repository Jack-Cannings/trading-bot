"""Optional independent float-based reference; never part of engine implementation."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pandas as pd
import pytest

from crypto_research.backtesting.engine import PreparedData, run
from crypto_research.backtesting.models import Instruction, PortfolioTarget, SimulationSettings
from crypto_research.contracts import (
    Candle,
    DatasetReference,
    Market,
    MarketData,
    Period,
    StrategyConfig,
    TargetExposure,
)


def test_vectorbt_explicit_next_open_orders() -> None:
    vbt = pytest.importorskip("vectorbt", reason="install the locked reference extra")
    start = datetime(2024, 1, 1, tzinfo=UTC)
    day = timedelta(days=1)
    market = Market(exchange="fixture", pair="BTC/USDT", timeframe="1d")
    config = StrategyConfig(strategy_id="manual", version="1", market=market)
    candles = tuple(
        Candle(
            market=market,
            open_time=start + i * day,
            close_time=start + (i + 1) * day,
            available_at=start + (i + 1) * day,
            open=p,
            high=p,
            low=p,
            close=p,
            volume=1.0,
            complete=True,
            dataset=DatasetReference(source="fixture", version="1", sha256="a" * 64),
        )
        for i, p in enumerate((100.0, 200.0, 300.0, 400.0))
    )
    h = MarketData(market=market, as_of=start + 4 * day, candles=candles)
    targets = tuple(
        PortfolioTarget(
            instructions=(
                Instruction(
                    config=config,
                    signal=TargetExposure(
                        strategy_id="manual",
                        market=market,
                        data_through=start + (i + 1) * day,
                        generated_at=start + (i + 1) * day,
                        effective_at=start + (i + 1) * day,
                        weight=weight,
                    ),
                ),
            )
        )
        for i, weight in enumerate((0.4, 0.0))
    )
    settings = SimulationSettings(
        experiment_id="reference",
        period=Period(start=start, end=start + 4 * day),
        starting_capital=Decimal(1000),
        fee_bps=Decimal(100),
        slippage_bps=Decimal(100),
    )
    result = run(PreparedData((h,)), targets, settings)
    close = pd.Series([100.0, 200.0, 300.0, 400.0])
    reference = vbt.Portfolio.from_orders(
        close,
        size=[0.0, 2.0, -2.0, 0.0],
        price=close,
        fees=0.01,
        slippage=0.01,
        init_cash=1000.0,
        direction="longonly",
        allow_partial=False,
        freq="1D",
    )
    assert result.equity[-1].equity == Decimal("1180.02")
    assert float(reference.value().iloc[-1]) == pytest.approx(
        float(result.equity[-1].equity), abs=1e-9
    )
    assert int(reference.orders.count()) == len(result.fills) == 2
