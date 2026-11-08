"""An entirely offline, predetermined buy-and-hold example."""

import argparse
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from crypto_research.backtesting.benchmarks import buy_and_hold
from crypto_research.backtesting.engine import PreparedData, run
from crypto_research.backtesting.models import SimulationSettings
from crypto_research.contracts import Candle, DatasetReference, Market, MarketData, Period


def demo() -> str:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    market = Market(exchange="synthetic", pair="BTC/USDT", timeframe="1d")
    candles = tuple(
        Candle(
            market=market,
            open_time=start + timedelta(days=i),
            close_time=start + timedelta(days=i + 1),
            available_at=start + timedelta(days=i + 1),
            open=p,
            high=p,
            low=p,
            close=p,
            volume=100.0,
            complete=True,
            dataset=DatasetReference(source="synthetic", version="1", sha256="a" * 64),
        )
        for i, p in enumerate((100.0, 100.0, 110.0, 120.0))
    )
    data = PreparedData(
        (MarketData(market=market, as_of=candles[-1].available_at, candles=candles),)
    )
    settings = SimulationSettings(
        experiment_id="demo",
        period=Period(start=start, end=start + timedelta(days=4)),
        starting_capital=Decimal("1000"),
        fee_bps=Decimal("10"),
        slippage_bps=Decimal("5"),
    )
    result = run(data, buy_and_hold(data, settings, ((market, Decimal(1)),)), settings)
    final = result.equity[-1]
    return (
        f"Starting capital: {settings.starting_capital:.2f} USDT\n"
        f"Final equity: {final.equity:.6f} USDT\n"
        f"Total return: {final.cumulative_return:.6%}\n"
        f"Fees: {final.fees_paid:.6f} USDT\nExecuted trades: {len(result.fills)}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline deterministic backtesting example")
    parser.add_argument("command", choices=["demo"])
    parser.parse_args(argv)
    print(demo())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
