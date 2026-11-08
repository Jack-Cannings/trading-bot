"""Predeclared causal passive comparator; no directional signal or fitting."""

import math
import statistics

from crypto_research.contracts import (
    Market,
    MarketData,
    Parameter,
    SignalBatch,
    StrategyConfig,
    TargetExposure,
)
from crypto_research.strategies.time_series_momentum import daily_window


def risk_configuration(market: Market) -> StrategyConfig:
    return StrategyConfig(
        strategy_id="passive_risk_target",
        version="1",
        market=market,
        parameters=(
            Parameter(name="volatility_days", value=30),
            Parameter(name="target_volatility", value=0.20),
        ),
    )


class RiskTarget:
    """Daily min(1, 20% / trailing 30-day sample log-vol * sqrt(365.25))."""

    def __init__(self, config: StrategyConfig) -> None:
        if config != risk_configuration(config.market) or config.market.timeframe != "1d":
            raise ValueError("risk benchmark requires the fixed daily 30-day/20% policy")
        self.config = config

    def generate(self, history: MarketData) -> SignalBatch:
        if history.market != self.config.market:
            raise ValueError("risk benchmark market mismatch")
        signals: tuple[TargetExposure, ...] = ()
        if daily_window(history, 31) is None:
            prices = [c.close for c in history.candles[-31:]]
            returns = [math.log(b / a) for a, b in zip(prices, prices[1:], strict=False)]
            volatility = statistics.stdev(returns) * math.sqrt(365.25)
            signals = (
                TargetExposure(
                    strategy_id=self.config.strategy_id,
                    market=history.market,
                    data_through=history.candles[-1].close_time,
                    generated_at=history.as_of,
                    effective_at=history.as_of,
                    strength=volatility,
                    weight=min(1.0, 0.20 / volatility) if volatility > 0 else 1.0,
                ),
            )
        return SignalBatch(config=self.config, history=history, signals=signals)
