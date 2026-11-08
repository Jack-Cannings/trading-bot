"""Passive initial allocations, using the same causal targets and execution engine."""

from decimal import Context, Decimal, localcontext

from crypto_research.backtesting.engine import PreparedData
from crypto_research.backtesting.models import Instruction, PortfolioTarget, SimulationSettings
from crypto_research.contracts import Market, Parameter, StrategyConfig, TargetExposure


def buy_and_hold(
    data: PreparedData, settings: SimulationSettings, weights: tuple[tuple[Market, Decimal], ...]
) -> tuple[PortfolioTarget, ...]:
    """Empty weights is all cash; one/many weights is single/fixed-weight buy-and-hold."""
    with localcontext(Context(prec=60)):
        if (
            len({m for m, _ in weights}) != len(weights)
            or any(
                not w.is_finite() or not 0 <= w <= 1 or m not in data.markets for m, w in weights
            )
            or sum((w for _, w in weights), Decimal(0)) > 1
        ):
            raise ValueError("benchmark weights must be unique supported markets without leverage")
        invested = [(m, w) for m, w in weights if w > 0]
        if not invested:
            return ()
        first = {}
        for market, _ in invested:
            available = [
                c.available_at
                for (m, _), c in data.sources.items()
                if m == market and settings.period.start <= c.available_at < settings.period.end
            ]
            if not available:
                raise ValueError("no completed benchmark source candle in simulation period")
            first[market] = min(available)
        decision = max(first.values())
        instructions = []
        for market, weight in invested:
            source = max(
                (
                    c
                    for (m, _), c in data.sources.items()
                    if m == market and c.available_at <= decision
                ),
                key=lambda c: c.close_time,
            )
            config = StrategyConfig(
                strategy_id="benchmark",
                version="1",
                market=market,
                parameters=(Parameter(name="weight", value=float(weight)),),
            )
            signal = TargetExposure(
                strategy_id=config.strategy_id,
                market=market,
                data_through=source.close_time,
                generated_at=decision,
                effective_at=decision,
                weight=float(weight),
            )
            instructions.append(Instruction(config=config, signal=signal))
        return (PortfolioTarget(instructions=tuple(instructions)),)
