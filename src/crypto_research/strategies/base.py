"""Signal generation only: no exchange client or order submission capability."""

from typing import Protocol, runtime_checkable

from crypto_research.contracts import MarketData, SignalBatch, StrategyConfig


@runtime_checkable
class Strategy(Protocol):
    @property
    def config(self) -> StrategyConfig: ...

    def generate(self, history: MarketData) -> SignalBatch:
        """Pure deterministic function of config and history, including history.as_of.

        Multiple markets/timeframes are handled via separate validated input batches.
        No wall clock, ambient RNG, network access, or order submission is permitted.
        """
        ...
