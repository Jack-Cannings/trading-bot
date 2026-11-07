"""Signal generation only: no exchange client or order submission capability."""

from collections.abc import Iterable
from datetime import datetime
from typing import Protocol, runtime_checkable

from crypto_research.contracts import MarketData, SignalBatch, StrategyConfig


@runtime_checkable
class Strategy(Protocol):
    @property
    def config(self) -> StrategyConfig: ...

    def generate(self, history: MarketData) -> SignalBatch:
        """Pure deterministic function of config and history, including history.as_of.

        Generate current decisions at history.as_of; reconstruct historical decisions
        with separate point-in-time snapshots using reconstruct().
        Multiple markets/timeframes are handled via separate validated input batches.
        No wall clock, ambient RNG, network access, or order submission is permitted.
        """
        ...


def reconstruct(strategy: Strategy, snapshots: Iterable[MarketData]) -> tuple[SignalBatch, ...]:
    """Replay separate point-in-time inputs without exposing a later snapshot early.

    Callers must construct each snapshot from the dataset vintage available then.
    This validates output boundaries, not arbitrary strategy implementation purity.
    """
    batches: list[SignalBatch] = []
    previous: datetime | None = None
    for snapshot in snapshots:
        if previous is not None and snapshot.as_of <= previous:
            raise ValueError("replay cutoffs must be strictly increasing")
        batch = strategy.generate(snapshot)
        if batch.history != snapshot or batch.config != strategy.config:
            raise ValueError("strategy output must retain its exact input and configuration")
        if any(signal.generated_at != snapshot.as_of for signal in batch.signals):
            raise ValueError("replay decisions must use the current snapshot cutoff")
        batches.append(batch)
        previous = snapshot.as_of
    return tuple(batches)
