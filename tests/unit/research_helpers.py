from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from crypto_research.backtesting.models import Instruction, PortfolioTarget, SimulationSettings
from crypto_research.contracts import (
    DatasetReference,
    Market,
    Parameter,
    Period,
    StrategyConfig,
    TargetExposure,
)
from crypto_research.data.models import Observation, Request
from crypto_research.data.storage import Store
from crypto_research.data.validation import validate
from crypto_research.experiments.models import ExperimentSpec
from crypto_research.experiments.registry import Registry
from crypto_research.experiments.runner import Runner

T = datetime(2023, 1, 1, tzinfo=UTC)
DAY = timedelta(days=1)
M = Market(exchange="fixture", pair="BTC/USDT", timeframe="1d")


def case(
    root: Path, count: int = 12
) -> tuple[Store, Registry, Runner, ExperimentSpec, tuple[PortfolioTarget, ...]]:
    source = Store(root / "data")
    rows = tuple(
        Observation(
            open_time=T + i * DAY,
            observed_at=T + (i + 1) * DAY,
            open=100.0,
            high=100.0,
            low=100.0,
            close=100.0,
            volume=10.0,
        )
        for i in range(count)
    )
    query = Request(market=M, period=Period(start=T, end=T + count * DAY))
    manifest = source.save(validate(query, rows), "fixture", T + count * DAY)
    dataset = DatasetReference(
        source="fixture", version=manifest.version, sha256=manifest.dataset_id
    )
    config = StrategyConfig(
        strategy_id="scripted",
        version="1",
        market=M,
        parameters=(Parameter(name="weight", value=1.0),),
    )
    settings = SimulationSettings(
        experiment_id="trial",
        period=Period(start=T + DAY, end=T + count * DAY),
        starting_capital=Decimal(1000),
        fee_bps=Decimal(100),
        slippage_bps=Decimal(0),
        shadow=True,
    )
    spec = ExperimentSpec(
        experiment_id="trial",
        strategy=config,
        dataset=dataset,
        code_revision="a" * 40,
        settings=settings,
        split="test",
    )
    target = PortfolioTarget(
        instructions=(
            Instruction(
                config=config,
                source_dataset=dataset,
                signal=TargetExposure(
                    strategy_id=config.strategy_id,
                    market=M,
                    data_through=T + DAY,
                    generated_at=T + DAY,
                    effective_at=T + DAY,
                    weight=1.0,
                ),
            ),
        )
    )
    registry = Registry(root / "research")
    return source, registry, Runner(source, registry), spec, (target,)


def renamed(spec: ExperimentSpec, experiment_id: str, **changes: object) -> ExperimentSpec:
    return ExperimentSpec.model_validate(
        spec.model_dump()
        | dict(
            experiment_id=experiment_id,
            settings=SimulationSettings.model_validate(
                spec.settings.model_dump() | {"experiment_id": experiment_id}
            ),
        )
        | changes
    )
