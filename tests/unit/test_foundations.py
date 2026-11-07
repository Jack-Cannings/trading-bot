from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from crypto_research.config.models import ExperimentConfig, load_config
from crypto_research.contracts import (
    Candle,
    DatasetReference,
    DecisionInputs,
    ExperimentMetadata,
    FutureOutcome,
    Market,
    MarketData,
    MarketFeature,
    Parameter,
    Period,
    ResultReference,
    SignalBatch,
    StrategyConfig,
    StrategyPerformance,
    TargetExposure,
    TradingCosts,
)
from crypto_research.strategies.base import Strategy, reconstruct

T = datetime(2025, 1, 1, tzinfo=UTC)
M = Market(exchange="test", pair="BTC/USDT", timeframe="1d")
D = DatasetReference(source="fixture", version="v1", sha256="c" * 64)


def candle(**changes: object) -> Candle:
    values: dict[str, object] = dict(
        market=M,
        open_time=T,
        close_time=T + timedelta(days=1),
        open=10.0,
        high=12.0,
        low=9.0,
        close=11.0,
        volume=0.0,
        complete=True,
        available_at=T + timedelta(days=1),
        dataset=D,
    )
    values.update(changes)
    return Candle.model_validate(values)


def history() -> MarketData:
    return MarketData(market=M, as_of=T + timedelta(days=1), candles=(candle(),))


def test_default_config() -> None:
    config = load_config("configs/research.yaml")
    assert config.starting_capital == 10000
    assert config.markets == ("BTC/USDT", "ETH/USDT", "SOL/USDT")
    assert config.fee_bps > 0 and config.slippage_bps > 0
    assert not config.live_trading and not config.leverage


@pytest.mark.parametrize(
    "replacement",
    [
        "fee_bps: -1",
        "fee_bps: .nan",
        "fee_bps: true",
        "fee_bps: 10000",
        "fee_bps: null",
        "fee_bps: invalid",
    ],
)
def test_bad_cost(tmp_path: Path, replacement: str) -> None:
    text = Path("configs/research.yaml").read_text()
    p = tmp_path / "bad.yaml"
    p.write_text(text.replace("fee_bps: 10.0", replacement))
    with pytest.raises(ValidationError):
        load_config(p)


@pytest.mark.parametrize(
    "old,new",
    [
        ("live_trading: false", "live_trading: true"),
        ("leverage: false", "leverage: true"),
        ("long_only: true", "long_only: false"),
        ("starting_capital: 10000.0", "starting_capital: 0"),
        ("fee_bps: 10.0", ""),
        ("base_currency: USDT", "base_currency: USD"),
        ("primary_timeframe: 1d", "primary_timeframe: 4h"),
        ("SOL/USDT", "BTC/USDT"),
        ("primary_timeframe: 1d", "primary_timeframe: 2h"),
    ],
)
def test_invalid_config(tmp_path: Path, old: str, new: str) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text(Path("configs/research.yaml").read_text().replace(old, new))
    with pytest.raises(ValidationError):
        load_config(p)


@pytest.mark.parametrize(
    "text", ["fee_bps: 1\nfee_bps: 2", "1: value", "[]", "", "!!python/object:bad {}"]
)
def test_invalid_yaml(tmp_path: Path, text: str) -> None:
    import yaml

    p = tmp_path / "bad.yaml"
    p.write_text(text)
    with pytest.raises((ValueError, yaml.YAMLError)):
        load_config(p)


@pytest.mark.parametrize(
    "changes",
    [
        {"open": 0},
        {"volume": -1},
        {"high": 10},
        {"low": 12},
        {"close": float("nan")},
        {"close_time": T},
        {"open_time": T.replace(tzinfo=None)},
        {"complete": "false"},
        {"volume": None},
    ],
)
def test_invalid_candles(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        candle(**changes)


def test_utc_normalization() -> None:
    c = candle(open_time="2025-01-01T02:00:00+02:00", close_time="2025-01-02T02:00:00+02:00")
    assert c.open_time == T and c.open_time.tzinfo == UTC
    assert Candle.model_validate_json(c.model_dump_json()) == c


def test_history_validation() -> None:
    c = candle()
    next_c = candle(
        open_time=T + timedelta(days=1),
        close_time=T + timedelta(days=2),
        available_at=T + timedelta(days=2),
    )
    assert len(MarketData(market=M, as_of=next_c.close_time, candles=(c, next_c)).candles) == 2
    for candles in [
        (next_c, c),
        (c, c),
        (candle(complete=False),),
        (candle(market=M.model_copy(update={"exchange": "other"})),),
    ]:
        with pytest.raises(ValidationError):
            MarketData(market=M, as_of=next_c.close_time, candles=candles)
    with pytest.raises(ValidationError):
        MarketData(market=M, as_of=T, candles=(c,))
    with pytest.raises(ValidationError):
        MarketData(market=M, as_of=T, candles=())


class StubStrategy:
    """Contract test double; does not implement a trading algorithm."""

    @property
    def config(self) -> StrategyConfig:
        return StrategyConfig(strategy_id="stub", version="1", market=M)

    def generate(self, history: MarketData) -> SignalBatch:
        return SignalBatch(config=self.config, history=history, signals=())


def test_strategy_protocol_and_determinism() -> None:
    strategy: Strategy = StubStrategy()
    assert isinstance(strategy, Strategy)
    assert not isinstance(object(), Strategy)
    assert (
        strategy.generate(history()).model_dump_json()
        == strategy.generate(history()).model_dump_json()
    )


def signal(**changes: object) -> TargetExposure:
    values: dict[str, object] = dict(
        strategy_id="stub",
        market=M,
        data_through=T + timedelta(days=1),
        generated_at=T + timedelta(days=1),
        effective_at=T + timedelta(days=1),
        weight=0.5,
    )
    values.update(changes)
    return TargetExposure.model_validate(values)


@pytest.mark.parametrize(
    "changes",
    [
        {"weight": -0.1},
        {"weight": 1.1},
        {"weight": float("inf")},
        {"generated_at": T},
        {"effective_at": T},
    ],
)
def test_invalid_signal(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        signal(**changes)


def test_batch_alignment() -> None:
    config = StrategyConfig(strategy_id="stub", version="1", market=M)
    s = signal()
    assert SignalBatch(config=config, history=history(), signals=(s,)).signals == (s,)
    for signals in [
        (s, s),
        (signal(strategy_id="wrong"),),
        (signal(data_through=T),),
        (signal(generated_at=T + timedelta(days=2), effective_at=T + timedelta(days=2)),),
        (signal(market=M.model_copy(update={"exchange": "other"})),),
    ]:
        with pytest.raises(ValidationError):
            SignalBatch(config=config, history=history(), signals=signals)


@pytest.mark.parametrize("value", [None, [], {}, float("nan"), float("inf")])
def test_bad_parameters(value: object) -> None:
    with pytest.raises(ValidationError):
        Parameter.model_validate(dict(name="period", value=value))


def test_experiment_contracts() -> None:
    p = Parameter(name="period", value=10)
    s = StrategyConfig(strategy_id="stub", version="1", market=M, parameters=(p,))
    with pytest.raises(ValidationError):
        StrategyConfig(strategy_id="stub", version="1", market=M, parameters=(p, p))
    config = load_config("configs/research.yaml")
    assert ExperimentConfig(experiment_id="run1", research=config, strategies=(s,)).strategies == (
        s,
    )
    with pytest.raises(ValidationError):
        ExperimentConfig(experiment_id="run1", research=config, strategies=(s, s))
    m = ExperimentMetadata(
        experiment_id="run1",
        created_at=T,
        code_revision="a" * 40,
        config_sha256="b" * 64,
        strategies=(s,),
        datasets=(D,),
        training=Period(start=T, end=T + timedelta(days=1)),
        validation=Period(start=T + timedelta(days=1), end=T + timedelta(days=2)),
        test=Period(start=T + timedelta(days=2), end=T + timedelta(days=3)),
        costs=TradingCosts(fee_bps=10.0, slippage_bps=5.0),
        status="succeeded",
        seed=0,
    )
    assert m.created_at == T
    assert ResultReference(
        experiment_id="run1", artifact_uri="artifacts/result.json", sha256="d" * 64
    )
    with pytest.raises(ValidationError):
        Market(exchange="test", pair="bad", timeframe="1d")


def test_publication_delay_and_revision() -> None:
    delayed = candle(available_at=T + timedelta(days=2))
    with pytest.raises(ValidationError, match="cutoff"):
        MarketData(market=M, as_of=T + timedelta(days=1), candles=(delayed,))
    h = MarketData(market=M, as_of=delayed.available_at, candles=(delayed,))
    with pytest.raises(ValidationError, match="available"):
        SignalBatch(config=StubStrategy().config, history=h, signals=(signal(),))
    for when in [T - timedelta(seconds=1), T]:
        with pytest.raises(ValidationError, match="availability|available"):
            candle(available_at=when)
    incomplete = candle(complete=False, available_at=T)
    assert not incomplete.complete


def test_historical_signals_and_replay() -> None:
    first = history()
    second_candle = candle(
        open_time=T + timedelta(days=1),
        close_time=T + timedelta(days=2),
        available_at=T + timedelta(days=2),
    )
    second = MarketData(
        market=M, as_of=second_candle.available_at, candles=(candle(), second_candle)
    )
    later = signal(
        data_through=second_candle.close_time, generated_at=second.as_of, effective_at=second.as_of
    )
    batch = SignalBatch(config=StubStrategy().config, history=second, signals=(signal(), later))
    assert batch.signals[0].generated_at < batch.signals[1].generated_at
    replay = reconstruct(StubStrategy(), (first, second))
    assert tuple(b.history.as_of for b in replay) == (first.as_of, second.as_of)
    with pytest.raises(ValueError, match="increasing"):
        reconstruct(StubStrategy(), (second, first))
    wrong = StrategyConfig(
        strategy_id="stub",
        version="1",
        market=Market(exchange="other", pair="BTC/USDT", timeframe="1d"),
    )
    with pytest.raises(ValidationError, match="configuration market"):
        SignalBatch(config=wrong, history=first, signals=())


def test_configuration_identity() -> None:
    a, b = Parameter(name="a", value=1), Parameter(name="b", value=2)
    first = StrategyConfig(strategy_id="stub", version="1", market=M, parameters=(a, b))
    reordered = StrategyConfig(strategy_id="stub", version="1", market=M, parameters=(b, a))
    assert first.identity == reordered.identity
    assert first.canonical_json() == reordered.canonical_json()
    updates: list[dict[str, object]] = [{"version": "2"}, {"parameters": (a,)}]
    for update in updates:
        assert StrategyConfig.model_validate(first.model_dump() | update).identity != first.identity
    experiment = ExperimentConfig(
        experiment_id="run1", research=load_config("configs/research.yaml"), strategies=(first,)
    )
    assert (
        ExperimentConfig.model_validate_json(experiment.model_dump_json()).identity
        == experiment.identity
    )
    with pytest.raises(ValidationError):
        first.version = "changed"


def test_features_performance_and_outcomes() -> None:
    h = history()
    feature = MarketFeature(
        market=M, name="observation", version="1", value=0.5, data_through=T, available_at=h.as_of
    )
    period = Period(start=T, end=h.as_of)
    performance = StrategyPerformance(
        config=StubStrategy().config,
        period=period,
        available_at=h.as_of,
        position=1.0,
        target_exposure=0.5,
        cash_allocation=100.0,
        gross_return=0.02,
        net_return=0.01,
        trading_costs=1.0,
        drawdown=0.1,
        shadow=True,
    )
    inputs = DecisionInputs(history=h, features=(feature,), past_performance=(performance,))
    assert inputs.past_performance[0].shadow
    outcome = FutureOutcome(
        decision=signal(),
        horizon=Period(start=h.as_of, end=h.as_of + timedelta(days=1)),
        available_at=h.as_of + timedelta(days=1),
        net_return=0.02,
        drawdown=0.1,
    )
    for field in ["features", "past_performance"]:
        with pytest.raises(ValidationError):
            DecisionInputs.model_validate({"history": h, field: (outcome,)})
    with pytest.raises(ValidationError):
        DecisionInputs.model_validate({"history": h, "outcomes": (outcome,)})
    with pytest.raises(ValidationError, match="not yet available"):
        DecisionInputs(
            history=h,
            past_performance=(
                StrategyPerformance.model_validate(
                    performance.model_dump() | {"available_at": outcome.available_at}
                ),
            ),
        )
    with pytest.raises(ValidationError, match="decision time"):
        DecisionInputs(
            history=h,
            features=(
                MarketFeature.model_validate(
                    feature.model_dump() | {"available_at": outcome.available_at}
                ),
            ),
        )
    with pytest.raises(ValidationError, match="period end"):
        StrategyPerformance.model_validate(performance.model_dump() | {"available_at": T})
    for value in [float("nan"), float("inf")]:
        with pytest.raises(ValidationError, match="finite"):
            MarketFeature.model_validate(feature.model_dump() | {"value": value})
    with pytest.raises(ValidationError, match="source data"):
        MarketFeature.model_validate(feature.model_dump() | {"available_at": T - timedelta(days=1)})
    with pytest.raises(ValidationError, match="horizon"):
        FutureOutcome.model_validate(outcome.model_dump() | {"available_at": T})
    with pytest.raises(ValidationError, match="start < end"):
        Period(start=T, end=T)


def experiment_metadata(**updates: object) -> ExperimentMetadata:
    values: dict[str, object] = dict(
        experiment_id="run1",
        created_at=T,
        code_revision="a" * 40,
        config_sha256="b" * 64,
        strategies=(StubStrategy().config,),
        datasets=(D,),
        training=Period(start=T, end=T + timedelta(days=1)),
        validation=Period(start=T + timedelta(days=1), end=T + timedelta(days=2)),
        test=Period(start=T + timedelta(days=2), end=T + timedelta(days=3)),
        costs=TradingCosts(fee_bps=10.0, slippage_bps=5.0),
        status="succeeded",
    )
    values.update(updates)
    return ExperimentMetadata.model_validate(values)


def test_failed_and_successful_experiments() -> None:
    assert experiment_metadata().seed is None
    assert experiment_metadata(status="failed", failure_reason="invalid input").status == "failed"
    invalid: list[dict[str, object]] = [
        {"status": "failed"},
        {"failure_reason": "unexpected"},
        {"validation": Period(start=T, end=T + timedelta(days=2))},
        {
            "artifacts": (
                ResultReference(experiment_id="other", artifact_uri="result.json", sha256="d" * 64),
            )
        },
    ]
    for updates in invalid:
        with pytest.raises(ValidationError):
            experiment_metadata(**updates)


@pytest.mark.parametrize("pair", ["btcusdt", "BTC-USDT", "btc/USDT", "BTC/USDT/ETH", "BTC/USDT\n"])
def test_market_identifiers(pair: str) -> None:
    with pytest.raises(ValidationError):
        Market(exchange="test", pair=pair, timeframe="1d")


def test_replay_rejects_wrong_snapshot_and_backdated_outputs() -> None:
    original = history()
    later = MarketData(market=M, as_of=original.as_of + timedelta(days=1), candles=original.candles)

    class FixedOutput:
        config = StubStrategy().config

        def __init__(self, output: SignalBatch) -> None:
            self.output = output

        def generate(self, history: MarketData) -> SignalBatch:
            return self.output

    wrong_history = FixedOutput(
        SignalBatch(config=StubStrategy().config, history=later, signals=())
    )
    with pytest.raises(ValueError, match="exact input"):
        reconstruct(wrong_history, (original,))
    backdated = FixedOutput(
        SignalBatch(config=StubStrategy().config, history=later, signals=(signal(),))
    )
    with pytest.raises(ValueError, match="current snapshot"):
        reconstruct(backdated, (later,))
    current = FixedOutput(
        SignalBatch(config=StubStrategy().config, history=original, signals=(signal(),))
    )
    assert reconstruct(current, (original,))[0].signals == (signal(),)
