from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from crypto_research.config.models import ExperimentConfig, load_config
from crypto_research.contracts import (
    Candle,
    ExperimentMetadata,
    Market,
    MarketData,
    Parameter,
    ResultReference,
    SignalBatch,
    StrategyConfig,
    TargetExposure,
)
from crypto_research.strategies.base import Strategy

T = datetime(2025, 1, 1, tzinfo=UTC)
M = Market(exchange="test", pair="BTC/USDT", timeframe="1d")


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
    next_c = candle(open_time=T + timedelta(days=1), close_time=T + timedelta(days=2))
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
        return StrategyConfig(strategy_id="stub")

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
    config = StrategyConfig(strategy_id="stub")
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
    s = StrategyConfig(strategy_id="stub", parameters=(p,))
    with pytest.raises(ValidationError):
        StrategyConfig(strategy_id="stub", parameters=(p, p))
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
        dataset_sha256="c" * 64,
        seed=0,
    )
    assert m.created_at == T
    assert ResultReference(
        experiment_id="run1", artifact_uri="artifacts/result.json", sha256="d" * 64
    )
    with pytest.raises(ValidationError):
        Market(exchange="test", pair="bad", timeframe="1d")
