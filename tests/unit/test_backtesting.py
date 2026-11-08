from datetime import UTC, datetime, timedelta
from decimal import ROUND_UP, localcontext
from decimal import Decimal as D

import pytest
from pydantic import ValidationError

from crypto_research.backtesting.accounting import Account
from crypto_research.backtesting.benchmarks import buy_and_hold
from crypto_research.backtesting.engine import (
    PreparedData,
    replay_targets,
    run,
    targets_from_batches,
)
from crypto_research.backtesting.models import (
    Instruction,
    PortfolioTarget,
    Side,
    SimulationSettings,
)
from crypto_research.contracts import (
    Candle,
    DatasetReference,
    Market,
    MarketData,
    Period,
    SignalBatch,
    StrategyConfig,
    TargetExposure,
)

T = datetime(2024, 1, 1, tzinfo=UTC)
DAY = timedelta(days=1)
M = Market(exchange="fixture", pair="BTC/USDT", timeframe="1d")


def settings(**changes: object) -> SimulationSettings:
    values: dict[str, object] = dict(
        experiment_id="test",
        period=Period(start=T, end=T + 4 * DAY),
        starting_capital=D("1000"),
        fee_bps=D("100"),
        slippage_bps=D("0"),
    )
    return SimulationSettings.model_validate(values | changes)


def test_accounting_round_trip_exact() -> None:
    account = Account(settings())
    buy, fill, ledger = account.execute("buy", "target", M, T, T, T, "buy", D("2"), D("100"))
    assert buy.status == "filled" and fill is not None and ledger is not None
    assert account.cash == D("798") and account.positions[M].basis == D("202")
    sell, _, entry = account.execute("sell", "target2", M, T, T, T, "sell", D("2"), D("120"))
    assert sell.status == "filled" and entry is not None
    assert account.cash == D("1035.6")
    assert account.positions[M].quantity == 0 and account.positions[M].basis == 0
    assert account.realized_pnl == D("35.6") and account.fees == D("4.4")


def test_insufficient_cash_rejected_without_ledger() -> None:
    account = Account(settings())
    order, fill, ledger = account.execute("o", "t", M, T, T, T, "buy", D("10"), D("100"))
    assert order.status == "rejected" and "cash" in (order.reason or "")
    assert fill is None and ledger is None and account.cash == D("1000")


def history(
    prices: tuple[float, ...] = (100.0, 200.0, 300.0, 400.0),
    market: Market = M,
    *,
    skip: tuple[int, ...] = (),
    delay: int = 0,
) -> MarketData:
    candles = tuple(
        Candle(
            market=market,
            open_time=T + i * DAY,
            close_time=T + (i + 1) * DAY,
            available_at=T + (i + 1) * DAY + timedelta(hours=delay),
            open=p,
            high=p,
            low=p,
            close=p,
            volume=10.0,
            complete=True,
            dataset=DatasetReference(source="fixture", version="1", sha256="a" * 64),
        )
        for i, p in enumerate(prices)
        if i not in skip
    )
    return MarketData(market=market, as_of=max(c.available_at for c in candles), candles=candles)


def instruction(
    h: MarketData,
    index: int,
    weight: float,
    *,
    effective: datetime | None = None,
    decision: datetime | None = None,
) -> Instruction:
    candle = h.candles[index]
    timestamp = decision or candle.available_at
    config = StrategyConfig(strategy_id="manual", version="1", market=h.market)
    signal = TargetExposure(
        strategy_id="manual",
        market=h.market,
        data_through=candle.close_time,
        generated_at=timestamp,
        effective_at=effective or timestamp,
        weight=weight,
    )
    return Instruction(config=config, signal=signal)


def target(h: MarketData, index: int, weight: float, **times: datetime) -> PortfolioTarget:
    return PortfolioTarget(instructions=(instruction(h, index, weight, **times),))


def free_settings(**changes: object) -> SimulationSettings:
    return settings(**({"fee_bps": D(0), "slippage_bps": D(0)} | changes))


def test_next_open_gap_and_liquidation_manual_reference() -> None:
    h = history()
    result = run(PreparedData((h,)), (target(h, 0, 1), target(h, 1, 0)), free_settings())
    assert [f.execution_price for f in result.fills] == [D(200), D(300)]
    assert [f.timestamp for f in result.fills] == [T + DAY, T + 2 * DAY]
    assert [f.quantity for f in result.fills] == [D(5), D(5)]
    assert result.equity[-1].equity == D(1500)
    assert result.equity[-1].realized_pnl == D(500)
    assert result.equity[-1].unrealized_pnl == 0
    assert result.equity[-1].cash == D(1500)
    assert len(result.fills) == len(result.ledger) == len(result.orders) == 2


@pytest.mark.parametrize(
    "prices,expected",
    [
        ((100.0, 100.0, 100.0, 100.0), D(1000)),
        ((50.0, 100.0, 150.0, 200.0), D(2000)),
        ((400.0, 200.0, 150.0, 100.0), D(500)),
        ((100.0, 200.0, 100.0, 200.0), D(1000)),
    ],
)
def test_passive_price_fixtures(prices: tuple[float, ...], expected: D) -> None:
    h = history(prices)
    data = PreparedData((h,))
    cfg = free_settings()
    result = run(data, buy_and_hold(data, cfg, ((M, D(1)),)), cfg)
    assert result.equity[-1].equity == expected
    assert len(result.fills) == 1
    assert result.equity[-1].cumulative_return == expected / D(1000) - 1
    assert result.equity[-1].gross_equity == expected


def test_effective_timestamp_and_publication_delay() -> None:
    h = history()
    result = run(
        PreparedData((h,)),
        (target(h, 0, 1, effective=T + DAY + timedelta(hours=1)),),
        free_settings(),
    )
    assert result.fills[0].timestamp == T + 2 * DAY
    assert result.fills[0].reference_price == D(300)
    delayed = history(delay=1)
    result = run(PreparedData((delayed,)), (target(delayed, 0, 1),), free_settings())
    assert result.fills[0].timestamp == T + 2 * DAY
    assert all(p.cash == D(1000) for p in result.equity if p.timestamp < T + 2 * DAY)
    with pytest.raises(ValidationError):
        instruction(delayed, 0, 1, decision=T + timedelta(hours=1))
    invalid = instruction(delayed, 0, 1, decision=T + DAY)
    with pytest.raises(ValueError, match="unavailable"):
        run(PreparedData((delayed,)), (PortfolioTarget(instructions=(invalid,)),), free_settings())


def test_missing_execution_and_end_of_data() -> None:
    h = history(skip=(1,))
    data = PreparedData((h,))
    result = run(data, (target(h, 0, 1),), free_settings(max_execution_delay_seconds=0))
    assert not result.fills and result.target_outcomes[0].status == "rejected"
    deferred = run(data, (target(h, 0, 1),), free_settings())
    assert deferred.fills[0].timestamp == T + 2 * DAY
    h = history()
    result = run(PreparedData((h,)), (target(h, 2, 1, effective=T + 4 * DAY),), free_settings())
    assert result.target_outcomes[0].status == "rejected" and not result.orders


def test_duplicate_out_of_order_and_colliding_targets() -> None:
    h = history()
    data = PreparedData((h,))
    for targets in ((target(h, 0, 1), target(h, 0, 1)), (target(h, 1, 1), target(h, 0, 0))):
        with pytest.raises(ValueError, match="increasing"):
            run(data, targets, free_settings())
    with pytest.raises(ValueError, match="same candle"):
        run(data, (target(h, 0, 1, effective=T + 2 * DAY), target(h, 1, 0)), free_settings())


def test_multi_asset_rotation_sells_first_and_cash_reconciles() -> None:
    eth = Market(exchange="fixture", pair="ETH/USDT", timeframe="1d")
    btc_h = history((100.0, 100.0, 200.0, 200.0))
    eth_h = history((10.0, 10.0, 20.0, 40.0), eth)
    data = PreparedData((eth_h, btc_h))
    rotate = PortfolioTarget(instructions=(instruction(btc_h, 1, 0.0), instruction(eth_h, 1, 1.0)))
    result = run(data, (target(btc_h, 0, 1), rotate), free_settings())
    assert [(f.market.pair, f.side, f.quantity) for f in result.fills] == [
        ("BTC/USDT", "buy", D(10)),
        ("BTC/USDT", "sell", D(10)),
        ("ETH/USDT", "buy", D(100)),
    ]
    assert result.equity[-1].equity == D(4000)
    assert result.equity[-1].realized_pnl == D(1000)
    assert result.equity[-1].unrealized_pnl == D(2000)
    assert all(p.cash >= 0 and p.equity == p.cash + p.holdings_value for p in result.equity)
    assert (
        run(
            PreparedData((btc_h, eth_h)), (target(btc_h, 0, 1), rotate), free_settings()
        ).canonical_json()
        == result.canonical_json()
    )


def test_partial_investment_and_fixed_weight_benchmark_drift() -> None:
    eth = Market(exchange="fixture", pair="ETH/USDT", timeframe="1d")
    h = history((100.0, 100.0, 100.0, 200.0))
    e = history((10.0, 10.0, 10.0, 10.0), eth)
    data = PreparedData((h, e))
    cfg = free_settings()
    targets = buy_and_hold(data, cfg, ((M, D("0.4")), (eth, D("0.4"))))
    result = run(data, targets, cfg)
    assert result.equity[-1].cash == D(200)
    assert result.equity[-1].equity == D(1400)
    assert len(result.fills) == 2
    btc = next(p for p in result.equity[-1].positions if p.market == M)
    assert btc.actual_weight > btc.target_weight
    cash = run(data, buy_and_hold(data, cfg, ()), cfg)
    assert cash.equity[-1].equity == D(1000) and not cash.fills


def test_fees_slippage_higher_costs_and_buy_scaling() -> None:
    h = history((100.0, 100.0, 100.0, 100.0))
    data = PreparedData((h,))
    targets = (target(h, 0, 1), target(h, 1, 0))
    low = run(data, targets, free_settings())
    high = run(data, targets, settings(fee_bps=D(100), slippage_bps=D(100)))
    assert high.fills[0].execution_price == D(101)
    assert high.fills[1].execution_price == D(99)
    assert D("9.802960494069208") < high.fills[0].quantity < D("9.802960494069210")
    assert D("960.788157") < high.equity[-1].equity < D("960.788159")
    assert high.equity[-1].equity < low.equity[-1].equity
    assert high.equity[-1].gross_equity == low.equity[-1].equity == D(1000)
    assert high.equity[-1].fees_paid == sum((entry.fee for entry in high.ledger), D(0))
    assert high.equity[-1].slippage_paid == sum((f.slippage for f in high.fills), D(0))
    assert all(p.cash >= 0 for p in high.equity)


def test_repeated_no_trade_and_ambient_decimal_context() -> None:
    h = history((100.0, 100.0, 100.0, 100.0))
    data = PreparedData((h,))
    targets = (target(h, 0, 1), target(h, 1, 1), target(h, 2, 1))
    expected = run(data, targets, free_settings())
    assert len(expected.orders) == 1
    assert [o.status for o in expected.target_outcomes] == ["filled", "no_trade", "no_trade"]
    with localcontext() as context:
        context.prec = 6
        context.rounding = ROUND_UP
        assert run(data, targets, free_settings()).canonical_json() == expected.canonical_json()


def test_shared_cash_rebalance_with_high_fees() -> None:
    eth = Market(exchange="fixture", pair="ETH/USDT", timeframe="1d")
    h, e = history(), history(market=eth)
    data = PreparedData((h, e))
    targets = tuple(
        PortfolioTarget(instructions=(instruction(h, n, weight), instruction(e, n, 1 - weight)))
        for n, weight in enumerate((0.5, 0.9, 0.1))
    )
    result = run(data, targets, settings(fee_bps=D(5000), slippage_bps=D(100)))
    assert all(o.status == "filled" for o in result.orders)
    assert all(p.cash >= 0 for p in result.equity)
    for point in result.equity:
        assert point.equity - D(1000) == point.realized_pnl + point.unrealized_pnl


def test_stale_prices_fail_explicitly() -> None:
    h = history((100.0, 100.0))
    with pytest.raises(ValueError, match="stale valuation"):
        run(PreparedData((h,)), (target(h, 0, 1),), free_settings())


def test_later_publication_cannot_replace_newer_open_price() -> None:
    h = history((100.0, 200.0, 300.0, 400.0), delay=1)
    result = run(PreparedData((h,)), (target(h, 0, 1),), free_settings())
    point = next(p for p in result.equity if p.timestamp == T + 2 * DAY + timedelta(hours=1))
    assert point.positions[0].price == D(300)


@pytest.mark.parametrize("weight", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_weights(weight: float) -> None:
    with pytest.raises(ValidationError):
        target(history(), 0, weight)


def test_leverage_duplicates_and_quote_validation() -> None:
    eth = Market(exchange="fixture", pair="ETH/USDT", timeframe="1d")
    h, e = history(), history(market=eth)
    for legs in (
        (instruction(h, 0, 0.6), instruction(e, 0, 0.6)),
        (instruction(h, 0, 0.5), instruction(h, 0, 0.5)),
    ):
        with pytest.raises(ValidationError):
            PortfolioTarget(instructions=legs)
    with pytest.raises(ValueError, match="quote currency"):
        run(PreparedData((h,)), (), free_settings(quote_currency="USD"))


def test_strategy_replay_adapter_uses_existing_interface() -> None:
    h = history()

    class Dummy:
        config = StrategyConfig(strategy_id="dummy", version="1", market=M)

        def generate(self, history: MarketData) -> SignalBatch:
            assert all(c.available_at <= history.as_of for c in history.candles)
            last = history.candles[-1]
            signal = TargetExposure(
                strategy_id="dummy",
                market=M,
                data_through=last.close_time,
                generated_at=history.as_of,
                effective_at=history.as_of,
                weight=0.5,
            )
            return SignalBatch(config=self.config, history=history, signals=(signal,))

    snapshots = (
        MarketData(market=M, as_of=T + DAY, candles=h.candles[:1]),
        MarketData(market=M, as_of=T + 2 * DAY, candles=h.candles[:2]),
    )
    targets = replay_targets(Dummy(), iter(snapshots))
    result = run(PreparedData((h,)), targets, free_settings())
    assert result.fills and targets[0].instructions[0].config.strategy_id == "dummy"
    batch = Dummy().generate(snapshots[0])
    assert targets_from_batches((batch,)) == targets[:1]


def test_shadow_simulation_is_independent() -> None:
    h = history()
    data = PreparedData((h,))
    targets = (target(h, 0, 1),)
    actual = run(data, targets, free_settings())
    shadow = run(data, targets, free_settings(shadow=True))
    assert shadow.settings.shadow and shadow.equity == actual.equity
    assert actual.run_id != shadow.run_id


def test_weighted_average_basis_and_partial_sale_exact() -> None:
    account = Account(settings())
    trades: tuple[tuple[str, Side, int, int], ...] = (
        ("a", "buy", 2, 100),
        ("b", "buy", 2, 200),
        ("c", "sell", 1, 300),
    )
    for oid, side, qty, price in trades:
        account.execute(oid, "target", M, T, T, T, side, D(qty), D(price))
    assert account.positions[M].quantity == D(3)
    assert account.positions[M].basis == D("454.5")
    assert account.cash == D(691) and account.realized_pnl == D("145.5")
    account.execute("d", "target", M, T, T, T, "sell", D(3), D(250))
    assert account.cash == D("1433.5") and account.realized_pnl == D("433.5")
    assert account.positions[M].basis == 0


@pytest.mark.parametrize("qty", [D(0), D("1.0000000000000000001")])
def test_precision_rejection_has_no_cash_effect(qty: D) -> None:
    account = Account(free_settings())
    order, fill, ledger = account.execute("a", "t", M, T, T, T, "buy", qty, D(100))
    assert order.status == "rejected" and "precision" in (order.reason or "")
    assert fill is None and ledger is None and account.cash == D(1000)


def test_overselling_and_order_id_reuse() -> None:
    account = Account(free_settings())
    order, fill, ledger = account.execute("a", "t", M, T, T, T, "sell", D(1), D(100))
    assert order.status == "rejected" and "quantity" in (order.reason or "")
    assert account.cash == D(1000) and not account.positions and fill is None and ledger is None
    with pytest.raises(ValueError, match="duplicate order"):
        account.execute("a", "t", M, T, T, T, "buy", D(1), D(100))


@pytest.mark.parametrize(
    "qty,price",
    [(D(-1), D(100)), (D("NaN"), D(100)), (D(1), D(0)), (D(1), D("Infinity")), (D("1e19"), D(1))],
)
def test_nonfinite_negative_or_unsupported_precision(qty: D, price: D) -> None:
    account = Account(free_settings())
    with pytest.raises(ValueError, match="precision bounds"):
        account.execute("a", "t", M, T, T, T, "buy", qty, price)
    assert account.cash == D(1000) and not account.positions


def test_order_timing_quote_and_minimum_capital() -> None:
    account = Account(free_settings())
    with pytest.raises(ValueError, match="precedes"):
        account.execute("a", "t", M, T, T + DAY, T, "buy", D(1), D(100))
    with pytest.raises(ValueError, match="timezone-aware"):
        account.execute("a", "t", M, T.replace(tzinfo=None), T, T, "buy", D(1), D(100))
    other = Market(exchange="fixture", pair="BTC/USD", timeframe="1d")
    with pytest.raises(ValueError, match="quote currency"):
        account.execute("a", "t", other, T, T, T, "buy", D(1), D(100))
    with pytest.raises(ValueError, match="below quote precision"):
        Account(free_settings(starting_capital=D("1e-20")))


@pytest.mark.parametrize(
    "field,value",
    [
        ("fee_bps", D(-1)),
        ("fee_bps", D(10000)),
        ("slippage_bps", D("NaN")),
        ("starting_capital", D(0)),
        ("max_price_age_seconds", 0),
    ],
)
def test_invalid_simulation_settings(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        settings(**{field: value})
    with pytest.raises(ValidationError):
        SimulationSettings.model_validate(settings().model_dump(exclude={"fee_bps"}))


def test_multi_market_delayed_availability_and_input_isolation() -> None:
    eth = Market(exchange="fixture", pair="ETH/USDT", timeframe="1d")
    h, e = history(), history(market=eth, delay=1)
    data = PreparedData((h, e))
    invalid = PortfolioTarget(
        instructions=(instruction(h, 0, 0.5), instruction(e, 0, 0.5, decision=T + DAY))
    )
    with pytest.raises(ValueError, match="unavailable"):
        run(data, (invalid,), free_settings())
    decision = T + DAY + timedelta(hours=1)
    valid = PortfolioTarget(
        instructions=(instruction(h, 0, 0.5, decision=decision), instruction(e, 0, 0.5))
    )
    before = valid.canonical_json()
    result = run(data, (valid,), free_settings())
    assert all(f.timestamp == T + 2 * DAY for f in result.fills)
    assert valid.canonical_json() == before and result.decisions == (valid,)
    with pytest.raises(ValidationError):
        PortfolioTarget.model_validate(valid.model_dump() | {"future_return": 0.5})


def test_prepared_data_validation_and_missing_source() -> None:
    h = history()
    with pytest.raises(ValueError, match="at least one"):
        PreparedData(())
    with pytest.raises(ValueError, match="one timeframe"):
        PreparedData((h, h))
    source = instruction(h, 0, 1)
    unknown = Instruction(
        config=source.config,
        signal=TargetExposure.model_validate(source.signal.model_dump() | {"data_through": T}),
    )
    with pytest.raises(ValueError, match="unknown"):
        run(PreparedData((h,)), (PortfolioTarget(instructions=(unknown,)),), free_settings())
    with pytest.raises(ValueError, match="outside simulation"):
        run(
            PreparedData((h,)),
            (target(h, 0, 1),),
            free_settings(period=Period(start=T + 2 * DAY, end=T + 4 * DAY)),
        )


def test_passive_benchmark_validation_and_missing_history() -> None:
    h = history()
    data = PreparedData((h,))
    for weights in (((M, D(-1)),), ((M, D("NaN")),), ((M, D(1)), (M, D(0)))):
        with pytest.raises(ValueError, match="benchmark weights"):
            buy_and_hold(data, free_settings(), weights)
    late = history(delay=100)
    with pytest.raises(ValueError, match="no completed benchmark"):
        buy_and_hold(PreparedData((late,)), free_settings(), ((M, D(1)),))


def test_demo_offline(capsys: pytest.CaptureFixture[str]) -> None:
    from crypto_research.backtesting.cli import main

    assert main(["demo"]) == 0
    text = capsys.readouterr().out
    assert "Final equity: 1198.202098" in text and "Executed trades: 1" in text


def test_unpublished_final_close_and_source_vintage() -> None:
    h = history(delay=1)
    changed = Candle.model_validate(h.candles[-1].model_dump() | {"close": 1000.0, "high": 1000.0})
    h = MarketData(market=M, as_of=h.as_of, candles=h.candles[:-1] + (changed,))
    data = PreparedData((h,))
    result = run(data, (target(h, 0, 1),), free_settings())
    # Delayed source forces entry at 300. The final 1000 close is still unpublished.
    assert result.equity[-1].positions[0].price == D(400)
    assert result.equity[-1].equity < D(1400)
    original = instruction(h, 0, 1)
    mismatch = Instruction(
        config=original.config,
        signal=original.signal,
        source_dataset=DatasetReference(source="other", version="1", sha256="b" * 64),
    )
    with pytest.raises(ValueError, match="provenance"):
        run(data, (PortfolioTarget(instructions=(mismatch,)),), free_settings())


def test_target_quantity_bound_fails_before_execution() -> None:
    h = history((1e-18, 1e-18, 1e-18, 1e-18))
    with pytest.raises(ValueError, match="target quantity"):
        run(PreparedData((h,)), (target(h, 0, 1),), free_settings())
