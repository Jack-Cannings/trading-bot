"""Offline chronological simulation. Strategy inputs never receive the broker tape."""

import hashlib
from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime, timedelta
from decimal import Context, Decimal, localcontext

from crypto_research.backtesting.accounting import (
    PRECISION,
    UNIT,
    ZERO,
    Account,
    Position,
    amount,
    quantity,
)
from crypto_research.backtesting.models import (
    BacktestResult,
    EquityPoint,
    Fill,
    Instruction,
    LedgerEntry,
    Order,
    PortfolioTarget,
    PositionValue,
    Side,
    SimulationSettings,
    TargetOutcome,
)
from crypto_research.contracts import Candle, Market, MarketData, SignalBatch
from crypto_research.data.models import aligned
from crypto_research.strategies.base import Strategy, reconstruct


class PreparedData:
    """Reusable validated in-memory tape; indexes are built once, not per strategy."""

    def __init__(self, histories: tuple[MarketData, ...]) -> None:
        if not histories:
            raise ValueError("at least one market history is required")
        self.histories = tuple(
            sorted(
                (MarketData.model_validate(h.model_dump()) for h in histories),
                key=lambda h: h.market.identity,
            )
        )
        self.markets = tuple(h.market for h in self.histories)
        instruments = {(m.exchange, m.pair) for m in self.markets}
        if len(instruments) != len(self.markets) or len({m.exchange for m in self.markets}) != 1:
            raise ValueError("use one exchange and one timeframe per instrument")
        self.opens: dict[datetime, list[Candle]] = defaultdict(list)
        self.publications: dict[datetime, list[Candle]] = defaultdict(list)
        self.sources: dict[tuple[Market, datetime], Candle] = {}
        common: set[datetime] | None = None
        for history in self.histories:
            openings = set()
            for candle in history.candles:
                if not aligned(candle.open_time, candle.market):
                    raise ValueError("candle is not aligned to UTC timeframe")
                for price in (candle.open, candle.close):
                    if not Decimal("1e-18") <= Decimal(str(price)) <= Decimal("1e18"):
                        raise ValueError("price is outside supported precision bounds")
                self.opens[candle.open_time].append(candle)
                self.publications[candle.available_at].append(candle)
                self.sources[(candle.market, candle.close_time)] = candle
                openings.add(candle.open_time)
            common = openings if common is None else common & openings
        self.common_opens = tuple(sorted(common or ()))
        self.timeline = tuple(sorted(self.opens.keys() | self.publications.keys()))
        self.identities = tuple(h.identity for h in self.histories)


def targets_from_batches(batches: Iterable[SignalBatch]) -> tuple[PortfolioTarget, ...]:
    grouped: dict[tuple[datetime, datetime], list[Instruction]] = defaultdict(list)
    for batch in batches:
        references = {c.close_time: c.dataset for c in batch.history.candles}
        for signal in batch.signals:
            grouped[(signal.generated_at, signal.effective_at)].append(
                Instruction(
                    config=batch.config,
                    signal=signal,
                    source_dataset=references[signal.data_through],
                )
            )
    return tuple(PortfolioTarget(instructions=tuple(legs)) for _, legs in sorted(grouped.items()))


def replay_targets(
    strategy: Strategy, snapshots: Iterable[MarketData]
) -> tuple[PortfolioTarget, ...]:
    return targets_from_batches(reconstruct(strategy, snapshots))


def _weights(target: PortfolioTarget) -> dict[Market, Decimal]:
    return {i.signal.market: Decimal(str(i.signal.weight)) for i in target.instructions}


def _schedule(
    data: PreparedData, targets: tuple[PortfolioTarget, ...], settings: SimulationSettings
) -> tuple[dict[datetime, PortfolioTarget], set[str]]:
    scheduled: dict[datetime, PortfolioTarget] = {}
    rejected: set[str] = set()
    previous: PortfolioTarget | None = None
    for target in targets:
        if previous and (
            target.decision_at <= previous.decision_at
            or target.effective_at < previous.effective_at
        ):
            raise ValueError(
                "targets must have unique, increasing decisions and ordered effective times"
            )
        previous = target
        if not settings.period.start <= target.decision_at < settings.period.end:
            raise ValueError("decision timestamp outside simulation period")
        for instruction in target.instructions:
            signal = instruction.signal
            candle = data.sources.get((signal.market, signal.data_through))
            if candle is None or candle.available_at > signal.generated_at:
                raise ValueError("signal references unknown or unavailable source data")
            if (
                instruction.source_dataset is not None
                and instruction.source_dataset != candle.dataset
            ):
                raise ValueError("signal source dataset differs from execution tape provenance")
        index = bisect_left(data.common_opens, target.effective_at)
        if index == len(data.common_opens):
            rejected.add(target.identity)
            continue
        execute_at = data.common_opens[index]
        if execute_at >= settings.period.end or execute_at - target.effective_at > timedelta(
            seconds=settings.max_execution_delay_seconds
        ):
            rejected.add(target.identity)
            continue
        if execute_at in scheduled:
            raise ValueError("contradictory targets would execute at the same candle open")
        scheduled[execute_at] = target
    return scheduled, rejected


def _holdings(
    account: Account, marks: dict[Market, tuple[Decimal, datetime]], timestamp: datetime
) -> Decimal:
    total = ZERO
    for market, position in account.positions.items():
        if position.quantity == 0:
            continue
        if market not in marks or timestamp - marks[market][1] > timedelta(
            seconds=account.settings.max_price_age_seconds
        ):
            raise ValueError(
                f"missing or stale valuation price for {market.pair} at {timestamp.isoformat()}"
            )
        price, _ = marks[market]
        total += amount(position.quantity * price)
    return total


def _rebalance(
    account: Account,
    target: PortfolioTarget,
    markets: tuple[Market, ...],
    marks: dict[Market, tuple[Decimal, datetime]],
    timestamp: datetime,
) -> tuple[list[Order], list[Fill], list[LedgerEntry]]:
    equity = account.cash + _holdings(account, marks, timestamp)
    weights = _weights(target)
    desired: dict[Market, Decimal] = {}
    for market in markets:
        raw_quantity = weights.get(market, ZERO) * equity / marks[market][0]
        if raw_quantity > Decimal("1e18"):
            raise ValueError("target quantity is outside supported precision bounds")
        desired[market] = quantity(raw_quantity)
    orders: list[Order] = []
    fills: list[Fill] = []
    ledger: list[LedgerEntry] = []

    def submit(market: Market, side: Side, qty: Decimal) -> None:
        result = account.execute(
            f"{target.identity}.{market.identity}.{side}",
            target.identity,
            market,
            target.decision_at,
            target.effective_at,
            timestamp,
            side,
            qty,
            marks[market][0],
        )
        order, fill, entry = result
        orders.append(order)
        if fill and entry:
            fills.append(fill)
            ledger.append(entry)

    for market in markets:
        held = account.positions.get(market, Position()).quantity
        if held > desired[market]:
            submit(market, "sell", held - desired[market])
    buys = {
        m: desired[m] - account.positions.get(m, Position()).quantity
        for m in markets
        if desired[m] > account.positions.get(m, Position()).quantity
    }
    required = sum(
        (sum(account.quote("buy", q, marks[m][0])[1:3], ZERO) for m, q in buys.items()), ZERO
    )
    scale = (
        (
            Decimal(1)
            if required <= account.cash
            else max(ZERO, account.cash - 2 * len(buys) * UNIT) / required
        )
        if required
        else ZERO
    )
    for market, qty in buys.items():
        sized = quantity(qty * scale)
        if sized:
            submit(market, "buy", sized)
    return orders, fills, ledger


def run(
    data: PreparedData, targets: tuple[PortfolioTarget, ...], settings: SimulationSettings
) -> BacktestResult:
    """Simulate a shared spot portfolio, never calling an exchange or strategy implicitly."""
    with localcontext(Context(prec=PRECISION)):
        return _run(data, targets, settings)


def _run(
    data: PreparedData, targets: tuple[PortfolioTarget, ...], settings: SimulationSettings
) -> BacktestResult:
    if any(m.pair.split("/")[1] != settings.quote_currency for m in data.markets):
        raise ValueError("all markets must use the portfolio quote currency")
    scheduled, rejected = _schedule(data, targets, settings)
    decisions = {t.decision_at: t for t in targets}
    account = Account(settings)
    gross = Account(
        SimulationSettings.model_validate(
            settings.model_dump() | {"fee_bps": ZERO, "slippage_bps": ZERO}
        )
    )
    marks: dict[Market, tuple[Decimal, datetime]] = {}
    # Warmup quotes are allowed only when their event occurred before the start.
    start_index = bisect_left(data.timeline, settings.period.start)
    end_index = bisect_right(data.timeline, settings.period.end)
    times = sorted(
        set(data.timeline[start_index:end_index])
        | set(decisions)
        | {settings.period.start, settings.period.end}
    )
    for history in data.histories:
        # One warmup pass per market; no repeated history scans within the event loop.
        candidates = [
            (c.open_time, Decimal(str(c.open)))
            for c in history.candles
            if c.open_time < settings.period.start
        ]
        candidates.extend(
            (c.close_time, Decimal(str(c.close)))
            for c in history.candles
            if c.available_at < settings.period.start
        )
        if candidates:
            price_at, price = max(candidates, key=lambda item: item[0])
            marks[history.market] = price, price_at
    orders: list[Order] = []
    fills: list[Fill] = []
    ledger: list[LedgerEntry] = []
    outcomes: list[TargetOutcome] = []
    points: list[EquityPoint] = []
    weights: dict[Market, Decimal] = {}
    prior_net = account.cash
    prior_gross = gross.cash
    initial = account.cash
    for timestamp in times:
        for candle in data.publications.get(timestamp, ()):
            previous = marks.get(candle.market)
            if previous is None or candle.close_time > previous[1]:
                marks[candle.market] = Decimal(str(candle.close)), candle.close_time
        if timestamp < settings.period.end:
            for candle in data.opens.get(timestamp, ()):
                marks[candle.market] = Decimal(str(candle.open)), candle.open_time
        if timestamp in decisions:
            target = decisions[timestamp]
            weights = _weights(target)
            if target.identity in rejected:
                outcomes.append(
                    TargetOutcome(
                        target_id=target.identity,
                        decision_at=target.decision_at,
                        effective_at=target.effective_at,
                        executed_at=None,
                        status="rejected",
                        reason=(
                            "no synchronized candle open within execution delay "
                            "and simulation period"
                        ),
                    )
                )
        if timestamp in scheduled:
            target = scheduled[timestamp]
            new_orders, new_fills, new_ledger = _rebalance(
                account, target, data.markets, marks, timestamp
            )
            _rebalance(gross, target, data.markets, marks, timestamp)
            orders.extend(new_orders)
            fills.extend(new_fills)
            ledger.extend(new_ledger)
            outcomes.append(
                TargetOutcome(
                    target_id=target.identity,
                    decision_at=target.decision_at,
                    effective_at=target.effective_at,
                    executed_at=timestamp,
                    status="rejected"
                    if any(o.status == "rejected" for o in new_orders)
                    else "filled"
                    if new_fills
                    else "no_trade",
                    reason="one or more orders rejected"
                    if any(o.status == "rejected" for o in new_orders)
                    else None,
                )
            )
        holdings = _holdings(account, marks, timestamp)
        equity = account.cash + holdings
        gross_equity = gross.cash + _holdings(gross, marks, timestamp)
        positions = tuple(
            PositionValue(
                market=m,
                quantity=account.positions.get(m, Position()).quantity,
                price=marks[m][0],
                price_at=marks[m][1],
                value=amount(account.positions.get(m, Position()).quantity * marks[m][0]),
                cost_basis=account.positions.get(m, Position()).basis,
                realized_pnl=account.positions.get(m, Position()).realized_pnl,
                unrealized_pnl=amount(account.positions.get(m, Position()).quantity * marks[m][0])
                - account.positions.get(m, Position()).basis,
                target_weight=weights.get(m, ZERO),
                actual_weight=amount(account.positions.get(m, Position()).quantity * marks[m][0])
                / equity
                if equity
                else ZERO,
            )
            for m in data.markets
            if m in marks
        )
        unrealized = sum((p.unrealized_pnl for p in positions), ZERO)
        if equity - initial != account.realized_pnl + unrealized:
            raise ArithmeticError("cash, positions and profit/loss do not reconcile")
        points.append(
            EquityPoint(
                timestamp=timestamp,
                cash=account.cash,
                holdings_value=holdings,
                equity=equity,
                period_return=equity / prior_net - 1 if prior_net else None,
                cumulative_return=equity / initial - 1,
                gross_equity=gross_equity,
                gross_period_return=gross_equity / prior_gross - 1 if prior_gross else None,
                gross_cumulative_return=gross_equity / initial - 1,
                fees_paid=account.fees,
                slippage_paid=account.slippage,
                turnover=account.turnover,
                realized_pnl=account.realized_pnl,
                unrealized_pnl=unrealized,
                positions=positions,
            )
        )
        prior_net, prior_gross = equity, gross_equity
    run_id = hashlib.sha256(
        (
            settings.identity + "".join(data.identities) + "".join(t.identity for t in targets)
        ).encode()
    ).hexdigest()
    return BacktestResult(
        run_id=run_id,
        settings=settings,
        data_identities=data.identities,
        decisions=targets,
        target_outcomes=tuple(sorted(outcomes, key=lambda o: o.decision_at)),
        orders=tuple(orders),
        fills=tuple(fills),
        ledger=tuple(ledger),
        equity=tuple(points),
    )
