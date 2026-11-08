"""Immutable decision, execution and outcome records for offline simulation."""

from datetime import datetime
from decimal import Context, Decimal, localcontext
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from crypto_research.contracts import (
    DatasetReference,
    Identifier,
    Market,
    Period,
    Record,
    StrategyConfig,
    TargetExposure,
)

Amount = Annotated[Decimal, Field(allow_inf_nan=False)]
Nonnegative = Annotated[Decimal, Field(ge=0, allow_inf_nan=False)]
Weight = Annotated[Decimal, Field(ge=0, le=1, allow_inf_nan=False)]
Side = Literal["buy", "sell"]


class SimulationSettings(Record):
    experiment_id: Identifier
    period: Period
    starting_capital: Annotated[Decimal, Field(gt=0, le=Decimal("1e18"), allow_inf_nan=False)]
    quote_currency: Identifier = "USDT"
    fee_bps: Annotated[Decimal, Field(ge=0, lt=10000, allow_inf_nan=False)]
    slippage_bps: Annotated[Decimal, Field(ge=0, lt=10000, allow_inf_nan=False)]
    max_price_age_seconds: Annotated[int, Field(gt=0, strict=True)] = 86400
    max_execution_delay_seconds: Annotated[int, Field(ge=0, strict=True)] = 86400
    shadow: bool = False


class Instruction(Record):
    config: StrategyConfig
    signal: TargetExposure
    source_dataset: DatasetReference | None = None

    @model_validator(mode="after")
    def identity_matches(self) -> Self:
        if (
            self.config.strategy_id != self.signal.strategy_id
            or self.config.market != self.signal.market
        ):
            raise ValueError("signal identity must match strategy configuration")
        return self


class PortfolioTarget(Record):
    """Full portfolio instruction. Omitted instruments target zero; remainder is cash."""

    instructions: tuple[Instruction, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def valid_allocations(self) -> Self:
        markets = [i.signal.market for i in self.instructions]
        times = {(i.signal.generated_at, i.signal.effective_at) for i in self.instructions}
        if len(markets) != len(set(markets)) or len(times) != 1:
            raise ValueError(
                "target requires unique markets and common decision/effective timestamps"
            )
        with localcontext(Context(prec=60)):
            if sum((Decimal(str(i.signal.weight)) for i in self.instructions), Decimal(0)) > 1:
                raise ValueError("total asset allocation cannot exceed one")
        object.__setattr__(
            self,
            "instructions",
            tuple(sorted(self.instructions, key=lambda i: i.signal.market.identity)),
        )
        return self

    @property
    def decision_at(self) -> datetime:
        return self.instructions[0].signal.generated_at

    @property
    def effective_at(self) -> datetime:
        return self.instructions[0].signal.effective_at


class Order(Record):
    order_id: Identifier
    target_id: Identifier
    market: Market
    decision_at: datetime
    earliest_at: datetime
    executed_at: datetime | None
    side: Side
    requested_quantity: Nonnegative
    filled_quantity: Nonnegative
    reference_price: Nonnegative | None
    execution_price: Nonnegative | None
    notional: Nonnegative
    fee: Nonnegative
    slippage: Nonnegative
    status: Literal["filled", "rejected"]
    lifecycle: tuple[Literal["proposed", "accepted", "filled", "rejected"], ...]
    reason: str | None = None


class Fill(Record):
    fill_id: Identifier
    order_id: Identifier
    target_id: Identifier
    market: Market
    timestamp: datetime
    side: Side
    quantity: Nonnegative
    reference_price: Nonnegative
    execution_price: Nonnegative
    notional: Nonnegative
    fee: Nonnegative
    slippage: Nonnegative


class LedgerEntry(Record):
    order_id: Identifier
    timestamp: datetime
    market: Market
    cash_change: Amount
    quantity_change: Amount
    cash_after: Nonnegative
    quantity_after: Nonnegative
    basis_after: Nonnegative
    realized_pnl: Amount
    fee: Nonnegative


class PositionValue(Record):
    market: Market
    quantity: Nonnegative
    price: Nonnegative
    price_at: datetime
    value: Nonnegative
    cost_basis: Nonnegative
    realized_pnl: Amount
    unrealized_pnl: Amount
    target_weight: Weight
    actual_weight: Weight


class EquityPoint(Record):
    timestamp: datetime
    pre_execution_equity: Nonnegative | None = None
    pre_execution_gross_equity: Nonnegative | None = None
    cash: Nonnegative
    holdings_value: Nonnegative
    equity: Nonnegative
    period_return: Amount | None
    cumulative_return: Amount
    gross_equity: Nonnegative
    gross_period_return: Amount | None
    gross_cumulative_return: Amount
    fees_paid: Nonnegative
    slippage_paid: Nonnegative
    turnover: Nonnegative
    realized_pnl: Amount
    unrealized_pnl: Amount
    positions: tuple[PositionValue, ...]


class TargetOutcome(Record):
    target_id: Identifier
    decision_at: datetime
    effective_at: datetime
    executed_at: datetime | None
    status: Literal["filled", "no_trade", "rejected"]
    reason: str | None = None


class DataGap(Record):
    market: Market
    period: Period


class BacktestResult(Record):
    schema_version: Literal[1] = 1
    run_id: Identifier
    settings: SimulationSettings
    data_identities: tuple[str, ...]
    data_gaps: tuple[DataGap, ...] = ()
    decisions: tuple[PortfolioTarget, ...]
    target_outcomes: tuple[TargetOutcome, ...]
    orders: tuple[Order, ...]
    fills: tuple[Fill, ...]
    ledger: tuple[LedgerEntry, ...]
    equity: tuple[EquityPoint, ...]
