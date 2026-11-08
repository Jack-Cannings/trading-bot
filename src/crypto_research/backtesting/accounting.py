"""Weighted-average cost basis. Quote-denominated fees enter basis once."""

from dataclasses import dataclass
from datetime import datetime
from decimal import ROUND_DOWN, ROUND_HALF_EVEN, Context, Decimal, localcontext

from crypto_research.backtesting.models import Fill, LedgerEntry, Order, Side, SimulationSettings
from crypto_research.contracts import Market

UNIT = Decimal("1e-18")
ZERO = Decimal(0)
PRECISION = 60


def amount(value: Decimal) -> Decimal:
    return value.quantize(UNIT, rounding=ROUND_HALF_EVEN)


def quantity(value: Decimal) -> Decimal:
    return value.quantize(UNIT, rounding=ROUND_DOWN)


@dataclass
class Position:
    quantity: Decimal = ZERO
    basis: Decimal = ZERO
    realized_pnl: Decimal = ZERO


class Account:
    def __init__(self, settings: SimulationSettings) -> None:
        self.settings = settings
        with localcontext(Context(prec=PRECISION)):
            self.cash = amount(settings.starting_capital)
        if self.cash <= 0:
            raise ValueError("starting capital is below quote precision")
        self.positions: dict[Market, Position] = {}
        self.fees = ZERO
        self.slippage = ZERO
        self.turnover = ZERO
        self.realized_pnl = ZERO
        self._ids: set[str] = set()

    def quote(
        self, side: Side, qty: Decimal, reference: Decimal
    ) -> tuple[Decimal, Decimal, Decimal, Decimal]:
        with localcontext(Context(prec=PRECISION)):
            slip = self.settings.slippage_bps / 10000
            price = reference * (1 + slip if side == "buy" else 1 - slip)
            notional = amount(qty * price)
            fee = amount(notional * self.settings.fee_bps / 10000)
            slippage = amount(abs(price - reference) * qty)
            return price, notional, fee, slippage

    def execute(
        self,
        order_id: str,
        target_id: str,
        market: Market,
        decision_at: datetime,
        earliest_at: datetime,
        execution_at: datetime,
        side: Side,
        requested: Decimal,
        reference: Decimal,
    ) -> tuple[Order, Fill | None, LedgerEntry | None]:
        """Full fill of a pre-sized order, or a rejection with no account mutation."""
        with localcontext(Context(prec=PRECISION)):
            if side not in {"buy", "sell"}:
                raise ValueError("unsupported order side")
            if any(
                t.tzinfo is None or t.utcoffset() is None
                for t in (decision_at, earliest_at, execution_at)
            ):
                raise ValueError("order timestamps must be timezone-aware")
            if order_id in self._ids:
                raise ValueError("duplicate order identifier")
            if not decision_at <= earliest_at <= execution_at:
                raise ValueError("order execution precedes decision or eligibility")
            if (
                not requested.is_finite()
                or requested < 0
                or requested > Decimal("1e18")
                or not reference.is_finite()
                or not Decimal("1e-18") <= reference <= Decimal("1e18")
            ):
                raise ValueError("quantity/price outside supported finite precision bounds")
            if market.pair.split("/")[1] != self.settings.quote_currency:
                raise ValueError("market quote currency differs from account")
            price, notional, fee, slip = self.quote(side, requested, reference)
            position = self.positions.get(market, Position())
            reason = None
            if requested == 0 or quantity(requested) != requested or notional == 0:
                reason = "quantity below precision or not quantized"
            elif side == "buy" and notional + fee > self.cash:
                reason = "insufficient cash including fees"
            elif side == "sell" and requested > position.quantity:
                reason = "insufficient asset quantity"
            self._ids.add(order_id)
            order = Order(
                order_id=order_id,
                target_id=target_id,
                market=market,
                decision_at=decision_at,
                earliest_at=earliest_at,
                executed_at=None if reason else execution_at,
                side=side,
                requested_quantity=requested,
                filled_quantity=ZERO if reason else requested,
                reference_price=reference,
                execution_price=None if reason else price,
                notional=ZERO if reason else notional,
                fee=ZERO if reason else fee,
                slippage=ZERO if reason else slip,
                status="rejected" if reason else "filled",
                lifecycle=("proposed", "rejected")
                if reason
                else ("proposed", "accepted", "filled"),
                reason=reason,
            )
            if reason:
                return order, None, None
            realized = ZERO
            if side == "buy":
                cash_change = -(notional + fee)
                quantity_change = requested
                position.basis += notional + fee
            else:
                released = (
                    position.basis
                    if requested == position.quantity
                    else amount(position.basis * requested / position.quantity)
                )
                cash_change = notional - fee
                quantity_change = -requested
                realized = cash_change - released
                position.basis -= released
                position.realized_pnl += realized
                self.realized_pnl += realized
            self.cash += cash_change
            position.quantity += quantity_change
            self.positions[market] = position
            self.fees += fee
            self.slippage += slip
            self.turnover += notional
            if self.cash < 0 or position.quantity < 0 or position.basis < 0:
                raise ArithmeticError("accounting invariant violation")
            fill = Fill(
                fill_id=f"{order_id}.fill",
                order_id=order_id,
                target_id=target_id,
                market=market,
                timestamp=execution_at,
                side=side,
                quantity=requested,
                reference_price=reference,
                execution_price=price,
                notional=notional,
                fee=fee,
                slippage=slip,
            )
            ledger = LedgerEntry(
                order_id=order_id,
                timestamp=execution_at,
                market=market,
                cash_change=cash_change,
                quantity_change=quantity_change,
                cash_after=self.cash,
                quantity_after=position.quantity,
                basis_after=position.basis,
                realized_pnl=realized,
                fee=fee,
            )
            return order, fill, ledger
