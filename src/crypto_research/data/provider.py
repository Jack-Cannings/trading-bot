"""Public spot market-data access only. Construction never contacts an exchange."""

import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol, cast

import ccxt

from crypto_research.contracts import Market


class ProviderError(RuntimeError):
    """Unavailable, unsupported or malformed exchange response."""


class TransientProviderError(ProviderError):
    """Retryable connectivity, rate-limit or temporary service error."""


class Provider(Protocol):
    exchange: str
    identity: str

    def capabilities(self) -> tuple[tuple[str, ...], tuple[str, ...]]: ...
    def server_time(self) -> datetime: ...
    def fetch(self, market: Market, since: int, limit: int, end: int) -> list[list[object]]: ...


class ExchangeClient(Protocol):
    timeframes: dict[str, str]

    def load_markets(self) -> dict[str, dict[str, object]]: ...
    def fetch_time(self) -> int: ...
    def fetch_ohlcv(
        self, symbol: str, timeframe: str, since: int, limit: int, params: dict[str, object]
    ) -> list[list[object]]: ...


def from_milliseconds(value: object) -> datetime:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value % 1 != 0:
        raise ValueError("exchange timestamps must be integer milliseconds")
    return datetime.fromtimestamp(value / 1000, UTC)


def retry[T](
    call: Callable[[], T], retries: int, backoff: float, sleep: Callable[[float], None] = time.sleep
) -> T:
    for attempt in range(retries + 1):
        try:
            return call()
        except TransientProviderError:
            if attempt == retries:
                raise
            sleep(min(60.0, backoff * 2**attempt))
    raise AssertionError("unreachable")


class BinanceProvider:
    exchange = "binance"
    identity = f"ccxt-{ccxt.__version__}-binance-public-spot"

    def __init__(self, client: ExchangeClient | None = None) -> None:
        self.client = client or cast(
            ExchangeClient,
            ccxt.binance(
                {
                    "enableRateLimit": True,
                    "timeout": 30000,
                    "options": {"defaultType": "spot", "fetchMarkets": {"types": ["spot"]}},
                }
            ),
        )

    def _call[T](self, call: Callable[[], T]) -> T:
        try:
            return call()
        except ccxt.NetworkError as exc:
            raise TransientProviderError(
                f"Binance public endpoint temporarily unavailable: {exc}"
            ) from exc
        except ccxt.BaseError as exc:
            raise ProviderError(
                f"Binance public endpoint failed; no exchange substitution: {exc}"
            ) from exc

    def capabilities(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        markets = self._call(self.client.load_markets)
        pairs = tuple(
            p
            for p in ("BTC/USDT", "ETH/USDT", "SOL/USDT")
            if p in markets and markets[p].get("spot") is True
        )
        return pairs, tuple(t for t in ("1d", "4h") if t in self.client.timeframes)

    def server_time(self) -> datetime:
        try:
            return from_milliseconds(self._call(self.client.fetch_time))
        except (ValueError, OverflowError, OSError) as exc:
            raise ProviderError("invalid Binance server timestamp") from exc

    def fetch(self, market: Market, since: int, limit: int, end: int) -> list[list[object]]:
        return self._call(
            lambda: self.client.fetch_ohlcv(
                market.pair,
                market.timeframe,
                since,
                limit,
                {"endTime": end - 1},
            )
        )
