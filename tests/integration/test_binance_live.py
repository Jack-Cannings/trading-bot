from pathlib import Path

import pytest

from crypto_research.contracts import Market, Period
from crypto_research.data.cli import date
from crypto_research.data.models import Request
from crypto_research.data.provider import BinanceProvider, ProviderError
from crypto_research.data.service import acquire
from crypto_research.data.storage import Store


@pytest.mark.live
def test_binance_public_smoke(tmp_path: Path, request: pytest.FixtureRequest) -> None:
    if not request.config.getoption("--live-binance"):
        pytest.skip("public Binance access requires --live-binance")
    query = Request(
        market=Market(exchange="binance", pair="BTC/USDT", timeframe="1d"),
        period=Period(start=date("2024-01-01"), end=date("2024-01-04")),
        retries=1,
    )
    store = Store(tmp_path)
    try:
        result = acquire(BinanceProvider(), store, query)
    except ProviderError as exc:
        pytest.skip(f"Binance endpoint inaccessible (separate from offline validation): {exc}")
    assert store.read(result.version)[2].status == "clean"
    print("Binance public endpoint accessible; three completed daily candles validated")
