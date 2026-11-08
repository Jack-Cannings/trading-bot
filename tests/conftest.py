"""Normal tests cannot open network connections."""

import socket

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--live-binance", action="store_true", help="opt in to a public Binance smoke test"
    )


@pytest.fixture(autouse=True)
def offline(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if request.node.get_closest_marker("live") and request.config.getoption("--live-binance"):
        return

    def blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError("network access is prohibited in offline tests")

    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
