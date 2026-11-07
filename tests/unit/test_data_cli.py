import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from test_market_data import FakeProvider, raw

from crypto_research.data import cli
from crypto_research.data.provider import ProviderError
from crypto_research.data.storage import Store


def test_dates() -> None:
    assert cli.date("2024-01-01") == datetime(2024, 1, 1, tzinfo=UTC)
    assert cli.date("2024-01-01T02:00:00+02:00") == cli.date("2024-01-01")
    for text in ("wrong", "2024-01-01T01:00:00", "2024-02-30"):
        with pytest.raises(argparse.ArgumentTypeError):
            cli.date(text)


def test_download_inspect_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    provider = FakeProvider([[raw(0)]])
    monkeypatch.setattr(cli, "BinanceProvider", lambda: provider)
    base = ["--root", str(tmp_path)]
    assert (
        cli.main(
            base
            + [
                "download",
                "--market",
                "BTC/USDT",
                "--timeframe",
                "1d",
                "--start",
                "2024-01-01",
                "--end",
                "2024-01-02",
            ]
        )
        == 0
    )
    output = capsys.readouterr()
    manifest = json.loads(output.out)
    version = manifest["version"]
    assert version in output.err
    for command in ("validate", "coverage", "quality", "list"):
        assert cli.main(base + [command] + ([] if command == "list" else [version])) == 0
        assert json.loads(capsys.readouterr().out)
    provider = FakeProvider([[raw(0), raw(1)]])
    monkeypatch.setattr(cli, "BinanceProvider", lambda: provider)
    assert cli.main(base + ["update", version, "--end", "2024-01-03"]) == 0
    updated = json.loads(capsys.readouterr().out)
    assert updated["change"] == "updated" and updated["parent_version"] == version


def test_cli_multiple_markets_timeframes_and_longest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    provider = FakeProvider(
        [[raw(0)], [raw(0)], [], [raw(0)], [raw(0)], [], [raw(0)], [raw(0)], []]
    )
    monkeypatch.setattr(cli, "BinanceProvider", lambda: provider)
    base = ["--root", str(tmp_path)]
    assert (
        cli.main(
            base
            + [
                "download",
                "--market",
                "BTC/USDT",
                "ETH/USDT",
                "SOL/USDT",
                "--timeframe",
                "1d",
                "4h",
                "--start",
                "2024-01-01",
                "--end",
                "2024-01-02",
            ]
        )
        == 0
    )
    results = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(results) == 6
    assert {m["request"]["market"]["pair"] for m in results} == {"BTC/USDT", "ETH/USDT", "SOL/USDT"}
    provider = FakeProvider([[raw(0)], []])
    monkeypatch.setattr(cli, "BinanceProvider", lambda: provider)
    assert (
        cli.main(
            base
            + [
                "download",
                "--market",
                "BTC/USDT",
                "--timeframe",
                "1d",
                "--longest",
                "--end",
                "2024-01-02",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert Store(tmp_path).read(result["version"])[2].status == "incomplete"
    assert provider.calls[0] == 0
    assert cli.main(base + ["validate", result["version"]]) == 2


@pytest.mark.parametrize(
    "failure,status", [(ProviderError("unavailable"), 2), (KeyboardInterrupt(), 130)]
)
def test_cli_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure: BaseException,
    status: int,
) -> None:
    monkeypatch.setattr(cli, "BinanceProvider", lambda: FakeProvider([failure]))
    assert (
        cli.main(
            [
                "--root",
                str(tmp_path),
                "download",
                "--market",
                "BTC/USDT",
                "--timeframe",
                "1d",
                "--start",
                "2024-01-01",
                "--end",
                "2024-01-02",
            ]
        )
        == status
    )
    assert capsys.readouterr().err


def test_cli_no_data_and_missing_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "BinanceProvider", lambda: FakeProvider([[]]))
    base = ["--root", str(tmp_path)]
    assert (
        cli.main(
            base
            + [
                "download",
                "--market",
                "BTC/USDT",
                "--timeframe",
                "1d",
                "--start",
                "2024-01-01",
                "--end",
                "2024-01-02",
            ]
        )
        == 2
    )
    assert "missing_intervals" in capsys.readouterr().err
    assert cli.main(base + ["quality", "0" * 64]) == 2
    assert "error:" in capsys.readouterr().err
