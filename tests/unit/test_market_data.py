from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import ccxt
import pytest
from pydantic import ValidationError

from crypto_research.contracts import Market, Period
from crypto_research.data.downloader import pages
from crypto_research.data.models import Issue, Manifest, Observation, Request
from crypto_research.data.provider import BinanceProvider, ProviderError, TransientProviderError
from crypto_research.data.service import NoDataError, acquire
from crypto_research.data.storage import IntegrityError, Store, content_hash
from crypto_research.data.validation import validate

T = datetime(2024, 1, 1, tzinfo=UTC)
DAY = timedelta(days=1)
M = Market(exchange="binance", pair="BTC/USDT", timeframe="1d")


def request(days: int = 3) -> Request:
    return Request(market=M, period=Period(start=T, end=T + days * DAY))


def row(day: int, **changes: object) -> Observation:
    values: dict[str, object] = dict(
        open_time=T + day * DAY,
        open=10.0,
        high=12.0,
        low=9.0,
        close=11.0,
        volume=2.0,
        observed_at=T + 10 * DAY,
    )
    return Observation.model_validate(values | changes)


def test_validation_reports_gaps_without_filling() -> None:
    result = validate(request(), (row(0), row(2)))
    assert result.report.status == "incomplete"
    assert result.report.missing_intervals == (Period(start=T + DAY, end=T + 2 * DAY),)
    assert len(result.observations) == 2


def test_validation_clean_and_cutoff() -> None:
    result = validate(request(1), (row(0),))
    assert result.report.status == "clean"
    assert result.report.expected_intervals == 1
    result = validate(request(1), (row(0, observed_at=T + timedelta(hours=1)),))
    assert result.report.incomplete_records == 1
    assert not result.observations


class FakeProvider:
    exchange = "binance"
    identity = "mock-public"

    def __init__(self, responses: Sequence[list[list[object]] | BaseException]) -> None:
        self.responses = iter(responses)
        self.calls: list[int] = []
        self.now = T + 10 * DAY
        self.supported: tuple[tuple[str, ...], tuple[str, ...]] = (
            ("BTC/USDT", "ETH/USDT", "SOL/USDT"),
            ("1d", "4h"),
        )

    def capabilities(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        return self.supported

    def server_time(self) -> datetime:
        return self.now

    def fetch(self, market: Market, since: int, limit: int, end: int) -> list[list[object]]:
        self.calls.append(since)
        result = next(self.responses, [])
        if isinstance(result, BaseException):
            raise result
        return result


def raw(day: int, **updates: object) -> list[object]:
    r = row(day, **updates)
    return [int(r.open_time.timestamp() * 1000), r.open, r.high, r.low, r.close, r.volume]


@pytest.mark.parametrize(
    "changes",
    [
        {"open": 0.0},
        {"close": -1.0},
        {"low": 13.0},
        {"high": 8.0},
        {"volume": -0.1},
        {"open": float("nan")},
        {"close": float("inf")},
        {"open_time": T + timedelta(hours=1)},
        {"open_time": T - DAY},
        {"open_time": T + 3 * DAY},
    ],
)
def test_invalid_observations(changes: dict[str, object]) -> None:
    result = validate(request(), (row(0, **changes),))
    assert result.report.status == "invalid"
    assert result.report.invalid_records == 1
    assert not result.observations


def test_duplicates_order_and_zero_volume() -> None:
    result = validate(request(), (row(2), row(0, volume=0.0), row(0, volume=0.0)))
    assert result.report.duplicates_detected == 1
    assert result.report.zero_volume_observations == 1
    assert any(a.kind == "order" for a in result.report.anomalies)
    assert tuple(r.open_time for r in result.observations) == (T, T + 2 * DAY)
    conflict = validate(request(1), (row(0), row(0, close=10.5), row(0)))
    assert conflict.report.status == "invalid" and not conflict.observations


def test_request_alignment_and_four_hour() -> None:
    with pytest.raises(ValidationError, match="align"):
        Request(market=M, period=Period(start=T + timedelta(hours=1), end=T + DAY))
    with pytest.raises(ValidationError, match="epoch"):
        Request(market=M, period=Period(start=datetime(1969, 1, 1, tzinfo=UTC), end=T))
    market = Market(exchange="binance", pair="SOL/USDT", timeframe="4h")
    r = Request(market=market, period=Period(start=T, end=T + DAY))
    result = validate(r, (row(0),))
    assert result.report.expected_intervals == 6
    assert result.report.retrieved == Period(start=T, end=T + timedelta(hours=4))


@pytest.mark.parametrize(
    "responses,expected",
    [
        ([[raw(0), raw(1), raw(2)]], 3),
        ([[raw(0)], [raw(1)], [raw(2)]], 3),
        ([[raw(0)], []], 1),
        ([[]], 0),
        ([[raw(0)], [raw(0), raw(1)], [raw(1), raw(2)]], 3),
        ([[raw(0)], [raw(0)]], 1),
    ],
)
def test_pagination(responses: list[list[list[object]]], expected: int) -> None:
    provider = FakeProvider(responses)
    result = list(pages(provider, request()))
    assert sum(len(p.observations) for p in result) == expected
    assert len(provider.calls) <= 4
    if responses == [[raw(0)], [raw(0)]]:
        assert any(a.kind == "pagination" for a in result[-1].issues)


def test_retry_backoff_and_persistent_failure() -> None:
    transient = TransientProviderError("429 rate limit")
    provider = FakeProvider([transient, transient, [raw(0)]])
    delays: list[float] = []
    assert list(pages(provider, request(1), delays.append))[0].observations
    assert delays == [1.0, 2.0]
    provider = FakeProvider([transient] * 4)
    with pytest.raises(TransientProviderError):
        list(pages(provider, request(1), delays.append))
    assert len(provider.calls) == 4


def test_unexpected_page_timestamps_and_malformed_data() -> None:
    provider = FakeProvider([[raw(1), raw(0), [None, 1, 2, 1, 2, 1], raw(4)], []])
    result = list(pages(provider, request()))
    assert len(result[0].observations) == 2
    assert sum(a.kind == "invalid" for a in result[0].issues) == 2
    assert any(a.kind == "order" for a in result[0].issues)
    for timestamp in [True, "bad", 1.5, float("inf")]:
        p = FakeProvider([[[timestamp, 1, 2, 1, 2, 1]]])
        assert any(i.kind == "invalid" for i in list(pages(p, request(1)))[0].issues)
    provider = FakeProvider([[raw(0)], [raw(0, close=10.5)], []])
    assert len(list(pages(provider, request()))[1].observations) == 2


@pytest.mark.parametrize("markets,timeframes", [((), ("1d",)), (("BTC/USDT",), ())])
def test_unsupported(markets: tuple[str, ...], timeframes: tuple[str, ...]) -> None:
    provider = FakeProvider([])
    provider.supported = (markets, timeframes)
    with pytest.raises(ProviderError, match="unsupported"):
        list(pages(provider, request()))
    assert not provider.calls


def test_clock_regression() -> None:
    class Regressing(FakeProvider):
        def fetch(self, market: Market, since: int, limit: int, end: int) -> list[list[object]]:
            self.now -= DAY
            return [raw(0)]

    with pytest.raises(ProviderError, match="clock"):
        list(pages(Regressing([]), request()))


class FakeClient:
    timeframes = {"1d": "1d", "4h": "4h"}

    def __init__(self) -> None:
        self.error: Exception | None = None
        self.params: dict[str, object] = {}

    def load_markets(self) -> dict[str, dict[str, object]]:
        return {"BTC/USDT": {"spot": True}, "ETH/USDT": {"spot": False}}

    def fetch_time(self) -> int:
        if self.error:
            raise self.error
        return int((T + 10 * DAY).timestamp() * 1000)

    def fetch_ohlcv(
        self, symbol: str, timeframe: str, since: int, limit: int, params: dict[str, object]
    ) -> list[list[object]]:
        self.params = params
        return [raw(0)]


def test_binance_public_adapter() -> None:
    client = FakeClient()
    provider = BinanceProvider(client)
    assert provider.capabilities() == (("BTC/USDT",), ("1d", "4h"))
    assert provider.server_time() == T + 10 * DAY
    assert provider.fetch(M, 0, 1, 100)[0] == raw(0)
    assert client.params == {"endTime": 99}
    for error in [ccxt.RateLimitExceeded("429"), ccxt.RequestTimeout("timeout")]:
        client.error = error
        with pytest.raises(TransientProviderError):
            provider.server_time()
    client.error = ccxt.BadSymbol("not available")
    with pytest.raises(ProviderError, match="no exchange substitution"):
        provider.server_time()


def test_parquet_round_trip_and_hashes(tmp_path: Path) -> None:
    store = Store(tmp_path)
    data = validate(request(), (row(0), row(1), row(2)))
    manifest = store.save(data, "mock-public", T + 10 * DAY)
    loaded, observations, report = store.read(manifest.version)
    assert observations == data.observations and report == data.report and loaded == manifest
    assert manifest.dataset_id == content_hash(M, observations)
    assert manifest.artifact_sha256 != manifest.dataset_id
    assert store.save(data, "mock-public", T + 10 * DAY) == manifest
    assert store.versions() == (manifest.version,)
    alternate = tuple(row(day, observed_at=T + 20 * DAY) for day in (2, 1, 0))
    assert content_hash(M, alternate) == manifest.dataset_id
    assert content_hash(M, (row(0, close=10.5), row(1), row(2))) != manifest.dataset_id
    df = store.dataframe(manifest.version, market=M, period=request().period, as_of=T + 10 * DAY)
    assert str(df.open_time.dtype) == "datetime64[us, UTC]"
    assert df.close.tolist() == [11.0, 11.0, 11.0]


def test_replay_and_market_isolation(tmp_path: Path) -> None:
    store = Store(tmp_path)
    rows = (row(0, observed_at=T + DAY), row(1, observed_at=T + 2 * DAY), row(2))
    manifest = store.save(validate(request(), rows), "mock-public", T + 10 * DAY)
    early = store.load(manifest.version, market=M, period=request().period, as_of=T + DAY)
    assert len(early.candles) == 1 and early.candles[0].available_at == T + DAY
    with pytest.raises(ValueError, match="missing or unavailable"):
        store.load(manifest.version, market=M, period=request().period, as_of=T + 3 * DAY)
    partial = store.load(
        manifest.version, market=M, period=request().period, as_of=T + 3 * DAY, allow_gaps=True
    )
    assert len(partial.candles) == 2
    with pytest.raises(ValueError, match="no completed"):
        store.load(
            manifest.version, market=M, period=request().period, as_of=T + timedelta(hours=1)
        )
    other = Market(exchange="kraken", pair="BTC/USDT", timeframe="1d")
    with pytest.raises(ValueError, match="market"):
        store.load(manifest.version, market=other, period=request().period, as_of=T + 10 * DAY)
    with pytest.raises(ValueError):
        store.manifest("../outside")


@pytest.mark.parametrize("filename", ["observations.parquet", "quality.json", "manifest.json"])
def test_corruption_detected(tmp_path: Path, filename: str) -> None:
    store = Store(tmp_path)
    manifest = store.save(validate(request(1), (row(0),)), "mock-public", T + 10 * DAY)
    path = tmp_path / "versions" / manifest.version / filename
    if filename == "manifest.json":
        path.write_text(path.read_text().replace('"observation_count":1', '"observation_count":2'))
    else:
        path.write_bytes(path.read_bytes() + b"corruption")
    with pytest.raises(IntegrityError):
        store.read(manifest.version)


def test_atomic_failure_is_not_published(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = Store(tmp_path)

    def fail(*args: object, **kwargs: object) -> None:
        raise OSError("disk write failed")

    monkeypatch.setattr("crypto_research.data.storage.pq.write_table", fail)
    with pytest.raises(OSError, match="disk write"):
        store.save(validate(request(1), (row(0),)), "mock-public", T + 10 * DAY)
    assert store.versions() == () and not list(tmp_path.glob(".staging-*"))


def test_incremental_update_revision_and_idempotence(tmp_path: Path) -> None:
    store = Store(tmp_path)
    original = acquire(FakeProvider([[raw(0), raw(1)]]), store, request(2))
    provider = FakeProvider([[raw(1, close=10.5), raw(2)]])
    provider.now = T + 20 * DAY
    updated = acquire(provider, store, request(), parent_version=original.version, overlap=1)
    assert provider.calls[0] == int((T + DAY).timestamp() * 1000)
    assert updated.change == "corrected" and updated.revised_opens == (T + DAY,)
    assert updated.parent_version == original.version
    assert store.read(original.version)[1][1].close == 11.0
    assert store.read(updated.version)[1][1].observed_at == T + 20 * DAY
    repeat_provider = FakeProvider([[raw(2)]])
    repeat_provider.now = T + 30 * DAY
    repeated = acquire(repeat_provider, store, request(), parent_version=updated.version, overlap=1)
    assert repeated.version == updated.version
    assert store.read(updated.version)[2].status == "clean"


def test_old_gaps_not_erased_and_restart(tmp_path: Path) -> None:
    store = Store(tmp_path)
    checkpoints: list[Manifest] = []
    with pytest.raises(KeyboardInterrupt):
        acquire(
            FakeProvider([[raw(0)], KeyboardInterrupt()]),
            store,
            request(),
            checkpoint=checkpoints.append,
        )
    assert len(checkpoints) == 1
    first = checkpoints[0]
    assert store.read(first.version)[2].status == "incomplete"
    resumed = acquire(
        FakeProvider([[raw(1), raw(2)]]), store, request(), parent_version=first.version, overlap=0
    )
    assert store.read(resumed.version)[2].status == "clean"
    gap = acquire(FakeProvider([[raw(0), raw(2)]]), store, request())
    provider = FakeProvider([[raw(2), raw(3)]])
    extended = acquire(provider, store, request(4), parent_version=gap.version, overlap=1)
    report = store.read(extended.version)[2]
    assert report.status == "incomplete"
    assert report.missing_intervals == (Period(start=T + DAY, end=T + 2 * DAY),)


def test_no_data_and_update_restrictions(tmp_path: Path) -> None:
    store = Store(tmp_path)
    with pytest.raises(NoDataError) as exc:
        acquire(FakeProvider([[]]), store, request())
    assert exc.value.report.status == "incomplete" and store.versions() == ()
    original = acquire(FakeProvider([[raw(0), raw(1)]]), store, request(2))
    with pytest.raises(ValueError, match="shrink"):
        acquire(FakeProvider([]), store, request(1), parent_version=original.version)
    with pytest.raises(ValueError, match="overlap"):
        acquire(FakeProvider([]), store, request(), overlap=-1)
    with pytest.raises(ValueError, match="preserve"):
        acquire(
            FakeProvider([]),
            store,
            Request(market=M, period=Period(start=T + DAY, end=T + 3 * DAY)),
            parent_version=original.version,
        )


def test_unresolved_old_anomaly_survives_incremental_update(tmp_path: Path) -> None:
    store = Store(tmp_path)
    issue = Issue(kind="invalid", message="historical conflict", open_time=T)
    original = store.save(
        validate(request(2), (row(0), row(1)), (issue,)), "mock-public", T + 10 * DAY
    )
    updated = acquire(
        FakeProvider([[raw(1), raw(2)]]),
        store,
        request(),
        parent_version=original.version,
        overlap=1,
    )
    assert store.read(updated.version)[2].status == "invalid"
    with pytest.raises(ValueError, match="unresolved invalid"):
        store.load(updated.version, market=M, period=request().period, as_of=T + 20 * DAY)


def test_atomic_publish_failure_and_clock_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = Store(tmp_path)
    original = acquire(FakeProvider([[raw(0)]]), store, request(1))

    def fail(*args: object, **kwargs: object) -> None:
        raise OSError("publish failed")

    monkeypatch.setattr("crypto_research.data.storage.os.rename", fail)
    with pytest.raises(OSError, match="publish"):
        store.save(validate(request(2), (row(0), row(1))), "mock-public", T + 10 * DAY)
    assert store.versions() == (original.version,)
    assert not list(tmp_path.glob(".staging-*"))
    provider = FakeProvider([[raw(0)]])
    provider.now = T + 9 * DAY
    with pytest.raises(ProviderError, match="prior acquisition"):
        acquire(provider, store, request(1), parent_version=original.version)


def test_revised_values_are_not_available_before_revision(tmp_path: Path) -> None:
    store = Store(tmp_path)
    original = store.save(
        validate(request(1), (row(0, observed_at=T + DAY),)), "mock-public", T + DAY
    )
    provider = FakeProvider([[raw(0, close=10.5)]])
    corrected = acquire(provider, store, request(1), parent_version=original.version)
    with pytest.raises(ValueError, match="no completed"):
        store.load(corrected.version, market=M, period=request(1).period, as_of=T + DAY)
    historical = store.load(original.version, market=M, period=request(1).period, as_of=T + DAY)
    assert historical.candles[0].close == 11.0


def test_manifest_and_quality_boundaries(tmp_path: Path) -> None:
    store = Store(tmp_path)
    data = validate(request(1), (row(0),))
    manifest = store.save(data, "mock-public", T + 10 * DAY)
    with pytest.raises(ValidationError, match="timezone-aware"):
        Manifest.model_validate(
            manifest.model_dump() | {"revised_opens": (T.replace(tzinfo=None),)}
        )
    bad_report = data.report.model_dump() | {"candle_count": 100}
    with pytest.raises(ValueError, match="matching quality"):
        store.save(
            type(data).model_validate({"observations": data.observations, "report": bad_report}),
            "mock-public",
            T + 10 * DAY,
        )


def test_logical_corruption_even_with_updated_artifact_checksum(tmp_path: Path) -> None:
    import json

    import pyarrow.parquet as pq

    from crypto_research.data.storage import digest, version_hash

    store = Store(tmp_path)
    manifest = store.save(validate(request(1), (row(0),)), "mock-public", T + 10 * DAY)
    folder = tmp_path / "versions" / manifest.version
    artifact = folder / "observations.parquet"
    table = pq.read_table(artifact)
    entries = table.to_pylist()
    entries[0]["close"] = 10.5
    pq.write_table(type(table).from_pylist(entries, schema=table.schema), artifact)
    changed = Manifest.model_validate(
        manifest.model_dump() | {"artifact_sha256": digest(artifact.read_bytes())}
    )
    version = version_hash(changed)
    (folder / "manifest.json").write_text(
        json.dumps(changed.model_dump(mode="json") | {"version": version})
    )
    folder.rename(folder.with_name(version))
    with pytest.raises(IntegrityError, match="logical content"):
        store.read(version)
