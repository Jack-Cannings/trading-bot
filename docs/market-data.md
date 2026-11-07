# Historical market data

The data subsystem acquires public Binance spot OHLCV and stores immutable local
versions. It implements no strategy, indicator, backtest, performance calculation
or order submission. Imports do not contact an exchange. Standard tests block
socket connections; the live integration test requires an explicit opt-in.

## Installation and commands

Python 3.12 and `uv sync --locked` install the data dependencies: CCXT, Pandas,
NumPy and PyArrow. There is no need to install TA-Lib, VectorBT or optimization
libraries. Optional research extras remain available for later work.

Run commands from the repository root. `--root` precedes the subcommand and
defaults to `data`. Date-only arguments mean UTC midnight; full timestamps must
include a timezone. Start/end boundaries must align to UTC candle opens.

```sh
uv run python -m crypto_research.data.cli download --exchange binance --market BTC/USDT --timeframe 1d --start 2022-01-01 --end 2023-01-01
uv run python -m crypto_research.data.cli download --market ETH/USDT --timeframe 4h --start 2022-01-01 --end 2023-01-01 --batch-size 500 --retries 3
```

This command requests all six initial market/timeframe combinations independently:

```sh
uv run python -m crypto_research.data.cli download --market BTC/USDT ETH/USDT SOL/USDT --timeframe 1d 4h --start 2022-01-01 --end 2023-01-01
```

Use `--longest` instead of `--start` to probe from the Unix epoch. The first
returned candle establishes the earliest retrieved coverage, not proof that the
endpoint exposed all historical observations. Pre-listing periods remain reported
as missing, without assuming that BTC, ETH and SOL share a listing date.

```sh
uv run python -m crypto_research.data.cli download --market BTC/USDT --timeframe 1d --longest --end 2026-11-01
uv run python -m crypto_research.data.cli list
uv run python -m crypto_research.data.cli coverage VERSION
uv run python -m crypto_research.data.cli quality VERSION
uv run python -m crypto_research.data.cli validate VERSION
uv run python -m crypto_research.data.cli update VERSION --end 2026-11-07 --overlap 3
```

Replace `VERSION` with the SHA-256 version printed by download/update. JSON
manifests go to stdout; durable checkpoint identifiers go to stderr. Inspect the
quality report even after a successful command: completion of acquisition does
not imply complete historical coverage. `validate` returns exit code 2 for
incomplete/invalid quality; errors also return 2, and interruption returns 130.
No-data errors emit a machine-readable quality report and create no empty Parquet
file. Coverage reports include requested and actual ranges and candle counts.

## Exchange boundary, pagination and retries

`Provider` exposes exchange identity, supported markets/timeframes, exchange time
and raw historical OHLCV retrieval. `BinanceProvider` uses CCXT's public spot
`fetch_ohlcv`, `load_markets` and `fetch_time` methods. The corresponding Binance
resources are `/api/v3/klines`, `/api/v3/exchangeInfo` and `/api/v3/time`. No API
keys, authentication, exchange substitution or private endpoints are configured.
Actual availability is checked through market metadata before fetching candles.

CCXT's built-in rate limiter is enabled. Connectivity and rate-limit failures
receive bounded retries with exponential delays capped at 60 seconds; unsupported
symbols and permanent exchange errors fail explicitly. Request size is 1–1000.
The downloader advances from returned opens, continues after short pages, detects
repeated/non-advancing pages and stops on an empty page or the exclusive end.
Duplicate boundaries are reported and deduplicated. Conflicting duplicates are
rejected rather than choosing one value. Out-of-order and unexpected timestamps
remain visible in reports even though accepted output is sorted.

Requested intervals are `[start,end)` over candle **opens**. Close is derived as
open plus the timeframe and is exclusive; Binance's inclusive last-millisecond
close convention is not copied into the research contract. End bounds supplied
to Binance use `endTime=end-1ms`. A candle is complete only after its full interval
has elapsed according to an exchange server timestamp retrieved after the page.
The local wall clock never substitutes for exchange time, and clock regressions
fail explicitly. Remote server-time accuracy itself is trusted, not independently
certified by this implementation.

The relevant external specifications are the
[CCXT manual](https://docs.ccxt.com/docs/manual) and
[Binance Spot REST API](https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md).
Binance endpoint access may be restricted by jurisdiction, connectivity, service
availability or rate limits. Public endpoints can expose incomplete history;
this tool reports missing coverage and does not silently use another exchange.

## Validation and quality

The validator rejects malformed, non-finite or nonpositive prices, negative
volume, inconsistent OHLC bounds, misaligned/out-of-range timestamps and
incomplete observations. Conflicting duplicates are removed; identical duplicates
are reported and collapsed. Zero volume is a warning, not an automatic rejection.
Missing periods are compressed into half-open ranges rather than a potentially
huge list of timestamps. `expected_intervals` includes the entire requested
range, including pre-listing and not-yet-completed periods. No prices are filled,
interpolated, smoothed or manufactured.

Quality is `clean` when coverage is complete and there are no structural
anomalies, `incomplete` for gaps or unresolved ordering/pagination/duplicate
anomalies, and `invalid` when rejected invalid records remain reported. The report
contains the full request (exchange, pair, timeframe, boundaries and settings),
actual coverage, accepted candle count, expected intervals, missing ranges,
duplicate/invalid/incomplete/zero-volume counts and detailed anomaly messages.
Rejected raw values are not persisted in research Parquet; diagnostics describe
the rejection. This is rejection rather than a quarantine/archive implementation.

## Immutable storage and identity

The layout groups a complete immutable version into one directory, simplifying
atomic publication and keeping its manifest, report and data together:

```text
data/
  versions/
    VERSION/
      observations.parquet
      manifest.json
      quality.json
  .staging-.../           # not discoverable as a committed dataset
```

Each version contains exactly one exchange/pair/timeframe. Select these through
the manifest and loader; versions never mix exchange sources. All downloaded
data under the default root is ignored by Git. Custom roots should also be kept
outside version control. No empty data files or directories are committed.

Parquet schema v1 uses string exchange/pair/timeframe, UTC microsecond open,
exclusive close and observation timestamps, float64 OHLCV and boolean
completeness. Rows are ascending and unique; Zstandard compression and 4096-row
groups support date predicate pushdown. Reads verify checksums before decoding.
Point-in-time reads filter both the requested opens and availability timestamps;
the integrity scan still reads all artifact bytes, but unrelated row groups need
not be decoded. The store is intended for local research, not a distributed
catalog or large-scale query service.

Logical content hashing uses UTF-8 compact JSON with sorted keys, schema version,
market identity and rows sorted by UTC open. Row values are open time in ISO
format followed by float64 open/high/low/close/volume; negative zero normalizes to
zero. Observation times and acquisition settings are excluded. Identical
normalized market observations therefore have the same `dataset_id` regardless
of batching or download time. This checksum is distinct from SHA-256 of the
actual Parquet bytes.

The manifest records logical identity, immutable version, provider/library
identity, schema, requested and actual coverage, observation count, exchange-time
acquisition timestamp, Parquet checksum, quality checksum and relative artifact
references, acquisition settings, availability policy, parent version, change
classification and revised candle opens. The version is SHA-256 of canonical
manifest fields excluding only its own version field. It therefore identifies
both content and provenance, including artifact bytes. Readers verify this
identity as well as file checksums, schema and logical contents on full validation.
Hashes detect corruption and inconsistency; they are not signatures authenticating
the exchange. An experiment must retain the specific version and logical checksum.

Writes stage all three files, flush them, then rename the entire directory on the
same filesystem. An existing nonempty version cannot be overwritten. A failed
write cleans its staging directory; a hard process termination may leave an
ignored staging directory that can be removed after confirming no writer uses it.
This does not claim transactional durability across every filesystem/power failure.

## Updates and restart

Every successfully processed page publishes an immutable checkpoint with quality
measured against the **full** requested range. If interrupted, use `update` with
the last checkpoint identifier; it requests remaining missing ranges and a
configurable recent overlap. Original versions remain unchanged. Updates also
retry historical gaps, but successful extension does not erase gaps that remain.
Adjacent/overlapping missing and overlap ranges are coalesced before fetching.

Overlapping equal OHLCV retains the earlier observed availability. Changed values
create a corrected version, record changed opens and receive the new observation
timestamp. Additions create updated content. Repeating an unchanged update with
the same coverage/quality returns the existing version. Metadata-only quality or
coverage changes may create a new version with the same logical dataset identity.
Earlier anomalies persist until the relevant timestamps are successfully
reobserved; an undated anomaly requires a fresh download for independent review.
Corrections outside requested missing/overlap ranges cannot be detected by that
update. Use a larger overlap or a fresh full-range download for a deeper audit.

For simplicity, checkpoints contain the cumulative accepted data. Long downloads
trade extra disk/I/O for reliable local restart; there is no mutable latest-version
pointer or automatic garbage collection. Versions must be chosen explicitly.

## Point-in-time research access

```python
from crypto_research.contracts import Market, Period
from crypto_research.data.cli import date
from crypto_research.data.storage import Store

store = Store("data")
market = Market(exchange="binance", pair="BTC/USDT", timeframe="1d")
period = Period(start=date("2022-01-01"), end=date("2023-01-01"))
# Choose an explicit cutoff at/after the recorded observation time for these values.
cutoff = date("2026-11-08")
history = store.load("VERSION", market=market, period=period, as_of=cutoff)
df = store.dataframe("VERSION", market=market, period=period, as_of=cutoff)
```

`load` returns existing `MarketData` and `Candle` contracts with dataset version
references. Wrong market/timeframe identities, naive cutoffs, unordered data and
unresolved invalid datasets fail. Missing/unavailable intervals fail by default;
`allow_gaps=True` explicitly permits a partial view, while never relaxing
completeness, availability or invalid-data checks. Pandas access applies the same
cutoff. Use separately constructed snapshots with `strategies.base.reconstruct`.

**Candle close does not prove actual data availability.** Binance historical REST
OHLCV provides no publication-vintage history. These downloads conservatively
set `available_at` to the exchange time sampled after observing the page. A
candle downloaded today is not exposed at a historical cutoff yesterday, and a
corrected value is unavailable before its correction was observed. Earlier
immutable versions can reproduce the original observed values. There is no
implicit close-time availability assumption or historical-vintage fabrication.

The system supports reproducible storage and causal access for future research.
It cannot retrospectively prove the first publication time of old exchange data.
Strict historical decision reconstruction needs prospectively captured vintages
or a separately justified, explicit availability model before backtesting early
periods. No such model or backtesting engine is included here.

## Tests

```sh
uv run --offline --no-sync pytest
uv run --no-sync ruff format --check .
uv run --no-sync ruff check .
uv run --no-sync mypy
```

The optional public smoke test is separate from ordinary offline CI:

```sh
uv run --no-sync pytest tests/integration/test_binance_live.py --live-binance --no-cov -rs -s
```

It downloads three historical BTC daily candles into a temporary directory,
validates them and reports accessibility. Endpoint restrictions/network errors
are reported as a skip with a reason; successful connectivity with invalid data
still fails validation. It requires no credentials and never places an order.
