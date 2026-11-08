# trading-bot

Python 3.12 cryptocurrency quantitative research infrastructure. The package is
`crypto_research`. It provides validated contracts, configuration, public Binance
OHLCV ingestion and immutable local Parquet storage. It contains no strategies,
backtesting calculations, order submission or profitability claims.

## Installation

Install Python 3.12 and [uv](https://docs.astral.sh/uv/getting-started/installation/),
then from a checkout run:

```sh
uv sync --locked
```

Runtime only: `uv sync --locked --no-dev`. CCXT, NumPy, Pandas and PyArrow are
now runtime dependencies for data work. Optional future research dependencies:
`uv sync --locked --extra research` includes TA-Lib, VectorBT, SciPy, scikit-learn
and CVXPY. The existing `ingestion` extra remains compatible. The lock resolves
all extras together; installing research extras may need platform-specific wheels.
Python is intentionally constrained to 3.12 until other versions are validated.

## Development and checks

```sh
uv run --no-sync ruff format .
uv run --no-sync ruff check .
uv run --no-sync mypy
uv run --offline --no-sync pytest
```

Tests use synthetic fixtures and require no network or credentials after install.
Coverage is printed and written to `coverage.xml`; the minimum is 90%.
CI runs these checks on Linux and Windows. To change dependencies, edit
`pyproject.toml`, run `uv lock`, and commit both files. `--locked` checks that the
manifest and lock agree; see [uv sync documentation](https://docs.astral.sh/uv/concepts/projects/sync/).

## Configuration

```python
from crypto_research.config.models import load_config

settings = load_config("configs/research.yaml")
```

Defaults select BTC/USDT, ETH/USDT and SOL/USDT, daily and four-hour candles,
10,000 simulated USDT, spot long-only exposures, no leverage and no live trading.
Fees (10 bps) and slippage (5 bps) are illustrative assumptions, not exchange
quotes. Both fields are required, finite and nonnegative; explicit zero is allowed
for deliberate sensitivity studies. No implicit zero costs are permitted.
Unknown settings and duplicate YAML keys are rejected. Do not store credentials.

See [architecture](docs/architecture.md) for contracts, limitations and next steps.

## Reproducibility and contracts

`uv.lock` pins the complete resolution, including optional dependencies. Runtime
versions in this lock are Pydantic 2.14.3 and PyYAML 6.0.3. Development tools are
pytest 8.4.2, pytest-cov 6.3.0, Ruff 0.17.4 and mypy 1.20.2. Build dependencies
use the declared Hatchling compatibility range. Data versions are CCXT 4.5.92,
NumPy 2.5.4, Pandas 2.3.3 and PyArrow 23.0.1. Remaining optional research
libraries are locked but are not installed by default.

Configuration records expose `canonical_json()` and SHA-256 `identity`. Use
`model_validate` or constructors at input boundaries; Pydantic's `model_construct`
and `model_copy(update=...)` deliberately bypass validation and must not be used
with untrusted data. UTC-normalized records can be exported with
`model_dump(mode="python")` for future tabular tools without importing them now.

Historical replay uses `strategies.base.reconstruct` with separate validated
`MarketData` snapshots for each cutoff. Decision inputs and research-only
`FutureOutcome` labels are distinct contracts; see the architecture for their
availability and joining rules. The protocol does not sandbox strategy code.

## Historical market data

See [the market-data guide](docs/market-data.md) for downloading BTC, ETH and SOL
on daily and four-hour timeframes, updates, quality checks, manifests, checksums,
restart points and the optional live smoke test.

```sh
uv run python -m crypto_research.data.cli download --market BTC/USDT ETH/USDT SOL/USDT --timeframe 1d 4h --start 2022-01-01 --end 2023-01-01
uv run python -m crypto_research.data.cli list
uv run python -m crypto_research.data.cli quality VERSION
uv run python -m crypto_research.data.cli update VERSION --end 2023-02-01 --overlap 3
```

Commands explicitly opt into public exchange requests. No credentials are used.
Dates mean UTC midnight; requested open-time intervals are `[start,end)`. Immutable
Parquet, JSON manifests and quality reports live under `data/versions/VERSION/`
and are ignored by Git. Missing data is reported, never filled. Normal tests block
network access; the live smoke test is skipped without `--live-binance`.

Historical REST data cannot establish original publication time. Research loaders
use recorded exchange observation times and require an explicit cutoff. Data
observed today cannot be used at yesterday's cutoff; prospective vintages or an
explicit, justified availability model are needed for strict early-period replay.
