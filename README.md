# trading-bot

Python 3.12 foundations for cryptocurrency quantitative research. The package is
`crypto_research`. This project contains contracts and configuration only; it does
not ingest market data, backtest, submit orders, or claim profitable results.

## Installation

Install Python 3.12 and [uv](https://docs.astral.sh/uv/getting-started/installation/),
then from a checkout run:

```sh
uv sync --locked
```

Runtime only: `uv sync --locked --no-dev`. Optional future research dependencies:
`uv sync --locked --extra research --extra ingestion`. NumPy, Pandas, TA-Lib,
VectorBT, SciPy, scikit-learn and CVXPY belong to `research`; CCXT belongs to
`ingestion`. Neither extra is imported by the foundation. The lock resolves all
extras together; installing extras may require platform-specific binary wheels.
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
