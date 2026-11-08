# Daily time-series momentum research

Prompt 5 adds one directional strategy family. `TimeSeriesMomentum` implements the
existing `Strategy` protocol; the separate fixed volatility-target comparator is
a passive risk benchmark, not a second alpha strategy. Nothing submits orders to
an exchange. Only explicit acquisition commands use the public network.

For L in **30, 60, 90, 120, 180**, compute `close[t] / close[t-L] - 1` from L+1
contiguous, completed, available daily candles. Strictly positive momentum targets
100% long; zero or negative targets cash. Lookback parameters are strict integers
from this set; bools, floats, strings and other candidates fail. The assessment
distinguishes warmup, missing intervals and stale data from a valid cash signal.
No prices are filled across gaps. Decimal division uses an independent 60-digit
context; momentum is stored as signal strength and explicitly projected alongside
lookback in `decisions.parquet` schema 2.

Signals carry source close in `data_through`, snapshot time in `generated_at` and
`effective_at`. The existing broker schedules the first eligible subsequent open.
At an exact prior-close/next-open boundary the established engine convention can
execute at that open. A signal arriving 60 seconds later must wait for the next
daily open. The source close is never used as its execution price. Future candles
and separate future outcomes are absent from strategy inputs.

## Acquisition and exact dataset selection

```powershell
uv sync --locked
uv run python -m crypto_research.data.cli download --market BTC/USDT ETH/USDT SOL/USDT --timeframe 1d --longest --end 2026-11-08
uv run python -m crypto_research.data.cli list
uv run python -m crypto_research.data.cli quality <VERSION>
```

Select explicit immutable versions; the campaign never silently picks a latest
version. `make_plan` verifies source provider, checksums, market, UTC alignment,
quality and actual coverage. Invalid records, duplicates, incomplete observations,
interior missing candles or unresolved anomalies reject that market. Pre-listing
gaps reported by `--longest` remain in provenance but do not exclude a contiguous
actual trading history. No alternative downloader or exchange credentials are used.

```powershell
uv run python -m crypto_research.experiments.cli run-momentum --dataset BTC/USDT=<BTC_VERSION> --dataset ETH/USDT=<ETH_VERSION> --dataset SOL/USDT=<SOL_VERSION>
```

Default replay respects actual observation timestamps. A REST archive acquired
today does **not** establish when those candle values were historically published
or revised. Such data cannot produce historical strategy results under the default
policy. All unavailable strategy-market combinations and benchmarks are recorded
without fabricated dataset references, prices, returns or successful executions.

For an explicitly conditional archival simulation:

```powershell
uv run python -m crypto_research.experiments.cli run-momentum --archival-close-plus-60s --dataset BTC/USDT=<BTC_VERSION> --dataset ETH/USDT=<ETH_VERSION> --dataset SOL/USDT=<SOL_VERSION>
```

`archival-close-plus-60s` is a named **unverified research assumption**: finalized
daily OHLCV is assumed published 60 seconds after close. This is not an assertion
about Binance publication latency. The raw observations, manifests and acquisition
timestamps remain unchanged. `ExperimentSpec.replay_availability` participates in
input/comparison hashes; assumed and observed results cannot be compared as the
same experiment. A derived in-memory candle tape and its content identity record
the assumption. Decisions record assumed availability, and their spec and report
identify the policy. Separate feature caches prevent policy contamination. This
small compatible extension allows conventional archival research without claiming
historical publication-vintage validation. Revision bias remains unmeasured.

For a fully synthetic fixture campaign, visibly labeled in every report:

```powershell
uv run python -m crypto_research.experiments.cli --root research/momentum-fixture --data-root data/momentum-fixture run-momentum --fixture
```

Fixtures have exchange `synthetic`, a synthetic provider and deterministic invented
prices. They verify software and cannot establish market performance.

## Predeclared campaign and chronological validation

The plan is committed to `campaign_plans` in the existing SQLite experiment registry
and written to `plan.json` **before** running any strategy. Each market independently
uses its actual coverage, with 201 warmup days for all lookbacks and 200-day features.
The final 365 days are reserved if at least 180 development and 180 validation days
remain. Otherwise reserve max(90 days, 20% of eligible history); require at least
180 development plus 90 validation days. Insufficient coverage is explicit.

Development is `min(365, max(180, available_non_holdout_days // 3))` days. Split
remaining validation time into two chronological windows if each can have 90 days,
otherwise one. All five candidates run in every window. There is no fitting,
parameter selection or threshold estimation. Evaluation splits use fresh 10,000
USDT accounts, with pre-window history solely for causal warmup. No positions or
capital are carried between validation folds. Future outcomes end at their fold
boundary; incomplete labels stay incomplete, so labels do not cross folds.

The per-market aggregate covers all eligible non-holdout history. When all three
markets have at least 90 overlapping eligible days, a common-period aggregate uses
that intersection. Coverage differences remain visible. Aggregates overlap the
development/validation folds and cannot be treated as independent validation.

`information_end` is the protected holdout start. The runner excludes later candles
from the broker and feature tapes, not merely from reported statistics. Raw data
integrity checks may read/hash source bytes including holdout bytes; those prices
are not used for decisions, fitting, evaluation or report plots. A last pre-holdout
close assumed published after the boundary is not available at the boundary;
valuation can retain the prior open within the engine's one-day staleness bound.
Every strategy and benchmark uses the same convention.

Two cost scenarios use `configs/research.yaml`: 10 bps fee and 5 bps adverse
slippage per fill, and 20/10 bps sensitivity. These are project assumptions, not a
measured Binance tier/spread. Starting capital is 10,000 USDT. Same accounting,
periods, valuation and costs apply to cash and same-coin buy-and-hold. The causal
risk comparator rebalances daily to `min(1, 0.20 / trailing_30_day_vol)` using sample
log-return standard deviation times sqrt(365.25), 31 completed daily closes and
the same gap/warmup protections. Zero estimated volatility targets the cap of one.
It targets fixed ex-ante risk and does not equalize realized momentum/passive risk.

There are 15 initial momentum-market combinations. Each scope, chronological fold,
cost multiplier and three benchmarks is a separately counted trial. All poor,
failed and unavailable results are retained. No automatic ranking or best-lookback
selection occurs. Comparing five lookbacks across three correlated coins creates
multiple-testing risk even before adding windows/cost scenarios.

## Registry, artifacts and repeatability

Completed/failed executable experiments use the existing registry lifecycle,
Runner, Decimal engine, metrics and immutable artifact store. `campaign_attempts`
links a durable attempt to its preallocated execution before running; missing
datasets have an explicit unavailable campaign record without inventing an
`ExperimentSpec.dataset`. Created/running attempts remain visible after abrupt
interruption. Repeating a plan creates new execution UUIDs for the same immutable
logical experiment configurations and should reproduce content IDs.

```powershell
uv run python -m crypto_research.experiments.cli run-momentum --plan research/reports/time_series_momentum/<RUN>/plan.json
uv run python -m crypto_research.experiments.cli list
uv run python -m crypto_research.experiments.cli compare
```

Use the same `--root` and `--data-root` for saved-plan replay. These global options
precede the subcommand. A clean committed checkout provides the default code SHA;
`--revision` is a caller-declared identifier, not a clean-tree attestation. Run IDs
and physical artifact hashes differ between executions; numerical results and
logical content IDs should match. Default-valued new replay fields preserve legacy
Prompt 4 experiment hashes. Artifact reader supports schemas 1 and 2.

Report directory: `research/reports/time_series_momentum/<PLAN_PREFIX>-<RUN>/`.
It contains the predeclared plan, all attempts, full results/undefined reasons,
matched benchmark differences, conditional diagnostics, deterministic numerical
results, standalone SVG figures, readable report and file-hash manifest. Research
outputs and market data stay ignored by Git. Per-execution artifacts remain at
`research/artifacts/<EXECUTION_UUID>/` with original manifests and registry anchors.

Decision records contain exact configuration/version/hash, market, source/checksum,
data cutoff, momentum, lookback, target and eight point-in-time features. Performance
schema 2 adds `quantity_before` alongside post-event `quantity`. Quantities are
exact decimal strings. Future 1/7/30-day simulated returns remain in a **separate**
outcomes table keyed by decision ID. They follow executable strategy decisions
throughout the horizon, not a frozen hypothetical signal. Half-open label windows
include entry costs and exclude trades at the endpoint. Read [research.md](research.md)
for metric, label and purged chronological join conventions.

Diagnostics analyze each axis separately: trailing 30-day return >0 vs <=0, and
annualized trailing 30-day volatility >=60% vs <60%. Thresholds are fixed before
evaluation, without full-sample quantiles. Missing features are explicit. Tables
report complete/incomplete label counts, distinct consecutive category episodes,
and a greedily thinned non-overlapping horizon count. Adjacent outcomes, markets
and repeated scenarios remain correlated. No regime prediction/selection model is
trained. Completed-roundtrip win rate, profit factor and holding duration are not
supported by the current summary and are not fabricated from fill counts.

## Validation

All ordinary tests block socket and DNS access. Tests cover independent momentum
references, all five parameter identities, warmup/gaps, delayed publication, no
future/incomplete inputs, next-open fills, cash/no-leverage reconciliation and
liquidation. End-to-end campaigns verify artifact hashes, decision/label separation,
benchmark matching, failure/unavailability retention, protected tape boundaries
and repeated content identity. Run the full suite and static checks:

```powershell
uv run --offline --no-sync pytest
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv sync --locked
uv build
```
