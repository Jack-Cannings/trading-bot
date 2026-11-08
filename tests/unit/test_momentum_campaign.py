import json
import math
import statistics
from pathlib import Path

import pytest
from research_helpers import DAY, T, case, renamed
from test_features import history

from crypto_research.contracts import Period
from crypto_research.experiments.momentum_plan import chronological_windows
from crypto_research.strategies.risk_target import RiskTarget, risk_configuration


def test_predeclared_splits_holdout_and_short_history() -> None:
    coverage = Period(start=T, end=T + 1500 * DAY)
    windows, holdout = chronological_windows(coverage)
    assert holdout is not None and holdout.end - holdout.start == 365 * DAY
    assert windows[0].name == "development"
    assert len(windows) >= 3
    assert all(w.period.end <= holdout.start for w in windows)
    assert all(a.period.end == b.period.start for a, b in zip(windows, windows[1:], strict=False))
    short, reserved = chronological_windows(Period(start=T, end=T + 220 * DAY))
    assert short == () and reserved is None


def test_risk_benchmark_manual_reference_and_causality() -> None:
    h = history(days=40)
    benchmark = RiskTarget(risk_configuration(h.market))
    batch = benchmark.generate(h)
    prices = [c.close for c in h.candles[-31:]]
    vol = statistics.stdev(
        [math.log(b / a) for a, b in zip(prices, prices[1:], strict=False)]
    ) * math.sqrt(365.25)
    assert batch.signals[0].weight == pytest.approx(min(1, 0.2 / vol))
    assert batch.signals[0].data_through == h.as_of
    assert not benchmark.generate(history(days=30)).signals


def test_archival_policy_explicit_hash_and_protected_tape(tmp_path: Path) -> None:
    source, registry, runner, spec, _ = case(tmp_path, count=240)
    spec = renamed(
        spec,
        "bounded",
        settings=spec.settings.model_dump()
        | {"experiment_id": "bounded", "period": Period(start=T + DAY, end=T + 220 * DAY)},
    )
    archival = renamed(
        spec,
        "archival",
        replay_availability="archival-close-plus-60s",
        information_end=T + 220 * DAY,
    )
    assert archival.input_hash != spec.input_hash
    data = runner._load(archival)
    assert data.histories[0].candles[-1].close_time == archival.information_end
    assert all(c.available_at == c.close_time + DAY / 1440 for c in data.histories[0].candles)
    _, rows, _ = source.read(spec.dataset.version)
    assert rows[0].observed_at == T + DAY  # raw observations untouched
    assert runner._load(spec).histories[0].candles[0].available_at == T + DAY
    assert runner._load(archival).histories[0].candles[0].available_at == T + DAY + DAY / 1440
    with pytest.raises(ValueError, match="protected"):
        renamed(spec, "bad-boundary", information_end=T + 10 * DAY)
    assert registry.list() == ()


def test_default_hash_remains_compatible(tmp_path: Path) -> None:
    import hashlib

    _, _, _, spec, _ = case(tmp_path)
    old = spec.model_dump(mode="json", exclude={"replay_availability", "information_end"})
    old.pop("experiment_id")
    old["settings"].pop("experiment_id")
    expected = hashlib.sha256(
        json.dumps(old, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert spec.input_hash == expected


def compact_plan(tmp_path: Path) -> tuple[object, ...]:
    from crypto_research.config.models import load_config
    from crypto_research.contracts import DatasetReference, Market
    from crypto_research.data.models import Observation, Request
    from crypto_research.data.storage import Store
    from crypto_research.data.validation import validate
    from crypto_research.experiments.momentum_plan import PAIRS, CampaignPlan, Selection, Window
    from crypto_research.experiments.registry import Registry

    source, selections = Store(tmp_path / "data"), []
    for pair in PAIRS:
        market = Market(exchange="synthetic", pair=pair, timeframe="1d")
        # Trend up, then down: forces actual entry and full liquidation.
        prices = [100.0 + i * 0.1 if i < 205 else 50.0 for i in range(240)]
        rows = tuple(
            Observation(
                open_time=T + i * DAY,
                observed_at=T + (i + 1) * DAY,
                open=p,
                high=p,
                low=p,
                close=p,
                volume=10.0,
            )
            for i, p in enumerate(prices)
        )
        request = Request(market=market, period=Period(start=T, end=T + 240 * DAY))
        manifest = source.save(validate(request, rows), "synthetic-fixture", request.period.end)
        dataset = DatasetReference(
            source=manifest.provider, version=manifest.version, sha256=manifest.dataset_id
        )
        selections.append(
            Selection(
                market=market,
                dataset=dataset,
                coverage=request.period,
                quality="clean",
                candle_count=240,
                windows=(
                    Window(
                        name="development",
                        period=Period(start=T + 201 * DAY, end=T + 204 * DAY),
                        split="training",
                    ),
                    Window(
                        name="validation-1",
                        period=Period(start=T + 204 * DAY, end=T + 239 * DAY),
                        split="validation",
                    ),
                ),
                holdout=Period(start=T + 239 * DAY, end=T + 240 * DAY),
            )
        )
    plan = CampaignPlan(
        code_revision="a" * 40,
        research=load_config("configs/research.yaml"),
        selections=tuple(selections),
        fixture=True,
        common=Period(start=T + 201 * DAY, end=T + 239 * DAY),
    )
    return source, Registry(tmp_path / "research"), plan


def test_complete_campaign_all_lookbacks_benchmarks_records_and_reproducibility(
    tmp_path: Path,
) -> None:
    import pyarrow.parquet as pq

    from crypto_research.data.storage import Store
    from crypto_research.experiments.momentum_campaign import CampaignLedger, execute_campaign
    from crypto_research.experiments.momentum_plan import CampaignPlan
    from crypto_research.experiments.registry import Registry
    from crypto_research.experiments.storage import checksum

    source, registry, plan = compact_plan(tmp_path)
    assert (
        isinstance(source, Store)
        and isinstance(registry, Registry)
        and isinstance(plan, CampaignPlan)
    )
    report = execute_campaign(plan, source, registry)
    trials = CampaignLedger(registry).list(plan.identity)
    assert len(trials) == 192 and all(a.status == "completed" for a in trials)
    assert {
        a.configuration.parameters[0].value
        for a in trials
        if a.configuration.strategy_id == "time_series_momentum"
    } == {30, 60, 90, 120, 180}
    assert (report / "equity-drawdown.svg").is_file()
    assert "SYNTHETIC FIXTURE" in (report / "report.md").read_text(encoding="utf-8")
    hashes = json.loads((report / "manifest.json").read_text())
    assert all(checksum(report / name) == digest for name, digest in hashes.items())
    summaries = json.loads((report / "numerical-results.json").read_text())
    holdout = plan.selections[0].holdout
    assert holdout is not None
    assert all(row["end"] <= holdout.start.isoformat() for row in summaries)
    first = next(
        a
        for a in trials
        if a.window == "per-market"
        and a.configuration.strategy_id == "time_series_momentum"
        and a.cost_multiplier == 1
    )
    root = registry.root / "artifacts" / str(first.execution_id)
    decisions = pq.read_table(root / "decisions.parquet").to_pylist()
    performance = pq.read_table(root / "performance.parquet").to_pylist()
    labels = pq.read_table(root / "outcomes.parquet").to_pylist()
    assert decisions[0]["momentum"] > 0 and decisions[0]["lookback"] == 30
    assert all(d["available_at"] <= d["timestamp"] < holdout.start for d in decisions)
    assert not {"net_return", "horizon_days"} & set(decisions[0])
    assert {o["horizon_days"] for o in labels} == {1, 7, 30}
    assert any(o["status"] == "incomplete" for o in labels)
    assert any(float(p["quantity_before"]) > 0 and float(p["quantity"]) == 0 for p in performance)
    assert all(float(p["cash"]) >= 0 and p["actual_exposure"] <= 1 for p in performance)
    assert len(pq.read_table(root / "trades.parquet")) >= 2
    # Rerun exactly one logical trial; all expensive bundle verification stays enabled.
    from crypto_research.experiments.runner import Runner
    from crypto_research.strategies.time_series_momentum import TimeSeriesMomentum

    spec = registry.spec(registry.get(str(first.execution_id)).experiment_id)
    repeat = Runner(source, registry).execute(spec, strategy=TimeSeriesMomentum(spec.strategy))
    assert repeat.content_id == first.content_id and repeat.run_id != first.execution_id
    import pandas as pd

    compared = pd.read_csv(report / "benchmark-differences.csv")
    assert set(compared.benchmark) == {"cash", "benchmark", "passive_risk_target"}
    assert len(compared) == 120 * 3
    conditions = pd.read_csv(report / "conditions.csv")
    assert set(conditions.axis) == {"direction", "volatility"}
    assert (conditions.nonoverlapping_windows <= conditions.complete).all()


def test_unavailable_grid_is_durable_and_never_performance(tmp_path: Path) -> None:
    from crypto_research.config.models import load_config
    from crypto_research.data.storage import Store
    from crypto_research.experiments.momentum_campaign import CampaignLedger, execute_campaign
    from crypto_research.experiments.momentum_plan import make_plan
    from crypto_research.experiments.registry import Registry

    source, registry = Store(tmp_path / "data"), Registry(tmp_path / "research")
    plan = make_plan(source, {}, "a" * 40, load_config("configs/research.yaml"))
    report = execute_campaign(plan, source, registry)
    attempts = CampaignLedger(registry).list(plan.identity)
    assert len(attempts) == 48 and all(
        a.status == "unavailable" and a.execution_id is None for a in attempts
    )
    assert len(registry.list()) == 0
    assert all(
        "total_net_return" not in row
        for row in json.loads((report / "numerical-results.json").read_text())
    )


def test_failed_trials_remain_linked_and_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from crypto_research.data.storage import Store
    from crypto_research.experiments.momentum_campaign import (
        Attempt,
        CampaignLedger,
        execute_campaign,
    )
    from crypto_research.experiments.momentum_plan import CampaignPlan
    from crypto_research.experiments.registry import Registry
    from crypto_research.experiments.runner import Runner

    source, registry, plan = compact_plan(tmp_path)
    assert (
        isinstance(source, Store)
        and isinstance(registry, Registry)
        and isinstance(plan, CampaignPlan)
    )

    def broken(*args: object, **kwargs: object) -> None:
        raise ValueError("regression: source unavailable after preflight")

    monkeypatch.setattr(Runner, "_load", broken)
    execute_campaign(plan, source, registry)
    attempts = CampaignLedger(registry).list(plan.identity)
    assert len(attempts) == 192 and all(a.status == "failed" and a.execution_id for a in attempts)
    assert all(
        r.status == "failed" and "source unavailable" in str(r.failure_reason)
        for r in registry.list()
    )
    with pytest.raises(ValueError, match="terminal"):
        CampaignLedger(registry).save(attempts[0])
    changed = Attempt.model_validate(attempts[0].model_dump() | {"window": "other"})
    with pytest.raises(ValueError, match="configuration"):
        CampaignLedger(registry).save(changed)


def test_fixture_plan_coverage_policy_and_guardrails(tmp_path: Path) -> None:
    from crypto_research.config.models import load_config
    from crypto_research.data.storage import Store
    from crypto_research.experiments.momentum_fixtures import fixture_versions
    from crypto_research.experiments.momentum_plan import CampaignPlan, Selection, make_plan

    source = Store(tmp_path / "data")
    versions = fixture_versions(source)
    assert versions == fixture_versions(source)
    plan = make_plan(source, versions, "a" * 40, load_config("configs/research.yaml"), fixture=True)
    assert all(s.windows and not s.unavailable_reason for s in plan.selections)
    assert plan.common is not None
    assert plan.selections[0].holdout is not None
    assert plan.selections[0].holdout.end - plan.selections[0].holdout.start == 365 * DAY
    assert plan.selections[-1].coverage != plan.selections[0].coverage
    wrong = make_plan(source, versions, "a" * 40, plan.research)
    assert all(s.unavailable_reason and not s.windows for s in wrong.selections)
    for changes in (
        {"lookbacks": (30,)},
        {"cost_multipliers": (1,)},
        {"volatility_threshold": 0.5},
        {"selections": plan.selections[:1]},
        {"fixture": False},
        {"common": Period(start=T, end=T + DAY)},
    ):
        with pytest.raises(ValueError):
            CampaignPlan.model_validate(plan.model_dump() | changes)
    selection = plan.selections[0]
    assert selection.coverage is not None
    with pytest.raises(ValueError, match="holdout"):
        Selection.model_validate(
            selection.model_dump()
            | {
                "holdout": Period(
                    start=selection.windows[0].period.start, end=selection.coverage.end
                )
            }
        )
    with pytest.raises(ValueError, match="source"):
        Selection.model_validate(selection.model_dump() | {"dataset": None})


def test_momentum_cli_validation_and_saved_plan(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from crypto_research.config.models import load_config
    from crypto_research.data.storage import Store
    from crypto_research.experiments.cli import main
    from crypto_research.experiments.momentum_plan import make_plan

    root = tmp_path / "research"
    args = ["--root", str(root), "--data-root", str(tmp_path / "data"), "run-momentum"]
    assert main(args + ["--revision", "a" * 40]) == 0
    output = capsys.readouterr().out
    report = Path(output.strip().splitlines()[-1])
    assert main(args + ["--plan", str(report / "plan.json")]) == 0
    assert capsys.readouterr().out
    assert main(args + ["--plan", str(report / "plan.json"), "--fixture"]) == 2
    assert "overrides" in capsys.readouterr().err
    assert main(args + ["--dataset", "OTHER=bad"]) == 2
    assert capsys.readouterr().err
    assert main(args + ["--fixture", "--dataset", "BTC/USDT=bad"]) == 2
    assert "overrides" in capsys.readouterr().err
    plan = make_plan(
        Store(tmp_path / "data"),
        {"BTC/USDT": "bad"},
        "a" * 40,
        load_config("configs/research.yaml"),
        archival=True,
    )
    assert plan.selections[0].unavailable_reason


def test_archival_signals_wait_for_next_open_and_source_remains_observed(tmp_path: Path) -> None:
    from decimal import Decimal

    from crypto_research.backtesting.models import BacktestResult, SimulationSettings
    from crypto_research.data.storage import Store
    from crypto_research.experiments.models import ExperimentSpec
    from crypto_research.experiments.momentum_plan import CampaignPlan
    from crypto_research.experiments.registry import Registry
    from crypto_research.experiments.runner import Runner
    from crypto_research.strategies.time_series_momentum import TimeSeriesMomentum, configuration

    source, registry, plan = compact_plan(tmp_path)
    assert (
        isinstance(source, Store)
        and isinstance(registry, Registry)
        and isinstance(plan, CampaignPlan)
    )
    selection = plan.selections[0]
    assert selection.dataset is not None and selection.holdout is not None
    config = configuration(selection.market, 30)
    spec = ExperimentSpec(
        experiment_id="delayed-momentum",
        strategy=config,
        dataset=selection.dataset,
        code_revision="a" * 40,
        settings=SimulationSettings(
            experiment_id="delayed-momentum",
            period=Period(start=T + 201 * DAY, end=T + 239 * DAY),
            starting_capital=Decimal(10000),
            fee_bps=Decimal(10),
            slippage_bps=Decimal(5),
            shadow=True,
        ),
        split="validation",
        replay_availability="archival-close-plus-60s",
        information_end=selection.holdout.start,
    )
    execution = Runner(source, registry).execute(spec, strategy=TimeSeriesMomentum(config))
    result = BacktestResult.model_validate_json(
        (registry.root / "artifacts" / execution.run_id / "result.json").read_text()
    )
    first = result.decisions[0].instructions[0].signal
    assert first.generated_at == T + 201 * DAY + DAY / 1440
    assert result.fills[0].timestamp == T + 202 * DAY
    assert result.fills[0].reference_price == Decimal("120.2")
    assert result.fills[-1].side == "sell" and all(
        p.quantity == 0 for p in result.equity[-1].positions
    )
    assert result.equity[-1].fees_paid > 0 and result.equity[-1].slippage_paid > 0
    _, rows, _ = source.read(selection.dataset.version)
    assert rows[200].observed_at == T + 201 * DAY
    assert all(
        fill.timestamp >= order.earliest_at
        for fill in result.fills
        for order in result.orders
        if fill.order_id == order.order_id
    )
    assert all(
        point.cash >= 0 and point.equity == point.cash + point.holdings_value
        for point in result.equity
    )


def test_momentum_projection_preserves_other_strategy_parameter_types(tmp_path: Path) -> None:
    from crypto_research.backtesting.models import Instruction, PortfolioTarget
    from crypto_research.contracts import Parameter, StrategyConfig

    source, registry, runner, spec, targets = case(tmp_path)
    config = StrategyConfig.model_validate(
        spec.strategy.model_dump()
        | {
            "parameters": (Parameter(name="lookback", value="calendar-month"),),
        }
    )
    spec = renamed(spec, "generic-string-parameter", strategy=config)
    instruction = Instruction.model_validate(
        targets[0].instructions[0].model_dump() | {"config": config}
    )
    execution = runner.execute(spec, targets=(PortfolioTarget(instructions=(instruction,)),))
    assert execution.manifest_sha256
    rows = runner.artifacts.table(
        execution.run_id, "decisions", expected_sha=execution.manifest_sha256
    ).to_pylist()
    assert rows[0]["lookback"] is None and rows[0]["momentum"] is None
