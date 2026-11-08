import sqlite3
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError
from research_helpers import DAY, M, T, case, renamed

from crypto_research.backtesting.engine import run
from crypto_research.backtesting.models import BacktestResult, Instruction, PortfolioTarget
from crypto_research.contracts import DatasetReference, StrategyConfig, TargetExposure
from crypto_research.experiments.models import ExperimentSpec
from crypto_research.experiments.registry import Registry
from crypto_research.experiments.runner import compare
from crypto_research.experiments.storage import ResearchStore
from crypto_research.features.snapshots import FeatureValue
from crypto_research.metrics.summary import from_backtest, summarize
from crypto_research.validation.datasets import DecisionRecord, OutcomeRecord, validate_label_split


def test_repeatable_runs_distinct_execution_and_artifact_identity(tmp_path: Path) -> None:
    _, registry, runner, spec, targets = case(tmp_path)
    first = runner.execute(spec, targets=targets)
    second = runner.execute(spec, targets=targets)
    assert first.status == second.status == "completed"
    assert first.run_id != second.run_id and first.content_id == second.content_id
    assert first.manifest_sha256 != second.manifest_sha256
    assert len(registry.list("trial")) == 2
    assert registry.summary(first.run_id) == registry.summary(second.run_id)
    assert first.manifest_sha256 is not None
    store = ResearchStore(registry.root / "artifacts")
    table = store.table(first.run_id, "decisions", expected_sha=first.manifest_sha256)
    row = table.to_pylist()[0]
    assert (
        row["strategy_id"] == spec.strategy.strategy_id
        and row["dataset_version"] == spec.dataset.version
    )
    assert row["timestamp"] == T + DAY and row["preceding_exposure"] == 0
    assert "net_return" not in table.column_names and "gross_return" not in table.column_names
    assert "return_7d" in table.column_names and row["return_7d"] is None
    assert row["return_7d_missing"] == "insufficient_history"
    assert table.schema.field("timestamp").type.tz == "UTC"
    selected = store.table(
        first.run_id,
        "performance",
        expected_sha=first.manifest_sha256,
        strategy_id=spec.strategy.strategy_id,
        pair=M.pair,
        start=T + 2 * DAY,
        end=T + 3 * DAY,
    )
    assert selected.num_rows == 1 and selected.to_pylist()[0]["scope"] == "standalone_shadow"
    frame = compare(registry)
    assert len(frame) == 2 and frame.comparison_group.nunique() == 1
    assert frame.overlapping_evaluation.all()


def test_execution_lifecycle_and_immutable_configs(tmp_path: Path) -> None:
    _, registry, _, spec, _ = case(tmp_path)
    registry.register(spec)
    registry.register(spec)
    execution = registry.create_run(spec.experiment_id)
    assert execution.status == "created"
    registry.transition(execution.run_id, "running")
    changed = StrategyConfig(strategy_id="scripted", version="2", market=M)
    with pytest.raises(ValueError, match="immutable"):
        registry.register(ExperimentSpec.model_validate(spec.model_dump() | {"strategy": changed}))
    with pytest.raises(ValueError, match="verified"):
        registry.transition(execution.run_id, "completed")
    with pytest.raises(ValueError, match="reason"):
        registry.transition(execution.run_id, "failed")
    assert registry.recover_interrupted(execution.run_id).status == "cancelled"
    with pytest.raises(ValueError, match="lifecycle"):
        registry.transition(execution.run_id, "running")
    restarted = Registry(registry.root)
    assert restarted.get(execution.run_id).failure_reason is not None
    with pytest.raises(KeyError):
        restarted.spec("does-not-exist")


def test_failure_is_preserved_and_next_attempt_can_succeed(tmp_path: Path) -> None:
    _, registry, runner, spec, targets = case(tmp_path)
    with pytest.raises(ValueError, match="require a strategy"):
        runner.execute(spec)
    failed = registry.list()[0]
    assert failed.status == "failed" and "ValueError" in (failed.failure_reason or "")
    good = runner.execute(spec, targets=targets)
    assert good.status == "completed" and registry.get(failed.run_id) == failed
    assert set(compare(registry).status) == {"failed", "completed"}


def test_wrong_dataset_checksum_rejected_including_cache_hit(tmp_path: Path) -> None:
    _, registry, runner, spec, targets = case(tmp_path)
    runner.execute(spec, targets=targets)
    wrong = DatasetReference(
        source=spec.dataset.source, version=spec.dataset.version, sha256="b" * 64
    )
    bad = renamed(spec, "wrong", dataset=wrong)
    with pytest.raises(ValueError, match="identity"):
        runner.execute(bad, targets=targets)
    assert registry.list("wrong")[0].status == "failed"


def test_atomic_artifact_failure_is_not_completed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, registry, runner, spec, targets = case(tmp_path)

    def fail(*args: object, **kwargs: object) -> None:
        raise OSError("disk failed during Parquet write")

    monkeypatch.setattr("crypto_research.experiments.storage.pq.write_table", fail)
    with pytest.raises(OSError, match="disk failed"):
        runner.execute(spec, targets=targets)
    execution = registry.list()[0]
    assert execution.status == "failed"
    assert not runner.artifacts.path(execution.run_id).exists()
    assert not list(runner.artifacts.root.glob(".staging-*"))


def test_database_completion_rollback(tmp_path: Path) -> None:
    _, registry, runner, spec, targets = case(tmp_path)
    with registry.transaction() as db:
        db.executescript("""CREATE TRIGGER block_completion BEFORE UPDATE ON executions
            WHEN NEW.status='completed' BEGIN SELECT RAISE(ABORT,'commit blocked'); END;""")
    with pytest.raises(sqlite3.IntegrityError, match="commit blocked"):
        runner.execute(spec, targets=targets)
    execution = registry.list()[0]
    assert execution.status == "failed"
    with registry.transaction() as db:
        assert db.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0] == 0
    assert runner.artifacts.path(execution.run_id).exists()
    assert compare(registry).iloc[0].status == "failed"


def test_interrupt_record_and_no_artifact_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, registry, runner, spec, targets = case(tmp_path)

    def interrupt(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt()

    monkeypatch.setattr(runner.artifacts, "publish", interrupt)
    with pytest.raises(KeyboardInterrupt):
        runner.execute(spec, targets=targets)
    assert registry.list()[0].status == "cancelled"


def test_half_open_outcomes_include_entry_exclude_end_exit(tmp_path: Path) -> None:
    _, registry, runner, spec, targets = case(tmp_path)
    sell = PortfolioTarget(
        instructions=(
            Instruction(
                config=spec.strategy,
                signal=TargetExposure(
                    strategy_id=spec.strategy.strategy_id,
                    market=M,
                    data_through=T + 2 * DAY,
                    generated_at=T + 2 * DAY,
                    effective_at=T + 2 * DAY,
                    weight=0.0,
                ),
            ),
        )
    )
    execution = runner.execute(spec, targets=(*targets, sell))
    assert execution.manifest_sha256 is not None
    store = runner.artifacts
    labels = store.table(
        execution.run_id, "outcomes", expected_sha=execution.manifest_sha256
    ).to_pylist()
    buy_one = next(o for o in labels if o["timestamp"] == T + DAY and o["horizon_days"] == 1)
    sell_one = next(o for o in labels if o["timestamp"] == T + 2 * DAY and o["horizon_days"] == 1)
    assert buy_one["gross_return"] == 0
    assert buy_one["net_return"] == pytest.approx(-1 / 101)
    assert sell_one["net_return"] == pytest.approx(-0.01)
    assert Decimal(buy_one["fees"]) == pytest.approx(Decimal("9.900990099009900990"))
    assert buy_one["scope"] == "standalone_shadow"
    buy_seven = next(o for o in labels if o["timestamp"] == T + DAY and o["horizon_days"] == 7)
    assert buy_seven["net_return"] == pytest.approx(-2 / 101)
    performance = store.table(
        execution.run_id, "performance", expected_sha=execution.manifest_sha256
    ).to_pylist()
    compounded = 1.0
    for row in performance:
        if row["net_return"] is not None:
            compounded *= 1 + float(row["net_return"])
    assert compounded - 1 == pytest.approx(
        float(Decimal(performance[-1]["equity"]) / Decimal(1000) - 1), abs=1e-14
    )
    assert float(performance[1]["net_return"]) == pytest.approx(-2 / 101)
    incomplete = next(o for o in labels if o["horizon_days"] == 30)
    assert incomplete["status"] == "incomplete" and incomplete["net_return"] is None
    assert incomplete["fees"] is None and incomplete["available_at"] is None


def test_deliberate_label_leakage_and_overlap_rejected(tmp_path: Path) -> None:
    _, registry, runner, spec, targets = case(tmp_path)
    execution = runner.execute(spec, targets=targets)
    assert execution.manifest_sha256 is not None
    decisions = runner.artifacts.table(
        execution.run_id, "decisions", expected_sha=execution.manifest_sha256
    )
    labels = runner.artifacts.table(
        execution.run_id, "outcomes", expected_sha=execution.manifest_sha256
    )
    decision = DecisionRecord.model_validate_json(decisions.to_pylist()[0]["record_json"])
    outcomes = tuple(
        OutcomeRecord.model_validate_json(r["record_json"]) for r in labels.to_pylist()
    )
    with pytest.raises(ValidationError):
        DecisionRecord.model_validate(
            decision.model_dump() | {"future_return": outcomes[0].net_return}
        )
    with pytest.raises(ValidationError):
        FeatureValue.model_validate(dict(name="return_7d", observation=outcomes[0]))
    with pytest.raises(ValueError, match="leak across"):
        validate_label_split((outcomes[0],), (outcomes[1],))
    adjacent = OutcomeRecord.model_validate(
        outcomes[0].model_dump()
        | {
            "start": outcomes[0].end,
            "end": outcomes[0].end + DAY,
            "available_at": outcomes[0].end + DAY,
        }
    )
    validate_label_split((outcomes[0],), (adjacent,))
    with pytest.raises(ValidationError, match="cutoff"):
        DecisionRecord.model_validate(
            decision.model_dump() | {"data_available_at": decision.signal.generated_at + DAY}
        )
    with pytest.raises(ValidationError, match="fully observed"):
        OutcomeRecord.model_validate(outcomes[0].model_dump() | {"available_at": T})


def test_artifact_corruption_and_no_overwrite(tmp_path: Path) -> None:
    _, registry, runner, spec, targets = case(tmp_path)
    execution = runner.execute(spec, targets=targets)
    root = runner.artifacts.path(execution.run_id)
    with pytest.raises(FileExistsError):
        result = BacktestResult.model_validate_json((root / "result.json").read_text())
        metrics = registry.summary(execution.run_id)
        runner.artifacts.publish(spec, execution.run_id, result, metrics, (), ())
    path = root / "metrics.json"
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        registry.summary(execution.run_id)
    with pytest.raises(ValueError, match="checksum"):
        runner.artifacts.verify(execution.run_id, execution.manifest_sha256)
    with pytest.raises(ValueError, match="UUID"):
        runner.artifacts.path("../outside")


def test_comparison_flags_different_assumptions(tmp_path: Path) -> None:
    _, registry, runner, spec, targets = case(tmp_path)
    runner.execute(spec, targets=targets)
    other = renamed(spec, "more-fees")
    other = ExperimentSpec.model_validate(
        other.model_dump() | {"settings": other.settings.model_dump() | {"fee_bps": Decimal(200)}}
    )
    runner.execute(other, targets=targets)
    table = compare(registry)
    assert table.comparison_group.nunique() == 2 and table.overlapping_evaluation.all()
    assert table.net_return.iloc[1] < table.net_return.iloc[0]
    result = BacktestResult.model_validate_json(
        (runner.artifacts.path(registry.list()[0].run_id) / "result.json").read_text()
    )
    changed = BacktestResult.model_validate_json(
        (runner.artifacts.path(registry.list()[1].run_id) / "result.json").read_text()
    )
    with pytest.raises(ValueError, match="do not match"):
        summarize(from_backtest(result), spec.metrics, changed)


def test_missing_intervals_do_not_receive_complete_outcomes(tmp_path: Path) -> None:
    from crypto_research.data.validation import validate
    from crypto_research.validation.datasets import decisions, outcomes

    source, _, runner, spec, targets = case(tmp_path)
    manifest, original, _ = source.read(spec.dataset.version)
    missing = tuple(row for row in original if row.open_time != T + 5 * DAY)
    revised = source.save(validate(manifest.request, missing), "fixture", manifest.acquired_at)
    reference = DatasetReference(
        source="fixture", version=revised.version, sha256=revised.dataset_id
    )
    permissive = renamed(spec, "gapped", dataset=reference, allow_data_gaps=True)
    data = runner._load(permissive)
    permissive_target = PortfolioTarget(
        instructions=(Instruction(config=spec.strategy, signal=targets[0].instructions[0].signal),)
    )
    result = run(data, (permissive_target,), permissive.settings)
    records = decisions(permissive, result, "1" * 32, runner.features)
    labels = outcomes(permissive, result, records)
    assert next(label for label in labels if label.horizon_days == 7).status == "incomplete"
    strict = renamed(permissive, "strict", allow_data_gaps=False)
    with pytest.raises(ValueError, match="quality"):
        runner._load(strict)


def test_semantically_forged_labels_cannot_complete(tmp_path: Path) -> None:
    from crypto_research.validation.datasets import decisions, outcomes

    _, registry, runner, spec, targets = case(tmp_path)
    registry.register(spec)
    execution = registry.create_run(spec.experiment_id)
    registry.transition(execution.run_id, "running")
    result = run(runner._load(spec), targets, spec.settings)
    metrics = summarize(from_backtest(result), spec.metrics, result)
    records = decisions(spec, result, execution.run_id, runner.features)
    labels = outcomes(spec, result, records)
    fake = OutcomeRecord.model_validate(labels[0].model_dump() | {"net_return": 0.99})
    runner.artifacts.publish(spec, execution.run_id, result, metrics, records, (fake, *labels[1:]))
    with pytest.raises(ValueError, match="simulated strategy performance"):
        registry.complete(execution.run_id)
    assert registry.get(execution.run_id).status == "running"
    with registry.transaction() as db:
        assert db.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0] == 0


def test_source_quality_is_checked_before_cached_reuse(tmp_path: Path) -> None:
    # The explicit cache test above uses data gaps; invalid anomalies never load at all.
    from crypto_research.data.models import Issue
    from crypto_research.data.validation import validate

    source, registry, runner, spec, targets = case(tmp_path)
    manifest, rows, _ = source.read(spec.dataset.version)
    invalid = source.save(
        validate(
            manifest.request,
            rows,
            (Issue(kind="invalid", open_time=T, message="conflicting source vintage"),),
        ),
        "fixture",
        manifest.acquired_at,
    )
    bad = renamed(
        spec,
        "invalid",
        dataset=DatasetReference(
            source="fixture", version=invalid.version, sha256=invalid.dataset_id
        ),
        allow_data_gaps=True,
    )
    with pytest.raises(ValueError, match="quality"):
        runner.execute(bad, targets=targets)
    assert registry.list("invalid")[0].status == "failed"


def test_four_hour_outcomes_and_periodic_shadow_records(tmp_path: Path) -> None:
    from datetime import timedelta

    from crypto_research.backtesting.models import SimulationSettings
    from crypto_research.contracts import Market, Period
    from crypto_research.data.models import Observation, Request
    from crypto_research.data.storage import Store
    from crypto_research.data.validation import validate
    from crypto_research.experiments.runner import Runner
    from crypto_research.metrics.summary import MetricSettings

    step = timedelta(hours=4)
    market = Market(exchange="fixture", pair="BTC/USDT", timeframe="4h")
    source = Store(tmp_path / "data")
    rows = tuple(
        Observation(
            open_time=T + i * step,
            observed_at=T + (i + 1) * step,
            open=100.0,
            high=100.0,
            low=100.0,
            close=100.0,
            volume=10.0,
        )
        for i in range(72)
    )
    query = Request(market=market, period=Period(start=T, end=T + 72 * step))
    saved = source.save(validate(query, rows), "fixture", T + 72 * step)
    cfg = StrategyConfig(strategy_id="scripted", version="1", market=market)
    settings = SimulationSettings(
        experiment_id="four-hour",
        period=Period(start=T + step, end=T + 72 * step),
        starting_capital=Decimal(1000),
        fee_bps=Decimal(100),
        slippage_bps=Decimal(0),
        max_price_age_seconds=14400,
        shadow=True,
    )
    spec = ExperimentSpec(
        experiment_id="four-hour",
        strategy=cfg,
        dataset=DatasetReference(source="fixture", version=saved.version, sha256=saved.dataset_id),
        settings=settings,
        code_revision="a" * 40,
        split="test",
        metrics=MetricSettings(frequency="4h"),
    )
    target = PortfolioTarget(
        instructions=(
            Instruction(
                config=cfg,
                signal=TargetExposure(
                    strategy_id=cfg.strategy_id,
                    market=market,
                    data_through=T + step,
                    generated_at=T + step,
                    effective_at=T + step,
                    weight=1.0,
                ),
            ),
        )
    )
    registry = Registry(tmp_path / "research")
    runner = Runner(source, registry)
    execution = runner.execute(spec, targets=(target,))
    assert execution.manifest_sha256 is not None
    labels = runner.artifacts.table(
        execution.run_id, "outcomes", expected_sha=execution.manifest_sha256
    ).to_pylist()
    one = next(row for row in labels if row["horizon_days"] == 1)
    assert one["end"] - one["timestamp"] == DAY and one["status"] == "complete"
    assert one["net_return"] == pytest.approx(-1 / 101)
    performance = runner.artifacts.table(
        execution.run_id, "performance", expected_sha=execution.manifest_sha256
    ).to_pylist()
    assert performance[0]["period_start"] is None and performance[0]["net_return"] is None
    assert performance[1]["period_start"] == performance[0]["timestamp"]
    assert float(performance[1]["net_return"]) == pytest.approx(-1 / 101)
    assert all(row["available_at"] == row["timestamp"] for row in performance)
    assert registry.summary(execution.run_id).periodic_observations == 71


def test_standalone_outcome_builder_rejects_shared_portfolio(tmp_path: Path) -> None:
    from crypto_research.validation.datasets import outcomes

    _, _, runner, spec, targets = case(tmp_path)
    result = run(runner._load(spec), targets, spec.settings)
    shared = BacktestResult.model_validate(
        result.model_dump() | {"data_identities": ("asset-one", "asset-two")}
    )
    with pytest.raises(ValueError, match="standalone"):
        outcomes(spec, shared, ())


def test_semantically_forged_metrics_cannot_complete(tmp_path: Path) -> None:
    from crypto_research.metrics.summary import PerformanceSummary
    from crypto_research.validation.datasets import decisions, outcomes

    _, registry, runner, spec, targets = case(tmp_path)
    registry.register(spec)
    execution = registry.create_run(spec.experiment_id)
    registry.transition(execution.run_id, "running")
    result = run(runner._load(spec), targets, spec.settings)
    metrics = summarize(from_backtest(result), spec.metrics, result)
    fake = PerformanceSummary.model_validate(metrics.model_dump() | {"total_net_return": 10.0})
    records = decisions(spec, result, execution.run_id, runner.features)
    runner.artifacts.publish(
        spec, execution.run_id, result, fake, records, outcomes(spec, result, records)
    )
    with pytest.raises(ValueError, match="metrics do not match"):
        registry.complete(execution.run_id)


def test_late_source_publication_is_not_backdated_for_strategy(tmp_path: Path) -> None:
    from crypto_research.contracts import MarketData, SignalBatch
    from crypto_research.data.validation import validate

    source, registry, runner, spec, _ = case(tmp_path)
    manifest, rows, _ = source.read(spec.dataset.version)
    delayed = tuple(
        type(row).model_validate(row.model_dump() | {"observed_at": manifest.acquired_at})
        for row in rows
    )
    updated = source.save(validate(manifest.request, delayed), "fixture", manifest.acquired_at)
    declared = renamed(
        spec,
        "delayed",
        dataset=DatasetReference(
            source="fixture", version=updated.version, sha256=updated.dataset_id
        ),
    )

    class Dummy:
        config = spec.strategy

        def generate(self, history: MarketData) -> SignalBatch:
            pytest.fail("strategy must not receive historically unavailable data")

    with pytest.raises(ValueError, match="not backdated"):
        runner.execute(declared, strategy=Dummy())
    assert registry.list("delayed")[0].status == "failed"


def test_query_filters_and_live_input_type_refuse_outcomes(tmp_path: Path) -> None:
    from crypto_research.contracts import DecisionInputs

    _, registry, runner, spec, targets = case(tmp_path)
    execution = runner.execute(spec, targets=targets)
    assert registry.list(strategy_id=spec.strategy.strategy_id, pair=M.pair, timeframe="1d") == (
        execution,
    )
    assert not registry.list(timeframe="4h")
    assert execution.manifest_sha256 is not None
    labels = runner.artifacts.table(
        execution.run_id, "outcomes", expected_sha=execution.manifest_sha256
    )
    label = OutcomeRecord.model_validate_json(labels.to_pylist()[0]["record_json"])
    history = runner.features.history_at(spec.dataset, T + DAY)
    with pytest.raises(ValidationError):
        DecisionInputs.model_validate(dict(history=history, features=(label,)))
    table = runner.artifacts.table(
        execution.run_id,
        "decisions",
        expected_sha=execution.manifest_sha256,
        exchange="fixture",
        timeframe="4h",
    )
    assert table.num_rows == 0
    with registry.transaction() as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 1
