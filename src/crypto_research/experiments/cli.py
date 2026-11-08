"""Offline synthetic research pipeline and simple local registry inspection."""

import argparse
import json
import sqlite3
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from crypto_research.backtesting.models import SimulationSettings
from crypto_research.contracts import (
    DatasetReference,
    Market,
    MarketData,
    Parameter,
    Period,
    SignalBatch,
    StrategyConfig,
    TargetExposure,
)
from crypto_research.data.models import Observation, Request
from crypto_research.data.storage import Store
from crypto_research.data.validation import validate
from crypto_research.experiments.models import ExperimentSpec
from crypto_research.experiments.registry import Registry
from crypto_research.experiments.runner import Runner, compare
from crypto_research.experiments.storage import ResearchStore


class ScriptedDemo:
    """One predetermined fixture instruction, independent of observed prices."""

    def __init__(self, config: StrategyConfig, decision: datetime) -> None:
        self.config, self.decision = config, decision

    def generate(self, history: MarketData) -> SignalBatch:
        signals: tuple[TargetExposure, ...] = ()
        if history.as_of == self.decision:
            signals = (
                TargetExposure(
                    strategy_id=self.config.strategy_id,
                    market=self.config.market,
                    data_through=history.candles[-1].close_time,
                    generated_at=history.as_of,
                    effective_at=history.as_of,
                    weight=0.5,
                ),
            )
        return SignalBatch(config=self.config, history=history, signals=signals)


def run_demo(registry: Registry, source: Store, revision: str) -> list[dict[str, object]]:
    start = datetime(2023, 1, 1, tzinfo=UTC)
    day = timedelta(days=1)
    market = Market(exchange="synthetic", pair="BTC/USDT", timeframe="1d")
    rows = tuple(
        Observation(
            open_time=start + i * day,
            observed_at=start + (i + 1) * day,
            open=100.0 if i < 201 else 100 + (i - 200) * 0.5,
            high=100.0 if i < 201 else 100 + (i - 200) * 0.5,
            low=100.0 if i < 201 else 100 + (i - 200) * 0.5,
            close=100.0 if i < 201 else 100 + (i - 200) * 0.5,
            volume=100.0,
        )
        for i in range(241)
    )
    request = Request(market=market, period=Period(start=start, end=start + 241 * day))
    manifest = source.save(validate(request, rows), "synthetic-fixture", start + 241 * day)
    dataset = DatasetReference(
        source=manifest.provider, version=manifest.version, sha256=manifest.dataset_id
    )
    runner = Runner(source, registry)
    output: list[dict[str, object]] = []
    for mode in ("strategy", "cash", "buy_hold"):
        experiment_id = f"demo-{revision[:8]}-{mode}"
        config = StrategyConfig(
            strategy_id="scripted-demo"
            if mode == "strategy"
            else "cash"
            if mode == "cash"
            else "benchmark",
            version="1",
            market=market,
            parameters=(Parameter(name="decision_at", value=(start + 201 * day).isoformat()),)
            if mode == "strategy"
            else (Parameter(name="weight", value=1.0),)
            if mode == "buy_hold"
            else (),
        )
        settings = SimulationSettings(
            experiment_id=experiment_id,
            period=Period(start=start + 201 * day, end=start + 241 * day),
            starting_capital=Decimal(1000),
            fee_bps=Decimal(10),
            slippage_bps=Decimal(5),
            shadow=True,
        )
        spec = ExperimentSpec.model_validate(
            dict(
                experiment_id=experiment_id,
                strategy=config,
                dataset=dataset,
                code_revision=revision,
                settings=settings,
                split="test",
                mode=mode,
            )
        )
        execution = runner.execute(
            spec,
            strategy=ScriptedDemo(config, settings.period.start) if mode == "strategy" else None,
        )
        output.append(
            dict(
                experiment_id=experiment_id,
                run_id=execution.run_id,
                status=execution.status,
                content_id=execution.content_id,
                metrics=registry.summary(execution.run_id).model_dump(
                    mode="json", exclude={"cumulative"}
                ),
            )
        )
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Offline research trials and comparison; returns are decimal fractions"
    )
    parser.add_argument("--root", type=Path, default=Path("research"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("run-demo")
    demo.add_argument("--revision", help="declared code SHA; defaults to current Git HEAD")
    momentum = commands.add_parser(
        "run-momentum", help="offline fixed momentum campaign; never downloads"
    )
    momentum.add_argument("--revision", help="declared code SHA; defaults to current Git HEAD")
    momentum.add_argument("--config", type=Path, default=Path("configs/research.yaml"))
    momentum.add_argument("--dataset", action="append", default=[], metavar="PAIR=VERSION")
    momentum.add_argument(
        "--archival-close-plus-60s",
        action="store_true",
        help="assume finalized daily bars published after 60s; preserve raw observation timestamps",
    )
    momentum.add_argument(
        "--fixture",
        action="store_true",
        help="explicit synthetic campaign, not real market evidence",
    )
    momentum.add_argument("--plan", type=Path, help="replay an immutable saved campaign plan")
    commands.add_parser("list")
    commands.add_parser("compare")
    show = commands.add_parser("show")
    show.add_argument("experiment_id")
    recover = commands.add_parser("recover")
    recover.add_argument("run_id")
    args = parser.parse_args(argv)
    try:
        registry = Registry(args.root)
        if args.command == "run-momentum":
            from crypto_research.config.models import load_config
            from crypto_research.experiments.momentum_campaign import execute_campaign
            from crypto_research.experiments.momentum_fixtures import fixture_versions
            from crypto_research.experiments.momentum_plan import PAIRS, CampaignPlan, make_plan

            source = Store(args.data_root)
            if args.plan:
                if args.dataset or args.fixture or args.archival_close_plus_60s or args.revision:
                    raise ValueError("saved plan cannot be combined with parameter overrides")
                plan = CampaignPlan.model_validate_json(args.plan.read_text(encoding="utf-8"))
            else:
                revision = (
                    args.revision
                    or subprocess.run(
                        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
                    ).stdout.strip()
                )
                versions = {}
                for value in args.dataset:
                    pair, separator, version = value.partition("=")
                    if not separator or pair not in PAIRS or pair in versions:
                        raise ValueError(
                            "datasets must be unique PAIR=VERSION entries for BTC, ETH and SOL/USDT"
                        )
                    versions[pair] = version
                if args.fixture:
                    if versions:
                        raise ValueError("synthetic fixture mode cannot use real dataset overrides")
                    versions = fixture_versions(source)
                plan = make_plan(
                    source,
                    versions,
                    revision,
                    load_config(args.config),
                    archival=args.archival_close_plus_60s,
                    fixture=args.fixture,
                )
            print(execute_campaign(plan, source, registry))
        elif args.command == "run-demo":
            revision = args.revision
            if revision is None:
                revision = subprocess.run(
                    ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
                ).stdout.strip()
            print(json.dumps(run_demo(registry, Store(args.data_root), revision), sort_keys=True))
        elif args.command == "list":
            print(json.dumps([r.model_dump(mode="json") for r in registry.list()], sort_keys=True))
        elif args.command == "compare":
            table = compare(registry)
            print(
                table.to_string(index=False, na_rep="undefined")
                if not table.empty
                else "No experiment executions recorded."
            )
        elif args.command == "recover":
            print(registry.recover_interrupted(args.run_id).canonical_json())
        else:
            spec = registry.spec(args.experiment_id)
            runs = registry.list(args.experiment_id)
            for execution in runs:
                if execution.status == "completed":
                    ResearchStore(args.root / "artifacts").verify(
                        execution.run_id, execution.manifest_sha256
                    )
            print(
                json.dumps(
                    dict(
                        spec=spec.model_dump(mode="json"),
                        input_hash=spec.input_hash,
                        comparison_group=spec.comparison_group,
                        runs=[
                            dict(
                                execution=r.model_dump(mode="json"),
                                metrics=registry.summary(r.run_id).model_dump(
                                    mode="json", exclude={"cumulative"}
                                )
                                if r.status == "completed"
                                else None,
                            )
                            for r in runs
                        ],
                    ),
                    sort_keys=True,
                )
            )
        return 0
    except KeyboardInterrupt:
        print("research execution cancelled", file=sys.stderr)
        return 130
    except (ValueError, KeyError, OSError, sqlite3.Error, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
