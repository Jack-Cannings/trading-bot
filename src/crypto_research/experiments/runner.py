"""Narrow isolated-strategy orchestration over exact immutable source versions."""

from collections.abc import Iterator
from decimal import Decimal

import pandas as pd

from crypto_research.backtesting.benchmarks import buy_and_hold
from crypto_research.backtesting.engine import PreparedData, replay_targets, run
from crypto_research.backtesting.models import PortfolioTarget
from crypto_research.contracts import DatasetReference, MarketData
from crypto_research.data.storage import Store
from crypto_research.experiments.models import Execution, ExperimentSpec
from crypto_research.experiments.registry import Registry
from crypto_research.experiments.storage import ResearchStore
from crypto_research.features.snapshots import FeatureCache
from crypto_research.metrics.summary import from_backtest, summarize
from crypto_research.strategies.base import Strategy
from crypto_research.validation.datasets import decisions, outcomes


class Runner:
    def __init__(self, source: Store, registry: Registry) -> None:
        self.source = source
        self.registry = registry
        self.artifacts = ResearchStore(registry.root / "artifacts")
        self.features = FeatureCache()
        self._data: dict[str, PreparedData] = {}
        self._quality: dict[str, str] = {}

    def _load(self, spec: ExperimentSpec) -> PreparedData:
        version = spec.dataset.version
        if version not in self._data:
            manifest, _, report = self.source.read(version)
            expected = DatasetReference(
                source=manifest.provider, version=manifest.version, sha256=manifest.dataset_id
            )
            if expected != spec.dataset or manifest.request.market != spec.strategy.market:
                raise ValueError("explicit dataset version/checksum/market identity mismatch")
            if report.status == "invalid" or (
                report.status != "clean" and not spec.allow_data_gaps
            ):
                raise ValueError("source dataset has unresolved quality anomalies")
            history = self.source.load(
                version,
                market=spec.strategy.market,
                period=manifest.request.period,
                as_of=manifest.acquired_at,
                allow_gaps=spec.allow_data_gaps,
            )
            self.features.register(history, spec.dataset)
            self._data[version] = PreparedData((history,))
            self._quality[version] = report.status
        if self._quality[version] != "clean" and not spec.allow_data_gaps:
            raise ValueError("source dataset has unresolved quality anomalies")
        data = self._data[version]
        history = data.histories[0]
        if history.market != spec.strategy.market or history.candles[0].dataset != spec.dataset:
            raise ValueError("cached dataset identity differs from requested immutable version")
        return data

    def execute(
        self,
        spec: ExperimentSpec,
        *,
        strategy: Strategy | None = None,
        targets: tuple[PortfolioTarget, ...] | None = None,
    ) -> Execution:
        self.registry.register(spec)
        execution = self.registry.create_run(spec.experiment_id)
        self.registry.transition(execution.run_id, "running")
        try:
            data = self._load(spec)
            if strategy is not None and targets is not None:
                raise ValueError("provide either a strategy or precomputed targets")
            if spec.mode == "strategy":
                if strategy is not None:
                    if strategy.config != spec.strategy:
                        raise ValueError(
                            "strategy configuration differs from registered experiment"
                        )

                    def snapshots() -> Iterator[MarketData]:
                        cutoffs = sorted(
                            {
                                c.available_at
                                for c in data.histories[0].candles
                                if spec.settings.period.start
                                <= c.available_at
                                < spec.settings.period.end
                            }
                        )
                        if not cutoffs:
                            raise ValueError(
                                "no decision-time data available; "
                                "historical availability is not backdated"
                            )
                        for cutoff in cutoffs:
                            yield self.features.history_at(spec.dataset, cutoff)

                    targets = replay_targets(strategy, snapshots())
                elif targets is None:
                    raise ValueError("strategy experiments require a strategy or explicit targets")
                if any(
                    i.config != spec.strategy for target in targets for i in target.instructions
                ):
                    raise ValueError(
                        "isolated experiment cannot attribute another strategy's trades"
                    )
            else:
                if strategy is not None or targets is not None:
                    raise ValueError("passive benchmark experiments do not accept external signals")
                targets = buy_and_hold(
                    data,
                    spec.settings,
                    () if spec.mode == "cash" else ((spec.strategy.market, Decimal(1)),),
                )
                if any(
                    i.config != spec.strategy for target in targets for i in target.instructions
                ):
                    raise ValueError(
                        "benchmark configuration differs from registered implementation"
                    )
            result = run(data, targets, spec.settings)
            metrics = summarize(from_backtest(result), spec.metrics, result)
            inputs = decisions(spec, result, execution.run_id, self.features)
            labels = outcomes(spec, result, inputs)
            self.artifacts.publish(spec, execution.run_id, result, metrics, inputs, labels)
            return self.registry.complete(execution.run_id)
        except (KeyboardInterrupt, SystemExit):
            if self.registry.get(execution.run_id).status in {"created", "running"}:
                self.registry.transition(execution.run_id, "cancelled", "execution interrupted")
            raise
        except Exception as exc:
            if self.registry.get(execution.run_id).status in {"created", "running"}:
                self.registry.transition(execution.run_id, "failed", f"{type(exc).__name__}: {exc}")
            raise


def compare(registry: Registry) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    intervals = []
    for execution in registry.list():
        spec = registry.spec(execution.experiment_id)
        metrics = registry.summary(execution.run_id) if execution.status == "completed" else None
        rows.append(
            dict(
                experiment_id=spec.experiment_id,
                run_id=execution.run_id,
                strategy=spec.strategy.strategy_id,
                configuration=spec.strategy.identity,
                market=spec.strategy.market.pair,
                timeframe=spec.strategy.market.timeframe,
                net_return=metrics.total_net_return if metrics else None,
                cagr=metrics.cagr if metrics else None,
                sharpe=metrics.sharpe_ratio if metrics else None,
                sortino=metrics.sortino_ratio if metrics else None,
                calmar=metrics.calmar_ratio if metrics else None,
                max_drawdown=metrics.maximum_drawdown if metrics else None,
                cvar=metrics.cvar_95 if metrics else None,
                trades=metrics.number_of_trades if metrics else None,
                costs=(metrics.total_fees + metrics.total_slippage)
                if metrics and metrics.total_fees is not None and metrics.total_slippage is not None
                else None,
                exposure=metrics.average_exposure if metrics else None,
                status=execution.status,
                comparison_group=spec.comparison_group,
                overlapping_evaluation=False,
            )
        )
        intervals.append((spec.settings.period.start, spec.settings.period.end, len(rows) - 1))
    ordered = sorted(intervals)
    maximum_end = None
    for index, (start, end, row) in enumerate(ordered):
        prior = maximum_end is not None and start < maximum_end
        following = index + 1 < len(ordered) and ordered[index + 1][0] < end
        rows[row]["overlapping_evaluation"] = prior or following
        maximum_end = max(maximum_end, end) if maximum_end is not None else end
    return pd.DataFrame.from_records(rows)
