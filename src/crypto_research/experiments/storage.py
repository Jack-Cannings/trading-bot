"""Immutable research bundles: fixed Parquet schemas, checksums and atomic publication."""

import hashlib
import json
import os
import re
import shutil
import tempfile
from datetime import datetime
from decimal import Context, Decimal, localcontext
from pathlib import Path
from typing import Literal

import pyarrow as pa
import pyarrow.parquet as pq

from crypto_research.backtesting.accounting import amount
from crypto_research.backtesting.models import BacktestResult
from crypto_research.experiments.models import ArtifactFile, ExperimentSpec, ResearchManifest
from crypto_research.features.snapshots import FEATURES
from crypto_research.metrics.summary import PerformanceSummary, from_backtest, summarize
from crypto_research.validation.datasets import (
    DecisionRecord,
    OutcomeRecord,
)
from crypto_research.validation.datasets import (
    outcomes as build_outcomes,
)

Kind = Literal["decisions", "outcomes", "performance", "trades"]
COMMON = [
    (name, pa.string())
    for name in (
        "experiment_id",
        "run_id",
        "strategy_id",
        "strategy_version",
        "config_hash",
        "exchange",
        "pair",
        "timeframe",
        "dataset_version",
        "dataset_checksum",
    )
]
COMMON.append(("simulated", pa.bool_()))
TIME = pa.timestamp("us", tz="UTC")
SCHEMAS = {
    "decisions": pa.schema(
        COMMON
        + [
            ("decision_id", pa.string()),
            ("timestamp", TIME),
            ("data_through", TIME),
            ("available_at", TIME),
            ("effective_at", TIME),
            ("target_exposure", pa.float64()),
            ("preceding_exposure", pa.float64()),
            ("signal_strength", pa.float64()),
            ("feature_schema", pa.int32()),
            ("record_json", pa.string()),
        ]
        + [(name, pa.float64()) for name in FEATURES]
        + [(name + "_missing", pa.string()) for name in FEATURES],
        metadata={b"schema_version": b"1", b"kind": b"decisions"},
    ),
    "outcomes": pa.schema(
        COMMON
        + [
            ("decision_id", pa.string()),
            ("timestamp", TIME),
            ("end", TIME),
            ("available_at", TIME),
            ("horizon_days", pa.int32()),
            ("gross_return", pa.float64()),
            ("net_return", pa.float64()),
            ("fees", pa.string()),
            ("slippage", pa.string()),
            ("average_exposure", pa.float64()),
            ("status", pa.string()),
            ("scope", pa.string()),
            ("record_json", pa.string()),
        ],
        metadata={b"schema_version": b"1", b"kind": b"outcomes"},
    ),
    "performance": pa.schema(
        COMMON
        + [
            ("timestamp", TIME),
            ("period_start", TIME),
            ("available_at", TIME),
            ("cash", pa.string()),
            ("equity", pa.string()),
            ("gross_equity", pa.string()),
            ("net_return", pa.string()),
            ("gross_return", pa.string()),
            ("cumulative_return", pa.string()),
            ("gross_cumulative_return", pa.string()),
            ("fees_cumulative", pa.string()),
            ("slippage_cumulative", pa.string()),
            ("turnover_cumulative", pa.string()),
            ("quantity", pa.string()),
            ("target_exposure", pa.float64()),
            ("actual_exposure", pa.float64()),
            ("fees", pa.string()),
            ("slippage", pa.string()),
            ("turnover", pa.string()),
            ("scope", pa.string()),
        ],
        metadata={b"schema_version": b"1", b"kind": b"performance"},
    ),
    "trades": pa.schema(
        COMMON
        + [
            ("timestamp", TIME),
            ("order_id", pa.string()),
            ("fill_id", pa.string()),
            ("decision_at", TIME),
            ("effective_at", TIME),
            ("side", pa.string()),
            ("quantity", pa.string()),
            ("reference_price", pa.string()),
            ("execution_price", pa.string()),
            ("notional", pa.string()),
            ("fees", pa.string()),
            ("slippage", pa.string()),
            ("record_json", pa.string()),
        ],
        metadata={b"schema_version": b"1", b"kind": b"trades"},
    ),
}


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _content_id(
    result: BacktestResult,
    metrics: PerformanceSummary,
    decisions: tuple[DecisionRecord, ...],
    outcomes: tuple[OutcomeRecord, ...],
) -> str:
    data = dict(
        result=result.model_dump(mode="json"),
        metrics=metrics.model_dump(mode="json", exclude={"cumulative"}),
        decisions=[r.model_dump(mode="json", exclude={"run_id"}) for r in decisions],
        outcomes=[r.model_dump(mode="json", exclude={"run_id"}) for r in outcomes],
    )
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _difference(current: Decimal, previous: Decimal) -> str:
    with localcontext(Context(prec=60)):
        return str(current - previous)


def _ratio(current: Decimal, previous: Decimal) -> str | None:
    with localcontext(Context(prec=60)):
        return str(current / previous - 1) if previous else None


def _rows(
    spec: ExperimentSpec,
    result: BacktestResult,
    run_id: str,
    decisions: tuple[DecisionRecord, ...],
    outcomes: tuple[OutcomeRecord, ...],
) -> dict[str, list[dict[str, object]]]:
    base: dict[str, object] = dict(
        experiment_id=spec.experiment_id,
        simulated=True,
        run_id=run_id,
        strategy_id=spec.strategy.strategy_id,
        strategy_version=spec.strategy.version,
        config_hash=spec.strategy.identity,
        exchange=spec.strategy.market.exchange,
        pair=spec.strategy.market.pair,
        timeframe=spec.strategy.market.timeframe,
        dataset_version=spec.dataset.version,
        dataset_checksum=spec.dataset.sha256,
    )
    decision_rows = [
        base
        | dict(
            decision_id=d.decision_id,
            timestamp=d.signal.generated_at,
            data_through=d.signal.data_through,
            available_at=d.data_available_at,
            effective_at=d.signal.effective_at,
            target_exposure=d.signal.weight,
            preceding_exposure=d.preceding_exposure,
            signal_strength=d.signal.strength,
            feature_schema=d.features.schema_version,
            record_json=d.canonical_json(),
        )
        for d in decisions
    ]
    for row, decision in zip(decision_rows, decisions, strict=True):
        for value in decision.features.values:
            row[value.name] = value.observation.value if value.observation else None
            row[value.name + "_missing"] = value.missing_reason
    outcome_rows = [
        base
        | dict(
            decision_id=o.decision_id,
            timestamp=o.start,
            end=o.end,
            available_at=o.available_at,
            horizon_days=o.horizon_days,
            gross_return=o.gross_return,
            net_return=o.net_return,
            fees=str(o.fees) if o.fees is not None else None,
            slippage=str(o.slippage) if o.slippage is not None else None,
            average_exposure=o.average_exposure,
            status=o.status,
            scope=o.scope,
            record_json=o.canonical_json(),
        )
        for o in outcomes
    ]
    performance_rows = []
    previous_time = None
    with localcontext(Context(prec=60)):
        previous_net = previous_gross = amount(spec.settings.starting_capital)
    previous_fees, previous_slippage, previous_turnover = Decimal(0), Decimal(0), Decimal(0)
    for point in result.equity:
        position = next((p for p in point.positions if p.market == spec.strategy.market), None)
        performance_rows.append(
            base
            | dict(
                timestamp=point.timestamp,
                period_start=previous_time,
                available_at=point.timestamp,
                cash=str(point.cash),
                equity=str(point.equity),
                gross_equity=str(point.gross_equity),
                net_return=_ratio(point.equity, previous_net)
                if previous_time is not None
                else None,
                gross_return=_ratio(point.gross_equity, previous_gross)
                if previous_time is not None
                else None,
                cumulative_return=str(point.cumulative_return),
                gross_cumulative_return=str(point.gross_cumulative_return),
                fees_cumulative=str(point.fees_paid),
                slippage_cumulative=str(point.slippage_paid),
                turnover_cumulative=str(point.turnover),
                quantity=str(position.quantity if position else Decimal(0)),
                target_exposure=float(position.target_weight) if position else 0.0,
                actual_exposure=float(position.actual_weight) if position else 0.0,
                fees=_difference(point.fees_paid, previous_fees),
                slippage=_difference(point.slippage_paid, previous_slippage),
                turnover=_difference(point.turnover, previous_turnover),
                scope="standalone_shadow" if spec.settings.shadow else "standalone_simulated",
            )
        )
        if previous_time is not None:
            previous_net, previous_gross = point.equity, point.gross_equity
        previous_time = point.timestamp
        previous_fees, previous_slippage, previous_turnover = (
            point.fees_paid,
            point.slippage_paid,
            point.turnover,
        )
    orders = {o.order_id: o for o in result.orders}
    trade_rows = [
        base
        | dict(
            timestamp=f.timestamp,
            order_id=f.order_id,
            fill_id=f.fill_id,
            decision_at=orders[f.order_id].decision_at,
            effective_at=orders[f.order_id].earliest_at,
            side=f.side,
            quantity=str(f.quantity),
            reference_price=str(f.reference_price),
            execution_price=str(f.execution_price),
            notional=str(f.notional),
            fees=str(f.fee),
            slippage=str(f.slippage),
            record_json=f.canonical_json(),
        )
        for f in result.fills
    ]
    return dict(
        decisions=decision_rows,
        outcomes=outcome_rows,
        performance=performance_rows,
        trades=trade_rows,
    )


class ResearchStore:
    def __init__(self, root: str | Path = "research/artifacts") -> None:
        self.root = Path(root)

    def path(self, run_id: str) -> Path:
        if re.fullmatch(r"[0-9a-f]{32}", run_id) is None:
            raise ValueError("execution ID must be a UUID hex string")
        return self.root / run_id

    def publish(
        self,
        spec: ExperimentSpec,
        run_id: str,
        result: BacktestResult,
        metrics: PerformanceSummary,
        decisions: tuple[DecisionRecord, ...],
        outcomes: tuple[OutcomeRecord, ...],
    ) -> ResearchManifest:
        destination = self.path(run_id)
        if destination.exists():
            raise FileExistsError("research execution artifacts cannot be overwritten")
        decision_ids = {d.decision_id for d in decisions}
        if (
            result.settings != spec.settings
            or any(d.run_id != run_id or d.experiment_id != spec.experiment_id for d in decisions)
            or any(o.run_id != run_id or o.decision_id not in decision_ids for o in outcomes)
        ):
            raise ValueError("research bundle identities differ")
        self.root.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=".staging-", dir=self.root))
        try:
            for name, model in (
                ("spec", spec),
                ("result", result),
                (
                    "metrics",
                    PerformanceSummary.model_validate(metrics.model_dump(exclude={"cumulative"})),
                ),
            ):
                (stage / f"{name}.json").write_text(model.canonical_json(), encoding="utf-8")
            rows = _rows(spec, result, run_id, decisions, outcomes)
            for kind, schema in SCHEMAS.items():
                pq.write_table(
                    pa.Table.from_pylist(rows[kind], schema=schema),
                    stage / f"{kind}.parquet",
                    compression="zstd",
                    row_group_size=4096,
                    version="2.6",
                )
            content_id = _content_id(result, metrics, decisions, outcomes)
            files = tuple(
                ArtifactFile.model_validate(
                    dict(name=p.name, sha256=checksum(p), size=p.stat().st_size)
                )
                for p in sorted(stage.iterdir())
            )
            manifest = ResearchManifest(
                run_id=run_id,
                experiment_id=spec.experiment_id,
                input_hash=spec.input_hash,
                content_id=content_id,
                files=files,
            )
            (stage / "manifest.json").write_text(manifest.canonical_json(), encoding="utf-8")
            for path in stage.iterdir():
                with path.open("r+b") as stream:
                    os.fsync(stream.fileno())
            os.rename(stage, destination)
            return manifest
        finally:
            if stage.exists():
                shutil.rmtree(stage)

    def verify(
        self, run_id: str, expected_sha: str | None = None, *, full: bool = True
    ) -> ResearchManifest:
        root = self.path(run_id)
        if expected_sha is not None and checksum(root / "manifest.json") != expected_sha:
            raise ValueError("research manifest checksum mismatch")
        manifest = ResearchManifest.model_validate_json(
            (root / "manifest.json").read_text(encoding="utf-8")
        )
        if manifest.run_id != run_id:
            raise ValueError("research manifest execution identity mismatch")
        for artifact in manifest.files:
            path = root / artifact.name
            if path.stat().st_size != artifact.size or checksum(path) != artifact.sha256:
                raise ValueError(f"research artifact checksum mismatch: {artifact.name}")
            if path.suffix == ".parquet" and not pq.read_schema(path).equals(
                SCHEMAS[path.stem], check_metadata=True
            ):
                raise ValueError("research Parquet schema mismatch")
        if not full:
            if expected_sha is None:
                raise ValueError("filtered reads require the registered manifest checksum")
            return manifest
        spec = ExperimentSpec.model_validate_json((root / "spec.json").read_text(encoding="utf-8"))
        result = BacktestResult.model_validate_json(
            (root / "result.json").read_text(encoding="utf-8")
        )
        metrics = PerformanceSummary.model_validate_json(
            (root / "metrics.json").read_text(encoding="utf-8")
        )
        decision_records = tuple(
            DecisionRecord.model_validate_json(text)
            for text in pq.read_table(root / "decisions.parquet", columns=["record_json"])[
                "record_json"
            ].to_pylist()
        )
        labels = tuple(
            OutcomeRecord.model_validate_json(text)
            for text in pq.read_table(root / "outcomes.parquet", columns=["record_json"])[
                "record_json"
            ].to_pylist()
        )
        if (
            spec.experiment_id != manifest.experiment_id
            or spec.input_hash != manifest.input_hash
            or result.settings != spec.settings
            or _content_id(result, metrics, decision_records, labels) != manifest.content_id
        ):
            raise ValueError("research logical content/identity mismatch")
        expected_decisions = {
            (i.config.identity, i.signal.identity)
            for target in result.decisions
            for i in target.instructions
        }
        if (
            len(expected_decisions) != len(decision_records)
            or expected_decisions
            != {(d.config.identity, d.signal.identity) for d in decision_records}
            or any(
                d.experiment_id != spec.experiment_id
                or d.run_id != run_id
                or d.input_hash != spec.input_hash
                or d.dataset != spec.dataset
                or d.config != spec.strategy
                for d in decision_records
            )
        ):
            raise ValueError("decision records do not match the completed simulation")
        if labels != build_outcomes(spec, result, decision_records):
            raise ValueError("future outcomes do not match simulated strategy performance")
        if metrics.model_dump(exclude={"cumulative"}) != summarize(
            from_backtest(result), spec.metrics, result
        ).model_dump(exclude={"cumulative"}):
            raise ValueError("performance metrics do not match simulated equity")
        expected_rows = _rows(spec, result, run_id, decision_records, labels)
        for kind, schema in SCHEMAS.items():
            expected = pa.Table.from_pylist(expected_rows[kind], schema=schema)
            if not pq.read_table(root / f"{kind}.parquet").equals(expected, check_metadata=True):
                raise ValueError("research table columns disagree with validated records")
        return manifest

    def table(
        self,
        run_id: str,
        kind: Kind,
        *,
        expected_sha: str,
        strategy_id: str | None = None,
        pair: str | None = None,
        exchange: str | None = None,
        timeframe: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pa.Table:
        if kind not in SCHEMAS:
            raise ValueError("unknown research dataset kind")
        self.verify(run_id, expected_sha, full=False)
        filters: list[tuple[str, str, object]] = []
        for name, value in (
            ("strategy_id", strategy_id),
            ("pair", pair),
            ("exchange", exchange),
            ("timeframe", timeframe),
        ):
            if value is not None:
                filters.append((name, "=", value))
        if start is not None:
            if start.tzinfo is None:
                raise ValueError("filter timestamps must be UTC-aware")
            filters.append(("timestamp", ">=", start))
        if end is not None:
            if end.tzinfo is None:
                raise ValueError("filter timestamps must be UTC-aware")
            filters.append(("timestamp", "<", end))
        return pq.read_table(self.path(run_id) / f"{kind}.parquet", filters=filters or None)
