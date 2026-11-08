"""SQLite lifecycle transactions; published artifacts precede successful DB commits."""

import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from crypto_research.experiments.models import Execution, ExperimentSpec, Status
from crypto_research.experiments.storage import ResearchStore, checksum
from crypto_research.metrics.summary import PerformanceSummary


def now() -> str:
    return datetime.now(UTC).isoformat()


class Registry:
    def __init__(self, root: str | Path = "research") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "experiments.sqlite"
        with self.transaction() as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version not in {0, 1}:
                raise ValueError("unsupported experiment registry schema version")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    input_hash TEXT NOT NULL, spec_json TEXT NOT NULL,
                    strategy_id TEXT NOT NULL, market TEXT NOT NULL, timeframe TEXT NOT NULL,
                    period_start TEXT NOT NULL, period_end TEXT NOT NULL, created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS executions (
                    run_id TEXT PRIMARY KEY,
                    experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id),
                    status TEXT NOT NULL CHECK(status IN (
                        'created','running','completed','failed','cancelled')),
                    created_at TEXT NOT NULL, started_at TEXT,
                    finished_at TEXT, failure_reason TEXT,
                    content_id TEXT, manifest_sha256 TEXT);
                CREATE TABLE IF NOT EXISTS artifacts (
                    run_id TEXT NOT NULL REFERENCES executions(run_id), name TEXT NOT NULL,
                    sha256 TEXT NOT NULL, size INTEGER NOT NULL, PRIMARY KEY(run_id,name));
                CREATE INDEX IF NOT EXISTS experiment_selection
                    ON experiments(strategy_id,market,timeframe,period_start,period_end);
                CREATE INDEX IF NOT EXISTS executions_selection
                    ON executions(experiment_id,status,created_at);
            """)

            db.execute("PRAGMA user_version = 1")

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.db_path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def register(self, spec: ExperimentSpec) -> None:
        with self.transaction() as db:
            previous = db.execute(
                "SELECT spec_json FROM experiments WHERE experiment_id=?", (spec.experiment_id,)
            ).fetchone()
            if previous:
                if ExperimentSpec.model_validate_json(previous["spec_json"]) != spec:
                    raise ValueError(
                        "experiment configurations are immutable; choose a new experiment ID"
                    )
                return
            db.execute(
                "INSERT INTO experiments VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    spec.experiment_id,
                    spec.input_hash,
                    spec.canonical_json(),
                    spec.strategy.strategy_id,
                    spec.strategy.market.pair,
                    spec.strategy.market.timeframe,
                    spec.settings.period.start.isoformat(),
                    spec.settings.period.end.isoformat(),
                    now(),
                ),
            )

    def spec(self, experiment_id: str) -> ExperimentSpec:
        with self.transaction() as db:
            row = db.execute(
                "SELECT spec_json FROM experiments WHERE experiment_id=?", (experiment_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown experiment: {experiment_id}")
        return ExperimentSpec.model_validate_json(row["spec_json"])

    def create_run(self, experiment_id: str) -> Execution:
        self.spec(experiment_id)
        run_id = uuid.uuid4().hex
        with self.transaction() as db:
            db.execute(
                "INSERT INTO executions(run_id,experiment_id,status,created_at) "
                "VALUES (?,?,'created',?)",
                (run_id, experiment_id, now()),
            )
        return self.get(run_id)

    def get(self, run_id: str) -> Execution:
        with self.transaction() as db:
            row = db.execute("SELECT * FROM executions WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown execution: {run_id}")
        return Execution.model_validate(dict(row))

    def list(
        self,
        experiment_id: str | None = None,
        *,
        strategy_id: str | None = None,
        pair: str | None = None,
        timeframe: str | None = None,
    ) -> tuple[Execution, ...]:
        clauses: list[str] = []
        values: list[str] = []
        for column, value in (
            ("e.experiment_id", experiment_id),
            ("e.strategy_id", strategy_id),
            ("e.market", pair),
            ("e.timeframe", timeframe),
        ):
            if value is not None:
                clauses.append(column + "=?")
                values.append(value)
        query = "SELECT r.* FROM executions r JOIN experiments e ON e.experiment_id=r.experiment_id"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY r.created_at,r.run_id"
        with self.transaction() as db:
            rows = db.execute(query, values).fetchall()
        return tuple(Execution.model_validate(dict(row)) for row in rows)

    def transition(self, run_id: str, status: Status, reason: str | None = None) -> Execution:
        if status == "completed":
            raise ValueError("completion requires verified published artifacts")
        allowed = {
            "created": {"running", "failed", "cancelled"},
            "running": {"failed", "cancelled"},
        }
        with self.transaction() as db:
            row = db.execute("SELECT status FROM executions WHERE run_id=?", (run_id,)).fetchone()
            if row is None or status not in allowed.get(row["status"], set()):
                raise ValueError("invalid experiment execution lifecycle transition")
            if status in {"failed", "cancelled"} and not reason:
                raise ValueError("unsuccessful executions require a reason")
            db.execute(
                "UPDATE executions SET status=?,started_at=COALESCE(started_at,?),"
                "finished_at=?,failure_reason=? WHERE run_id=? AND status=?",
                (
                    status,
                    now() if status == "running" else None,
                    now() if status != "running" else None,
                    reason,
                    run_id,
                    row["status"],
                ),
            )
            if db.execute("SELECT changes()").fetchone()[0] != 1:
                raise ValueError("concurrent lifecycle transition; execution state changed")
        return self.get(run_id)

    def complete(self, run_id: str) -> Execution:
        artifacts = ResearchStore(self.root / "artifacts")
        manifest = artifacts.verify(run_id)
        execution = self.get(run_id)
        spec = self.spec(execution.experiment_id)
        if (
            manifest.experiment_id != execution.experiment_id
            or manifest.input_hash != spec.input_hash
        ):
            raise ValueError("published artifacts do not match registered experiment")
        with self.transaction() as db:
            row = db.execute("SELECT status FROM executions WHERE run_id=?", (run_id,)).fetchone()
            if row is None or row["status"] != "running":
                raise ValueError("only a running execution can complete")
            for artifact in manifest.files:
                db.execute(
                    "INSERT INTO artifacts VALUES (?,?,?,?)",
                    (run_id, artifact.name, artifact.sha256, artifact.size),
                )
            db.execute(
                "UPDATE executions SET status='completed',finished_at=?,content_id=?,"
                "manifest_sha256=? WHERE run_id=? AND status='running'",
                (
                    now(),
                    manifest.content_id,
                    checksum(artifacts.path(run_id) / "manifest.json"),
                    run_id,
                ),
            )
            if db.execute("SELECT changes()").fetchone()[0] != 1:
                raise ValueError("concurrent completion rejected; execution is no longer running")
        return self.get(run_id)

    def recover_interrupted(self, run_id: str) -> Execution:
        return self.transition(
            run_id,
            "cancelled",
            "interrupted execution explicitly recovered; artifacts not promoted",
        )

    def summary(self, run_id: str) -> PerformanceSummary:
        execution = self.get(run_id)
        if execution.status != "completed":
            raise ValueError("only completed experiments have performance summaries")
        with self.transaction() as db:
            artifact = db.execute(
                "SELECT sha256 FROM artifacts WHERE run_id=? AND name='metrics.json'", (run_id,)
            ).fetchone()
        path = ResearchStore(self.root / "artifacts").path(run_id) / "metrics.json"
        if artifact is None or checksum(path) != artifact["sha256"]:
            raise ValueError("registered metrics checksum mismatch")
        return PerformanceSummary.model_validate_json(path.read_text(encoding="utf-8"))
