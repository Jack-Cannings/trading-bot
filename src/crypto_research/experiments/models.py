"""Research identities: immutable logical experiment, distinct executions/artifacts."""

import hashlib
import json
from datetime import datetime
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from crypto_research.backtesting.models import SimulationSettings
from crypto_research.contracts import Checksum, DatasetReference, Identifier, Record, StrategyConfig
from crypto_research.metrics.summary import MetricSettings

Status = Literal["created", "running", "completed", "failed", "cancelled"]


class ExperimentSpec(Record):
    schema_version: Literal[1] = 1
    experiment_id: Identifier
    strategy: StrategyConfig
    dataset: DatasetReference
    code_revision: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    settings: SimulationSettings
    metrics: MetricSettings = MetricSettings()
    split: Literal["training", "validation", "test"]
    seed: Annotated[int, Field(ge=0, strict=True)] | None = None
    mode: Literal["strategy", "cash", "buy_hold"] = "strategy"
    outcome_days: tuple[Annotated[int, Field(gt=0, le=3650, strict=True)], ...] = (1, 7, 30)
    allow_data_gaps: bool = False
    strategy_lookback_days: Literal[201] = 201

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if self.settings.experiment_id != self.experiment_id:
            raise ValueError("experiment identity differs from simulation settings")
        if len(self.outcome_days) != len(set(self.outcome_days)):
            raise ValueError("outcome horizons must be unique")
        object.__setattr__(self, "outcome_days", tuple(sorted(self.outcome_days)))
        return self

    @property
    def input_hash(self) -> str:
        data = self.model_dump(mode="json")
        data.pop("experiment_id")
        data["settings"].pop("experiment_id")
        return hashlib.sha256(
            json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @property
    def comparison_group(self) -> str:
        data = self.model_dump(mode="json", exclude={"experiment_id", "strategy", "seed", "mode"})
        data["market"] = self.strategy.market.model_dump(mode="json")
        data["settings"].pop("experiment_id")
        return hashlib.sha256(
            json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


class Execution(Record):
    run_id: Identifier
    experiment_id: Identifier
    status: Status
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    failure_reason: str | None = None
    content_id: Checksum | None = None
    manifest_sha256: Checksum | None = None


class ArtifactFile(Record):
    name: Literal[
        "spec.json",
        "result.json",
        "metrics.json",
        "decisions.parquet",
        "outcomes.parquet",
        "performance.parquet",
        "trades.parquet",
    ]
    sha256: Checksum
    size: Annotated[int, Field(ge=0)]


class ResearchManifest(Record):
    schema_version: Literal[1] = 1
    run_id: Identifier
    experiment_id: Identifier
    input_hash: Checksum
    content_id: Checksum
    files: tuple[ArtifactFile, ...]

    @model_validator(mode="after")
    def all_files(self) -> Self:
        required = {
            "spec.json",
            "result.json",
            "metrics.json",
            "decisions.parquet",
            "outcomes.parquet",
            "performance.parquet",
            "trades.parquet",
        }
        if {f.name for f in self.files} != required or len(self.files) != len(required):
            raise ValueError("manifest must contain exactly the versioned research artifact set")
        return self
