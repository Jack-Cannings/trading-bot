"""Immutable Parquet versions. A manifest is published only after a complete commit."""

import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import cast

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from crypto_research.contracts import Candle, DatasetReference, Market, MarketData, Period
from crypto_research.data.models import (
    Manifest,
    Observation,
    QualityReport,
    Request,
    ValidatedData,
    duration,
)
from crypto_research.data.validation import validate, values

SCHEMA = pa.schema(
    [
        ("exchange", pa.string()),
        ("pair", pa.string()),
        ("timeframe", pa.string()),
        ("open_time", pa.timestamp("us", tz="UTC")),
        ("close_time", pa.timestamp("us", tz="UTC")),
        ("open", pa.float64()),
        ("high", pa.float64()),
        ("low", pa.float64()),
        ("close", pa.float64()),
        ("volume", pa.float64()),
        ("complete", pa.bool_()),
        ("observed_at", pa.timestamp("us", tz="UTC")),
    ]
)


class IntegrityError(ValueError):
    """Stored data, metadata or hashes disagree."""


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def content_hash(market: Market, rows: tuple[Observation, ...]) -> str:
    """Logical identity excludes observation time, batch settings and Parquet bytes."""
    canonical = {
        "schema": 1,
        "market": market.model_dump(mode="json"),
        "rows": [
            [r.open_time.isoformat(), *[0.0 if v == 0 else v for v in values(r)]]
            for r in sorted(rows, key=lambda r: r.open_time)
        ],
    }
    return digest(
        json.dumps(canonical, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )


def version_hash(manifest: Manifest) -> str:
    return digest(
        json.dumps(
            manifest.model_dump(mode="json", exclude={"version"}),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )


def _frame(market: Market, rows: tuple[Observation, ...]) -> pd.DataFrame:
    return cast(pd.DataFrame, _table(market, rows).to_pandas())


def _table(market: Market, rows: tuple[Observation, ...]) -> pa.Table:
    return pa.Table.from_pylist(
        [
            market.model_dump()
            | r.model_dump()
            | {
                "close_time": r.open_time + duration(market),
                "complete": True,
            }
            for r in rows
        ],
        schema=SCHEMA,
    )


class Store:
    def __init__(self, root: str | Path = "data") -> None:
        self.root = Path(root)

    def _path(self, version: str) -> Path:
        # Validate before using user-provided identifiers as path components.
        if len(version) != 64 or any(c not in "0123456789abcdef" for c in version):
            raise ValueError("version must be a lowercase SHA-256 identifier")
        return self.root / "versions" / version

    def versions(self) -> tuple[str, ...]:
        root = self.root / "versions"
        return (
            tuple(sorted(p.name for p in root.iterdir() if (p / "manifest.json").is_file()))
            if root.exists()
            else ()
        )

    def manifest(self, version: str) -> Manifest:
        manifest = Manifest.model_validate_json(
            (self._path(version) / "manifest.json").read_text(encoding="utf-8")
        )
        if manifest.version != version or version_hash(manifest) != version:
            raise IntegrityError("manifest version does not match its path")
        return manifest

    def save(
        self,
        data: ValidatedData,
        provider: str,
        acquired_at: datetime,
        parent: Manifest | None = None,
        revised: tuple[datetime, ...] = (),
    ) -> Manifest:
        rows, report = data.observations, data.report
        if not rows:
            raise ValueError("no valid completed observations; no empty dataset was stored")
        refreshed = validate(
            report.request, rows, tuple(i for i in report.anomalies if i.kind != "zero_volume")
        )
        if refreshed.observations != rows or refreshed.report.model_dump(
            exclude={"anomalies"}
        ) != report.model_dump(exclude={"anomalies"}):
            raise ValueError(
                "storage requires validated observations and a matching quality report"
            )
        if any(row.observed_at > acquired_at for row in rows):
            raise ValueError("acquisition timestamp precedes observation")
        logical = content_hash(report.request.market, rows)
        if parent and parent.request.market != report.request.market:
            raise ValueError("parent market identity differs")
        quality_bytes = report.canonical_json().encode()
        metadata = dict(
            dataset_id=logical,
            provider=provider,
            request=report.request,
            actual_coverage=report.retrieved,
            observation_count=len(rows),
            acquired_at=acquired_at,
            quality_sha256=digest(quality_bytes),
            parent_version=parent.version if parent else None,
            change="corrected"
            if revised
            else "updated"
            if parent and logical != parent.dataset_id
            else "unchanged"
            if parent
            else "original",
            revised_opens=revised,
        )
        # Logical identity excludes batching; version authenticates complete provenance and bytes.
        self.root.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=".staging-", dir=self.root))
        try:
            artifact = stage / "observations.parquet"
            pq.write_table(
                _table(report.request.market, rows),
                artifact,
                compression="zstd",
                version="2.6",
                row_group_size=4096,
            )
            manifest = Manifest.model_validate(
                metadata
                | {
                    "version": "0" * 64,
                    "artifact_sha256": digest(artifact.read_bytes()),
                }
            )
            version = version_hash(manifest)
            manifest = Manifest.model_validate(manifest.model_dump() | {"version": version})
            destination = self._path(version)
            (stage / "quality.json").write_bytes(quality_bytes)
            (stage / "manifest.json").write_text(manifest.canonical_json(), encoding="utf-8")
            for path in stage.iterdir():
                with path.open("r+b") as stream:
                    os.fsync(stream.fileno())
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                # Rename whole directory: readers never see a partial artifact/manifest pair.
                os.rename(stage, destination)
            except OSError:
                if not destination.exists():
                    raise
                existing, old_rows, old_report = self.read(version)
                if old_rows != rows or old_report != report:
                    raise IntegrityError("version collision; refusing overwrite") from None
                return existing
            return manifest
        finally:
            if stage.exists():
                shutil.rmtree(stage)

    def read(
        self, version: str, *, period: Period | None = None, as_of: datetime | None = None
    ) -> tuple[Manifest, tuple[Observation, ...], QualityReport]:
        location = self._path(version)
        manifest = self.manifest(version)
        artifact = location / "observations.parquet"
        quality_bytes = (location / "quality.json").read_bytes()
        if (
            digest(artifact.read_bytes()) != manifest.artifact_sha256
            or digest(quality_bytes) != manifest.quality_sha256
        ):
            raise IntegrityError("artifact or quality report checksum mismatch")
        report = QualityReport.model_validate_json(quality_bytes)
        filters = None
        if period is not None:
            filters = [("open_time", ">=", period.start), ("open_time", "<", period.end)]
            if as_of is not None:
                filters.extend([("close_time", "<=", as_of), ("observed_at", "<=", as_of)])
        table = pq.read_table(artifact, filters=filters)
        if table.schema != SCHEMA:
            raise IntegrityError("unexpected Parquet schema")
        rows: list[Observation] = []
        for entry in table.to_pylist():
            if any(
                entry[key] != value for key, value in manifest.request.market.model_dump().items()
            ):
                raise IntegrityError("stored market identity mismatch")
            if entry.pop("complete") is not True:
                raise IntegrityError("incomplete stored candle")
            close = entry.pop("close_time")
            for key in ("exchange", "pair", "timeframe"):
                entry.pop(key)
            row = Observation.model_validate(entry)
            if close != row.open_time + duration(manifest.request.market):
                raise IntegrityError("stored close time mismatch")
            rows.append(row)
        result = tuple(rows)
        checked = validate(manifest.request, result)
        if period is None:
            regenerated = validate(
                manifest.request,
                result,
                tuple(i for i in report.anomalies if i.kind != "zero_volume"),
            )
            if regenerated.report.model_dump(exclude={"anomalies"}) != report.model_dump(
                exclude={"anomalies"}
            ):
                raise IntegrityError("quality report does not match stored observations")
        if (
            checked.observations != result
            or checked.report.invalid_records
            or (period is None and len(result) != manifest.observation_count)
            or (
                period is None
                and content_hash(manifest.request.market, result) != manifest.dataset_id
            )
            or report.request != manifest.request
            or report.retrieved != manifest.actual_coverage
            or report.candle_count != manifest.observation_count
            or (period is None and checked.report.retrieved != report.retrieved)
        ):
            raise IntegrityError("logical content or metadata mismatch")
        return manifest, result, report

    def load(
        self,
        version: str,
        *,
        market: Market,
        period: Period,
        as_of: datetime,
        allow_gaps: bool = False,
    ) -> MarketData:
        Request(market=market, period=period)  # reject misaligned ranges even when allowing gaps
        as_of = Period(start=period.start, end=as_of).end
        manifest, rows, report = self.read(version, period=period, as_of=as_of)
        if report.status == "invalid":
            raise ValueError("dataset contains unresolved invalid observations")
        if market != manifest.request.market:
            raise ValueError("requested market differs from dataset")
        selected = tuple(
            r
            for r in rows
            if period.start <= r.open_time < period.end
            and r.open_time + duration(market) <= as_of
            and r.observed_at <= as_of
        )
        if not selected:
            raise ValueError("no completed and available candles at the requested cutoff")
        if not allow_gaps:
            end = min(period.end, as_of)
            # Only whole candles can be expected at an unaligned decision timestamp.
            end -= (end - period.start) % duration(market)
            if end > period.start:
                requested = Request(market=market, period=Period(start=period.start, end=end))
                if validate(requested, selected).report.status != "clean":
                    raise ValueError(
                        "research interval has missing or unavailable candles; "
                        "use allow_gaps explicitly"
                    )
        reference = DatasetReference(
            source=manifest.provider, version=version, sha256=manifest.dataset_id
        )
        return MarketData(
            market=market,
            as_of=as_of,
            candles=tuple(
                Candle(
                    market=market,
                    open_time=r.open_time,
                    close_time=r.open_time + duration(market),
                    open=r.open,
                    high=r.high,
                    low=r.low,
                    close=r.close,
                    volume=r.volume,
                    complete=True,
                    available_at=r.observed_at,
                    dataset=reference,
                )
                for r in selected
            ),
        )

    def dataframe(
        self,
        version: str,
        *,
        market: Market,
        period: Period,
        as_of: datetime,
        allow_gaps: bool = False,
    ) -> pd.DataFrame:
        """The same point-in-time boundary as load(), exported to typed Pandas columns."""
        snapshot = self.load(
            version, market=market, period=period, as_of=as_of, allow_gaps=allow_gaps
        )
        observations = tuple(
            Observation(
                open_time=c.open_time,
                open=c.open,
                high=c.high,
                low=c.low,
                close=c.close,
                volume=c.volume,
                observed_at=c.available_at,
            )
            for c in snapshot.candles
        )
        return _frame(market, observations)
