"""A fixed, offline research campaign using the existing Runner and SQLite registry."""

import json
import uuid
from decimal import Decimal
from pathlib import Path
from typing import Literal, Self

from pydantic import model_validator

from crypto_research.backtesting.models import SimulationSettings
from crypto_research.contracts import Parameter, Period, Record, StrategyConfig
from crypto_research.data.storage import Store
from crypto_research.experiments.models import ExperimentSpec
from crypto_research.experiments.momentum_plan import CampaignPlan, Selection, Window
from crypto_research.experiments.registry import Registry
from crypto_research.experiments.runner import Runner
from crypto_research.strategies.risk_target import RiskTarget, risk_configuration
from crypto_research.strategies.time_series_momentum import TimeSeriesMomentum, configuration


class Attempt(Record):
    attempt_id: str
    campaign_id: str
    pair: str
    configuration: StrategyConfig
    window: str
    period: Period | None
    cost_multiplier: Literal[1, 2]
    status: Literal["created", "running", "completed", "failed", "unavailable"]
    reason: str | None = None
    execution_id: str | None = None
    content_id: str | None = None

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if self.pair != self.configuration.market.pair:
            raise ValueError("attempt market identity mismatch")
        if self.status in {"failed", "unavailable"} and not self.reason:
            raise ValueError("unsuccessful attempt requires a reason")
        if self.status == "completed" and (
            not self.execution_id or not self.content_id or not self.period
        ):
            raise ValueError("completed attempt requires a verified execution and period")
        if self.status == "unavailable" and (self.execution_id or self.content_id):
            raise ValueError("unavailable attempt cannot claim executed performance")
        return self


class CampaignLedger:
    """Campaign plans and unavailable trials share the existing registry database.

    Missing datasets have no fabricated DatasetReference or performance. A created
    attempt is durable before execution; an interruption leaves explicit evidence.
    """

    def __init__(self, registry: Registry) -> None:
        self.registry = registry
        with registry.transaction() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS campaign_plans (
                    campaign_id TEXT PRIMARY KEY, plan_json TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS campaign_attempts (
                    attempt_id TEXT PRIMARY KEY,
                    campaign_id TEXT NOT NULL REFERENCES campaign_plans(campaign_id),
                    record_json TEXT NOT NULL);
            """)

    def declare(self, plan: CampaignPlan) -> None:
        with self.registry.transaction() as db:
            previous = db.execute(
                "SELECT plan_json FROM campaign_plans WHERE campaign_id=?", (plan.identity,)
            ).fetchone()
            if previous and previous[0] != plan.canonical_json():
                raise ValueError("immutable campaign plan conflict")
            db.execute(
                "INSERT OR IGNORE INTO campaign_plans VALUES (?,?)",
                (plan.identity, plan.canonical_json()),
            )

    def save(self, attempt: Attempt) -> None:
        if attempt.execution_id:
            execution = self.registry.get(attempt.execution_id)
            spec = self.registry.spec(execution.experiment_id)
            if spec.strategy != attempt.configuration or spec.settings.period != attempt.period:
                raise ValueError("campaign attempt differs from linked execution")
            if attempt.status == "completed" and (
                execution.status != "completed" or execution.content_id != attempt.content_id
            ):
                raise ValueError("campaign completion requires verified registered artifacts")
        with self.registry.transaction() as db:
            previous = db.execute(
                "SELECT record_json FROM campaign_attempts WHERE attempt_id=?",
                (attempt.attempt_id,),
            ).fetchone()
            if previous:
                old = Attempt.model_validate_json(previous[0])
                immutable = {"status", "reason", "execution_id", "content_id"}
                if old.model_dump(exclude=immutable) != attempt.model_dump(exclude=immutable):
                    raise ValueError("attempt configuration cannot change")
                if old.status not in {"created", "running"}:
                    raise ValueError("terminal campaign attempt cannot change")
                allowed = {
                    "created": {"running", "unavailable", "failed"},
                    "running": {"completed", "failed"},
                }
                if attempt.status not in allowed[old.status]:
                    raise ValueError("invalid campaign attempt transition")
                db.execute(
                    "UPDATE campaign_attempts SET record_json=? "
                    "WHERE attempt_id=? AND record_json=?",
                    (attempt.canonical_json(), attempt.attempt_id, previous[0]),
                )
                if db.execute("SELECT changes()").fetchone()[0] != 1:
                    raise ValueError("concurrent campaign attempt transition")
            else:
                if attempt.status != "created":
                    raise ValueError("new campaign attempt must start created")
                db.execute(
                    "INSERT INTO campaign_attempts VALUES (?,?,?)",
                    (attempt.attempt_id, attempt.campaign_id, attempt.canonical_json()),
                )

    def list(self, campaign_id: str) -> tuple[Attempt, ...]:
        with self.registry.transaction() as db:
            rows = db.execute(
                "SELECT record_json FROM campaign_attempts WHERE campaign_id=? ORDER BY rowid",
                (campaign_id,),
            ).fetchall()
        return tuple(Attempt.model_validate_json(row[0]) for row in rows)


def configurations(selection: Selection) -> tuple[StrategyConfig, ...]:
    market = selection.market
    return tuple(configuration(market, n) for n in (30, 60, 90, 120, 180)) + (
        StrategyConfig(strategy_id="cash", version="1", market=market),
        StrategyConfig(
            strategy_id="benchmark",
            version="1",
            market=market,
            parameters=(Parameter(name="weight", value=1.0),),
        ),
        risk_configuration(market),
    )


def execute_campaign(
    plan: CampaignPlan, source: Store, registry: Registry, *, report_root: Path | None = None
) -> Path:
    """Never downloads data. Never schedules a trial in the protected holdout."""
    ledger = CampaignLedger(registry)
    ledger.declare(plan)  # committed BEFORE any target generation or metrics
    report_root = report_root or registry.root / "reports" / "time_series_momentum"
    destination = report_root / (plan.identity[:16] + "-" + uuid.uuid4().hex[:8])
    destination.mkdir(parents=True)
    (destination / "plan.json").write_text(plan.canonical_json(), encoding="utf-8")
    runner = Runner(source, registry)
    attempts = []
    for selection in plan.selections:
        windows = list(selection.windows)
        if windows and selection.holdout:
            windows.insert(
                0,
                Window(
                    name="per-market",
                    period=Period(start=windows[0].period.start, end=selection.holdout.start),
                    split="validation",
                ),
            )
            if plan.common:
                windows.append(Window(name="common-period", period=plan.common, split="validation"))
        # Every unavailable combination is still represented, without invented dates.
        schedules: list[Window | None] = list(windows) if windows else [None]
        for window in schedules:
            for multiplier in plan.cost_multipliers:
                for config in configurations(selection):
                    attempt = Attempt(
                        attempt_id=uuid.uuid4().hex,
                        campaign_id=plan.identity,
                        pair=selection.market.pair,
                        configuration=config,
                        window=window.name if window else "per-market",
                        period=window.period if window else None,
                        cost_multiplier=multiplier,
                        status="created",
                    )
                    ledger.save(attempt)
                    if window is None or selection.dataset is None or selection.holdout is None:
                        attempt = Attempt.model_validate(
                            attempt.model_dump()
                            | dict(
                                status="unavailable",
                                reason=selection.unavailable_reason
                                or "no eligible research period",
                            )
                        )
                    else:
                        attempt = _execute(
                            plan, selection, window, multiplier, config, attempt, runner, ledger
                        )
                    ledger.save(attempt)
                    attempts.append(attempt)
                    print(
                        f"{selection.market.pair} {attempt.window} {config.strategy_id} "
                        f"{dict((p.name, p.value) for p in config.parameters)} "
                        f"costs={multiplier} {attempt.status}",
                        flush=True,
                    )
    (destination / "attempts.json").write_text(
        json.dumps([a.model_dump(mode="json") for a in attempts], sort_keys=True, indent=2),
        encoding="utf-8",
    )
    from crypto_research.experiments.momentum_report import write_report

    write_report(plan, tuple(attempts), registry, destination)
    return destination


def _execute(
    plan: CampaignPlan,
    selection: Selection,
    window: Window,
    multiplier: Literal[1, 2],
    config: StrategyConfig,
    attempt: Attempt,
    runner: Runner,
    ledger: CampaignLedger,
) -> Attempt:
    name = f"mom-{plan.identity[:12]}-{config.identity[:10]}-{window.name}-{multiplier}"
    if selection.dataset is None or selection.holdout is None:
        raise ValueError("scheduled trial requires source and holdout")
    mode: Literal["strategy", "cash", "buy_hold"] = (
        "cash"
        if config.strategy_id == "cash"
        else "buy_hold"
        if config.strategy_id == "benchmark"
        else "strategy"
    )
    spec = ExperimentSpec(
        experiment_id=name,
        strategy=config,
        dataset=selection.dataset,
        code_revision=plan.code_revision,
        settings=SimulationSettings(
            experiment_id=name,
            period=window.period,
            starting_capital=Decimal(10000),
            fee_bps=Decimal(str(plan.research.fee_bps)) * multiplier,
            slippage_bps=Decimal(str(plan.research.slippage_bps)) * multiplier,
            shadow=True,
        ),
        split=window.split,
        mode=mode,
        seed=plan.research.seed,
        replay_availability=plan.replay_availability,
        information_end=selection.holdout.start,
        # The plan rejects ALL interior gaps; only reported pre-listing gaps remain.
        allow_data_gaps=bool(selection.missing_intervals),
    )
    strategy = (
        TimeSeriesMomentum(config)
        if config.strategy_id == "time_series_momentum"
        else (RiskTarget(config) if config.strategy_id == "passive_risk_target" else None)
    )
    runner.registry.register(spec)
    # Create and link execution before calling the runner, so interruption cannot
    # lose the relation between campaign and the failed/running execution.
    execution = runner.registry.create_run(name)
    attempt = Attempt.model_validate(
        attempt.model_dump() | dict(status="running", execution_id=execution.run_id)
    )
    ledger.save(attempt)
    try:
        execution = runner.execute(spec, strategy=strategy, execution_id=execution.run_id)
        return Attempt.model_validate(
            attempt.model_dump() | dict(status="completed", content_id=execution.content_id)
        )
    except Exception as exc:
        return Attempt.model_validate(
            attempt.model_dump() | dict(status="failed", reason=f"{type(exc).__name__}: {exc}")
        )
