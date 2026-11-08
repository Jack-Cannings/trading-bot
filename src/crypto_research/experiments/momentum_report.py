"""Descriptive reporting only: offline joins cannot become strategy inputs."""

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from crypto_research.experiments.momentum_campaign import Attempt
from crypto_research.experiments.momentum_plan import PAIRS, CampaignPlan
from crypto_research.experiments.registry import Registry
from crypto_research.experiments.storage import checksum


def trial_rows(attempts: tuple[Attempt, ...], registry: Registry) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for attempt in attempts:
        row: dict[str, object] = dict(
            market=attempt.pair,
            strategy=attempt.configuration.strategy_id,
            lookback=next(
                (p.value for p in attempt.configuration.parameters if p.name == "lookback"), None
            ),
            configuration=attempt.configuration.identity,
            window=attempt.window,
            start=attempt.period.start.isoformat() if attempt.period else None,
            end=attempt.period.end.isoformat() if attempt.period else None,
            cost_multiplier=attempt.cost_multiplier,
            status=attempt.status,
            reason=attempt.reason,
            execution_id=attempt.execution_id,
            content_id=attempt.content_id,
        )
        if attempt.execution_id and attempt.status == "completed":
            metrics = registry.summary(attempt.execution_id)
            spec = registry.spec(registry.get(attempt.execution_id).experiment_id)
            row.update(
                metrics.model_dump(
                    mode="json",
                    exclude={
                        "settings",
                        "period",
                        "initial_equity",
                        "source_run_id",
                        "cumulative",
                        "undefined",
                    },
                )
            )
            row.update(
                experiment_id=spec.experiment_id,
                input_hash=spec.input_hash,
                comparison_group=spec.comparison_group,
                dataset_version=spec.dataset.version,
                dataset_checksum=spec.dataset.sha256,
                replay_availability=spec.replay_availability,
                undefined=json.dumps([v.model_dump() for v in metrics.undefined], sort_keys=True),
                total_costs=(metrics.total_fees or 0) + (metrics.total_slippage or 0),
            )
        rows.append(row)
    return rows


def diagnostic_rows(attempt: Attempt, registry: Registry) -> list[dict[str, object]]:
    if (
        attempt.execution_id is None
        or attempt.status != "completed"
        or attempt.configuration.strategy_id != "time_series_momentum"
    ):
        return []
    root = registry.root / "artifacts" / attempt.execution_id
    execution = registry.get(attempt.execution_id)
    # The registry anchors this bundle; report joins happen only after execution.
    from crypto_research.experiments.storage import ResearchStore

    ResearchStore(registry.root / "artifacts").verify(
        attempt.execution_id, execution.manifest_sha256, full=False
    )
    decisions = pq.read_table(root / "decisions.parquet").to_pylist()
    labels = pq.read_table(root / "outcomes.parquet").to_pylist()
    grouped: dict[tuple[str, str, int], list[dict[str, object]]] = {}
    by_id = {d["decision_id"]: d for d in decisions}
    for axis, feature in (("direction", "return_30d"), ("volatility", "volatility_30d")):
        episode, previous_category, previous_time = 0, None, None
        categories = {}
        for d in decisions:
            value = d[feature]
            category = (
                "missing"
                if value is None
                else ("positive" if value > 0 else "nonpositive")
                if axis == "direction"
                else "missing"
                if value is None
                else "high"
                if value >= 0.60
                else "low"
            )
            if (
                category != previous_category
                or previous_time is None
                or (d["timestamp"] - previous_time).total_seconds() > 86400
            ):
                episode += 1
            categories[d["decision_id"]] = (category, episode)
            previous_category, previous_time = category, d["timestamp"]
        for label in labels:
            category, episode = categories[label["decision_id"]]
            key = (axis, category, label["horizon_days"])
            grouped.setdefault(key, []).append(dict(label, episode=episode))
    rows: list[dict[str, object]] = []
    for (axis, category, horizon), observations in sorted(grouped.items()):
        complete = [o for o in observations if o["status"] == "complete"]
        returns = [float(str(o["net_return"])) for o in complete]
        # Count a greedily thinned subset, not an independence assertion.
        nonoverlap, last_end = 0, None
        for observation in complete:
            start = by_id[observation["decision_id"]]["timestamp"]
            if last_end is None or start >= last_end:
                nonoverlap += 1
                last_end = observation["end"]
        rows.append(
            dict(
                execution_id=attempt.execution_id,
                market=attempt.pair,
                configuration=attempt.configuration.identity,
                window=attempt.window,
                cost_multiplier=attempt.cost_multiplier,
                axis=axis,
                category=category,
                horizon_days=horizon,
                observations=len(observations),
                complete=len(complete),
                incomplete=len(observations) - len(complete),
                episodes=len({o["episode"] for o in complete}),
                nonoverlapping_windows=nonoverlap,
                mean_net_return=float(np.mean(returns)) if returns else None,
                median_net_return=float(np.median(returns)) if returns else None,
            )
        )
    return rows


def comparisons(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, momentum in table.iterrows():
        if momentum["status"] != "completed" or momentum["strategy"] != "time_series_momentum":
            continue
        matched = table[
            (table["comparison_group"] == momentum["comparison_group"])
            & (table["window"] == momentum["window"])
            & (table["strategy"].isin(["cash", "benchmark", "passive_risk_target"]))
            & (table["status"] == "completed")
        ]
        for _, benchmark in matched.iterrows():
            row = {
                k: momentum[k]
                for k in ("market", "lookback", "window", "cost_multiplier", "comparison_group")
            }
            row["benchmark"] = benchmark["strategy"]
            for metric in (
                "total_net_return",
                "maximum_drawdown",
                "sharpe_ratio",
                "average_exposure",
            ):
                a, b = momentum[metric], benchmark[metric]
                row[metric + "_difference"] = float(a - b) if pd.notna(a) and pd.notna(b) else None
            rows.append(row)
    return pd.DataFrame.from_records(rows)


def descriptive_findings(
    table: pd.DataFrame, matched: pd.DataFrame, diagnostics: list[dict[str, object]]
) -> list[str]:
    """Report all predefined candidates; no ranking, selection or significance test."""
    base = table[
        (table["status"] == "completed")
        & (table["strategy"] == "time_series_momentum")
        & (table["cost_multiplier"] == 1)
    ]
    if base.empty:
        return []
    lines = ["", "### Descriptive findings (no lookback selection)", ""]
    for pair in PAIRS:
        sample = base[(base["market"] == pair) & (base["window"] == "per-market")]
        validation = base[(base["market"] == pair) & base["window"].str.startswith("validation")]
        if sample.empty:
            lines.append(f"{pair}: no completed market results.")
            continue
        cagr, returns = sample["cagr"].dropna(), sample["total_net_return"].dropna()
        lines.append(
            f"{pair}: all five per-market baseline candidates have net-return range "
            f"{returns.min():.2%} to {returns.max():.2%}; CAGR range "
            f"{cagr.min():.2%} to {cagr.max():.2%}. "
            f"{int((returns > 0).sum())}/{len(returns)} positive net returns. "
            f"Chronological validation: {int((validation['total_net_return'] > 0).sum())}"
            f"/{len(validation)} positive candidate-window returns; range "
            f"{validation['total_net_return'].min():.2%} to "
            f"{validation['total_net_return'].max():.2%}."
        )
        if not matched.empty:
            scope = (
                "common-period" if (matched["window"] == "common-period").any() else "per-market"
            )
            groups = matched[
                (matched["market"] == pair)
                & (matched["window"] == scope)
                & (matched["cost_multiplier"] == 1)
            ]
            for benchmark, group in groups.groupby("benchmark"):
                difference = group["total_net_return_difference"].dropna()
                sharpe = group["sharpe_ratio_difference"].dropna()
                lines.append(
                    f"{pair}, {scope}, vs {benchmark}: "
                    f"{int((difference > 0).sum())}/{len(difference)} have higher net return; "
                    f"{int((sharpe > 0).sum())}/{len(sharpe)} have higher defined Sharpe. "
                    "Undefined cash Sharpe is not treated as zero. "
                    "See matched drawdown/exposure differences."
                )
        doubled = table[
            (table["market"] == pair)
            & (table["window"] == "per-market")
            & (table["strategy"] == "time_series_momentum")
            & (table["status"] == "completed")
            & (table["cost_multiplier"] == 2)
        ]
        paired = sample.merge(doubled, on="configuration", suffixes=("_base", "_double"))
        if not paired.empty:
            change = paired["total_net_return_double"] - paired["total_net_return_base"]
            lines.append(
                f"{pair}: doubled-minus-baseline cost net-return change range "
                f"{change.min():.2%} to {change.max():.2%} across the five candidates."
            )
    condition_table = pd.DataFrame.from_records(diagnostics)
    if not condition_table.empty:
        sample = condition_table[
            (condition_table["window"] == "per-market")
            & (condition_table["cost_multiplier"] == 1)
            & (condition_table["horizon_days"] == 7)
        ]
        for (group_pair, axis, category), group in sample.groupby(["market", "axis", "category"]):
            means = group["mean_net_return"].dropna()
            if not means.empty:
                lines.append(
                    f"{group_pair}, {axis}={category}: seven-day conditional mean-return range "
                    f"{means.min():.3%} to {means.max():.3%} across candidates; "
                    f"complete observations per candidate {group['complete'].min()}"
                    f"–{group['complete'].max()}, "
                    f"episodes {group['episodes'].min()}–{group['episodes'].max()}. "
                    "Descriptive overlapping strategy outcomes, not independent event estimates."
                )
    return lines


def _save(fig: Figure, destination: Path) -> None:
    fig.tight_layout()
    fig.savefig(destination, metadata={"Date": None})
    plt.close(fig)


def plots(table: pd.DataFrame, registry: Registry, root: Path, fixture: bool) -> None:
    completed = table[table["status"] == "completed"]
    if completed.empty:
        return
    plt.rcParams["svg.hashsalt"] = "time-series-momentum-v1"
    scope = "common-period" if (completed["window"] == "common-period").any() else "per-market"
    sample = completed[(completed["window"] == scope) & (completed["cost_multiplier"] == 1)]
    fig, axes = plt.subplots(3, 2, figsize=(15, 12), squeeze=False)
    for index, pair in enumerate(PAIRS):
        for _, row in sample[sample["market"] == pair].iterrows():
            frame = pq.read_table(
                registry.root / "artifacts" / str(row["execution_id"]) / "performance.parquet"
            ).to_pandas()
            values = frame["equity"].astype(float).to_numpy()
            dates = frame["timestamp"]
            label = (
                f"TSMOM-{int(row['lookback'])}"
                if pd.notna(row["lookback"])
                else str(row["strategy"])
            )
            axes[index, 0].plot(dates, values, label=label, linewidth=1)
            # Include starting capital in peak, including first-entry costs.
            peaks = np.maximum.accumulate(np.r_[10000.0, values])[1:]
            axes[index, 1].plot(dates, values / peaks - 1, label=label, linewidth=1)
        axes[index, 0].set_title(f"{pair}: net equity (USDT)")
        axes[index, 1].set_title(f"{pair}: drawdown (fraction)")
        axes[index, 0].legend(fontsize=7)
    fig.suptitle(
        ("SYNTHETIC FIXTURE — " if fixture else "Conditional archival simulation — ") + scope
    )
    _save(fig, root / "equity-drawdown.svg")
    momentum = completed[completed["strategy"] == "time_series_momentum"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), squeeze=False)
    for index, pair in enumerate(PAIRS):
        for cost, group in momentum[
            (momentum["market"] == pair) & (momentum["window"] == scope)
        ].groupby("cost_multiplier"):
            ordered = group.sort_values("lookback")
            axes[0, index].plot(
                ordered["lookback"], ordered["total_net_return"], marker="o", label=f"costs x{cost}"
            )
        axes[0, index].set(
            title=pair, xlabel="Predefined lookback (days)", ylabel="Net return (fraction)"
        )
        axes[0, index].legend()
    _save(fig, root / "lookback-cost-sensitivity.svg")
    validation = momentum[
        (momentum["window"].str.startswith("validation")) & (momentum["cost_multiplier"] == 1)
    ]
    if not validation.empty:
        grid = validation.pivot_table(
            index=["market", "lookback"], columns="window", values="total_net_return"
        )
        ax = grid.plot.bar(
            figsize=(15, 5),
            ylabel="Net return (fraction)",
            title="Chronological validation stability; no holdout evaluation",
        )
        figure = ax.get_figure()
        if isinstance(figure, Figure):
            _save(figure, root / "validation-stability.svg")
    comparison = comparisons(sample)
    if not comparison.empty:
        grid = comparison.pivot_table(
            index=["market", "lookback"], columns="benchmark", values="total_net_return_difference"
        )
        ax = grid.plot.bar(
            figsize=(15, 5),
            ylabel="Net return difference (fraction)",
            title="Matched-period and matched-cost benchmark differences",
        )
        figure = ax.get_figure()
        if isinstance(figure, Figure):
            _save(figure, root / "benchmark-differences.svg")


def write_report(
    plan: CampaignPlan, attempts: tuple[Attempt, ...], registry: Registry, root: Path
) -> None:
    rows = trial_rows(attempts, registry)
    table = pd.DataFrame.from_records(rows)
    table.to_csv(root / "results.csv", index=False)
    matched = comparisons(table)
    matched.to_csv(root / "benchmark-differences.csv", index=False)
    diagnostics = [row for attempt in attempts for row in diagnostic_rows(attempt, registry)]
    pd.DataFrame.from_records(diagnostics).to_csv(root / "conditions.csv", index=False)
    numeric = [{k: v for k, v in row.items() if k not in {"execution_id"}} for row in rows]
    (root / "numerical-results.json").write_text(
        json.dumps(numeric, sort_keys=True, indent=2, allow_nan=False), encoding="utf-8"
    )
    plots(table, registry, root, plan.fixture)
    lines = [
        "# Time-series momentum research",
        "",
        "**SYNTHETIC FIXTURE: no empirical market evidence.**"
        if plan.fixture
        else "**Conditional archival simulation; historical publication times unverified.**"
        if plan.replay_availability != "observed"
        else "Observed-time replay. Unavailable trials have no fabricated performance.",
        "",
        f"Campaign `{plan.identity}`; code `{plan.code_revision}`. "
        "Plan saved before strategy evaluation.",
        "Lookbacks: 30/60/90/120/180 daily candles. M = "
        "close[t]/close[t-L]-1; strictly positive => 100% long, otherwise "
        "cash. L+1 contiguous completed observations required.",
        "Starting cash 10,000 USDT per trial. No shorts/leverage. All trials "
        "use the existing Decimal portfolio engine and metrics module.",
        f"Baseline modeled taker fee {plan.research.fee_bps} bps and "
        f"adverse slippage {plan.research.slippage_bps} bps per fill; "
        f"sensitivity doubles both. These are configured assumptions, not "
        f"an exchange fee-tier quotation. Buy costs reduce affordable "
        f"quantity; sells charge fees; fees/slippage are already included "
        f"in net equity.",
        f"Replay policy: `{plan.replay_availability}`. Raw source "
        f"observations and hashes are preserved. Archival mode explicitly "
        f"ASSUMES finalized bars were published 60 seconds after close, "
        f"then waits for the next daily open. Historical "
        f"revision/publication timing is unverified; this assumption is not "
        f"a measured timestamp and default observed-time replay never "
        f"backdates candles.",
        "Cash, same-coin buy-and-hold and causal passive risk-target "
        "benchmark share periods, starting cash, costs and valuation. The "
        "risk benchmark rebalances daily to min(1, 20% / trailing 30-day "
        "sample log volatility annualized by sqrt(365.25)); zero volatility "
        "=> cap 1. No leverage or ex-post risk matching. It does not "
        "establish equal realized risk with momentum.",
        "",
        "## Coverage and protected splits",
        "",
        "201-day warmup. Development is 180–365 days by the predeclared "
        "coverage rule; subsequent validation is split into two windows of at"
        " least 90 days where possible. Reserve final 365 days if feasible; "
        "otherwise max(90 days, 20% of eligible history), requiring 180 "
        "development + 90 validation days. No evaluation of the holdout. "
        "Integrity verification hashes source bytes, but heldout prices are "
        "excluded from the engine/strategy tape. Aggregate per-market/common "
        "results overlap development/validation and are exploratory, not "
        "additional independent evidence.",
        "",
    ]
    for selection in plan.selections:
        lines.extend(
            [
                f"### {selection.market.pair}",
                "",
                f"Source: "
                f"`{selection.dataset.version if selection.dataset else 'unavailable'}`; "
                f"checksum "
                f"`{selection.dataset.sha256 if selection.dataset else 'unavailable'}`.",
                f"Coverage: {selection.coverage}; candles {selection.candle_count}; "
                f"quality {selection.quality}; acquired {selection.acquired_at}; "
                f"artifact SHA {selection.artifact_sha256}.",
                f"Reported missing spans: {selection.missing_intervals}. Only "
                f"pre-listing gaps are permissible; interior gaps/anomalies reject "
                f"the selection.",
                f"Unavailable reason: {selection.unavailable_reason or 'none'}.",
                f"Protected holdout: {selection.holdout}.",
                *[
                    f"- {w.name}: {w.period.start.isoformat()} — "
                    f"{w.period.end.isoformat()} ({w.split})"
                    for w in selection.windows
                ],
                "",
            ]
        )
    lines.extend(
        [
            f"Common period: {plan.common or 'insufficient overlapping eligible coverage'}.",
            "",
            "## All trials",
            "",
            f"Attempts: {len(attempts)}; completed: "
            f"{sum(a.status == 'completed' for a in attempts)}; failed: "
            f"{sum(a.status == 'failed' for a in attempts)}; unavailable: "
            f"{sum(a.status == 'unavailable' for a in attempts)}. These counts "
            f"include all costs, windows and three benchmarks, in addition to "
            f"the initial 15 strategy-market combinations. Repeated runs are "
            f"new executions of the same logical configurations.",
            "All results, undefined-metric reasons and hashes: "
            "[results.csv](results.csv). Full metric records include "
            "annualized volatility, cumulative returns, drawdown duration, "
            "turnover and time invested. Fills are counted as trades; "
            "completed-roundtrip win rate, profit factor and holding duration"
            " are unsupported by the current accounting summary and are not "
            "fabricated.",
            "",
            "| Market | Scope | Costs | Strategy/lookback | Status | Net return | CAGR"
            " | Sharpe | Sortino | Calmar | MDD | CVaR95 | Costs USDT | "
            "Exposure | Fills |",
            "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        label = str(row["strategy"]) + (f"-{row['lookback']}" if row["lookback"] else "")
        values = [
            row.get(name)
            for name in (
                "total_net_return",
                "cagr",
                "sharpe_ratio",
                "sortino_ratio",
                "calmar_ratio",
                "maximum_drawdown",
                "cvar_95",
                "total_costs",
                "average_exposure",
                "number_of_trades",
            )
        ]
        cells = [
            "undefined"
            if value is None
            else f"{value:.6g}"
            if isinstance(value, (int, float))
            else str(value)
            for value in values
        ]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["market"]),
                    str(row["window"]),
                    str(row["cost_multiplier"]),
                    label,
                    str(row["status"]),
                    *cells,
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Observable conditions",
            "",
            "[conditions.csv](conditions.csv) joins only completed "
            "simulations to separate 1/7/30-day outcome labels. Direction: "
            "trailing 30-day return > 0 vs <= 0. Volatility: trailing "
            "annualized 30-day volatility >= 60% vs < 60%. Thresholds fixed "
            "before evaluation, never full-sample fitted. Each axis is "
            "analyzed separately. Missing features/outcomes remain explicit. "
            "Counts include distinct consecutive category episodes and a "
            "greedily thinned non-overlapping horizon count. Neither "
            "establishes statistical independence. Long horizons overlap; "
            "markets, costs and fold/aggregate runs are correlated. No regime"
            " model or conditional allocation is trained.",
            "",
            "## Figures",
            "",
            "![Equity and drawdown](equity-drawdown.svg)",
            "![Benchmarks](benchmark-differences.svg)",
            "![Chronological validation](validation-stability.svg)",
            "![Lookback and costs](lookback-cost-sensitivity.svg)",
            "",
            "## Research conclusions",
            "",
        ]
    )
    completed = [
        r
        for r in rows
        if r["status"] == "completed"
        and r["strategy"] == "time_series_momentum"
        and r["window"] == "per-market"
        and r["cost_multiplier"] == 1
    ]
    if not completed or plan.fixture:
        lines.extend(
            [
                "1. Positive net market performance: not established. "
                "Fixtures verify machinery only.",
                "2. Fair-benchmark outperformance: no empirical conclusion.",
                "3. Chronological stability: no empirical conclusion.",
                "4. Promising cryptocurrencies: none can be identified from these runs.",
                "5. Observable conditions: descriptive fixture/unavailable output only.",
                "6. Lookback/cost sensitivity: implementation verified; "
                "empirical evidence incomplete.",
            ]
        )
    else:
        positive = sum(float(str(r["total_net_return"])) > 0 for r in completed)
        lines.extend(
            [
                f"1. {positive}/{len(completed)} baseline per-market momentum "
                f"trials have positive net returns under the declared archival "
                f"assumption. This is historical evidence, not future "
                f"profitability.",
                "2. Matched benchmark differences, including Sharpe, drawdown"
                " and exposure, are in benchmark-differences.csv; lower "
                "drawdown alone is not signal advantage.",
                "3. Validation-window results and validation-stability.svg "
                "expose time variation. Overlapping aggregates cannot be "
                "counted as new replications.",
                "4. Per-market and common-period tables show market "
                "dependence. No market is designated a proven winner or a "
                "deployment recommendation.",
                "5. conditions.csv describes stronger/weaker category means "
                "with episode/sample counts. Correlation, sparse episodes and"
                " overlapping horizons prevent reliable conditional-edge "
                "claims.",
                "6. lookback-cost-sensitivity.svg and complete "
                "baseline/doubled-cost rows show predefined parameter and "
                "cost sensitivity. No optimal lookback is selected.",
            ]
        )
    lines.extend(
        [
            "7. Missing evidence: verified historical publication/revision "
            "vintages, executable spread/liquidity/lot-size constraints, "
            "independent market samples, formal multiple-testing correction, "
            "and an untouched final-holdout assessment. Source "
            "coverage/failed trials above may impose further limits. No "
            "hypothetical unavailability is converted into a cash return.",
            "8. Retain momentum as a transparent candidate for further "
            "research, conditional on evidence and future formal validation; "
            "no strategy or lookback is approved for trading. Five lookbacks "
            "× three markets plus windows/costs create multiple comparisons; "
            "best observed results do not prove an edge.",
            "",
            "## Artifacts and reproducibility",
            "",
            f"Registry: `{registry.db_path}`. Source: explicit immutable "
            f"versions in plan.json. Per-execution bundles: "
            f"`{registry.root / 'artifacts'}/<execution_id>/` containing "
            f"decisions, performance, trades, separate outcomes, spec, result, "
            f"metrics and checksummed manifest. Decisions expose "
            f"momentum/lookback plus all point-in-time features; performance "
            f"exposes quantity_before/quantity after execution. Simulated "
            f"outcomes never enter decision inputs.",
            "Repeat the saved plan with run-momentum --plan plan.json. New "
            "UUIDs identify executions; identical logical inputs must "
            "reproduce content IDs and numerical-results.json. Report "
            "manifest hashes every generated file. No acquisition occurs in "
            "the campaign. Code SHA is caller-declared; use a clean committed"
            " checkout for reproducibility.",
        ]
    )
    if any(a.status == "completed" for a in attempts):
        lines.extend(descriptive_findings(table, matched, diagnostics))
    (root / "report.md").write_text("\n\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        p.name: checksum(p)
        for p in sorted(root.iterdir())
        if p.is_file() and p.name != "manifest.json"
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8"
    )
