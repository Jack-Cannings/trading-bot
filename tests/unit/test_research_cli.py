import json
from pathlib import Path

import pytest

from crypto_research.experiments.cli import main
from crypto_research.experiments.registry import Registry


def test_cli_complete_demo_and_inspection(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = ["--root", str(tmp_path / "research"), "--data-root", str(tmp_path / "data")]
    assert main(args + ["run-demo", "--revision", "a" * 40]) == 0
    result = json.loads(capsys.readouterr().out)
    assert len(result) == 3 and {r["status"] for r in result} == {"completed"}
    scripted = next(r for r in result if r["experiment_id"].endswith("strategy"))
    cash = next(r for r in result if r["experiment_id"].endswith("cash"))
    assert cash["metrics"]["total_net_return"] == 0 and cash["metrics"]["sharpe_ratio"] is None
    assert scripted["metrics"]["number_of_trades"] == 1
    assert 0.09 < scripted["metrics"]["total_net_return"] < 0.1
    assert main(args + ["show", scripted["experiment_id"]]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["runs"][0]["metrics"]["number_of_trades"] == 1
    assert len(shown["input_hash"]) == 64
    assert main(args + ["list"]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 3
    assert main(args + ["compare"]) == 0
    table = capsys.readouterr().out
    assert "comparison_group" in table and "cash" in table and "undefined" in table
    assert main(args + ["run-demo", "--revision", "a" * 40]) == 0
    repeated = json.loads(capsys.readouterr().out)
    assert [r["content_id"] for r in repeated] == [r["content_id"] for r in result]
    assert {r["run_id"] for r in repeated}.isdisjoint({r["run_id"] for r in result})


def test_cli_empty_missing_and_recovery(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "research"
    args = ["--root", str(root)]
    assert main(args + ["compare"]) == 0
    assert "No experiment" in capsys.readouterr().out
    assert main(args + ["show", "missing"]) == 2
    assert "unknown experiment" in capsys.readouterr().err
    assert main(args + ["recover", "missing"]) == 2
    assert capsys.readouterr().err
    from research_helpers import case

    _, registry, _, spec, _ = case(tmp_path)
    registry.register(spec)
    execution = registry.create_run(spec.experiment_id)
    assert main(args + ["recover", execution.run_id]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "cancelled"
    assert Registry(root).get(execution.run_id).status == "cancelled"


def test_invalid_demo_revision_rejected(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        main(
            [
                "--root",
                str(tmp_path / "research"),
                "--data-root",
                str(tmp_path / "data"),
                "run-demo",
                "--revision",
                "invalid",
            ]
        )
        == 2
    )
    assert "code_revision" in capsys.readouterr().err
