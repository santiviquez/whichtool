import json
import sys
from types import SimpleNamespace

from whichtool import config as C
from whichtool.config import MODELS, STUDIES
from whichtool.harness import Config
from whichtool.run import expected_cost, keep_console_log, observed_costs, observed_reserves, record_outcome
from whichtool.taskspec import Variant


def test_console_output_is_kept_in_the_run_folder(tmp_path):
    out, err = sys.stdout, sys.stderr
    try:
        keep_console_log(tmp_path)
        print("plan line")
        print("Stopping something", file=sys.stderr)
    finally:
        sys.stdout, sys.stderr = out, err
    keep_console_log(tmp_path)  # a second session appends instead of overwriting
    sys.stdout, sys.stderr = out, err
    log = (tmp_path / "console.log").read_text()
    assert "plan line" in log and "Stopping something" in log
    assert log.count("===== ") == 2


def test_each_session_records_how_it_ended(tmp_path):
    (tmp_path / "run.json").write_text(json.dumps({"sessions": [{"started_at": "a"}, {"started_at": "b"}]}))
    writer = SimpleNamespace(n_written=6, n_errors=1, n_skipped=0, spent_usd=0.8,
                             broken={("headline", "gpt-5.6-sol", "medium"): "APIError: no credits"})
    record_outcome(tmp_path, writer, left=1)
    sessions = json.loads((tmp_path / "run.json").read_text())["sessions"]
    assert "finished_at" not in sessions[0]  # earlier sessions are left as they were
    assert sessions[1]["calls_failed"] == 1 and sessions[1]["calls_without_success_left"] == 1
    assert sessions[1]["stopped_configurations"] == {"headline gpt-5.6-sol medium": "APIError: no credits"}


def test_expected_cost_uses_the_mean_of_earlier_calls(tmp_path):
    run = tmp_path / "20260101T000000Z-headline"
    run.mkdir()
    (run / "calls.csv").write_text(
        "study,model,effort,status,cost_usd\n"
        "headline,opus-5.5,high,ok,0.10\n"
        "headline,opus-5.5,high,ok,0.30\n"
        "headline,opus-5.5,high,api_error,0\n"
    )
    costs = observed_costs(tmp_path)
    assert costs == {("headline", "opus-5.5", "high"): [0.10, 0.30]}
    assert observed_reserves(tmp_path) == {("headline", "opus-5.5", "high"): 0.30}
    v = Variant("j9-v00", "j9", "text")
    seen = Config(STUDIES["headline"], MODELS["opus-5.5"], "high", [v], 1)
    new = Config(STUDIES["headline"], MODELS["opus-5.5"], "medium", [v], 1)
    todo = [(seen, v, 1), (seen, v, 2), (new, v, 1)]
    assert abs(expected_cost(todo, costs) - (0.20 * 2 + C.RESERVE_USD)) < 1e-9


def test_each_task_has_its_own_budget(tmp_path, monkeypatch, capsys):
    from pathlib import Path

    from whichtool import run
    from whichtool.taskspec import load_task

    fixtures = Path(__file__).parent / "fixtures"
    monkeypatch.setattr(run, "load_task", lambda name: load_task(name, fixtures))
    monkeypatch.setattr(run, "RESULTS_DIR", tmp_path)
    for task, cost in (("example", "0.25"), ("another-task", "180")):
        d = tmp_path / task / "20260101T000000Z-headline"
        d.mkdir(parents=True)
        (d / "calls.csv").write_text(
            "call_id,run_kind,study,model,effort,status,cost_usd\n"
            f"c1,study,headline,opus-5.5,high,ok,{cost}\n"
        )
    run.main(["--task", "example", "--allow-unfrozen", "--dry-run"])
    assert f"Spent so far on example: $0.25 of ${C.BUDGET_USD:.0f}" in capsys.readouterr().out
