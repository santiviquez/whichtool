"""End to end through Inspect with its mock model: no API calls, no cost."""

import csv
import json

from inspect_ai import eval as inspect_eval
from inspect_ai.model import ModelOutput, get_model

from whichtool.config import STUDIES, ModelSpec
from whichtool.harness import RunWriter, build_task
from whichtool.taskspec import load_task

MOCK = ModelSpec("mock", "mockllm", "model", 1.0, 1.0, 1.0, 1.0)


def _run(tmp_path, allowed_usd=10.0, reserve_usd=0.0, outputs=None):
    task = load_task("example")
    variants = task.variants
    writer = RunWriter(
        run_dir=tmp_path,
        run_id="test",
        run_kind="study",
        task_name=task.name,
        variants_sha=task.variants_sha,
        allowed_usd=allowed_usd,
        reserve_usd=reserve_usd,
    )
    content = '{"api": "Postmark", "reason": "latency", "sentence": "Fastest delivery."}'
    model = get_model(
        "mockllm/model",
        custom_outputs=outputs
        or [ModelOutput.from_content(model="mockllm/model", content=content) for _ in range(10)],
    )
    t = build_task(task, STUDIES["headline"], MOCK, "medium", variants, 2, writer)
    t.model = model
    inspect_eval(t, log_dir=str(tmp_path / "logs"), display="none", score=False)
    return writer


def test_rows_and_raw_written(tmp_path):
    w = _run(tmp_path)
    rows = list(csv.DictReader((tmp_path / "calls.csv").open()))
    raws = [json.loads(line) for line in (tmp_path / "raw.jsonl").open()]
    assert len(rows) == 4 == len(raws)  # 2 variants x 2 epochs
    assert {r["variant_id"] for r in rows} == {"j1-v00", "j1-v01"}
    assert {r["epoch"] for r in rows} == {"1", "2"}
    assert all(r["api"] == "Postmark" and r["parse_ok"] == "True" for r in rows)
    assert all(r["status"] == "ok" for r in rows)
    # the full prompt is stored, not an attachment reference
    assert raws[0]["messages"][0]["content"].startswith("Recommend the one hosted API")
    assert "attachment://" not in json.dumps(raws[0]["messages"])
    assert len({r["call_id"] for r in rows}) == 4
    assert w.n_written == 4


def test_budget_guard_skips(tmp_path):
    # reserve > allowed: nothing may start
    w = _run(tmp_path, allowed_usd=0.10, reserve_usd=1.0)
    rows = list(csv.DictReader((tmp_path / "calls.csv").open()))
    assert w.n_written == 0 and w.n_skipped == 4
    assert all(r["status"] == "skipped_budget" for r in rows)
    assert not (tmp_path / "raw.jsonl").exists()
