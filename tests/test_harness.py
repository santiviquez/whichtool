"""End to end with fake SDK clients: no API calls, no cost."""

import asyncio
import csv
import json

import pytest

from anthropic.types import Message
from openai.types.responses import Response

from whichtool.config import MODELS, STUDIES
from whichtool.harness import Config, RunWriter, build_request, run_plan
from whichtool.taskspec import load_task

ANSWER = '{"api": "Postmark", "reason": "latency", "sentence": "Fastest delivery."}'
USAGE = {"input_tokens": 100, "output_tokens": 50}


def claude_msg(content, stop_reason="end_turn", **usage):
    return Message.model_validate({
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
        "content": content, "stop_reason": stop_reason, "stop_sequence": None,
        "usage": {**USAGE, **usage},
    })


class FakeAnthropic:
    """Answers each stream() call with the next message in the list (the last one repeats)."""

    def __init__(self, messages=None, error=None):
        self.messages = self
        self.queue = list(messages or [claude_msg([{"type": "text", "text": ANSWER}])])
        self.error = error
        self.requests = []

    def stream(self, **req):
        self.requests.append(req)
        fake = self

        class _Stream:
            async def __aenter__(self):
                if fake.error:
                    raise fake.error
                return self

            async def __aexit__(self, *exc):
                return False

            async def get_final_message(self):
                return fake.queue.pop(0) if len(fake.queue) > 1 else fake.queue[0]

        return _Stream()


class FakeOpenAI:
    def __init__(self):
        self.responses = self
        self.requests = []

    async def create(self, **req):
        self.requests.append(req)
        return Response.model_validate({
            "id": "resp_1", "object": "response", "created_at": 0, "model": "gpt-6-astra",
            "status": "completed", "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
            "output": [
                {"type": "web_search_call", "id": "ws_1", "status": "completed",
                 "action": {"type": "search", "query": "email api latency",
                            "sources": [{"type": "url", "url": "https://example.com"}]}},
                {"type": "message", "id": "m_1", "role": "assistant", "status": "completed",
                 "content": [{"type": "output_text", "text": ANSWER, "annotations": []}]},
            ],
            "usage": {"input_tokens": 100, "output_tokens": 50,
                      "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                      "output_tokens_details": {"reasoning_tokens": 30}, "total_tokens": 150},
        })


def _run(tmp_path, clients, model="opus-5.5", study="headline", epochs=2,
         allowed_usd=10.0, reserve_usd=0.0):
    task = load_task("example")
    writer = RunWriter(run_dir=tmp_path, run_id="test", run_kind="study", task_name=task.name,
                       variants_sha=task.variants_sha, allowed_usd=allowed_usd,
                       reserve_usd=reserve_usd)
    plan = [Config(STUDIES[study], MODELS[model], "high", task.variants, epochs)]
    asyncio.run(run_plan(clients, task, plan, writer))
    rows = list(csv.DictReader((tmp_path / "calls.csv").open()))
    raw_path = tmp_path / "raw.jsonl"
    raws = [json.loads(line) for line in raw_path.open()] if raw_path.exists() else []
    return writer, rows, raws


def test_rows_and_raw_written(tmp_path):
    fake = FakeAnthropic()
    w, rows, raws = _run(tmp_path, {"anthropic": fake})
    assert len(rows) == 4 == len(raws)  # 2 variants x 2 epochs
    assert {r["variant_id"] for r in rows} == {"j1-v00", "j1-v01"}
    assert {r["epoch"] for r in rows} == {"1", "2"}
    assert all(r["api"] == "Postmark" and r["parse_ok"] == "True" for r in rows)
    assert all(r["status"] == "ok" and r["model_returned"] == "claude-opus-5-5" for r in rows)
    assert len({r["call_id"] for r in rows}) == 4
    # Opus 5.5 at $4 / $20: 100 in + 50 out
    assert float(rows[0]["cost_usd"]) == (100 * 4 + 50 * 20) / 1e6
    # the stored request is exactly what the client received
    assert raws[0]["request"] in fake.requests
    assert raws[0]["request"]["system"].startswith("Recommend the one hosted API")
    assert w.n_written == 4 and w.n_errors == 0


def test_claude_request_shape():
    spec, study = MODELS["fable-5.1"], STUDIES["search"]
    req = build_request(spec, study, "high", "instr", "job text")
    assert req["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert req["output_config"]["effort"] == "high"
    assert req["output_config"]["format"]["type"] == "json_schema"
    assert req["tools"] == [{"type": "web_search_20260209", "name": "web_search"}]  # no web_fetch
    code = build_request(spec, STUDIES["code"], "high", "instr", "job text")
    assert "format" not in code["output_config"] and "tools" not in code
    no_schema = build_request(spec, study, "high", "instr", "job text", schema=False)
    assert "format" not in no_schema["output_config"]


def test_openai_request_and_search(tmp_path):
    fake = FakeOpenAI()
    _, rows, raws = _run(tmp_path, {"openai": fake}, model="gpt-6-astra", study="search", epochs=1)
    req = fake.requests[0]
    assert req["store"] is False and "previous_response_id" not in req
    assert req["reasoning"] == {"effort": "high", "summary": "auto"}
    assert req["text"]["format"]["strict"] is True
    assert req["tools"] == [{"type": "web_search"}]
    assert rows[0]["api"] == "Postmark" and rows[0]["searched"] == "True"
    assert rows[0]["n_searches"] == "1" and rows[0]["reasoning_tokens"] == "30"
    assert raws[0]["urls"] == ["https://example.com"]


def test_pause_turn_is_continued(tmp_path):
    paused = claude_msg(
        [{"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search",
          "input": {"query": "transactional email api"}}],
        stop_reason="pause_turn", server_tool_use={"web_search_requests": 1, "web_fetch_requests": 0},
    )
    done = claude_msg([{"type": "text", "text": ANSWER}])
    fake = FakeAnthropic([paused, done])
    task = load_task("example")
    writer = RunWriter(run_dir=tmp_path, run_id="t", run_kind="study", task_name=task.name,
                       variants_sha="", allowed_usd=10, reserve_usd=0)
    plan = [Config(STUDIES["search"], MODELS["opus-5.5"], "high", task.variants[:1], 1)]
    asyncio.run(run_plan({"anthropic": fake}, task, plan, writer))
    row = next(csv.DictReader((tmp_path / "calls.csv").open()))
    assert row["n_api_calls"] == "2" and row["api"] == "Postmark"
    assert row["n_searches"] == "1" and row["search_queries"] == "transactional email api"
    # the second request sends the paused turn back, with no extra user message
    second = fake.requests[1]["messages"]
    assert [m["role"] for m in second] == ["user", "assistant"]
    # usage is summed over both calls: 200 in, 100 out, 1 search
    assert float(row["cost_usd"]) == round((200 * 4 + 100 * 20) / 1e6 + 0.01, 6)


def test_api_error_is_recorded(tmp_path):
    w, rows, raws = _run(tmp_path, {"anthropic": FakeAnthropic(error=RuntimeError("boom"))}, epochs=1)
    assert all(r["status"] == "api_error" and "boom" in r["error"] for r in rows)
    assert float(rows[0]["cost_usd"]) == 0 and w.n_errors == 2
    assert raws[0]["request"]["model"] == "claude-opus-5-5"


def test_budget_guard_skips(tmp_path):
    # reserve > allowed: nothing may start
    w, rows, raws = _run(tmp_path, {"anthropic": FakeAnthropic()}, allowed_usd=0.10, reserve_usd=1.0)
    assert w.n_written == 0 and w.n_skipped == 4
    assert all(r["status"] == "skipped_budget" for r in rows)
    assert raws == []


class FlakyAnthropic(FakeAnthropic):
    """Fails the first `n_fail` calls, then answers."""

    def __init__(self, n_fail):
        super().__init__()
        self.n_fail = n_fail

    def stream(self, **req):
        s = super().stream(**req)
        if self.n_fail > 0:
            self.n_fail -= 1
            fake_error = RuntimeError("overloaded")

            class _Failing:
                async def __aenter__(self):
                    raise fake_error

                async def __aexit__(self, *exc):
                    return False

            return _Failing()
        return s


def test_resume_reruns_only_unfinished_calls(tmp_path):
    from whichtool.harness import done_slots, slot_key, slots
    from whichtool.score import drop_superseded

    task = load_task("example")
    plan = [Config(STUDIES["headline"], MODELS["opus-5.5"], "high", task.variants, 2)]

    def writer():
        return RunWriter(run_dir=tmp_path, run_id="r", run_kind="study", task_name=task.name,
                         variants_sha="", allowed_usd=10, reserve_usd=0)

    # first session: one call fails
    fake = FlakyAnthropic(n_fail=1)
    asyncio.run(run_plan({"anthropic": fake}, task, plan, writer(), max_connections=1))
    assert len(done_slots(tmp_path / "calls.csv")) == 3

    # resume: only the failed slot is called again, and it lands in the same calls.csv
    again = FakeAnthropic()
    skip = done_slots(tmp_path / "calls.csv")
    asyncio.run(run_plan({"anthropic": again}, task, plan, writer(), skip=skip))
    assert len(again.requests) == 1
    assert done_slots(tmp_path / "calls.csv") == {slot_key(*s) for s in slots(plan)}

    rows = list(csv.DictReader((tmp_path / "calls.csv").open()))
    assert [r["status"] for r in rows].count("api_error") == 1 and len(rows) == 5
    kept = drop_superseded(rows)  # the scorer drops the failure the resume fixed
    assert len(kept) == 4 and all(r["status"] == "ok" for r in kept)


def test_stopping_starts_nothing_new(tmp_path):
    task = load_task("example")
    w = RunWriter(run_dir=tmp_path, run_id="r", run_kind="study", task_name=task.name,
                  variants_sha="", allowed_usd=10, reserve_usd=0)
    w.stopping = True
    fake = FakeAnthropic()
    plan = [Config(STUDIES["headline"], MODELS["opus-5.5"], "high", task.variants, 1)]
    asyncio.run(run_plan({"anthropic": fake}, task, plan, w))
    assert fake.requests == [] and w.n_skipped == 0
    assert len(list(csv.DictReader((tmp_path / "calls.csv").open()))) == 0  # nothing marked as skipped


def test_ledger_counts_calls_that_started_but_were_never_written(tmp_path):
    from whichtool import config
    from whichtool.run import spent_so_far

    _, rows, _ = _run(tmp_path / "t" / "run1", {"anthropic": FakeAnthropic()}, epochs=1)
    written = sum(float(r["cost_usd"]) for r in rows)
    started = tmp_path / "t" / "run1" / "started.txt"
    assert len(started.read_text().split()) == 2  # every call is marked before it is sent
    with started.open("a") as f:
        f.write("lost-call-id\n")  # e.g. the process was killed while this call ran
    assert spent_so_far(tmp_path) == pytest.approx(written + config.RESERVE_USD)


def test_every_request_setting_comes_from_config(monkeypatch):
    from whichtool import config

    monkeypatch.setitem(config.MAX_TOKENS, "high", 1234)
    monkeypatch.setattr(config, "CLAUDE_THINKING", {"type": "adaptive", "display": "omitted"})
    monkeypatch.setattr(config, "OPENAI_REASONING_SUMMARY", "detailed")
    claude = build_request(MODELS["opus-5.5"], STUDIES["headline"], "high", "i", "t")
    gpt = build_request(MODELS["gpt-6-astra"], STUDIES["headline"], "high", "i", "t")
    assert claude["max_tokens"] == 1234 and claude["thinking"]["display"] == "omitted"
    assert gpt["max_output_tokens"] == 1234 and gpt["reasoning"]["summary"] == "detailed"
    snap = json.loads(json.dumps(config.snapshot()))  # run.json gets all of it
    assert snap["experiment"]["max_tokens"]["high"] == 1234
    assert snap["execution"]["max_connections"] == config.MAX_CONNECTIONS
