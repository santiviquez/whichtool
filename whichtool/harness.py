"""Make the API calls: one fresh call per (configuration, variant, epoch),
written to disk as soon as it finishes.

The requests are built here, in plain dicts, and sent with the vendors' own
SDKs, so what raw.jsonl shows as the request is exactly what was sent.

Isolation: every call is a new, stateless request with only the fixed system
instruction and the job text. Nothing is shared between calls, epochs or
models: no conversation ids, no previous_response_id, store=false on OpenAI,
no memory, no tools except web search in the search study.
"""

from __future__ import annotations

import asyncio
import csv
import json
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import MAX_TOKENS, ModelSpec, Study
from .prompts import ANSWER_SCHEMA
from .records import CALLS_COLUMNS, cost_usd, extract, merge_extracts, parse_answer
from .taskspec import TaskSpec, Variant

# Anthropic's server-side search loop can stop with pause_turn; we send the
# turn back so it continues, at most this many times.
MAX_CONTINUATIONS = 5


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


# ------------------------------------------------------------------ requests


def build_request(
    spec: ModelSpec, study: Study, effort: str, instruction: str, text: str, schema: bool = True
) -> dict[str, Any]:
    """The exact keyword arguments for one call, for either vendor."""
    structured = study.answer == "structured" and schema
    if spec.vendor == "anthropic":
        output_config: dict[str, Any] = {"effort": effort}
        if structured:
            output_config["format"] = {"type": "json_schema", "schema": ANSWER_SCHEMA}
        req = {
            "model": spec.api_id,
            "max_tokens": MAX_TOKENS[effort],
            "system": instruction,
            "messages": [{"role": "user", "content": text}],
            "thinking": {"type": "adaptive", "display": "summarized"},
            "output_config": output_config,
        }
        if study.web_search:
            req["tools"] = [{"type": "web_search_20260209", "name": "web_search"}]
        return req

    req = {
        "model": spec.api_id,
        "instructions": instruction,
        "input": text,
        "reasoning": {"effort": effort, "summary": "auto"},
        "max_output_tokens": MAX_TOKENS[effort],
        "store": False,
    }
    if structured:
        req["text"] = {
            "format": {"type": "json_schema", "name": "answer", "schema": ANSWER_SCHEMA, "strict": True}
        }
    if study.web_search:
        req["tools"] = [{"type": "web_search"}]
        req["include"] = ["web_search_call.action.sources"]
    return req


async def send(client: Any, vendor: str, req: dict[str, Any]) -> tuple[list[dict], str]:
    """Send one request. Returns the raw response of every API call it took, and the answer text.

    Anthropic calls stream (long reasoning at high max_tokens would otherwise hit
    HTTP timeouts) and continue after pause_turn. OpenAI calls are one request.
    """
    if vendor == "anthropic":
        responses: list[dict] = []
        so_far: list[Any] = []  # assistant content blocks of every paused turn
        for _ in range(MAX_CONTINUATIONS + 1):
            messages = req["messages"] + ([{"role": "assistant", "content": so_far}] if so_far else [])
            async with client.messages.stream(**{**req, "messages": messages}) as stream:
                msg = await stream.get_final_message()
            responses.append(msg.to_dict())
            so_far += msg.content
            if msg.stop_reason != "pause_turn":
                break
        text = "".join(b.text for b in so_far if b.type == "text")
        return responses, text

    resp = await client.responses.create(**req)
    return [resp.to_dict()], resp.output_text or ""


# -------------------------------------------------------------------- writer


@dataclass
class RunWriter:
    """Appends one row to calls.csv and one record to raw.jsonl per call,
    and enforces the run's spending cap. Everything runs on one event loop,
    so no locks are needed."""

    run_dir: Path
    run_id: str
    run_kind: str  # "probe" | "study"
    task_name: str
    variants_sha: str
    allowed_usd: float  # this run may spend at most this much
    reserve_usd: float  # assumed cost of a call that is still in flight
    spent_usd: float = 0.0
    in_flight: int = 0
    n_written: int = 0
    n_skipped: int = 0
    n_errors: int = 0

    def __post_init__(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.calls_path = self.run_dir / "calls.csv"
        self.raw_path = self.run_dir / "raw.jsonl"
        if not self.calls_path.exists():
            with self.calls_path.open("w", newline="") as f:
                csv.DictWriter(f, fieldnames=CALLS_COLUMNS).writeheader()

    async def acquire(self) -> bool:
        """Wait for budget headroom. False means the run's budget is spent."""
        while True:
            if self.spent_usd + self.reserve_usd > self.allowed_usd:
                return False
            if self.spent_usd + (self.in_flight + 1) * self.reserve_usd <= self.allowed_usd:
                self.in_flight += 1
                return True
            await asyncio.sleep(0.5)

    def write(self, row: dict[str, Any], raw: dict[str, Any] | None, acquired: bool) -> None:
        if acquired:
            self.in_flight -= 1
        self.spent_usd += float(row.get("cost_usd") or 0)
        if row.get("status") == "skipped_budget":
            self.n_skipped += 1
        else:
            self.n_written += 1
            self.n_errors += row.get("status") != "ok"
        with self.calls_path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=CALLS_COLUMNS).writerow({k: row.get(k, "") for k in CALLS_COLUMNS})
        if raw is not None:
            with self.raw_path.open("a") as f:
                f.write(json.dumps(raw, ensure_ascii=False, default=str) + "\n")


# ----------------------------------------------------------------------- run


@dataclass(frozen=True)
class Config:
    """One (study, model, effort) configuration and the calls it makes."""

    study: Study
    spec: ModelSpec
    effort: str
    variants: list[Variant]
    epochs: int


async def run_call(
    client: Any,
    task: TaskSpec,
    cfg: Config,
    variant: Variant,
    epoch: int,
    writer: RunWriter,
    schema: bool = True,
) -> None:
    study, spec, effort = cfg.study, cfg.spec, cfg.effort
    base = {
        "call_id": uuid.uuid4().hex,
        "run_id": writer.run_id,
        "run_kind": writer.run_kind,
        "task": writer.task_name,
        "study": study.name,
        "condition": study.condition,
        "job_id": variant.job_id,
        "variant_id": variant.id,
        "epoch": epoch,
        "model": spec.label,
        "vendor": spec.vendor,
        "model_requested": spec.api_id,
        "effort": effort,
        "variants_sha": writer.variants_sha,
    }
    if not await writer.acquire():
        writer.write({**base, "status": "skipped_budget", "cost_usd": 0}, None, False)
        return

    instruction = task.code_instruction if study.answer == "code" else task.answer_instruction
    req = build_request(spec, study, effort, instruction, variant.text, schema)
    started, t0 = _now(), time.monotonic()
    responses: list[dict] = []
    text, error = "", ""
    try:
        responses, text = await send(client, spec.vendor, req)
    except Exception as e:  # the SDKs already retried; record the failure and move on
        error = f"{type(e).__name__}: {e}"

    try:
        row, raw = _record(base, spec, study, req, responses, text, error, started, t0)
    except Exception as e:  # never lose a call because of a bug here
        row = {**base, "status": "record_error", "started_at": started, "finished_at": _now(),
               "error": f"{error} | record: {type(e).__name__}: {e}"}
        raw = {**row, "request": req, "responses": responses, "output_text": text}
    writer.write(row, raw, True)


def _record(base, spec, study, req, responses, text, error, started, t0):
    x = merge_extracts([extract(spec.vendor, r) for r in responses])
    ans = (
        parse_answer(text)
        if study.answer == "structured"
        else {"api": "", "reason": "", "sentence": "", "parse_ok": ""}
    )
    row = {
        **base,
        "model_returned": x["model_returned"],
        "status": "api_error" if error else "ok",
        "stop_reason": x["stop_reason"],
        "refused": x["refused"],
        "parse_ok": ans["parse_ok"],
        "api": ans["api"],
        "reason": ans["reason"],
        "sentence": ans["sentence"],
        "searched": x["n_searches"] > 0 or bool(x["search_queries"]),
        "n_searches": x["n_searches"],
        "search_queries": " | ".join(q.replace("\n", " ")[:300] for q in x["search_queries"]),
        "input_tokens": x["input_tokens"],
        "output_tokens": x["output_tokens"],
        "reasoning_tokens": x["reasoning_tokens"],
        "cache_read_tokens": x["cache_read_tokens"],
        "cache_write_tokens": x["cache_write_tokens"],
        "cost_usd": cost_usd(spec, x),
        "latency_s": round(time.monotonic() - t0, 2),
        "n_api_calls": len(responses),
        "started_at": started,
        "finished_at": _now(),
        "error": error,
    }
    raw = {
        **row,
        "output_text": text,
        "reasoning_summary": x["reasoning_summary"],
        "search_queries": x["search_queries"],
        "urls": x["urls"],
        "request": req,
        "responses": responses,
    }
    return row, raw


async def run_plan(
    clients: dict[str, Any],
    task: TaskSpec,
    plan: list[Config],
    writer: RunWriter,
    max_connections: int = 8,
    schema: bool = True,
) -> None:
    """Run every call of every configuration, at most max_connections at once per model."""
    limits = {cfg.spec.label: asyncio.Semaphore(max_connections) for cfg in plan}

    async def one(cfg: Config, variant: Variant, epoch: int) -> None:
        async with limits[cfg.spec.label]:
            await run_call(clients[cfg.spec.vendor], task, cfg, variant, epoch, writer, schema)

    await asyncio.gather(
        *(one(cfg, v, e) for cfg in plan for e in range(1, cfg.epochs + 1) for v in cfg.variants)
    )


def make_clients(vendors: set[str], max_retries: int = 6, timeout: float = 1200.0) -> dict[str, Any]:
    """Async SDK clients. The SDKs retry 429s, 5xx and dropped connections with backoff."""
    clients: dict[str, Any] = {}
    if "anthropic" in vendors:
        import anthropic

        clients["anthropic"] = anthropic.AsyncAnthropic(max_retries=max_retries, timeout=timeout)
    if "openai" in vendors:
        import openai

        clients["openai"] = openai.AsyncOpenAI(max_retries=max_retries, timeout=timeout)
    return clients
