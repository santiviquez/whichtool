"""The Inspect AI part: one Task per (model, effort), and a solver that makes
one fresh API call per sample and writes it to disk as soon as it finishes.

Isolation: every sample is a new, stateless API call with only the fixed
system instruction and the job text. Nothing is shared between samples,
epochs or models: no conversation ids, no previous_response_id, store=false
on OpenAI, no memory, no tools except web search in the search study.
"""

from __future__ import annotations

import asyncio
import csv
import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from inspect_ai import Task
from inspect_ai.dataset import Sample
from inspect_ai.event import ModelEvent
from inspect_ai.log import transcript
from inspect_ai.model import (
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    ResponseSchema,
)
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.tool import web_search
from inspect_ai.util import JSONSchema

from .config import MAX_CONNECTIONS, MAX_TOKENS, ModelSpec, Study
from .prompts import ANSWER_SCHEMA
from .records import (
    CALLS_COLUMNS,
    cost_usd,
    extract,
    merge_extracts,
    parse_answer,
)
from .taskspec import TaskSpec, Variant


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


_ATTACHMENT = "attachment://"


def _resolve_attachments(obj: Any, lookup) -> Any:
    """Inspect swaps long strings for attachment://<hash> refs; put the text back."""
    if isinstance(obj, str) and obj.startswith(_ATTACHMENT):
        found = lookup(obj[len(_ATTACHMENT) :])
        return found if found is not None else obj
    if isinstance(obj, dict):
        return {k: _resolve_attachments(v, lookup) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_attachments(v, lookup) for v in obj]
    return obj


@dataclass
class RunWriter:
    """Appends one row to calls.csv and one record to raw.jsonl per call,
    and enforces the run's spending cap."""

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
    _lock: threading.Lock = field(default_factory=threading.Lock)

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
            with self._lock:
                if self.spent_usd + self.reserve_usd > self.allowed_usd:
                    return False
                if self.spent_usd + (self.in_flight + 1) * self.reserve_usd <= self.allowed_usd:
                    self.in_flight += 1
                    return True
            await asyncio.sleep(0.5)

    def write(self, row: dict[str, Any], raw: dict[str, Any] | None, acquired: bool) -> None:
        with self._lock:
            if acquired:
                self.in_flight -= 1
            self.spent_usd += float(row.get("cost_usd") or 0)
            if row.get("status") == "skipped_budget":
                self.n_skipped += 1
            else:
                self.n_written += 1
            with self.calls_path.open("a", newline="") as f:
                csv.DictWriter(f, fieldnames=CALLS_COLUMNS).writerow(
                    {k: row.get(k, "") for k in CALLS_COLUMNS}
                )
                f.flush()
            if raw is not None:
                with self.raw_path.open("a") as f:
                    f.write(json.dumps(raw, ensure_ascii=False, default=str) + "\n")
                    f.flush()


@solver
def answer_and_record(
    writer: RunWriter, spec: ModelSpec, study: Study, effort: str
):
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        meta = state.metadata
        call_id = uuid.uuid4().hex
        base = {
            "call_id": call_id,
            "run_id": writer.run_id,
            "run_kind": writer.run_kind,
            "task": writer.task_name,
            "study": study.name,
            "condition": study.condition,
            "job_id": meta["job_id"],
            "variant_id": meta["variant_id"],
            "epoch": state.epoch,
            "model": spec.label,
            "vendor": spec.vendor,
            "model_requested": spec.api_id,
            "effort": effort,
            "variants_sha": writer.variants_sha,
        }

        acquired = await writer.acquire()
        if not acquired:
            writer.write({**base, "status": "skipped_budget", "cost_usd": 0}, None, False)
            state.completed = True
            return state

        if study.web_search:
            state.tools = [web_search(spec.vendor)]

        started = _now()
        t0 = time.monotonic()
        error = ""
        try:
            state = await generate(state)
        except Exception as e:  # recorded, then re-raised so Inspect logs it too
            error = f"{type(e).__name__}: {e}"
            raise
        finally:
            try:
                row, raw = _build_record(base, state, spec, study, started, t0, error)
            except Exception as e:  # never lose a call because of a bug here
                row = {**base, "status": "record_error", "started_at": started,
                       "finished_at": _now(), "error": f"{error} | record: {type(e).__name__}: {e}"}
                raw = {**row}
            writer.write(row, raw, True)
        return state

    return solve


def _build_record(
    base: dict[str, Any],
    state: TaskState,
    spec: ModelSpec,
    study: Study,
    started: str,
    t0: float,
    error: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    latency = time.monotonic() - t0
    tr = transcript()
    events = [e for e in tr.events if isinstance(e, ModelEvent)]
    api_calls = []
    parts = []
    retries = 0
    for ev in events:
        retries += ev.retries or 0
        req = _resolve_attachments(ev.call.request, tr.attachments.get) if ev.call else None
        resp = _resolve_attachments(ev.call.response, tr.attachments.get) if ev.call else None
        api_calls.append(
            {
                "request": req,
                "response": resp,
                "time": ev.call.time if ev.call else None,
                "error": ev.error,
            }
        )
        if isinstance(resp, dict):
            parts.append(extract(spec.vendor, resp))
    x = merge_extracts(parts)
    text = "" if error else (state.output.completion or "")
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
        "latency_s": round(latency, 2),
        "n_api_calls": len(events),
        "retries": retries,
        "started_at": started,
        "finished_at": _now(),
        "error": error,
    }
    raw = {
        **row,
        "messages": [{"role": m.role, "content": m.text} for m in state.messages[:2]],
        "output_text": text,
        "reasoning_summary": x["reasoning_summary"],
        "search_queries": x["search_queries"],
        "urls": x["urls"],
        "api_calls": api_calls,
    }
    return row, raw


def build_dataset(task: TaskSpec, study: Study, variants: list[Variant]) -> list[Sample]:
    instruction = task.code_instruction if study.answer == "code" else task.answer_instruction
    return [
        Sample(
            id=v.id,
            input=[ChatMessageSystem(content=instruction), ChatMessageUser(content=v.text)],
            metadata={"job_id": v.job_id, "variant_id": v.id},
        )
        for v in variants
    ]


def build_task(
    task: TaskSpec,
    study: Study,
    spec: ModelSpec,
    effort: str,
    variants: list[Variant],
    epochs: int,
    writer: RunWriter,
    max_connections: int = MAX_CONNECTIONS,
) -> Task:
    config = GenerateConfig(
        reasoning_effort=effort,  # type: ignore[arg-type]
        max_tokens=MAX_TOKENS[effort],
        max_connections=max_connections,
        max_retries=6,
        attempt_timeout=1200,
    )
    if study.answer == "structured":
        config = config.merge(
            GenerateConfig(
                response_schema=ResponseSchema(
                    name="answer",
                    json_schema=JSONSchema.model_validate(ANSWER_SCHEMA),
                    strict=True,
                )
            )
        )
    return Task(
        name=f"{task.name}_{study.name}_{spec.label}_{effort}",
        dataset=build_dataset(task, study, variants),
        solver=answer_and_record(writer, spec, study, effort),
        model=spec.inspect_id,
        config=config,
        epochs=epochs,
        metadata={"study": study.name, "model": spec.label, "effort": effort},
    )
