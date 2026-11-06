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
import signal
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from . import config as C
from .config import ModelSpec, Study
from .records import CALLS_COLUMNS, cost_usd, extract, merge_extracts, parse_answer
from .taskspec import TaskSpec, Variant


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


# ------------------------------------------------------------------ requests


def build_request(
    spec: ModelSpec, study: Study, effort: str, instruction: str, text: str,
    schema: bool = C.STRUCTURED_OUTPUT,
) -> dict[str, Any]:
    """The exact keyword arguments for one call, for either vendor. Every setting comes from config.py."""
    structured = study.answer == "structured" and schema
    if spec.vendor == "anthropic":
        output_config: dict[str, Any] = {"effort": effort}
        if structured:
            output_config["format"] = {"type": "json_schema", "schema": C.ANSWER_SCHEMA}
        req = {
            "model": spec.api_id,
            "max_tokens": spec.max_output_tokens,
            "stream": True,
            "system": instruction,
            "messages": [{"role": "user", "content": text}],
            "thinking": dict(C.CLAUDE_THINKING),
            "output_config": output_config,
        }
        if study.web_search:
            req["tools"] = [dict(C.CLAUDE_WEB_SEARCH_TOOL)]
        return req

    req = {
        "model": spec.api_id,
        "instructions": instruction,
        "input": text,
        "reasoning": {"effort": effort, "summary": C.OPENAI_REASONING_SUMMARY},
        "max_output_tokens": spec.max_output_tokens,
        "store": False,
        "stream": True,
    }
    if structured:
        req["text"] = {
            "format": {"type": "json_schema", "name": "answer", "schema": C.ANSWER_SCHEMA, "strict": True}
        }
    if study.web_search:
        req["tools"] = [dict(C.OPENAI_WEB_SEARCH_TOOL)]
        req["include"] = ["web_search_call.action.sources"]
    return req


class CallFailed(Exception):
    """The vendor returned a failed response; keeps what came back for raw.jsonl."""

    def __init__(self, message: str, responses: list[dict]):
        super().__init__(message)
        self.responses = responses


async def send(client: Any, vendor: str, req: dict[str, Any]) -> tuple[list[dict], str]:
    """Send one request. Returns the raw response of every API call it took, and the answer text.

    Both vendors stream, so a long answer never sits on an idle connection
    until a timeout. Claude calls continue after pause_turn.
    """
    if vendor == "anthropic":
        body = {k: v for k, v in req.items() if k != "stream"}  # the stream helper sets it
        responses: list[dict] = []
        so_far: list[Any] = []  # assistant content blocks of every paused turn
        for _ in range(C.CLAUDE_MAX_CONTINUATIONS + 1):
            messages = body["messages"] + ([{"role": "assistant", "content": so_far}] if so_far else [])
            async with client.messages.stream(**{**body, "messages": messages}) as stream:
                msg = await stream.get_final_message()
            responses.append(msg.to_dict())
            so_far += msg.content
            if msg.stop_reason != "pause_turn":
                break
        text = "".join(b.text for b in so_far if b.type == "text")
        return responses, text

    final = None
    async for event in await client.responses.create(**req):
        if event.type in ("response.completed", "response.incomplete", "response.failed"):
            final = event.response
    if final is None:
        raise RuntimeError("the response stream ended without a final response")
    responses = [final.to_dict()]
    if final.status == "failed":
        raise CallFailed(f"response failed: {final.error}", responses)
    return responses, final.output_text or ""


# -------------------------------------------------------------------- writer


@dataclass
class RunWriter:
    """Appends one row to calls.csv and one record to raw.jsonl per call,
    and enforces the run's spending cap. Everything runs on one event loop,
    so no locks are needed.

    started.txt gets a call's id and set-aside amount just before it is sent.
    A call that started but never reached calls.csv (the process died) is
    counted by spent_so_far at that amount, so the ledger never forgets money
    that may have been spent.
    """

    run_dir: Path
    run_id: str
    run_kind: str  # "probe" | "study"
    task_name: str
    variants_sha: str
    allowed_usd: float  # this run may spend at most this much
    spent_usd: float = 0.0
    in_flight: int = 0
    in_flight_usd: float = 0.0  # money set aside for running calls
    n_written: int = 0
    n_skipped: int = 0
    n_errors: int = 0
    stopping: bool = False  # set by Ctrl-C: start nothing new, finish what is running
    broken: dict = field(default_factory=dict)  # configuration -> error that will repeat on every call

    def __post_init__(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.calls_path = self.run_dir / "calls.csv"
        self.raw_path = self.run_dir / "raw.jsonl"
        self.started_path = self.run_dir / "started.txt"
        if not self.calls_path.exists():
            with self.calls_path.open("w", newline="") as f:
                csv.DictWriter(f, fieldnames=CALLS_COLUMNS).writeheader()

    async def acquire(self, reserve_of: Callable[[], float]) -> float | None:
        """Wait until the amount to set aside for one more call fits under the cap, and set it aside.

        reserve_of() is asked again each time, so a call that waits picks up
        what finished calls have taught. Returns the amount set aside, or None
        if it can never fit or the run is stopping.
        """
        while True:
            if self.stopping:
                return None
            reserve = reserve_of()
            if self.spent_usd + reserve > self.allowed_usd:
                return None
            if self.spent_usd + self.in_flight_usd + reserve <= self.allowed_usd:
                self.in_flight += 1
                self.in_flight_usd += reserve
                return reserve
            await asyncio.sleep(0.5)

    def release(self, reserve: float) -> None:
        self.in_flight -= 1
        self.in_flight_usd -= reserve

    def mark_started(self, call_id: str, reserve: float) -> None:
        with self.started_path.open("a") as f:
            f.write(f"{call_id},{reserve:.6f}\n")

    def write(self, row: dict[str, Any], raw: dict[str, Any] | None, reserve: float | None) -> None:
        """reserve is what acquire() set aside for this call, or None if it never started."""
        if reserve is not None:
            self.release(reserve)
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


# Errors that will repeat on every call of a configuration: a rejected
# request, a bad key or model id, or an account with no credits.
_FATAL_STATUS = {400, 401, 403, 404, 422}


def is_fatal(e: Exception) -> bool:
    return getattr(e, "status_code", None) in _FATAL_STATUS or getattr(e, "code", None) == "insufficient_quota"


def config_key(cfg: Config) -> tuple[str, str, str]:
    return (cfg.study.name, cfg.spec.label, cfg.effort)


async def run_call(
    client: Any,
    task: TaskSpec,
    cfg: Config,
    variant: Variant,
    epoch: int,
    writer: RunWriter,
    reserve_of: Callable[[], float],
    schema: bool = C.STRUCTURED_OUTPUT,
) -> float | None:
    """Make one call and write it. Returns its cost if it succeeded."""
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
    # Left out entirely (no row), so a resume runs it: the run is stopping, or
    # an earlier call of this configuration hit an error that would repeat.
    if writer.stopping or config_key(cfg) in writer.broken:
        return None
    reserve = await writer.acquire(reserve_of)
    if reserve is None:
        if not writer.stopping:
            writer.write({**base, "status": "skipped_budget", "cost_usd": 0}, None, None)
        return None
    if config_key(cfg) in writer.broken:  # it broke while this call waited for budget
        writer.release(reserve)
        return None

    instruction = task.code_instruction if study.answer == "code" else task.answer_instruction
    req = build_request(spec, study, effort, instruction, variant.text, schema)
    writer.mark_started(base["call_id"], reserve)
    started, t0 = _now(), time.monotonic()
    responses: list[dict] = []
    text, error = "", ""
    try:
        responses, text = await send(client, spec.vendor, req)
    except Exception as e:  # the SDKs already retried; record the failure and move on
        error = f"{type(e).__name__}: {e}"
        responses = getattr(e, "responses", [])
        if is_fatal(e) and config_key(cfg) not in writer.broken:
            writer.broken[config_key(cfg)] = error
            print(f"\nStopping {' '.join(config_key(cfg))}: {error[:300]}", flush=True)

    try:
        row, raw = _record(base, spec, study, req, responses, text, error, started, t0)
    except Exception as e:  # never lose a call because of a bug here
        row = {**base, "status": "record_error", "started_at": started, "finished_at": _now(),
               "error": f"{error} | record: {type(e).__name__}: {e}"}
        raw = {**row, "request": req, "responses": responses, "output_text": text}
    writer.write(row, raw, reserve)
    return float(row["cost_usd"]) if row.get("status") == "ok" else None


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


Slot = tuple[str, str, str, str, int]  # study, model, effort, variant id, epoch


def slots(plan: list[Config]) -> list[tuple[Config, Variant, int]]:
    """Every call the plan makes."""
    return [(cfg, v, e) for cfg in plan for e in range(1, cfg.epochs + 1) for v in cfg.variants]


def slot_key(cfg: Config, variant: Variant, epoch: int) -> Slot:
    return (cfg.study.name, cfg.spec.label, cfg.effort, variant.id, epoch)


def done_slots(calls_path: Path) -> set[Slot]:
    """Slots that already have a successful call in a run's calls.csv."""
    if not calls_path.exists():
        return set()
    with calls_path.open() as f:
        return {
            (r["study"], r["model"], r["effort"], r["variant_id"], int(r["epoch"]))
            for r in csv.DictReader(f)
            if r.get("status") == "ok"
        }


async def run_plan(
    clients: dict[str, Any],
    task: TaskSpec,
    plan: list[Config],
    writer: RunWriter,
    reserves: dict[tuple[str, str, str], float] | None = None,
    max_connections: int = C.MAX_CONNECTIONS,
    schema: bool = C.STRUCTURED_OUTPUT,
    skip: set[Slot] = frozenset(),
) -> None:
    """Run every call of the plan except the slots in skip, at most max_connections at once per model.

    reserves gives the money set aside per call of each configuration: the
    most expensive successful call seen so far, or RESERVE_USD before there
    is one. It is updated as calls finish, so the probe's first calls of a
    configuration set the amount for the rest. The first Ctrl-C stops new calls and lets the
    running ones finish and be written; the second quits at once.
    """
    seen = dict(reserves or {})
    limits = {cfg.spec.label: asyncio.Semaphore(max_connections) for cfg in plan}
    loop = asyncio.get_running_loop()

    def interrupt() -> None:
        writer.stopping = True
        loop.remove_signal_handler(signal.SIGINT)  # a second Ctrl-C quits at once
        print(f"\nStopping: no new calls; waiting for {writer.in_flight} running calls to be written. "
              "Ctrl-C again to quit now. Resume later with --resume.", flush=True)

    try:
        loop.add_signal_handler(signal.SIGINT, interrupt)
    except (NotImplementedError, RuntimeError, ValueError):  # not the main thread, or no signals here
        pass

    async def one(cfg: Config, variant: Variant, epoch: int) -> None:
        async with limits[cfg.spec.label]:
            key = config_key(cfg)
            cost = await run_call(clients[cfg.spec.vendor], task, cfg, variant, epoch, writer,
                                  lambda: seen.get(key, C.RESERVE_USD), schema)
            if cost is not None:
                seen[key] = max(seen.get(key, 0.0), cost)

    try:
        await asyncio.gather(*(one(c, v, e) for c, v, e in slots(plan) if slot_key(c, v, e) not in skip))
    finally:
        try:
            loop.remove_signal_handler(signal.SIGINT)
        except (NotImplementedError, RuntimeError, ValueError):
            pass


async def preflight(clients: dict[str, Any], specs: list[ModelSpec]) -> dict[str, str]:
    """Look up every model with its vendor's models endpoint: free, and catches a
    bad key or a wrong model id before any money is spent. Returns label -> error."""
    errors: dict[str, str] = {}
    for spec in specs:
        try:
            await clients[spec.vendor].models.retrieve(spec.api_id)
        except Exception as e:
            errors[spec.label] = f"{type(e).__name__}: {e}"
    return errors


def make_clients(
    vendors: set[str], max_retries: int = C.MAX_RETRIES, timeout: float = C.TIMEOUT_S
) -> dict[str, Any]:
    """Async SDK clients. The SDKs retry 429s, 5xx and dropped connections with backoff."""
    clients: dict[str, Any] = {}
    if "anthropic" in vendors:
        import anthropic

        clients["anthropic"] = anthropic.AsyncAnthropic(max_retries=max_retries, timeout=timeout)
    if "openai" in vendors:
        import openai

        clients["openai"] = openai.AsyncOpenAI(max_retries=max_retries, timeout=timeout)
    return clients
