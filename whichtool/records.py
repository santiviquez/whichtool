"""Turn raw API requests/responses into one flat calls.csv row.

These are pure functions over the raw JSON each vendor returned, so the same
code can be rerun on raw.jsonl later without calling any API.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .config import WEB_SEARCH_USD_PER_CALL, ModelSpec

CALLS_COLUMNS = [
    "call_id",
    "run_id",
    "run_kind",
    "task",
    "study",
    "condition",
    "job_id",
    "variant_id",
    "epoch",
    "model",
    "vendor",
    "model_requested",
    "model_returned",
    "effort",
    "status",  # ok | api_error | skipped_budget
    "stop_reason",
    "refused",
    "parse_ok",
    "api",
    "reason",
    "sentence",
    "searched",
    "n_searches",
    "search_queries",
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "cost_usd",
    "latency_s",
    "n_api_calls",
    "retries",
    "started_at",
    "finished_at",
    "variants_sha",
    "error",
]


# ---------------------------------------------------------------- answers

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def parse_answer(text: str) -> dict[str, Any]:
    """Parse the structured answer. Tolerates code fences and stray prose."""
    out = {"api": "", "reason": "", "sentence": "", "parse_ok": False}
    if not text:
        return out
    candidates = [text.strip()]
    candidates += [m.strip() for m in _FENCE.findall(text)]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for c in candidates:
        try:
            obj = json.loads(c)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict) and "api" in obj:
            out["api"] = str(obj.get("api") or "").strip()
            out["reason"] = str(obj.get("reason") or "").strip()
            out["sentence"] = str(obj.get("sentence") or "").strip()
            out["parse_ok"] = True
            return out
    return out


# ------------------------------------------------------- vendor responses


def _empty_extract() -> dict[str, Any]:
    return {
        "model_returned": "",
        "stop_reason": "",
        "refused": False,
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "n_searches": 0,
        "search_queries": [],
        "urls": [],
        "reasoning_summary": [],
    }


def extract_anthropic(resp: dict) -> dict[str, Any]:
    """Pull usage, model id, stop reason and search activity from a Messages response."""
    x = _empty_extract()
    x["model_returned"] = resp.get("model") or ""
    x["stop_reason"] = resp.get("stop_reason") or ""
    x["refused"] = x["stop_reason"] == "refusal"
    u = resp.get("usage") if isinstance(resp.get("usage"), dict) else {}
    x["input_tokens"] = int(u.get("input_tokens") or 0)
    x["output_tokens"] = int(u.get("output_tokens") or 0)
    x["cache_read_tokens"] = int(u.get("cache_read_input_tokens") or 0)
    x["cache_write_tokens"] = int(u.get("cache_creation_input_tokens") or 0)
    stu = u.get("server_tool_use") or {}
    x["n_searches"] = int(stu.get("web_search_requests") or 0)
    content = resp.get("content")
    for block in content if isinstance(content, list) else []:
        if not isinstance(block, dict):
            continue
        t = block.get("type")
        if t == "thinking" and block.get("thinking"):
            x["reasoning_summary"].append(block["thinking"])
        elif t == "server_tool_use":
            inp = block.get("input") or {}
            if block.get("name") == "web_search" and inp.get("query"):
                x["search_queries"].append(inp["query"])
            elif inp.get("code"):  # dynamic filtering runs searches from code
                x["search_queries"].append(f"[code] {inp['code']}")
        elif t == "web_search_tool_result":
            content = block.get("content")
            if isinstance(content, list):
                x["urls"] += [r.get("url") for r in content if r.get("url")]
        elif t == "text":
            for c in block.get("citations") or []:
                if c.get("url"):
                    x["urls"].append(c["url"])
    return x


def extract_openai(resp: dict) -> dict[str, Any]:
    """Pull usage, model id, stop reason and search activity from a Responses API response."""
    x = _empty_extract()
    x["model_returned"] = resp.get("model") or ""
    status = resp.get("status") or ""
    reason = (resp.get("incomplete_details") or {}).get("reason")
    x["stop_reason"] = f"{status}:{reason}" if reason else status
    u = resp.get("usage") if isinstance(resp.get("usage"), dict) else {}
    x["input_tokens"] = int(u.get("input_tokens") or 0)
    x["output_tokens"] = int(u.get("output_tokens") or 0)
    x["cache_read_tokens"] = int((u.get("input_tokens_details") or {}).get("cached_tokens") or 0)
    x["reasoning_tokens"] = int((u.get("output_tokens_details") or {}).get("reasoning_tokens") or 0)
    output = resp.get("output")
    for item in output if isinstance(output, list) else []:
        if not isinstance(item, dict):
            continue
        t = item.get("type")
        if t == "reasoning":
            for s in item.get("summary") or []:
                if s.get("text"):
                    x["reasoning_summary"].append(s["text"])
        elif t == "web_search_call":
            action = item.get("action") or {}
            if action.get("type", "search") == "search":
                x["n_searches"] += 1
            q = action.get("query") or action.get("url")
            if q:
                x["search_queries"].append(q)
            for s in action.get("sources") or []:
                if s.get("url"):
                    x["urls"].append(s["url"])
        elif t == "message":
            for c in item.get("content") or []:
                if c.get("type") == "refusal":
                    x["refused"] = True
                for a in c.get("annotations") or []:
                    if a.get("url"):
                        x["urls"].append(a["url"])
    return x


def extract(vendor: str, resp: dict) -> dict[str, Any]:
    return extract_anthropic(resp) if vendor == "anthropic" else extract_openai(resp)


def merge_extracts(parts: list[dict[str, Any]]) -> dict[str, Any]:
    """Sum usage over several API calls that made up one answer (e.g. pause_turn)."""
    x = _empty_extract()
    for p in parts:
        for k in (
            "input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "n_searches",
        ):
            x[k] += p[k]
        for k in ("search_queries", "urls", "reasoning_summary"):
            x[k] += p[k]
        x["refused"] = x["refused"] or p["refused"]
    if parts:
        x["model_returned"] = parts[-1]["model_returned"]
        x["stop_reason"] = parts[-1]["stop_reason"]
    x["urls"] = list(dict.fromkeys(x["urls"]))
    return x


def cost_usd(spec: ModelSpec, x: dict[str, Any]) -> float:
    """Cost of one answer from token usage and search count.

    Anthropic reports input_tokens excluding cache reads/writes; OpenAI
    reports input_tokens including cached tokens. Reasoning tokens are part
    of output_tokens for both vendors.
    """
    cr, cw = x["cache_read_tokens"], x["cache_write_tokens"]
    if spec.vendor == "anthropic":
        fresh_in = x["input_tokens"]
    else:
        fresh_in = max(x["input_tokens"] - cr, 0)
    usd = (
        fresh_in * spec.usd_in
        + cr * spec.usd_cache_read
        + cw * spec.usd_cache_write
        + x["output_tokens"] * spec.usd_out
    ) / 1e6
    usd += x["n_searches"] * WEB_SEARCH_USD_PER_CALL
    return round(usd, 6)
