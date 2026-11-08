"""Classify every saved answer: which hosted API it names or calls. No API calls.

    python -m whichtool.classify --task doc-processing

Reads only:
    results/<task>/*/calls.csv and raw.jsonl   (what the harness wrote)
    results/<task>/review/reviews_*.csv        (the owner's review of the classification; optional)
    tasks/<task>/aliases.yaml                  (API names and code signatures -> API ids)
    tasks/<task>/manual_apis.yaml              (calls classified by hand; optional)
Writes:
    results/<task>/results.csv    one row per call: question, answer, the APIs it
                                  names or calls, search data, cost and status.
                                  Only the latest successful call of each slot; the
                                  run folders keep every attempt.
Answers no alias matches are listed on the console: add them to aliases.yaml and rerun.

Statuses:
    mapped       the answer names (or the code calls) at least one API: api_1 to api_3
    hedge        "it depends" or similar, and no API named
    refusal      the model declined (refusal stop reason or refusal text)
    truncated    no answer because the model used its whole output limit, or
                 Claude's search was still running after the last continuation
    format_error the answer could not be parsed, or the code had no code
    unmapped     no alias matched; add it to aliases.yaml and rerun
    api_error    the call failed
    record_error the call finished but the harness failed to record it (a harness bug)

An answer that names or calls more than one API keeps every one: api_1 is the one
named first (stated answers) or listed first in aliases.yaml (code). Self-hosted
helpers found in code (such as PyMuPDF) are dropped when the code also calls a
hosted API. A call listed in manual_apis.yaml gets the APIs listed there instead
(match_method "manual").

Grading against a ground truth is not part of this step. The grading code that
existed before 2026-11-08 is in git history (commit 71f80b6).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .config import HEDGE_PHRASES, REFUSAL_PHRASES, RESULTS_DIR
from .taskspec import TaskSpec, load_aliases, load_manual_apis, load_task

ERROR_STATUSES = ("api_error", "record_error")
# stop_reason values (as calls.csv writes them) that mean the answer never finished
CUT_OFF = ("max_tokens", "pause_turn", "incomplete:max_output_tokens")

FIELDS = [
    "canonical_api",  # API 1
    "canonical_api_2",
    "canonical_api_3",
    "n_apis",
    "matched_names",
    "match_method",
    "status",
]


def norm(s: str) -> str:
    s = s.lower().replace("&", " and ")
    s = re.sub(r"[™®©]", "", s)
    s = re.sub(r"[^a-z0-9+.# ]+", " ", s)
    s = re.sub(r"(?<![a-z0-9])\.|\.(?![a-z0-9])", " ", s)  # dots only inside tokens (e.g. "api.x.com")
    return re.sub(r"\s+", " ", s).strip()


@dataclass
class Match:
    api_id: str | None  # API 1
    names: list[str]
    method: str  # exact | contains | multiple | none | code | code_multiple | manual
    ids: list[str] = field(default_factory=list)  # every API, API 1 first


class AliasTable:
    def __init__(self, apis: dict):
        self.apis = apis
        self.names: list[tuple[str, str]] = []  # (normalized alias, api id)
        self.exact: list[tuple[str, str]] = []  # aliases too generic to match inside longer text
        self.code: list[tuple[str, str]] = []  # (lowercased code signature, api id)
        for api_id, entry in apis.items():
            entry = entry or {}
            for n in [api_id.replace("_", " ")] + list(entry.get("names") or []):
                if norm(n):
                    self.names.append((norm(n), api_id))
            for n in entry.get("exact") or []:
                if norm(n):
                    self.exact.append((norm(n), api_id))
            for c in entry.get("code") or []:
                self.code.append((str(c).lower(), api_id))
        # longest aliases first so "azure ai document intelligence" beats "document intelligence"
        self.names.sort(key=lambda t: -len(t[0]))

    def hosted(self, api_id: str) -> bool:
        return bool((self.apis.get(api_id) or {}).get("hosted", True))

    def match_name(self, answer: str) -> Match:
        a = norm(answer)
        if not a:
            return Match(None, [], "none")
        for n, api_id in self.names + self.exact:
            if a == n:
                return Match(api_id, [n], "exact", [api_id])
        spans: list[tuple[int, int, str, str]] = []
        for n, api_id in self.names:
            for m in re.finditer(rf"(?<![a-z0-9]){re.escape(n)}(?![a-z0-9])", a):
                s, e = m.span()
                if any(s >= s2 and e <= e2 for s2, e2, _, _ in spans):
                    continue  # inside a longer alias already found
                spans.append((s, e, n, api_id))
        spans.sort()  # API 1 is the one named first
        ids = list(dict.fromkeys(api_id for _, _, _, api_id in spans))
        names = [n for _, _, n, _ in spans]
        if len(ids) == 1:
            return Match(ids[0], names, "contains", ids)
        if len(ids) > 1:
            return Match(ids[0], names, "multiple", ids)
        return Match(None, [], "none")

    def match_code(self, text: str) -> Match:
        t = code_only(text).lower()
        hits = [(c, api_id) for c, api_id in self.code if c in t]
        # API 1 is the one listed first in aliases.yaml (document APIs come
        # before general LLM APIs there), not the one the code calls first.
        ids = list(dict.fromkeys(api_id for _, api_id in hits))
        # Helpers such as PyMuPDF for page rendering don't count as a second
        # API when the code also calls a hosted API.
        if any(self.hosted(i) for i in ids):
            ids = [i for i in ids if self.hosted(i)]
            hits = [(c, i) for c, i in hits if i in ids]
        names = [c for c, _ in hits]
        if len(ids) == 1:
            return Match(ids[0], names, "code", ids)
        if len(ids) > 1:
            return Match(ids[0], names, "code_multiple", ids)
        return Match(None, [], "none")


_CODE_BLOCK = re.compile(r"```[a-zA-Z0-9_+-]*\n(.*?)```", re.S)


def code_only(text: str) -> str:
    """The code the model wrote, without prose or comments that name alternatives."""
    blocks = _CODE_BLOCK.findall(text)
    code = "\n".join(blocks) if blocks else text
    lines = []
    for line in code.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        # drop trailing comments (" # ..."), leaving URLs with fragments alone
        line = re.sub(r"\s#\s.*$", "", line)
        lines.append(line)
    return "\n".join(lines)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12] if path.exists() else ""


def load_calls(task: TaskSpec, results_dir: Path, include_probe: bool) -> tuple[list[dict], dict[str, dict]]:
    """The calls to classify, and each call's raw record (output_text, urls) by call id."""
    rows: list[dict] = []
    raws: dict[str, dict] = {}
    for calls in sorted((results_dir / task.name).glob("*/calls.csv")):
        with calls.open() as f:
            for r in csv.DictReader(f):
                if r.get("status") == "skipped_budget":
                    continue
                if r.get("run_kind") == "probe" and not include_probe:
                    continue
                rows.append(r)
        raw = calls.parent / "raw.jsonl"
        if raw.exists():
            with raw.open() as f:
                for line in f:
                    rec = json.loads(line)
                    raws[rec["call_id"]] = {"output_text": rec.get("output_text") or "", "urls": rec.get("urls") or []}
    return drop_superseded(rows), raws


def load_reviews(task: TaskSpec, results_dir: Path) -> dict[str, str]:
    """The owner's review decision per call id (accepted, changed, accepted_in_bulk); later files win."""
    out: dict[str, str] = {}
    for path in sorted((results_dir / task.name / "review").glob("reviews_*.csv")):
        with path.open() as f:
            for r in csv.DictReader(f):
                for cid in r["call_ids"].split():
                    out[cid] = r["decision"]
    return out


RESULT_COLUMNS = [
    "call_id", "experiment", "model", "effort", "job", "wording", "wording_by", "repeat",
    "question", "answer", "reason", "sentence",
    "api_1", "api_2", "api_3", "n_apis", "match_method", "matched_on", "review", "status",
    "searched", "n_searches", "search_queries", "cited_urls",
    "input_tokens", "output_tokens", "reasoning_tokens", "cache_read_tokens", "cache_write_tokens",
    "cost_usd", "latency_s", "model_returned", "stop_reason", "call_status", "error",
    "run_id", "started_at", "finished_at", "variants_sha", "aliases_sha", "manual_apis_sha",
]


def result_row(r: dict, extra: dict, raw: dict, question: str, review: str, shas: dict | None = None) -> dict:
    """One row of results.csv: the call, its question and answer, and its classification."""
    code = r.get("study") == "code"
    same = ["reason", "sentence", "searched", "n_searches", "search_queries", "input_tokens", "output_tokens",
            "reasoning_tokens", "cache_read_tokens", "cache_write_tokens", "cost_usd", "latency_s", "model_returned",
            "stop_reason", "error", "run_id", "started_at", "finished_at", "variants_sha"]
    return {
        "call_id": r["call_id"], "experiment": r.get("study", ""), "model": r.get("model", ""),
        "effort": r.get("effort", ""), "job": r.get("job_id", ""), "wording": r.get("variant_id", ""),
        "wording_by": "ChatGPT" if r.get("variant_id", "").endswith("v00") else "Claude",
        "repeat": r.get("epoch", ""), "question": question,
        # stated answers: the api field; code: the whole reply
        "answer": raw.get("output_text", "") if code else r.get("api", ""),
        **{k: r.get(k, "") for k in same},
        "api_1": extra["canonical_api"], "api_2": extra["canonical_api_2"], "api_3": extra["canonical_api_3"],
        "n_apis": extra["n_apis"], "match_method": extra["match_method"], "matched_on": extra["matched_names"],
        "review": review, "status": extra["status"], "cited_urls": " ".join(raw.get("urls") or []),
        "call_status": r.get("status", ""),
        "aliases_sha": (shas or {}).get("aliases", ""), "manual_apis_sha": (shas or {}).get("manual_apis", ""),
    }


def _slot(r: dict) -> tuple:
    return (r.get("run_id"), r.get("study"), r.get("model"), r.get("effort"), r.get("variant_id"), r.get("epoch"))


def drop_superseded(rows: list[dict]) -> list[dict]:
    """Drop failed rows of a slot (run, configuration, variant, repeat) that a resumed run later got right."""
    ok = {_slot(r) for r in rows if r.get("status") == "ok"}
    return [r for r in rows if r.get("status") == "ok" or _slot(r) not in ok]


def classify(row: dict, text: str, table: AliasTable, manual: list[str] | None = None) -> dict:
    out = {k: "" for k in FIELDS}
    refused = str(row.get("refused")).lower() == "true"
    if row.get("status") in ERROR_STATUSES:
        out["status"] = row["status"]
    elif refused:
        out["status"] = "refusal"
    else:
        cut_off = row.get("stop_reason") in CUT_OFF
        if row.get("study") == "code" or row.get("condition") == "write_code":
            if "```" not in text and "import " not in text:
                out["status"] = "truncated" if cut_off else "format_error"
                m = Match(None, [], "none")
            else:
                m = table.match_code(text)
        else:
            if str(row.get("parse_ok")).lower() != "true":
                out["status"] = "truncated" if cut_off else "format_error"
                m = Match(None, [], "none")
            else:
                m = table.match_name(row.get("api", ""))
        if manual and not out["status"]:
            m = Match(manual[0], m.names, "manual", list(manual))
        out["matched_names"] = "; ".join(m.names)
        out["match_method"] = m.method
        if not out["status"]:
            answer = norm(row.get("api", ""))
            if any(norm(p) in answer for p in REFUSAL_PHRASES):
                out["status"] = "refusal"
            elif m.api_id is None and any(f" {norm(p)} " in f" {answer} " for p in HEDGE_PHRASES):
                out["status"] = "hedge"
            elif m.api_id is None:
                out["status"] = "unmapped"
            else:
                ids = m.ids or [m.api_id]
                for col, api_id in zip(("canonical_api", "canonical_api_2", "canonical_api_3"), ids):
                    out[col] = api_id
                out["n_apis"] = len(ids)
                out["status"] = "mapped"

    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True)
    p.add_argument("--include-probe", action="store_true", help="also classify cost-probe calls")
    p.add_argument("--results-dir", default=str(RESULTS_DIR))
    args = p.parse_args(argv)

    task = load_task(args.task)
    results_dir = Path(args.results_dir)
    table = AliasTable(load_aliases(task))
    manual = load_manual_apis(task)
    rows, raws = load_calls(task, results_dir, args.include_probe)
    reviews = load_reviews(task, results_dir)
    questions = {v.id: v.text for v in task.variants}
    if not rows:
        sys.exit(f"No calls found under {results_dir / task.name}")

    shas = {"aliases": _sha(task.path / "aliases.yaml"), "manual_apis": _sha(task.path / "manual_apis.yaml")}
    unknown = set(manual) - {r["call_id"] for r in rows}
    if unknown:
        sys.exit(f"manual_apis.yaml lists calls that are not in the results: {sorted(unknown)}")
    results = []
    for r in rows:
        raw = raws.get(r["call_id"], {})
        extra = classify(r, raw.get("output_text", ""), table, manual.get(r["call_id"]))
        question = questions.get(r.get("variant_id", ""), "")
        results.append(result_row(r, extra, raw, question, reviews.get(r["call_id"], ""), shas))

    out = results_dir / task.name / "results.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        w.writeheader()
        w.writerows(results)

    counts: dict[str, int] = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(f"Classified {len(results)} calls -> {out}")
    print("  " + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])))
    unmapped = [r for r in results if r["status"] == "unmapped"]
    if unmapped:
        print(f"  {len(unmapped)} answers no alias matched (add them to aliases.yaml and rerun):")
        for r in unmapped:
            print(f"    {r['experiment']}  {r['call_id']}  {r['answer'][:100]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
