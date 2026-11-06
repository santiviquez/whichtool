"""Score saved calls against the ground truth. No API calls.

Reads only:
    results/<task>/*/calls.csv and raw.jsonl   (what the harness wrote)
    tasks/<task>/aliases.yaml                  (API names -> canonical ids)
    tasks/<task>/ground_truth.yaml             (best API and alternates per job)
Writes:
    results/<task>/scored.csv     one row per call, with its status
    results/<task>/unmapped.csv   answers no alias matched, for review

Statuses (every one except correct/alternate is a miss, except the two
error statuses: a call that failed is left out of accuracy, because it says
nothing about the model's choice. Rerun it with `run --resume`; once a slot
has a successful call, its earlier failed rows are dropped):
    correct      the ground-truth best API
    alternate    an acceptable alternate (counts only for the lenient score)
    wrong        another API from the comparison
    off_list     a known API outside the comparison, or a self-hosted tool
    hedge        more than one API named, or "it depends"
    refusal      the model declined (refusal stop reason or refusal text)
    truncated    no answer because the model used its whole output limit, or
                 Claude's search was still running after the last continuation
    format_error the answer could not be parsed, or the code had no code
    api_error    the call failed
    record_error the call finished but the harness failed to record it (a harness bug)
    unmapped     no alias matched; add it to aliases.yaml and rescore
    unscored     mapped, but there is no ground truth yet
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import HEDGE_PHRASES, REFUSAL_PHRASES, RESULTS_DIR
from .taskspec import TaskSpec, load_aliases, load_ground_truth, load_task

ERROR_STATUSES = ("api_error", "record_error")
# stop_reason values (as calls.csv writes them) that mean the answer never finished
CUT_OFF = ("max_tokens", "pause_turn", "incomplete:max_output_tokens")

SCORED_EXTRA = [
    "canonical_api",
    "matched_names",
    "match_method",
    "in_comparison",
    "status",
    "strict_correct",
    "lenient_correct",
    "aliases_sha",
    "ground_truth_sha",
]


def norm(s: str) -> str:
    s = s.lower().replace("&", " and ")
    s = re.sub(r"[™®©]", "", s)
    s = re.sub(r"[^a-z0-9+.# ]+", " ", s)
    s = re.sub(r"(?<![a-z0-9])\.|\.(?![a-z0-9])", " ", s)  # dots only inside tokens (e.g. "api.x.com")
    return re.sub(r"\s+", " ", s).strip()


@dataclass
class Match:
    api_id: str | None
    names: list[str]
    method: str  # exact | contains | multiple | none | code | code_multiple


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
                return Match(api_id, [n], "exact")
        spans: list[tuple[int, int, str, str]] = []
        for n, api_id in self.names:
            for m in re.finditer(rf"(?<![a-z0-9]){re.escape(n)}(?![a-z0-9])", a):
                s, e = m.span()
                if any(s >= s2 and e <= e2 for s2, e2, _, _ in spans):
                    continue  # inside a longer alias already found
                spans.append((s, e, n, api_id))
        ids = list(dict.fromkeys(api_id for _, _, _, api_id in spans))
        names = [n for _, _, n, _ in spans]
        if len(ids) == 1:
            return Match(ids[0], names, "contains")
        if len(ids) > 1:
            return Match(None, names, "multiple")
        return Match(None, [], "none")

    def match_code(self, text: str) -> Match:
        t = code_only(text).lower()
        hits = [(c, api_id) for c, api_id in self.code if c in t]
        ids = list(dict.fromkeys(api_id for _, api_id in hits))
        # Helpers such as PyMuPDF for page rendering don't make it a hedge
        # when the code also calls a hosted API.
        if any(self.hosted(i) for i in ids):
            ids = [i for i in ids if self.hosted(i)]
            hits = [(c, i) for c, i in hits if i in ids]
        names = [c for c, _ in hits]
        if len(ids) == 1:
            return Match(ids[0], names, "code")
        if len(ids) > 1:
            return Match(None, names, "code_multiple")
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


def load_calls(task: TaskSpec, results_dir: Path, include_probe: bool) -> tuple[list[dict], dict[str, str]]:
    rows: list[dict] = []
    texts: dict[str, str] = {}
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
                    texts[rec["call_id"]] = rec.get("output_text") or ""
    return drop_superseded(rows), texts


def _slot(r: dict) -> tuple:
    return (r.get("run_id"), r.get("study"), r.get("model"), r.get("effort"), r.get("variant_id"), r.get("epoch"))


def drop_superseded(rows: list[dict]) -> list[dict]:
    """Drop failed rows of a slot (run, configuration, variant, repeat) that a resumed run later got right."""
    ok = {_slot(r) for r in rows if r.get("status") == "ok"}
    return [r for r in rows if r.get("status") == "ok" or _slot(r) not in ok]


def classify(row: dict, text: str, table: AliasTable, gt: dict | None) -> dict:
    out = {k: "" for k in SCORED_EXTRA}
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
        out["matched_names"] = "; ".join(m.names)
        out["match_method"] = m.method
        if not out["status"]:
            answer = norm(row.get("api", ""))
            if m.method in ("multiple", "code_multiple"):
                out["status"] = "hedge"
            elif any(norm(p) in answer for p in REFUSAL_PHRASES):
                out["status"] = "refusal"
            elif m.api_id is None and any(f" {norm(p)} " in f" {answer} " for p in HEDGE_PHRASES):
                out["status"] = "hedge"
            elif m.api_id is None:
                out["status"] = "unmapped"
            else:
                out["canonical_api"] = m.api_id
                out["status"] = _grade(m.api_id, row.get("job_id", ""), table, gt)
                if gt:
                    out["in_comparison"] = m.api_id in set(gt.get("comparison") or [])

    if gt and out["status"] not in ("unscored", *ERROR_STATUSES):
        out["strict_correct"] = int(out["status"] == "correct")
        out["lenient_correct"] = int(out["status"] in ("correct", "alternate"))
    return out


def _grade(api_id: str, job_id: str, table: AliasTable, gt: dict | None) -> str:
    if not gt:
        return "unscored"
    job = (gt.get("jobs") or {}).get(job_id)
    if not job:
        return "unscored"
    if not table.hosted(api_id):
        return "off_list"
    if api_id == job.get("best"):
        return "correct"
    if api_id in set(job.get("alternates") or []):
        return "alternate"
    if api_id in set(gt.get("comparison") or []):
        return "wrong"
    return "off_list"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True)
    p.add_argument("--include-probe", action="store_true", help="also score cost-probe calls")
    p.add_argument("--results-dir", default=str(RESULTS_DIR))
    args = p.parse_args(argv)

    task = load_task(args.task)
    results_dir = Path(args.results_dir)
    table = AliasTable(load_aliases(task))
    gt = load_ground_truth(task)
    rows, texts = load_calls(task, results_dir, args.include_probe)
    if not rows:
        sys.exit(f"No calls found under {results_dir / task.name}")

    a_sha, g_sha = _sha(task.path / "aliases.yaml"), _sha(task.path / "ground_truth.yaml")
    scored = []
    for r in rows:
        extra = classify(r, texts.get(r["call_id"], ""), table, gt)
        extra["aliases_sha"], extra["ground_truth_sha"] = a_sha, g_sha
        scored.append({**r, **extra})

    out_dir = results_dir / task.name
    columns = list(rows[0].keys()) + SCORED_EXTRA
    with (out_dir / "scored.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(scored)

    unmapped: dict[tuple[str, str], int] = {}
    for r in scored:
        if r["status"] == "unmapped":
            key = (r["study"], r["api"] if r["study"] != "code" else f"[code] call {r['call_id']}")
            unmapped[key] = unmapped.get(key, 0) + 1
    with (out_dir / "unmapped.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["study", "answer", "count"])
        for (study, ans), n in sorted(unmapped.items(), key=lambda kv: -kv[1]):
            w.writerow([study, ans, n])

    counts: dict[str, int] = {}
    for r in scored:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(f"Scored {len(scored)} calls -> {out_dir / 'scored.csv'}")
    print("  " + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])))
    if not gt:
        print("  No ground_truth.yaml yet: answers are mapped but not graded.")
    if unmapped:
        print(f"  {sum(unmapped.values())} unmapped answers -> {out_dir / 'unmapped.csv'} (add them to aliases.yaml)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
