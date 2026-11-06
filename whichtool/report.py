"""Build results/<task>/report.md from results/<task>/scored.csv. No API calls.

Pre-registered headline comparisons (from the design doc):
  1. each configuration's accuracy with a 95% interval
  2. medium vs xhigh within each model
  3. the best configuration of each vendor, head to head
Everything else in the report is exploratory and uncorrected for multiple tests.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .config import EFFORTS, MODELS, RESULTS_DIR
from .stats import clustered_mean_ci, mean_pass_hat_k, paired_diff_ci

MISS_STATUSES = ["wrong", "off_list", "hedge", "refusal", "truncated", "format_error", "unmapped"]
# Failed calls are not misses: they are left out of accuracy and shown on their own.
ERROR_STATUSES = ["api_error", "record_error"]


def _f(x: float, pct: bool = True) -> str:
    if x != x:  # nan
        return "n/a"
    return f"{100 * x:.0f}%" if pct else f"{x:.2f}"


def _ci(r: dict) -> str:
    if r["n"] == 0:
        return "n/a"
    if r["lo"] != r["lo"]:
        return f"{_f(r['mean'])}"
    return f"{_f(r['mean'])} ({_f(r['lo'])} to {_f(r['hi'])})"


def _diff(r: dict) -> str:
    if r["variants"] == 0:
        return "n/a"
    d = f"{100 * r['diff']:+.0f} pts"
    if r["lo"] != r["lo"]:
        return d
    lo, hi = max(-1.0, r["lo"]), min(1.0, r["hi"])
    return f"{d} ({100 * lo:+.0f} to {100 * hi:+.0f})"


def _num(x: float, fmt: str) -> str:
    return "n/a" if x != x else fmt.format(x)


def _p(x: float) -> str:
    return "n/a" if x != x else (f"{x:.3f}" if x >= 0.001 else "<0.001")


def load(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def order_key(model: str, effort: str) -> tuple:
    m = list(MODELS).index(model) if model in MODELS else 99
    e = EFFORTS.index(effort) if effort in EFFORTS else 99
    return (m, e)


class Config:
    def __init__(self, rows: list[dict], metric: str):
        self.rows = rows
        self.metric = metric
        self.graded = [r for r in rows if r.get(metric) not in ("", None)]
        self.values = [float(r[metric]) for r in self.graded]
        self.clusters = [r["variant_id"] for r in self.graded]
        self.by_variant: dict[str, list[float]] = defaultdict(list)
        for v, c in zip(self.values, self.clusters):
            self.by_variant[c].append(v)

    def ci(self) -> dict:
        return clustered_mean_ci(self.values, self.clusters)

    def mean(self, col: str) -> float:
        xs = [float(r[col]) for r in self.rows if r.get(col) not in ("", None)]
        return sum(xs) / len(xs) if xs else float("nan")

    def total(self, col: str) -> float:
        return sum(float(r[col]) for r in self.rows if r.get(col) not in ("", None))


def group(rows: list[dict], study: str) -> dict[tuple[str, str], list[dict]]:
    g: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        if r["study"] == study:
            g[(r["model"], r["effort"])].append(r)
    return dict(sorted(g.items(), key=lambda kv: order_key(*kv[0])))


def accuracy_table(groups: dict, epochs_k: int) -> list[str]:
    out = [
        "| Model | Effort | Scored runs | Strict accuracy (95% CI) | Lenient accuracy | "
        f"pass^{epochs_k} | Misses: wrong / off-list / hedge / refusal / cut off / format / unmapped | "
        "Failed calls (not scored) | Mean output tokens | Mean cost | Cost per correct |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for (model, effort), rows in groups.items():
        strict, lenient = Config(rows, "strict_correct"), Config(rows, "lenient_correct")
        st = Counter(r["status"] for r in rows)
        misses = " / ".join(str(st.get(s, 0)) for s in MISS_STATUSES)
        failed = sum(st.get(s, 0) for s in ERROR_STATUSES)
        n_correct = sum(strict.values)
        cost = strict.total("cost_usd")
        cpc = f"${cost / n_correct:.3f}" if n_correct else "n/a"
        pk = mean_pass_hat_k(strict.by_variant, epochs_k) if epochs_k >= 2 else float("nan")
        out.append(
            f"| {model} | {effort} | {len(strict.values)} | {_ci(strict.ci())} | {_ci(lenient.ci())} | {_f(pk)} | "
            f"{misses} | {failed} | {_num(strict.mean('output_tokens'), '{:.0f}')} | {_num(strict.mean('cost_usd'), '${:.4f}')} | {cpc} |"
        )
    return out


def picks_table(rows: list[dict], top: int = 6) -> list[str]:
    jobs = sorted({r["job_id"] for r in rows})
    out: list[str] = []
    for job in jobs:
        jr = [r for r in rows if r["job_id"] == job]
        overall = Counter(r["canonical_api"] or f"({r['status']})" for r in jr)
        cols = [k for k, _ in overall.most_common(top)]
        out += [f"**Job {job}**", "", "| Configuration | " + " | ".join(cols) + " | other |", "|---" * (len(cols) + 2) + "|"]
        g: dict[tuple, list[dict]] = defaultdict(list)
        for r in jr:
            g[(r["study"], r["model"], r["effort"])].append(r)
        for (study, model, effort), rr in sorted(g.items(), key=lambda kv: (kv[0][0], order_key(kv[0][1], kv[0][2]))):
            c = Counter(r["canonical_api"] or f"({r['status']})" for r in rr)
            other = len(rr) - sum(c.get(k, 0) for k in cols)
            out.append(f"| {study} {model} {effort} | " + " | ".join(str(c.get(k, 0)) for k in cols) + f" | {other} |")
        out.append("")
    return out


def build(rows: list[dict], task: str) -> str:
    graded = any(r.get("strict_correct") not in ("", None) for r in rows)
    epochs = max((int(r["epoch"]) for r in rows if r.get("epoch")), default=1)
    lines = [
        f"# whichtool results: {task}",
        "",
        f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC from scored.csv: {len(rows)} calls, "
        f"aliases {rows[0].get('aliases_sha') or 'n/a'}, ground truth {rows[0].get('ground_truth_sha') or 'n/a'}.",
        "",
        "Strict accuracy counts only the ground-truth best API; lenient also accepts the listed alternates. "
        "Intervals are 95% and clustered by prompt variant. Hedges, refusals, off-list picks and format "
        "failures all count as misses and are tallied separately.",
        "",
    ]
    if not graded:
        lines += ["**Not graded yet:** there is no ground_truth.yaml, so only picks, tokens and cost are shown.", ""]

    head = group(rows, "headline")
    if head:
        lines += ["## 1. Headline: accuracy per configuration (no web search)", ""]
        lines += accuracy_table(head, epochs) + [""]

        lines += ["## 2. Medium vs xhigh within each model (paired by variant)", ""]
        lines += ["| Model | medium | xhigh | xhigh minus medium (95% CI) | p | Variants |", "|---|---|---|---|---|---|"]
        for model in MODELS:
            a, b = head.get((model, "xhigh")), head.get((model, "medium"))
            if not a or not b:
                continue
            ca, cb = Config(a, "strict_correct"), Config(b, "strict_correct")
            d = paired_diff_ci(ca.by_variant, cb.by_variant)
            lines.append(f"| {model} | {_f(cb.ci()['mean'])} | {_f(ca.ci()['mean'])} | {_diff(d)} | {_p(d['p'])} | {d['variants']} |")
        lines.append("")

        lines += ["## 3. Best configuration of each vendor, head to head", ""]
        best = {}
        for (model, effort), rr in head.items():
            vendor = MODELS[model].vendor if model in MODELS else "?"
            acc = Config(rr, "strict_correct").ci()["mean"]
            if acc == acc and (vendor not in best or acc > best[vendor][0]):
                best[vendor] = (acc, model, effort)
        if len(best) == 2 and graded:
            (va, (acc_a, ma, ea)), (vb, (acc_b, mb, eb)) = sorted(best.items())
            d = paired_diff_ci(Config(head[(ma, ea)], "strict_correct").by_variant, Config(head[(mb, eb)], "strict_correct").by_variant)
            lines += [
                f"Best {va}: {ma} at {ea} ({_f(acc_a)}). Best {vb}: {mb} at {eb} ({_f(acc_b)}).",
                "",
                f"{ma} {ea} minus {mb} {eb}: {_diff(d)}, p {_p(d['p'])}, over {d['variants']} variants. "
                "Note the best configuration is picked from the same data, so this comparison is slightly optimistic.",
                "",
            ]
        else:
            lines += ["Needs graded results from both vendors.", ""]

    search = group(rows, "search")
    if search:
        lines += ["## 4. Side study: web search available (high effort)", ""]
        lines += accuracy_table(search, epochs) + [""]
        lines += ["| Model | Runs that searched | Mean searches | Search minus no search (95% CI) | p |", "|---|---|---|---|---|"]
        for (model, effort), rr in search.items():
            searched = sum(1 for r in rr if str(r.get("searched")).lower() == "true") / len(rr)
            base = head.get((model, effort)) if head else None
            d = (
                paired_diff_ci(Config(rr, "strict_correct").by_variant, Config(base, "strict_correct").by_variant)
                if base
                else {"variants": 0, "p": float("nan")}
            )
            lines.append(f"| {model} | {_f(searched)} | {Config(rr, 'n_searches').mean('n_searches'):.1f} | {_diff(d)} | {_p(d['p'])} |")
        lines.append("")

    code = group(rows, "code")
    if code:
        lines += ["## 5. Side study: the API used in integration code (high effort)", ""]
        lines += accuracy_table(code, epochs) + [""]
        if head:
            lines += [
                "Saying versus doing: for each variant, the API the model recommended most often (headline, same effort) "
                "against the API its code used most often.",
                "",
                "| Model | Variants compared | Same API |",
                "|---|---|---|",
            ]
            for (model, effort), rr in code.items():
                base = head.get((model, effort)) or []
                said, did = defaultdict(Counter), defaultdict(Counter)
                for r in base:
                    said[r["variant_id"]][r["canonical_api"] or r["status"]] += 1
                for r in rr:
                    did[r["variant_id"]][r["canonical_api"] or r["status"]] += 1
                common = sorted(set(said) & set(did))
                same = sum(1 for v in common if said[v].most_common(1)[0][0] == did[v].most_common(1)[0][0])
                lines.append(f"| {model} | {len(common)} | {_f(same / len(common)) if common else 'n/a'} |")
            lines.append("")

    lines += ["## 6. What each configuration picked, per job (exploratory)", ""]
    lines += picks_table(rows)

    lines += ["## 7. Spend", "", "| Study | Model | Calls | Cost |", "|---|---|---|---|"]
    spend: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        spend[(r["study"], r["model"])].append(float(r.get("cost_usd") or 0))
    for (study, model), xs in sorted(spend.items()):
        lines.append(f"| {study} | {model} | {len(xs)} | ${sum(xs):.2f} |")
    lines.append(f"| all | all | {len(rows)} | ${sum(float(r.get('cost_usd') or 0) for r in rows):.2f} |")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True)
    p.add_argument("--results-dir", default=str(RESULTS_DIR))
    args = p.parse_args(argv)
    path = Path(args.results_dir) / args.task / "scored.csv"
    if not path.exists():
        sys.exit(f"{path} not found; run `python -m whichtool.score --task {args.task}` first")
    rows = load(path)
    if not rows:
        sys.exit("scored.csv is empty")
    out = path.parent / "report.md"
    out.write_text(build(rows, args.task))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
