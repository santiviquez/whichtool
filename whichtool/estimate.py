"""Turn the cost probe into run counts that fit the remaining budget. No API calls.

    python -m whichtool.estimate --task doc-processing

Reads every probe run's calls.csv for the task, takes the mean cost per call
of each configuration, and says how many runs per configuration each study
can afford with its share of what is left of the budget. The shares and
the margin held back are STUDY_SHARES and SAFETY_MARGIN in config.py.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

from .config import BUDGET_USD, RESERVE_USD, RESULTS_DIR, SAFETY_MARGIN, STUDIES, STUDY_SHARES
from .run import spent_so_far
from .taskspec import load_task


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True)
    p.add_argument("--results-dir", default=str(RESULTS_DIR))
    args = p.parse_args(argv)

    task = load_task(args.task)
    results_dir = Path(args.results_dir)
    costs: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    tokens: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for path in (results_dir / task.name).glob("*/calls.csv"):
        with path.open() as f:
            for r in csv.DictReader(f):
                if r.get("run_kind") != "probe" or r.get("status") != "ok":
                    continue
                key = (r["study"], r["model"], r["effort"])
                costs[key].append(float(r["cost_usd"] or 0))
                tokens[key].append(float(r["output_tokens"] or 0))
    if not costs:
        sys.exit("No successful probe calls found. Run `python -m whichtool.run --task ... --probe` first.")

    print("Mean cost per call from the probe:")
    print(f"  {'study':9s} {'model':12s} {'effort':6s} {'calls':>5s} {'out tok':>8s} {'$/call':>8s}")
    for key in sorted(costs):
        c, t = costs[key], tokens[key]
        print(f"  {key[0]:9s} {key[1]:12s} {key[2]:6s} {len(c):5d} {sum(t)/len(t):8.0f} {sum(c)/len(c):8.4f}")

    worst = max(max(c) for c in costs.values())
    print(f"\nMost expensive probe call: ${worst:.4f}. RESERVE_USD in config.py is ${RESERVE_USD:.2f}"
          + ("; raise it to at least that." if worst > RESERVE_USD else "."))

    spent = spent_so_far(results_dir)
    remaining = (BUDGET_USD - spent) * (1 - SAFETY_MARGIN)
    shares = STUDY_SHARES
    n_variants = len(task.variants)
    print(f"\nSpent ${spent:.2f}. Planning with ${remaining:.2f} ({SAFETY_MARGIN:.0%} held back). "
          f"The task has {n_variants} variants across {len({v.job_id for v in task.variants})} jobs.")
    for study_name, share in shares.items():
        study = STUDIES[study_name]
        keys = [k for k in costs if k[0] == study_name]
        if not keys:
            print(f"\n{study_name}: no probe data")
            continue
        per_round = sum(sum(costs[k]) / len(costs[k]) for k in keys)  # one call in every configuration
        runs = math.floor(remaining * share / per_round) if per_round > 0 else 0
        epochs = runs // n_variants if n_variants else 0
        print(
            f"\n{study_name}: {len(keys)} configurations, ${per_round:.3f} for one call in each. "
            f"${remaining * share:.2f} buys {runs} runs per configuration"
            + (f" = all {n_variants} variants x {epochs} epochs" if epochs >= 1 else
               f" (fewer than one pass over the {n_variants} variants)")
        )
        if study.efforts != ("high",) and epochs < 2:
            print("  Warning: under 2 epochs per variant, pass^k and consistency can't be measured.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
