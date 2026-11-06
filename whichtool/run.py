"""Run a study (or the cost probe) for one task.

Examples:
    python -m whichtool.run --task doc-processing --probe
    python -m whichtool.run --task doc-processing --study headline --epochs 3
    python -m whichtool.run --task doc-processing --study search --models gpt-6-astra

Every call is appended to results/<task>/<run_id>/calls.csv and raw.jsonl as
soon as it finishes.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .config import (
    BUDGET_USD,
    MAX_CONNECTIONS,
    MODELS,
    PROBE_BUDGET_USD,
    RESULTS_DIR,
    STUDIES,
)
from .harness import Config, RunWriter, make_clients, run_plan
from .taskspec import load_task

# The cloud environment reserves ANTHROPIC_API_KEY for Claude Code's own
# login, so the benchmark key is stored under a different name.
ANTHROPIC_KEY_VARS = ("WHICHTOOL_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY")
OPENAI_KEY_VARS = ("WHICHTOOL_OPENAI_API_KEY", "OPENAI_API_KEY")


def setup_keys() -> dict[str, bool]:
    """Export the vendor keys under the names the SDKs read. Returns which vendors have one."""
    have = {}
    for vendor, names, target in (
        ("anthropic", ANTHROPIC_KEY_VARS, "ANTHROPIC_API_KEY"),
        ("openai", OPENAI_KEY_VARS, "OPENAI_API_KEY"),
    ):
        value = next((os.environ[n] for n in names if os.environ.get(n)), "")
        if value:
            os.environ[target] = value
        have[vendor] = bool(value)
    return have


def spent_so_far(results_dir: Path = RESULTS_DIR, kind: str | None = None) -> float:
    """Total cost_usd over every calls.csv under results/ (optionally one run kind)."""
    total = 0.0
    for path in results_dir.glob("**/calls.csv"):
        with path.open() as f:
            for row in csv.DictReader(f):
                if kind and row.get("run_kind") != kind:
                    continue
                try:
                    total += float(row.get("cost_usd") or 0)
                except ValueError:
                    pass
    return total


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:
        return "unknown"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, help="folder name under tasks/")
    p.add_argument("--study", choices=sorted(STUDIES), default="headline")
    p.add_argument("--probe", action="store_true", help="cost probe: 2 calls per configuration of every study")
    p.add_argument("--models", default=",".join(MODELS), help="comma-separated model labels")
    p.add_argument("--efforts", default=None, help="comma-separated effort levels (default: the study's)")
    p.add_argument("--epochs", type=int, default=1, help="repeats of every variant")
    p.add_argument("--jobs", default=None, help="only these job ids (comma-separated)")
    p.add_argument("--variants", default=None, help="only these variant ids (comma-separated)")
    p.add_argument("--max-usd", type=float, default=None, help="cap for this run (also capped by the total budget)")
    p.add_argument("--reserve-usd", type=float, default=0.30, help="assumed cost of a call in flight, for the budget guard")
    p.add_argument("--max-connections", type=int, default=MAX_CONNECTIONS, help="concurrent calls per configuration")
    p.add_argument("--allow-unfrozen", action="store_true", help="run a study before variants.yaml is frozen")
    p.add_argument("--no-schema", action="store_true",
                   help="don't enforce the JSON answer with structured output (the instruction still asks for JSON)")
    p.add_argument("--label", default="", help="free-text note stored in run.json")
    p.add_argument("--dry-run", action="store_true", help="print the plan and exit without calling any API")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    keys = setup_keys()
    task = load_task(args.task)

    labels = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in labels if m not in MODELS]
    if unknown:
        sys.exit(f"Unknown models {unknown}; choose from {list(MODELS)}")
    specs = [MODELS[m] for m in labels]
    missing = sorted({s.vendor for s in specs if not keys[s.vendor]})
    if missing and not args.dry_run:
        hint = {
            "anthropic": "set WHICHTOOL_ANTHROPIC_API_KEY",
            "openai": "set OPENAI_API_KEY",
        }
        sys.exit("Missing API keys: " + "; ".join(f"{v} ({hint[v]})" for v in missing))

    variants = task.variants
    if args.jobs:
        keep = set(args.jobs.split(","))
        variants = [v for v in variants if v.job_id in keep]
    if args.variants:
        keep = set(args.variants.split(","))
        variants = [v for v in variants if v.id in keep]
    if not variants:
        sys.exit("No variants selected. Has the task got jobs.yaml?")

    plan: list[Config] = []
    if args.probe:
        # 2 calls per configuration: the original text of two different jobs,
        # or one job twice if there is only one.
        originals = [v for v in variants if v.id.endswith("-v00")] or variants
        probe_variants, probe_epochs = (originals[:2], 1) if len(originals) >= 2 else (originals[:1], 2)
        for study in STUDIES.values():
            for spec in specs:
                for effort in study.efforts:
                    plan.append(Config(study, spec, effort, probe_variants, probe_epochs))
        run_kind = "probe"
        cap = PROBE_BUDGET_USD - spent_so_far(kind="probe")
    else:
        if not task.variants_frozen and not args.allow_unfrozen:
            sys.exit(
                "variants.yaml is not frozen. Freeze it (frozen: true) before a study run, "
                "or pass --allow-unfrozen for a test run."
            )
        study = STUDIES[args.study]
        efforts = args.efforts.split(",") if args.efforts else list(study.efforts)
        for spec in specs:
            for effort in efforts:
                plan.append(Config(study, spec, effort, variants, args.epochs))
        run_kind = "study"
        cap = float("inf")

    total_spent = spent_so_far()
    allowed = min(cap, BUDGET_USD - total_spent)
    if args.max_usd is not None:
        allowed = min(allowed, args.max_usd)

    n_calls = sum(len(c.variants) * c.epochs for c in plan)
    print(f"Task {task.name} | {run_kind} | {len(plan)} configurations | {n_calls} calls planned")
    for c in plan:
        print(f"  {c.study.name:9s} {c.spec.label:12s} {c.effort:6s} "
              f"{len(c.variants)} variants x {c.epochs} epochs")
    print(f"Spent so far ${total_spent:.2f} of ${BUDGET_USD:.0f}; this run may spend up to ${allowed:.2f}")
    if args.dry_run:
        return 0
    if allowed <= 0:
        sys.exit("No budget left for this run.")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{stamp}-{run_kind if args.probe else args.study}"
    run_dir = RESULTS_DIR / task.name / run_id
    writer = RunWriter(
        run_dir=run_dir,
        run_id=run_id,
        run_kind=run_kind,
        task_name=task.name,
        variants_sha=task.variants_sha,
        allowed_usd=allowed,
        reserve_usd=args.reserve_usd,
    )
    import anthropic
    import openai

    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "run_kind": run_kind,
                "task": task.name,
                "variants_sha": task.variants_sha,
                "variants_frozen": task.variants_frozen,
                "git_commit": git_commit(),
                "anthropic_sdk_version": anthropic.__version__,
                "openai_sdk_version": openai.__version__,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "allowed_usd": allowed,
                "label": args.label,
                "structured_output": not args.no_schema,
                "plan": [
                    {
                        "study": c.study.name,
                        "model": c.spec.label,
                        "model_id": c.spec.api_id,
                        "effort": c.effort,
                        "variants": [v.id for v in c.variants],
                        "epochs": c.epochs,
                    }
                    for c in plan
                ],
            },
            indent=2,
        )
    )

    clients = make_clients({c.spec.vendor for c in plan})
    asyncio.run(
        run_plan(clients, task, plan, writer, max_connections=args.max_connections,
                 schema=not args.no_schema)
    )
    print(
        f"\nRun {run_id}: {writer.n_written} calls written ({writer.n_errors} errors), "
        f"{writer.n_skipped} skipped for budget, ${writer.spent_usd:.2f} spent. Files in {run_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
