"""Run a study (or the cost probe) for one task.

Examples:
    python -m whichtool.run --task doc-processing --probe
    python -m whichtool.run --task doc-processing --study headline --epochs 3
    python -m whichtool.run --task doc-processing --study search --models gpt-6-astra
    python -m whichtool.run --task doc-processing --resume 20261107T010203Z-headline

Every call is appended to results/<task>/<run_id>/calls.csv and raw.jsonl as
soon as it finishes. Every setting comes from config.py; the flags here only
choose what to run. --resume runs the calls of an earlier run that have no
successful row yet: calls that failed, were skipped for budget, or never ran
because the run stopped.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import config as C
from .config import BUDGET_USD, MODELS, PROBE_BUDGET_USD, RESULTS_DIR, STUDIES
from .harness import (
    Config,
    RunWriter,
    config_key,
    done_slots,
    make_clients,
    preflight,
    run_plan,
    slot_key,
    slots,
)
from .taskspec import TaskSpec, load_task

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


def _run_kind(run_dir: Path) -> str:
    try:
        return json.loads((run_dir / "run.json").read_text()).get("run_kind", "")
    except (OSError, ValueError):
        return ""


def spent_so_far(results_dir: Path = RESULTS_DIR, kind: str | None = None) -> float:
    """The ledger: cost_usd over every calls.csv under results/ (optionally one run kind),
    plus the set-aside amount of every call that started but was never written."""
    total = 0.0
    for path in results_dir.glob("**/calls.csv"):
        written = set()
        with path.open() as f:
            for row in csv.DictReader(f):
                written.add(row.get("call_id"))
                if kind and row.get("run_kind") != kind:
                    continue
                try:
                    total += float(row.get("cost_usd") or 0)
                except ValueError:
                    pass
        started = path.parent / "started.txt"
        if started.exists() and (not kind or _run_kind(path.parent) == kind):
            for line in started.read_text().splitlines():
                call_id, _, reserve = line.strip().partition(",")
                if call_id and call_id not in written:
                    total += float(reserve) if reserve else C.RESERVE_USD
    return total


def observed_reserves(task_dir: Path) -> dict[tuple[str, str, str], float]:
    """The most expensive successful call seen so far for each configuration of a task."""
    out: dict[tuple[str, str, str], float] = {}
    for path in task_dir.glob("*/calls.csv"):
        with path.open() as f:
            for r in csv.DictReader(f):
                if r.get("status") != "ok":
                    continue
                key = (r["study"], r["model"], r["effort"])
                out[key] = max(out.get(key, 0.0), float(r.get("cost_usd") or 0))
    return out


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
    p.add_argument("--probe", action="store_true",
                   help=f"cost probe: {C.PROBE_CALLS_PER_CONFIG} calls per configuration of every study")
    p.add_argument("--resume", metavar="RUN_ID", default=None,
                   help="rerun the calls of an earlier run that have no successful row yet")
    p.add_argument("--models", default=",".join(MODELS), help="comma-separated model labels")
    p.add_argument("--efforts", default=None, help="comma-separated effort levels (default: the study's)")
    p.add_argument("--epochs", type=int, default=1, help="repeats of every variant")
    p.add_argument("--jobs", default=None, help="only these job ids (comma-separated)")
    p.add_argument("--variants", default=None, help="only these variant ids (comma-separated)")
    p.add_argument("--max-usd", type=float, default=None, help="cap for this run (also capped by the total budget)")
    p.add_argument("--allow-unfrozen", action="store_true", help="run a study before variants.yaml is frozen")
    p.add_argument("--label", default="", help="free-text note stored in run.json")
    p.add_argument("--dry-run", action="store_true", help="print the plan and exit without calling any API")
    return p.parse_args(argv)


def new_plan(args: argparse.Namespace, task: TaskSpec) -> tuple[list[Config], str]:
    labels = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in labels if m not in MODELS]
    if unknown:
        sys.exit(f"Unknown models {unknown}; choose from {list(MODELS)}")
    specs = [MODELS[m] for m in labels]

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
        # PROBE_CALLS_PER_CONFIG calls per configuration: the original text of
        # different jobs, repeating them when there are fewer jobs than calls.
        n = C.PROBE_CALLS_PER_CONFIG
        originals = ([v for v in variants if v.id.endswith("-v00")] or variants)[:n]
        epochs = math.ceil(n / len(originals))
        for study in STUDIES.values():
            for spec in specs:
                for effort in study.efforts:
                    plan.append(Config(study, spec, effort, originals, epochs))
        return plan, "probe"

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
    return plan, "study"


def resumed_plan(run_dir: Path, task: TaskSpec) -> tuple[list[Config], str, dict]:
    """The plan of an earlier run, after checking nothing that changes results has changed since."""
    try:
        saved = json.loads((run_dir / "run.json").read_text())
    except OSError:
        sys.exit(f"No run.json in {run_dir}")
    if saved.get("variants_sha") != task.variants_sha:
        sys.exit("The task's prompts changed since this run started, so it can't be resumed. Start a new run.")
    now = json.loads(json.dumps(C.snapshot()["experiment"]))
    if saved.get("config", {}).get("experiment") != now:
        sys.exit("Experiment settings in config.py changed since this run started, so it can't be resumed. "
                 "Compare run.json's \"config\" with config.py, or start a new run.")
    by_id = {v.id: v for v in task.variants}
    plan = [
        Config(STUDIES[c["study"]], MODELS[c["model"]], c["effort"], [by_id[i] for i in c["variants"]], c["epochs"])
        for c in saved["plan"]
    ]
    return plan, saved["run_kind"], saved


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    keys = setup_keys()
    task = load_task(args.task)

    saved = None
    if args.resume:
        run_id = args.resume
        run_dir = RESULTS_DIR / task.name / run_id
        plan, run_kind, saved = resumed_plan(run_dir, task)
    else:
        plan, run_kind = new_plan(args, task)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{stamp}-{run_kind if run_kind == 'probe' else args.study}"
        run_dir = RESULTS_DIR / task.name / run_id

    missing = sorted({c.spec.vendor for c in plan if not keys[c.spec.vendor]})
    if missing and not args.dry_run:
        hint = {"anthropic": "set WHICHTOOL_ANTHROPIC_API_KEY", "openai": "set OPENAI_API_KEY"}
        sys.exit("Missing API keys: " + "; ".join(f"{v} ({hint[v]})" for v in missing))

    skip = done_slots(run_dir / "calls.csv")
    todo = [s for s in slots(plan) if slot_key(*s) not in skip]

    total_spent = spent_so_far()
    cap = PROBE_BUDGET_USD - spent_so_far(kind="probe") if run_kind == "probe" else float("inf")
    allowed = min(cap, BUDGET_USD - total_spent)
    if args.max_usd is not None:
        allowed = min(allowed, args.max_usd)

    print(f"Task {task.name} | {run_kind} | {run_id} | {len(plan)} configurations | {len(todo)} calls to make"
          + (f" ({len(skip)} already done)" if skip else ""))
    for c in plan:
        print(f"  {c.study.name:9s} {c.spec.label:12s} {c.effort:6s} {len(c.variants)} variants x {c.epochs} epochs")
    print(f"Spent so far ${total_spent:.2f} of ${BUDGET_USD:.0f}; this run may spend up to ${allowed:.2f}")
    seen = observed_reserves(RESULTS_DIR / task.name)
    reserves = {config_key(c): seen.get(config_key(c), C.RESERVE_USD) for c in plan}
    from_data = sum(config_key(c) in seen for c in plan)
    print(f"Money set aside per running call: from earlier calls for {from_data} of {len(plan)} configurations, "
          f"${C.RESERVE_USD:.2f} (RESERVE_USD) for the rest")
    if args.dry_run:
        return 0
    if not todo:
        print("Nothing left to run.")
        return 0
    if allowed <= 0:
        sys.exit("No budget left for this run.")

    writer = asyncio.run(execute(args, task, plan, run_kind, run_id, run_dir, saved, allowed, len(todo), reserves, skip))
    if writer is None:
        return 1
    left = len([s for s in slots(plan) if slot_key(*s) not in done_slots(run_dir / "calls.csv")])
    print(
        f"\nRun {run_id}: {writer.n_written} calls written ({writer.n_errors} failed), "
        f"{writer.n_skipped} skipped for budget, ${writer.spent_usd:.2f} spent. Files in {run_dir}"
    )
    for key, err in writer.broken.items():
        print(f"Stopped {' '.join(key)} after an error that would repeat on every call: {err[:300]}")
    if left:
        print(f"{left} calls have no successful result yet. Run them with: "
              f"python -m whichtool.run --task {args.task} --resume {run_id}")
    return 0


async def execute(args, task, plan, run_kind, run_id, run_dir, saved, allowed, n_todo, reserves, skip):
    """Pre-flight check, run.json, then the calls, all on one event loop (the SDK clients belong to it)."""
    import anthropic
    import openai

    clients = make_clients({c.spec.vendor for c in plan})
    specs = list({c.spec.label: c.spec for c in plan}.values())
    errors = await preflight(clients, specs)
    if errors:
        print("Pre-flight check failed, nothing was spent:\n" + "\n".join(f"  {k}: {v}" for k, v in errors.items()))
        return None

    if saved is None:
        info = {
            "run_id": run_id,
            "run_kind": run_kind,
            "task": task.name,
            "variants_sha": task.variants_sha,
            "variants_frozen": task.variants_frozen,
            "answer_instruction": task.answer_instruction,
            "code_instruction": task.code_instruction,
            "label": args.label,
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
            "config": C.snapshot(),
            "sessions": [],
        }
    else:
        info = saved
    info["sessions"].append({
        "started_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "anthropic_sdk_version": anthropic.__version__,
        "openai_sdk_version": openai.__version__,
        "allowed_usd": allowed,
        "calls_to_make": n_todo,
        "reserves_usd": {" ".join(k): v for k, v in reserves.items()},
    })
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(json.dumps(info, indent=2))

    writer = RunWriter(
        run_dir=run_dir,
        run_id=run_id,
        run_kind=run_kind,
        task_name=task.name,
        variants_sha=task.variants_sha,
        allowed_usd=allowed,
    )
    await run_plan(clients, task, plan, writer, reserves=reserves, skip=skip)
    return writer


if __name__ == "__main__":
    raise SystemExit(main())
