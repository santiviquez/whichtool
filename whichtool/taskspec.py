"""Load a task folder: tasks/<name>/.

A task folder holds everything specific to one benchmark task:

- task.yaml          name, description, optional instruction overrides
- jobs.yaml          the original job descriptions (written by the task owner)
- variants.yaml      rephrased variants of each job, frozen before any run
- aliases.yaml       API names and code signatures -> canonical API ids
- manual_apis.yaml   calls classified by hand (optional)

Each job and variant may say who wrote its wording (`by`). Without it, a
job's original text (v00) is credited to ChatGPT and a variant to Claude,
which is how j1 to j8 of doc-processing were written.

Adding a new task means adding a new folder; nothing else changes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .config import TASKS_DIR
from .config import ANSWER_INSTRUCTION, CODE_INSTRUCTION


DEFAULT_JOB_BY = "ChatGPT"
DEFAULT_VARIANT_BY = "Claude"


@dataclass(frozen=True)
class Variant:
    id: str
    job_id: str
    text: str
    by: str = ""  # who wrote this wording; not part of variants_sha


@dataclass
class TaskSpec:
    name: str
    path: Path
    description: str
    answer_instruction: str
    code_instruction: str
    variants: list[Variant]
    variants_frozen: bool
    meta: dict = field(default_factory=dict)

    @property
    def variants_sha(self) -> str:
        """Short hash of the exact prompts, written into every run."""
        blob = json.dumps(
            [[v.id, v.job_id, v.text] for v in self.variants]
            + [self.answer_instruction, self.code_instruction],
            ensure_ascii=False,
        )
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def load_task(name: str, tasks_dir: Path = TASKS_DIR) -> TaskSpec:
    path = tasks_dir / name
    if not path.is_dir():
        raise FileNotFoundError(f"No task folder at {path}")
    meta = _read_yaml(path / "task.yaml")
    jobs = _read_yaml(path / "jobs.yaml").get("jobs") or []
    vfile = _read_yaml(path / "variants.yaml")

    variants: list[Variant] = []
    # The original job text is always variant v00 of its job. A job with no
    # text (its original wording was not kept) only has its variants.
    for job in jobs:
        if job.get("text"):
            variants.append(Variant(f"{job['id']}-v00", job["id"], job["text"].strip(), job.get("by", DEFAULT_JOB_BY)))
    for v in vfile.get("variants") or []:
        variants.append(Variant(v["id"], v["job_id"], v["text"].strip(), v.get("by", DEFAULT_VARIANT_BY)))

    ids = [v.id for v in variants]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"Duplicate variant ids in {path}: {sorted(dupes)}")
    job_ids = {j["id"] for j in jobs}
    orphans = {v.job_id for v in variants} - job_ids
    if orphans:
        raise ValueError(f"Variants refer to unknown jobs {sorted(orphans)} in {path}")

    return TaskSpec(
        name=meta.get("name", name),
        path=path,
        description=meta.get("description", ""),
        answer_instruction=meta.get("answer_instruction", ANSWER_INSTRUCTION).strip(),
        code_instruction=meta.get("code_instruction", CODE_INSTRUCTION).strip(),
        variants=variants,
        variants_frozen=bool(vfile.get("frozen", False)),
        meta=meta,
    )


def load_aliases(task: TaskSpec) -> dict:
    return _read_yaml(task.path / "aliases.yaml").get("apis") or {}


def load_manual_apis(task: TaskSpec) -> dict[str, list[str]]:
    """Calls classified by hand (manual_apis.yaml): call id -> APIs, API 1 first."""
    calls = _read_yaml(task.path / "manual_apis.yaml").get("calls") or {}
    return {str(cid): list((entry or {}).get("apis") or []) for cid, entry in calls.items()}
