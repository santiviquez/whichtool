# whichtool

How often do frontier models pick the best tool for a job?

Each prompt describes a real job (the first task: processing documents) without naming any tool. The model names the one hosted API it would use, and code scores that pick against a ground truth built from a structured comparison of the candidates. No LLM judges anything.

- **Models:** Claude Opus 5.5, Claude Fable 5.1, GPT-6 Astra, GPT-5.6 Sol.
- **Effort:** medium, high and xhigh on every model, set explicitly on every call (12 configurations).
- **Headline study:** all 12 configurations, no web search, the stated recommendation.
- **Side studies, at high effort:** each vendor's own web search available (not requested), and the API the model actually calls when asked to write the integration code.
- **Budget:** at most $100 of API calls in total, enforced by the harness.

Everything is public: prompts, every request and response, scores and the report.

## How a run works

The harness is a thin layer on [Inspect AI](https://inspect.aisi.org.uk/). Every (model, effort) pair is one Inspect task; every sample is one stateless API call:

- The system message is one fixed answer instruction, the same for every model ([`whichtool/prompts.py`](whichtool/prompts.py)). The job text is the user message. No persona, no history, no memory.
- Nothing carries over between calls: no conversation ids, no `previous_response_id`, `store: false` on OpenAI.
- Stated answers come back as structured output: `{"api", "reason", "sentence"}`, with `reason` one of accuracy, latency, cost, ease_of_integration, reputation, other.
- No sampling parameters are set (none of these models accept them at these effort levels), so repeats use each vendor's default sampling.
- Claude refusal fallbacks are left off on purpose: a fallback would silently answer with a different model.

As soon as a call finishes it is appended to two files in `results/<task>/<run_id>/`:

| File | What it holds |
|---|---|
| `calls.csv` | One flat row per call: ids, study, job, variant, epoch, model requested and returned, effort, the answer fields, whether it searched, tokens, cost, latency, errors |
| `raw.jsonl` | The same plus the full prompt, the exact request and raw response of every API call, reasoning summaries, search queries and cited URLs |
| `run.json` | The plan, the git commit, the Inspect version and a hash of the prompts |
| `inspect-logs/` | Inspect's own logs, for `inspect view --log-dir results/<task>/<run_id>/inspect-logs` |

Scoring and reporting read only these files, so they can be rerun at any time without the harness or any API call.

## Setup

```bash
uv venv && source .venv/bin/activate
uv pip install -e '.[dev]'
pytest
```

API keys come from the environment:

| Variable | Used for |
|---|---|
| `OPENAI_API_KEY` | GPT-6 Astra and GPT-5.6 Sol |
| `WHICHTOOL_ANTHROPIC_API_KEY` | Opus 5.5 and Fable 5.1. Claude Code cloud environments reserve `ANTHROPIC_API_KEY` for their own login, so the harness reads this name first and falls back to `ANTHROPIC_API_KEY`. |

## Workflow for a task

1. **Jobs.** The task owner writes 4 to 6 jobs in `tasks/<task>/jobs.yaml`, each with a different priority. The original text of each job is variant `v00`.
2. **Variants.** Rephrased variants go in `tasks/<task>/variants.yaml`, written without seeing the ground truth. The task owner checks them for vendor-specific terms, then sets `frozen: true`. Study runs refuse to start before that.
3. **Cost probe.** Two calls per configuration of every study, capped at $5:
   ```bash
   python -m whichtool.run --task doc-processing --probe
   python -m whichtool.estimate --task doc-processing   # runs per configuration that fit the budget
   ```
4. **Studies.**
   ```bash
   python -m whichtool.run --task doc-processing --study headline --epochs 3
   python -m whichtool.run --task doc-processing --study search --epochs 3
   python -m whichtool.run --task doc-processing --study code --epochs 3
   ```
   `--dry-run` prints the plan and the remaining budget without calling anything. `--models`, `--efforts`, `--jobs`, `--variants` and `--max-usd` narrow a run.
5. **Ground truth.** After the variants are frozen, the task owner adds `tasks/<task>/ground_truth.yaml` (see `ground_truth.example.yaml`): the APIs in the comparison, and the best API plus acceptable alternates for each job.
6. **Score and report.**
   ```bash
   python -m whichtool.score --task doc-processing    # -> results/<task>/scored.csv, unmapped.csv
   python -m whichtool.report --task doc-processing   # -> results/<task>/report.md
   ```
   Answers no alias matches land in `unmapped.csv`. Add them to `tasks/<task>/aliases.yaml` and score again.

## Scoring

The scorer maps each answer to a canonical API id through `aliases.yaml` (names for stated answers, code signatures such as imports and hostnames for the code study), then grades it:

| Status | Meaning | Strict | Lenient |
|---|---|---|---|
| `correct` | the ground-truth best API | hit | hit |
| `alternate` | an acceptable near-tie | miss | hit |
| `wrong` | another API from the comparison | miss | miss |
| `off_list` | an API outside the comparison, or a self-hosted tool | miss | miss |
| `hedge` | more than one API, or "it depends" | miss | miss |
| `refusal` | the model declined | miss | miss |
| `format_error` | the answer could not be parsed, or no code was written | miss | miss |
| `api_error` | the call failed | miss | miss |
| `unmapped` | no alias matched yet; fix the alias table and rescore | miss | miss |

The report covers the three pre-registered comparisons (each configuration's accuracy with a 95% interval, medium vs xhigh within each model, and the best configuration of each vendor head to head), then the side studies, pass^k consistency, cost per correct pick and every configuration's picks per job. Intervals are clustered by prompt variant, and comparisons are paired by variant.

## Budget guard

`results/` is the ledger: before a run starts, the harness adds up `cost_usd` over every `calls.csv` and refuses to spend past $100 in total (or past $5 across probe runs). During a run, a call only starts if the money already spent plus an allowance for calls in flight stays under the cap; otherwise it is recorded as `skipped_budget`. Cost is computed from the token counts in each raw response at the list prices in [`whichtool/config.py`](whichtool/config.py), plus $10 per 1,000 web searches.

## Adding a task

Create `tasks/<name>/` with `task.yaml`, `jobs.yaml`, `variants.yaml`, `aliases.yaml` and, once the variants are frozen, `ground_truth.yaml`. Nothing else changes. `tasks/example/` is a made-up smoke-test task, not part of the benchmark.

## Notes on vendor differences

- In the search study, Inspect's native Anthropic web search also gives Claude a web fetch tool; OpenAI's web search tool can open pages itself. Both are each vendor's own search, as designed.
- Claude models return summarized thinking and OpenAI models return reasoning summaries; both are saved in `raw.jsonl` for error analysis and billed the same either way.
- GPT-5.6 Sol's $4 / $20 price is promotional through at least 2026-11-21; runs after that need the price in `config.py` updated.
