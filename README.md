# whichtool

How often do frontier models pick the best tool for a job?

Each prompt describes a real job (the first task: processing documents) without naming any tool. The model names the one hosted API it would use, and code scores that pick against a ground truth built from a structured comparison of the candidates. No LLM judges anything.

- **Models:** Claude Opus 5.5, Claude Fable 5.1, GPT-6 Astra, GPT-5.6 Sol.
- **Effort:** medium, high and xhigh on every model, set explicitly on every call (12 configurations).
- **Headline study:** all 12 configurations, no web search, the stated recommendation.
- **Side studies, at high effort:** each vendor's own web search available (not requested), and the API the model actually calls when asked to write the integration code.
- **Budget:** at most $100 of API calls in total, enforced by the harness.

Everything is public: prompts, every request and response, scores and the report.

## Where the settings live

Two places, nothing else:

- **[`whichtool/config.py`](whichtool/config.py)** holds every setting of the experiment: the models and prices, effort levels, studies, the instructions and answer schema every model sees, the exact request settings (max tokens, thinking, search tools, structured output), how calls run (concurrency, retries, timeout), the budget, and the scoring phrase lists. Command-line flags only choose what to run (task, study, models, efforts, jobs, variants, repeats).
- **`tasks/<task>/`** holds everything about one task: jobs, variants, aliases and ground truth.

Every run saves a full copy of `config.py`'s settings in its `run.json`.

## How a run works

The harness calls the Anthropic and OpenAI Python SDKs directly, so the request saved for each call is exactly what was sent. Every (study, model, effort) is one configuration; every call is one stateless API call:

- The system message is one fixed answer instruction, the same for every model ([`whichtool/config.py`](whichtool/config.py)). The job text is the user message. No persona, no history, no memory.
- Nothing carries over between calls: no conversation ids, no `previous_response_id`, `store: false` on OpenAI.
- Stated answers come back as structured output: `{"api", "reason", "sentence"}`, with `reason` one of accuracy, latency, cost, ease_of_integration, reputation, other.
- No sampling parameters are set (none of these models accept them at these effort levels), so repeats use each vendor's default sampling.
- Claude: adaptive thinking with summarized display, `output_config.effort` and `output_config.format` (JSON schema). OpenAI: the Responses API with `reasoning.effort`, `reasoning.summary: auto` and a strict JSON schema. Both stream, so a long answer never sits on an idle connection until a timeout.
- Every call may use the model's full output limit (128K tokens on all four), so an answer is only cut off where the model itself would stop. The limit is a ceiling: only tokens used are billed.
- Before anything is spent, a pre-flight check looks up every model with its vendor's models endpoint (free), which catches a bad key or model id.
- The SDKs retry rate limits, server errors and dropped connections (6 retries, 20-minute timeout). A call that still fails is recorded as `api_error` and the run goes on. An error that would repeat on every call of a configuration (a rejected request, a bad key or model id, no credits) stops that configuration; its remaining calls stay open for `--resume`.
- The first Ctrl-C stops new calls and lets the running ones finish and be written; a second one quits at once. `--resume <run_id>` then runs every call of that run that has no successful row yet (failed, skipped for budget, or never started), appending to the same files. A run can only be resumed if the task's prompts and the experiment settings in `config.py` are unchanged.
- Claude refusal fallbacks are left off on purpose: a fallback would silently answer with a different model.

As soon as a call finishes it is appended to two files in `results/<task>/<run_id>/`:

| File | What it holds |
|---|---|
| `calls.csv` | One flat row per call: ids, study, job, variant, epoch, model requested and returned, effort, the answer fields, whether it searched, tokens, cost, latency, errors |
| `raw.jsonl` | The same plus the full prompt, the exact request and raw response of every API call, reasoning summaries, search queries and cited URLs |
| `run.json` | The plan, the full settings from `config.py`, the instructions used, a hash of the prompts, and for each session (the first run and every resume) the git commit and SDK versions |
| `started.txt` | The id of every call, written just before it is sent (see the budget guard) |

Scoring and reporting read only these files, so they can be rerun at any time without any API call.

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

1. **Jobs.** The task owner writes 4 to 6 jobs in `tasks/<task>/jobs.yaml`, each stressing a different need (accuracy, latency, cost, ease of integration). Only the `id` and `text` are read, and the text is all a model sees. The original text of each job is variant `v00`.
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
   `--dry-run` prints the plan and the remaining budget without calling anything. `--models`, `--efforts`, `--jobs`, `--variants` and `--max-usd` narrow a run. If a run stops or some calls fail, finish it with `python -m whichtool.run --task doc-processing --resume <run_id>` (the command is printed at the end of the run).
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
| `truncated` | no answer: the model used its whole output limit, or Claude's search was still running after the last continuation | miss | miss |
| `format_error` | the answer could not be parsed, or no code was written | miss | miss |
| `api_error` | the call failed after all retries | not scored | not scored |
| `unmapped` | no alias matched yet; fix the alias table and rescore | miss | miss |

A failed call says nothing about the model's choice, so it is left out of accuracy and counted in the report's "failed calls" column instead. Rerun it with `--resume`; once that call succeeds, the earlier failure is dropped.

The report covers the three pre-registered comparisons (each configuration's accuracy with a 95% interval, medium vs xhigh within each model, and the best configuration of each vendor head to head), then the side studies, pass^k consistency, cost per correct pick and every configuration's picks per job. Intervals are clustered by job and comparisons are paired by job, because rewordings of one job mostly get the same answer: with 8 jobs the intervals are honest but wide.

## Budget guard

`results/` is the ledger: before a run starts, the harness adds up `cost_usd` over every `calls.csv` and refuses to spend past $100 in total (or past $5 across probe runs). During a run, a call only starts if the money already spent plus an amount set aside for every running call stays under the cap; otherwise it is recorded as `skipped_budget`. The amount set aside for a call is the most expensive successful call seen so far for its configuration (from the probe, an earlier run, or earlier in the same run), or `RESERVE_USD` ($0.30) before there is one. Because running calls are only estimated, a cap can be overshot by the difference between those estimates and what the calls really cost. A call that started (its id is in `started.txt`) but never reached `calls.csv`, because the process was killed, is counted at its set-aside amount, so the ledger never forgets money that may have been spent. Cost is computed from the token counts in each raw response at the list prices in [`whichtool/config.py`](whichtool/config.py), plus $10 per 1,000 web searches. A spend limit in each vendor's console is a good backstop.

## Adding a task

Create `tasks/<name>/` with `task.yaml`, `jobs.yaml`, `variants.yaml`, `aliases.yaml` and, once the variants are frozen, `ground_truth.yaml`. Nothing else changes. `tasks/example/` is a made-up smoke-test task, not part of the benchmark.

## Notes on vendor differences

- In the search study, Claude gets its web search tool only (`web_search_20260209`, which can filter results with code) and no separate page-fetch tool; OpenAI's web search tool can open pages itself. Both are each vendor's own search, as designed.
- Claude's search runs a server-side loop that can stop with `pause_turn`; the harness sends the turn back so it continues (up to 5 times), and the usage of every request is added up for the call.
- Claude models return summarized thinking and OpenAI models return reasoning summaries; both are saved in `raw.jsonl` for error analysis and billed the same either way.
- GPT-5.6 Sol's $4 / $20 price is promotional through at least 2026-11-21; runs after that need the price in `config.py` updated.
