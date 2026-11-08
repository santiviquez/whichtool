# whichtool

How often do frontier models pick the best tool for a job?

Each prompt describes a real job (the first task: processing documents) without naming any tool. The model names the one hosted API it would use, and code maps each answer to an API id. No LLM judges anything. Grading the picks against a ground truth comes later, after the results are analyzed.

- **Models:** Claude Opus 5.5, Claude Fable 5.1, GPT-6 Astra, GPT-5.6 Sol.
- **Effort:** medium, high and xhigh on every model, set explicitly on every call (12 configurations).
- **Headline study:** all 12 configurations, no web search, the stated recommendation.
- **Side studies, at high effort:** each vendor's own web search available (not requested), and the API the model actually calls when asked to write the integration code.
- **Budget:** at most $140 of API calls in total (`BUDGET_USD` in config.py, raised from $100), enforced by the harness.

Everything is public: prompts, every request and response, and the classified results.

## What's in the repo

| Path | What it is |
|---|---|
| [`whichtool/`](whichtool) | The code: `run.py` runs the experiments (with `harness.py`, `records.py`, `estimate.py`), `classify.py` turns the saved answers into `results.csv`, `config.py` holds every setting |
| [`tasks/doc-processing/`](tasks/doc-processing) | The task: jobs, frozen variants, the API aliases, and the calls classified by hand |
| [`results/doc-processing/results.csv`](results/doc-processing/results.csv) | **The results: one row per call, everything together** (see below) |
| `results/doc-processing/<run_id>/` | The raw record of each run (probe, headline, search, code): every request and response, failed and repeated calls included |
| [`results/doc-processing/review/`](results/doc-processing/review) | Santiago's review of the classification (2026-11-08) |
| [`tests/`](tests) | Tests, with a made-up task in `tests/fixtures/example/` |

## Where the settings live

Two places, nothing else:

- **[`whichtool/config.py`](whichtool/config.py)** holds every setting of the experiment: the models and prices, effort levels, studies, the instructions and answer schema every model sees, the exact request settings (max tokens, thinking, search tools, structured output), how calls run (concurrency, retries, timeout), the budget, and the phrase lists the classifier uses. Command-line flags only choose what to run (task, study, models, efforts, jobs, variants, repeats).
- **`tasks/<task>/`** holds everything about one task: jobs, variants, aliases and calls classified by hand.

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
| `run.json` | The plan, the full settings from `config.py`, the instructions used, a hash of the prompts, and for each session (the first run and every resume) the git commit, SDK versions and how it ended: calls written, failed and skipped, money spent, configurations stopped |
| `started.txt` | The id of every call, written just before it is sent (see the budget guard) |
| `console.log` | Everything the run printed, errors and tracebacks included, one block per session |

Nothing in a run folder is ever overwritten or deleted. Failed calls keep their rows, `--resume` appends to the same files, and a rerun of the same study is a new folder. `results.csv` keeps only the latest successful call per slot, but the files above keep every attempt. The classifier reads only these files, so it can be rerun at any time without any API call.

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
5. **Classify.**
   ```bash
   python -m whichtool.classify --task doc-processing   # -> results/<task>/results.csv
   ```
   Answers no alias matches are printed. Add them to `tasks/<task>/aliases.yaml` and run it again.

## Results

[`results/doc-processing/results.csv`](results/doc-processing/results.csv) is the one file to analyse: one row per call (2,784: 2,400 headline, 192 search, 192 code; the cost probe is left out), with

- **the call:** `call_id`, `experiment` (headline, search, code), `model`, `effort`, `job`, `wording` (`j1-v00` to `j8-v04`), `wording_by` (ChatGPT wrote v00, Claude v01 to v04), `repeat`;
- **what the model saw and said:** `question` (the user message), `answer` (the api field of a stated answer, or the whole reply in the code experiment), `reason`, `sentence`;
- **the classification:** `api_1` to `api_3`, `n_apis`, `match_method` and `matched_on` (how it was matched), `review` (Santiago's review: `accepted`, `changed` or `accepted_in_bulk`), `status`;
- **search:** `searched`, `n_searches`, `search_queries`, `cited_urls`;
- **cost and run data:** tokens, `cost_usd`, `latency_s`, `model_returned`, `stop_reason`, `call_status`, `error`, `run_id`, timestamps, and the hashes of the prompts and alias files used.

`classify.py` writes it from the run folders. Each answer is mapped to an API id through `aliases.yaml`: names for stated answers, code signatures such as imports and hostnames for the code experiment. A call listed in `manual_apis.yaml` gets the APIs listed there instead (`match_method` "manual"); for doc-processing these are the 2 answers Santiago reclassified in his review.

| Status | Meaning |
|---|---|
| `mapped` | the answer names (or the code calls) at least one API: `api_1` to `api_3` |
| `hedge` | "it depends" or similar, with no API named |
| `refusal` | the model declined |
| `truncated` | no answer: the model used its whole output limit, or Claude's search was still running after the last continuation |
| `format_error` | the answer could not be parsed, or no code was written |
| `unmapped` | no alias matched yet; fix the alias table and rerun |
| `api_error` | the call failed after all retries |

All 2,784 calls are currently `mapped`.

An answer that names or calls more than one API (for example code that extracts with Azure Document Intelligence and then structures the result with the OpenAI API) keeps every one: in a stated answer `api_1` is the one named first; in code it is the one listed first in `aliases.yaml`. Self-hosted helpers found in code (such as PyMuPDF for rendering pages) are dropped when the code also calls a hosted API.

Grading against a ground truth is not in the repo yet. The grading and report code written before the runs (statuses correct, alternate, wrong and off-list; intervals clustered by job) is in git history at commit 71f80b6.

## Budget guard

`results/` is the ledger: before a run starts, the harness adds up `cost_usd` over every `calls.csv` and refuses to spend past `BUDGET_USD` ($140) in total (or past $5 across probe runs). During a run, a call only starts if the money already spent plus an amount set aside for every running call stays under the cap; otherwise it is recorded as `skipped_budget`. The amount set aside for a call is the most expensive successful call seen so far for its configuration (from the probe, an earlier run, or earlier in the same run), or `RESERVE_USD` ($0.30) before there is one. Because running calls are only estimated, a cap can be overshot by the difference between those estimates and what the calls really cost. A call that started (its id is in `started.txt`) but never reached `calls.csv`, because the process was killed, is counted at its set-aside amount, so the ledger never forgets money that may have been spent. Cost is computed from the token counts in each raw response at the list prices in [`whichtool/config.py`](whichtool/config.py), plus $10 per 1,000 web searches. A spend limit in each vendor's console is a good backstop.

## Adding a task

Create `tasks/<name>/` with `task.yaml`, `jobs.yaml`, `variants.yaml` and `aliases.yaml` (and `manual_apis.yaml` if any call is classified by hand). Nothing else changes. `tests/fixtures/example/` is a made-up task for the tests, not part of the benchmark.

## Notes on vendor differences

- In the search study, Claude gets its web search tool only (`web_search_20260209`, which can filter results with code) and no separate page-fetch tool; OpenAI's web search tool can open pages itself. Both are each vendor's own search, as designed.
- Claude's search runs a server-side loop that can stop with `pause_turn`; the harness sends the turn back so it continues (up to 5 times), and the usage of every request is added up for the call.
- Claude models return summarized thinking and OpenAI models return reasoning summaries; both are saved in `raw.jsonl` for error analysis and billed the same either way.
- GPT-5.6 Sol's $4 / $20 price is promotional through at least 2026-11-21; runs after that need the price in `config.py` updated.
