# whichtool

**Do AI agents recommend the best tool for the job?**

whichtool asks frontier models which hosted API they would use for a real job, without naming any tool in the prompt, and records what they pick. The first study is about document-processing APIs: 3,360 calls to Claude Opus 5.5, Claude Fable 5.1, GPT-6 Astra and GPT-5.6 Sol, asking for a recommendation, letting them search the web, and asking them to write the code.

Everything is in this repo: the prompts, every request and response, the code that maps each answer to an API, and the results. You can rerun it with your own prompts or on a different kind of tool.

**Read the write-up: [3,360 experiments to understand the future of product marketing](https://www.santiviquez.com/blog/3360-experiments-to-understand-the-future-of-product-marketing)**

[![Bar chart of which document-processing API each of the four models recommended when asked directly, out of 720 answers per model. Azure Document Intelligence, Google Document AI, the Gemini API and AWS Textract get most of the picks; startups such as Reducto, Extend and LlamaParse get few or none.](docs/which-api-each-model-picks.png)](https://www.santiviquez.com/blog/3360-experiments-to-understand-the-future-of-product-marketing)

## What the models picked

- **The big clouds win by default.** Asked directly, without web search, 2,282 of 2,880 answers (79%) recommended Azure Document Intelligence, Google Document AI, the Gemini API or AWS Textract.
- **Web search moves it a little.** With search available, the big clouds got 60% of the answers to the same prompts.
- **When writing code, models reach for their own API.** Opus 5.5 called the Claude API in 58 of its 60 implementations. GPT-6 Astra called the OpenAI API in 27 of 60.

This repo records what the models picked, not whether each pick was the right one. The [blog post](https://www.santiviquez.com/blog/3360-experiments-to-understand-the-future-of-product-marketing) gets into that.

## The study

| | |
|---|---|
| **Jobs** | 10 document-processing jobs: invoices, financial tables, receipts, printed paper forms and redlined contracts. Each one stresses one need (accuracy, latency, cost or ease of integration) and names no tool or vendor ([`jobs.yaml`](tasks/doc-processing/jobs.yaml)). |
| **Wordings** | 4 or 5 wordings of each job, 48 prompts in total ([`variants.yaml`](tasks/doc-processing/variants.yaml)). |
| **Models** | Claude Opus 5.5, Claude Fable 5.1, GPT-6 Astra, GPT-5.6 Sol. |
| **Effort** | Medium, high and xhigh, set explicitly on every call. |
| **Asked directly** | All 4 models at all 3 efforts, every prompt 5 times, no web search: 2,880 calls. |
| **With web search** | Each vendor's own web search available (not requested), high effort, 3 wordings per job, twice each: 240 calls. |
| **Writing code** | The model writes the integration code and the API it calls is its pick. High effort, same prompts as the search study: 240 calls. |
| **Isolation** | Every call is one stateless API call: one fixed instruction, the job as the user message, no persona, history or memory. |
| **Classification** | Code maps each answer to an API id with an alias table. No LLM judges anything. |
| **Cost** | $195.56 of API calls, failed and repeated calls included. |

## Run it with your own prompts or use case

Nothing in the harness is specific to documents. A task is a folder of YAML files, and the instruction every model sees is generic: *"Recommend the one hosted API you would use for the job described in the user's message."*

**1. Install and set your keys**

```bash
git clone https://github.com/santiviquez/whichtool && cd whichtool
uv venv && source .venv/bin/activate
uv pip install -e '.[dev]'
pytest

export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...   # WHICHTOOL_ANTHROPIC_API_KEY wins if both are set
```

**2. Write a task**

Copy the made-up example task (`cp -r tests/fixtures/example tasks/my-task`), or copy `tasks/doc-processing` to try new wordings of the document jobs. A task folder holds:

| File | What goes in it |
|---|---|
| `task.yaml` | The task's name and description. Optional `answer_instruction` and `code_instruction` replace the default instructions from `config.py`. |
| `jobs.yaml` | The jobs, each an `id` and a `text`. The text is all a model sees (it is sent as wording `v00`), so don't name any tool in it. |
| `variants.yaml` | Rewordings of each job (`id`, `job_id`, `text`). Check them for vendor names, then set `frozen: true`: study runs refuse to start before that. |
| `aliases.yaml` | The APIs you expect, the names models use for them, and code signatures (imports, package names, hostnames) for the code study. It doesn't need to be complete: the classifier prints every answer it can't match. |
| `manual_apis.yaml` | Optional. Calls you classified by hand. |

Jobs and variants can say who wrote them with `by`; it ends up in the `wording_by` column. Without it, `v00` is credited to ChatGPT and the rest to Claude, which is how this study's wordings were written.

**3. Check the cost, then run**

```bash
python -m whichtool.run --task my-task --probe       # 2 calls per configuration, capped at $5
python -m whichtool.estimate --task my-task          # how many repeats fit your budget
python -m whichtool.run --task my-task --study headline --epochs 5 --dry-run
python -m whichtool.run --task my-task --study headline --epochs 5
python -m whichtool.run --task my-task --study search --epochs 2
python -m whichtool.run --task my-task --study code --epochs 2
```

`--dry-run` prints the plan, the budget left and the expected cost without calling anything. `--models`, `--efforts`, `--jobs`, `--variants` and `--max-usd` narrow a run (this study ran search and code on `--variants` with 3 wordings per job). If a run stops or some calls fail, `--resume <run_id>` finishes it; the command is printed at the end of the run.

**4. Map the answers to APIs**

```bash
python -m whichtool.classify --task my-task          # -> results/my-task/results.csv
```

Add any answer it couldn't match to `aliases.yaml` and run it again. It reads only the saved run files, so it never calls an API.

**Budget and settings.** Each task has its own budget: `BUDGET_USD` in [`whichtool/config.py`](whichtool/config.py) ($210) caps what the runs in `results/<task>/` may spend, and `PROBE_BUDGET_USD` ($5) caps the probe. Set them to what you're willing to spend, and set a spend limit in each vendor's console too. Everything else about the experiment is in the same file: models and prices, effort levels, studies, the instructions and answer schema, request settings, retries and timeouts, and the phrase lists the classifier uses. Command-line flags only choose what to run. The harness supports Anthropic and OpenAI models; another vendor needs a client in `harness.py`.

## What's in the repo

| Path | What it is |
|---|---|
| [`whichtool/`](whichtool) | The code: `run.py` runs the experiments (with `harness.py`, `records.py`, `estimate.py`), `classify.py` turns the saved answers into `results.csv`, `config.py` holds every setting |
| [`tasks/doc-processing/`](tasks/doc-processing) | The task: jobs, frozen wordings, the API aliases, and the calls classified by hand |
| [`results/doc-processing/results.csv`](results/doc-processing/results.csv) | **The results: one row per call, everything together** (see below) |
| `results/doc-processing/<run_id>/` | The raw record of each run (probe, headline, search, code): every request and response, failed and repeated calls included |
| [`results/doc-processing/review/`](results/doc-processing/review) | The author's review of the classification (2026-11-08) |
| [`tests/`](tests) | Tests, with a made-up task in `tests/fixtures/example/` |

## The results file

[`results/doc-processing/results.csv`](results/doc-processing/results.csv) is the one file to analyse: one row per call (3,360: 2,880 asked directly, 240 with web search, 240 writing code; the cost probe is left out), with

- **the call:** `call_id`, `experiment` (headline, search, code), `model`, `effort`, `job`, `wording` (`j1-v00` to `j10-v04`), `wording_by` (for j1 to j8, ChatGPT wrote v00 and Claude v01 to v04; ChatGPT wrote all of j9 and j10), `repeat`;
- **what the model saw and said:** `question` (the user message), `answer` (the api field of a stated answer, or the whole reply in the code experiment), `reason`, `sentence`;
- **the classification:** `api_1` to `api_3`, `n_apis`, `match_method` and `matched_on` (how it was matched), `review` (the author's review: `accepted`, `changed` or `accepted_in_bulk`), `status`;
- **search:** `searched`, `n_searches`, `search_queries`, `cited_urls`;
- **cost and run data:** tokens, `cost_usd`, `latency_s`, `model_returned`, `stop_reason`, `call_status`, `error`, `run_id`, timestamps, and the hashes of the prompts and alias files used.

`classify.py` writes it from the run folders. Each answer is mapped to an API id through `aliases.yaml`: names for stated answers, code signatures such as imports and hostnames for the code experiment. A call listed in `manual_apis.yaml` gets the APIs listed there instead (`match_method` "manual"). For doc-processing these are 2 answers reclassified in the review of j1 to j8, and one j9 code answer counted as ABBYY only, because its OpenAI call is the job's own AI agent rather than the extraction API.

| Status | Meaning |
|---|---|
| `mapped` | the answer names (or the code calls) at least one API: `api_1` to `api_3` |
| `hedge` | "it depends" or similar, with no API named |
| `refusal` | the model declined |
| `truncated` | no answer: the model used its whole output limit, or Claude's search was still running after the last continuation |
| `format_error` | the answer could not be parsed, or no code was written |
| `unmapped` | no alias matched yet; fix the alias table and rerun |
| `api_error` | the call failed after all retries |

All 3,360 calls are `mapped`.

An answer that names or calls more than one API (for example code that extracts with Azure Document Intelligence and then structures the result with the OpenAI API) keeps every one: in a stated answer `api_1` is the one named first; in code it is the one listed first in `aliases.yaml`. Self-hosted helpers found in code (such as PyMuPDF for rendering pages) are dropped when the code also calls a hosted API.

**j9 and j10** (redlined contracts, as digital PDFs and as scans) were added on 2026-11-09 and run at the same settings as j1 to j8. ChatGPT wrote all their wordings. Only the four rewordings of each job were provided, so they have no `v00`: the headline study uses `v01` to `v04`, and the search and code studies use `v01` to `v03` where j1 to j8 use `v00` to `v02`.

The grading and report code written before the runs (statuses correct, alternate, wrong and off-list; intervals clustered by job) is not used and lives in git history at commit 71f80b6.

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

As soon as a call finishes it is appended to the files in `results/<task>/<run_id>/`:

| File | What it holds |
|---|---|
| `calls.csv` | One flat row per call: ids, study, job, variant, epoch, model requested and returned, effort, the answer fields, whether it searched, tokens, cost, latency, errors |
| `raw.jsonl` | The same plus the full prompt, the exact request and raw response of every API call, reasoning summaries, search queries and cited URLs |
| `run.json` | The plan, the full settings from `config.py`, the instructions used, a hash of the prompts, and for each session (the first run and every resume) the git commit, SDK versions and how it ended: calls written, failed and skipped, money spent, configurations stopped |
| `started.txt` | The id of every call, written just before it is sent (see the budget guard) |
| `console.log` | Everything the run printed, errors and tracebacks included, one block per session |

Nothing in a run folder is ever overwritten or deleted. Failed calls keep their rows, `--resume` appends to the same files, and a rerun of the same study is a new folder. `results.csv` keeps only the latest successful call per slot, but the files above keep every attempt.

## Budget guard

Each task's `results/<task>/` folder is its ledger: before a run starts, the harness adds up `cost_usd` over every `calls.csv` in it and refuses to spend past `BUDGET_USD` (or past `PROBE_BUDGET_USD` across probe runs). During a run, a call only starts if the money already spent plus an amount set aside for every running call stays under the cap; otherwise it is recorded as `skipped_budget`. The amount set aside for a call is the most expensive successful call seen so far for its configuration (from the probe, an earlier run, or earlier in the same run), or `RESERVE_USD` ($0.30) before there is one. Because running calls are only estimated, a cap can be overshot by the difference between those estimates and what the calls really cost. A call that started (its id is in `started.txt`) but never reached `calls.csv`, because the process was killed, is counted at its set-aside amount, so the ledger never forgets money that may have been spent. Cost is computed from the token counts in each raw response at the list prices in [`whichtool/config.py`](whichtool/config.py), plus $10 per 1,000 web searches.

## Notes on vendor differences

- In the search study, Claude gets its web search tool only (`web_search_20260209`, which can filter results with code) and no separate page-fetch tool; OpenAI's web search tool can open pages itself. Both are each vendor's own search, as designed.
- Claude's search runs a server-side loop that can stop with `pause_turn`; the harness sends the turn back so it continues (up to 10 times), and the usage of every request is added up for the call.
- Claude models return summarized thinking and OpenAI models return reasoning summaries; both are saved in `raw.jsonl` for error analysis and billed the same either way.
- GPT-5.6 Sol's $4 / $20 price is promotional through at least 2026-11-21; runs after that need the price in `config.py` updated.

## License

[MIT](LICENSE). If you build on the results, a link back to this repo or the [blog post](https://www.santiviquez.com/blog/3360-experiments-to-understand-the-future-of-product-marketing) is appreciated.
