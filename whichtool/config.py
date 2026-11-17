"""Every setting of the experiment, in one file.

Task-specific inputs (jobs, variants, aliases) live in tasks/<task>/.
Everything else a run depends on is here: models and prices, effort levels,
studies, the prompts every model sees, the exact request settings, how calls
are executed, the budget, and the phrase lists the classifier uses.
A copy of all of it is saved in each run's run.json (see snapshot()).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TASKS_DIR = ROOT / "tasks"
RESULTS_DIR = ROOT / "results"


# ============================================================== models


@dataclass(frozen=True)
class ModelSpec:
    label: str  # short name used in results and reports
    vendor: str  # "anthropic" | "openai"
    api_id: str  # model id sent to the vendor API
    usd_in: float  # per 1M input tokens
    usd_out: float  # per 1M output tokens (reasoning tokens are billed as output)
    usd_cache_read: float  # per 1M cached input tokens
    usd_cache_write: float  # per 1M cache-write input tokens (Anthropic only)
    max_output_tokens: int  # the model's own output limit (reasoning included)


# Prices checked 2026-11-02 (see the design doc's "Provider settings" table).
# Output limits checked 2026-11-06: 128K on all four (Anthropic model table,
# OpenAI model pages).
# GPT-5.6 Sol's $4 / $20 is promotional through at least 2026-11-21.
# OpenAI cached-input prices for these two models are not used by our
# prompts (they are far below the caching minimum), so the full input price
# is used as a conservative stand-in.
MODELS: dict[str, ModelSpec] = {
    m.label: m
    for m in [
        ModelSpec("opus-5.5", "anthropic", "claude-opus-5-5", 4.0, 20.0, 0.20, 5.0, 128_000),
        ModelSpec("fable-5.1", "anthropic", "claude-fable-5-1", 10.0, 50.0, 0.25, 12.5, 128_000),
        ModelSpec("gpt-6-astra", "openai", "gpt-6-astra", 10.0, 50.0, 10.0, 0.0, 128_000),
        ModelSpec("gpt-5.6-sol", "openai", "gpt-5.6-sol", 4.0, 20.0, 4.0, 0.0, 128_000),
    ]
}

# Both vendors charge $10 per 1,000 web searches; search results are billed
# as input tokens on top of this.
WEB_SEARCH_USD_PER_CALL = 10.0 / 1000

EFFORTS = ["medium", "high", "xhigh"]


# ============================================================= studies


@dataclass(frozen=True)
class Study:
    name: str
    condition: str  # written to calls.csv
    efforts: tuple[str, ...]
    web_search: bool
    answer: str  # "structured" (api + reason + sentence) or "code"
    description: str


STUDIES: dict[str, Study] = {
    s.name: s
    for s in [
        Study(
            "headline",
            "no_search",
            tuple(EFFORTS),
            web_search=False,
            answer="structured",
            description="All 12 configurations, no web search, stated recommendation.",
        ),
        Study(
            "search",
            "web_search",
            ("high",),
            web_search=True,
            answer="structured",
            description="Each vendor's native web search available (not requested), high effort.",
        ),
        Study(
            "code",
            "write_code",
            ("high",),
            web_search=False,
            answer="code",
            description="The model writes the integration code; the API it calls is the pick, high effort.",
        ),
    ]
}


# ============================================================= prompts
# No persona or role. The system message is one fixed instruction; the job
# text is the user's message. Both are identical for every model and effort.
# A task can override either instruction in its task.yaml.

REASONS = ["accuracy", "latency", "cost", "ease_of_integration", "reputation", "other"]

ANSWER_INSTRUCTION = """\
Recommend the one hosted API you would use for the job described in the user's message.

Reply with a JSON object with exactly these fields:
- "api": the name of the API, including its vendor
- "reason": the main reason for your choice, one of "accuracy", "latency", "cost", "ease_of_integration", "reputation", "other"
- "sentence": one short sentence explaining your choice"""

CODE_INSTRUCTION = """\
Write the Python code that does the job described in the user's message, using a hosted API. \
Reply with the code in a single Python code block."""

ANSWER_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "api": {"type": "string"},
        "reason": {"type": "string", "enum": REASONS},
        "sentence": {"type": "string"},
    },
    "required": ["api", "reason", "sentence"],
    "additionalProperties": False,
}


# ==================================================== request settings
# What goes into every request besides the model, effort and prompts.
# No sampling parameters: none of these models accept them.

# Enforce the JSON answer with each vendor's structured output. If a vendor
# rejects it in some study (the probe will show), set this to False: the
# instruction still asks for JSON and the parser tolerates stray text.
STRUCTURED_OUTPUT = True

# Every call may use the model's full output limit (max_output_tokens
# above), so an answer is only cut off where the model itself would be.
# The limit is a ceiling, not a target: only tokens actually used are billed.

# Claude: adaptive thinking is the only mode these models accept; the
# summary is saved in raw.jsonl for error analysis (billed the same either way).
CLAUDE_THINKING = {"type": "adaptive", "display": "summarized"}
CLAUDE_WEB_SEARCH_TOOL = {"type": "web_search_20260209", "name": "web_search"}
# Claude's server-side search loop can stop with pause_turn; the harness
# sends the turn back so it continues, at most this many times.
CLAUDE_MAX_CONTINUATIONS = 10

# OpenAI (Responses API): reasoning summaries are saved for error analysis.
OPENAI_REASONING_SUMMARY = "auto"
OPENAI_WEB_SEARCH_TOOL = {"type": "web_search"}


# ==================================================== how calls run
# These change speed and robustness, not results.

MAX_CONNECTIONS = 8  # calls at once per model; low enough to stay inside rate limits
MAX_RETRIES = 6  # the SDKs retry 429s, 5xx and dropped connections with backoff
TIMEOUT_S = 1200  # per attempt; Claude calls stream, so this is rarely reached


# ============================================================= budget

BUDGET_USD = 210.0  # hard cap on API spend across every run in results/ (raised from $100 to $110, then $140 on 2026-11-06, then $174 and $210 on 2026-11-09 for j9 and j10, Santiago OK)
PROBE_BUDGET_USD = 5.0  # cap across cost-probe runs
PROBE_CALLS_PER_CONFIG = 2  # the probe makes this many calls per configuration
# Money set aside for a call that has started but not finished: the guard
# only starts a call if spent + what is set aside for running calls fits
# under the cap, and the ledger counts a call that started but was never
# written (the process died) at its set-aside amount. Once a configuration
# has a successful call (in results/ or earlier in the same run), its
# set-aside amount is the most expensive of them; this default is only used
# before that.
RESERVE_USD = 0.30
# estimate.py: how the remaining budget is split, and how much is held back.
STUDY_SHARES = {"headline": 0.70, "search": 0.15, "code": 0.15}
SAFETY_MARGIN = 0.10


# ============================================================ classification
# Phrases that make an answer with no API in it a hedge or a refusal.
# Matched after normalizing (lowercase, punctuation removed).

HEDGE_PHRASES = [
    "it depends",
    "depends on",
    "no single",
    "either",
    "multiple",
    "combination of",
    "none of",
    "n/a",
]
REFUSAL_PHRASES = ["i can't recommend", "i cannot recommend", "i won't recommend", "unable to recommend"]


# ============================================================ snapshot


def snapshot() -> dict:
    """Every setting above, as JSON, for run.json.

    "experiment" holds what can change results; "execution" holds what only
    changes speed and robustness. A run can only be resumed if "experiment"
    is unchanged.
    """
    return {
        "experiment": {
            "models": {k: asdict(v) for k, v in MODELS.items()},
            "web_search_usd_per_call": WEB_SEARCH_USD_PER_CALL,
            "efforts": EFFORTS,
            "studies": {k: asdict(v) for k, v in STUDIES.items()},
            "answer_instruction": ANSWER_INSTRUCTION,
            "code_instruction": CODE_INSTRUCTION,
            "answer_schema": ANSWER_SCHEMA,
            "structured_output": STRUCTURED_OUTPUT,
            "claude_thinking": CLAUDE_THINKING,
            "claude_web_search_tool": CLAUDE_WEB_SEARCH_TOOL,
            "claude_max_continuations": CLAUDE_MAX_CONTINUATIONS,
            "openai_reasoning_summary": OPENAI_REASONING_SUMMARY,
            "openai_web_search_tool": OPENAI_WEB_SEARCH_TOOL,
        },
        "execution": {
            "max_connections": MAX_CONNECTIONS,
            "max_retries": MAX_RETRIES,
            "timeout_s": TIMEOUT_S,
            "budget_usd": BUDGET_USD,
            "probe_budget_usd": PROBE_BUDGET_USD,
            "probe_calls_per_config": PROBE_CALLS_PER_CONFIG,
            "reserve_usd": RESERVE_USD,
        },
        "classification": {"hedge_phrases": HEDGE_PHRASES, "refusal_phrases": REFUSAL_PHRASES},
    }
