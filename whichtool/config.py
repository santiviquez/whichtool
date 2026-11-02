"""Models, effort levels, studies, prices and the budget.

Everything a run depends on that is not task-specific lives here, so the
whole setup can be read in one place and is written into every results row.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TASKS_DIR = ROOT / "tasks"
RESULTS_DIR = ROOT / "results"

# Hard cap on API spend across every run in results/, in USD.
BUDGET_USD = 100.0
# Cap for the cost probe (2 calls per configuration).
PROBE_BUDGET_USD = 5.0

# Both vendors charge $10 per 1,000 web searches; search results are billed
# as input tokens on top of this.
WEB_SEARCH_USD_PER_CALL = 10.0 / 1000


@dataclass(frozen=True)
class ModelSpec:
    label: str  # short name used in results and reports
    vendor: str  # "anthropic" | "openai"
    api_id: str  # model id sent to the vendor API
    usd_in: float  # per 1M input tokens
    usd_out: float  # per 1M output tokens (reasoning tokens are billed as output)
    usd_cache_read: float  # per 1M cached input tokens
    usd_cache_write: float  # per 1M cache-write input tokens (Anthropic only)

    @property
    def inspect_id(self) -> str:
        return f"{self.vendor}/{self.api_id}"


# Prices checked 2026-11-02 (see the design doc's "Provider settings" table).
# GPT-5.6 Sol's $4 / $20 is promotional through at least 2026-11-21.
# OpenAI cached-input prices for these two models are not used by our
# prompts (they are far below the caching minimum), so the full input price
# is used as a conservative stand-in.
MODELS: dict[str, ModelSpec] = {
    m.label: m
    for m in [
        ModelSpec("opus-5.5", "anthropic", "claude-opus-5-5", 4.0, 20.0, 0.20, 5.0),
        ModelSpec("fable-5.1", "anthropic", "claude-fable-5-1", 10.0, 50.0, 0.25, 12.5),
        ModelSpec("gpt-6-astra", "openai", "gpt-6-astra", 10.0, 50.0, 10.0, 0.0),
        ModelSpec("gpt-5.6-sol", "openai", "gpt-5.6-sol", 4.0, 20.0, 4.0, 0.0),
    ]
}

EFFORTS = ["medium", "high", "xhigh"]


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

# Generous output ceilings so long reasoning at xhigh is never cut off.
# max_tokens is a ceiling, not a target: unused tokens are not billed.
MAX_TOKENS = {"medium": 32_000, "high": 64_000, "xhigh": 64_000}

# Per-model concurrency. Low enough to stay well inside rate limits, high
# enough that a few hundred calls finish in minutes.
MAX_CONNECTIONS = 8
