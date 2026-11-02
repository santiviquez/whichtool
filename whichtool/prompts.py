"""The fixed instructions every model sees.

No persona or role. The system message is one fixed instruction about the
answer; the job text is the user's message. Both are identical for all
models and effort levels. A task can override either instruction in its
task.yaml, but the default is meant to work for any "pick a hosted API" task.
"""

from __future__ import annotations

REASONS = [
    "accuracy",
    "latency",
    "cost",
    "ease_of_integration",
    "reputation",
    "other",
]

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
