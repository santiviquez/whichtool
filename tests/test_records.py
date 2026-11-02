from whichtool.config import MODELS
from whichtool.records import cost_usd, extract_anthropic, extract_openai, merge_extracts, parse_answer


def test_parse_plain_json():
    a = parse_answer('{"api": "Postmark", "reason": "latency", "sentence": "Fast."}')
    assert a == {"api": "Postmark", "reason": "latency", "sentence": "Fast.", "parse_ok": True}


def test_parse_fenced_json_with_prose():
    text = 'Here you go:\n```json\n{"api": "Amazon SES", "reason": "cost", "sentence": "Cheap."}\n```\nThanks'
    a = parse_answer(text)
    assert a["api"] == "Amazon SES" and a["parse_ok"]


def test_parse_garbage():
    a = parse_answer("I would use Postmark.")
    assert not a["parse_ok"] and a["api"] == ""


ANTHROPIC_RESP = {
    "model": "claude-opus-5-5",
    "stop_reason": "end_turn",
    "content": [
        {"type": "thinking", "thinking": "Consider latency."},
        {"type": "server_tool_use", "name": "web_search", "input": {"query": "fast transactional email"}},
        {
            "type": "web_search_tool_result",
            "content": [{"type": "web_search_result", "url": "https://a.example", "title": "A"}],
        },
        {
            "type": "text",
            "text": '{"api": "Postmark"}',
            "citations": [{"type": "web_search_result_location", "url": "https://b.example"}],
        },
    ],
    "usage": {
        "input_tokens": 1000,
        "output_tokens": 500,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "server_tool_use": {"web_search_requests": 1},
    },
}

OPENAI_RESP = {
    "model": "gpt-5.6-sol-2026-10-01",
    "status": "completed",
    "output": [
        {"type": "reasoning", "summary": [{"type": "summary_text", "text": "Latency matters."}]},
        {
            "type": "web_search_call",
            "action": {"type": "search", "query": "email api latency", "sources": [{"url": "https://c.example"}]},
        },
        {
            "type": "message",
            "content": [
                {
                    "type": "output_text",
                    "text": '{"api": "Postmark"}',
                    "annotations": [{"type": "url_citation", "url": "https://d.example"}],
                }
            ],
        },
    ],
    "usage": {
        "input_tokens": 2000,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 800,
        "output_tokens_details": {"reasoning_tokens": 600},
    },
}


def test_extract_anthropic():
    x = extract_anthropic(ANTHROPIC_RESP)
    assert x["model_returned"] == "claude-opus-5-5"
    assert x["n_searches"] == 1
    assert x["search_queries"] == ["fast transactional email"]
    assert x["urls"] == ["https://a.example", "https://b.example"]
    assert x["reasoning_summary"] == ["Consider latency."]
    assert not x["refused"]


def test_extract_anthropic_refusal():
    x = extract_anthropic({**ANTHROPIC_RESP, "stop_reason": "refusal"})
    assert x["refused"]


def test_extract_openai():
    x = extract_openai(OPENAI_RESP)
    assert x["model_returned"] == "gpt-5.6-sol-2026-10-01"
    assert x["n_searches"] == 1
    assert x["reasoning_tokens"] == 600
    assert x["urls"] == ["https://c.example", "https://d.example"]
    assert x["stop_reason"] == "completed"


def test_cost_includes_searches():
    x = merge_extracts([extract_anthropic(ANTHROPIC_RESP)])
    # Opus 5.5: 1000 * $4/M + 500 * $20/M + 1 search * $0.01
    assert abs(cost_usd(MODELS["opus-5.5"], x) - (0.004 + 0.010 + 0.01)) < 1e-9


def test_cost_openai():
    x = extract_openai(OPENAI_RESP)
    # GPT-6 Astra: 2000 * $10/M + 800 * $50/M + 1 search * $0.01
    assert abs(cost_usd(MODELS["gpt-6-astra"], x) - (0.02 + 0.04 + 0.01)) < 1e-9


def test_merge_sums_usage():
    a = extract_anthropic(ANTHROPIC_RESP)
    m = merge_extracts([a, a])
    assert m["input_tokens"] == 2000 and m["n_searches"] == 2
