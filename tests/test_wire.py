"""Our requests through the real SDKs to a fake HTTP server: checks what goes on the wire."""

import asyncio
import json

import anthropic
import httpx2
import openai

from whichtool.config import MODELS, STUDIES
from whichtool.harness import build_request, send

ANSWER = '{"api": "Postmark", "reason": "latency", "sentence": "Fast."}'


def _sse(events):
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)


def _claude_stream(stop_reason, block):
    start = {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
             "content": [], "stop_reason": None, "stop_sequence": None,
             "usage": {"input_tokens": 10, "output_tokens": 1}}
    if block["type"] == "text":
        blocks = [{"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                  {"type": "content_block_delta", "index": 0,
                   "delta": {"type": "text_delta", "text": block["text"]}}]
    else:
        blocks = [{"type": "content_block_start", "index": 0, "content_block": block}]
    return _sse([{"type": "message_start", "message": start}, *blocks,
                 {"type": "content_block_stop", "index": 0},
                 {"type": "message_delta", "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                  "usage": {"output_tokens": 5}},
                 {"type": "message_stop"}])


def test_claude_wire_format_and_pause_turn():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        if len(bodies) == 1:  # first call pauses mid-search
            text = _claude_stream("pause_turn", {"type": "server_tool_use", "id": "srvtoolu_1",
                                                 "name": "web_search", "input": {"query": "q"}})
        else:
            text = _claude_stream("end_turn", {"type": "text", "text": ANSWER})
        return httpx2.Response(200, text=text, headers={"content-type": "text/event-stream"})

    client = anthropic.AsyncAnthropic(
        api_key="test", http_client=anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler)))
    req = build_request(MODELS["opus-5.5"], STUDIES["search"], "xhigh", "instr", "job")
    responses, text = asyncio.run(send(client, "anthropic", req))

    assert len(responses) == 2 and text == ANSWER
    first = bodies[0]
    assert first["stream"] is True and first["max_tokens"] == 128_000
    assert first["output_config"]["effort"] == "xhigh"
    assert first["output_config"]["format"]["type"] == "json_schema"  # not the deprecated output_format
    assert "output_format" not in first and "temperature" not in first
    assert first["tools"] == [{"type": "web_search_20260209", "name": "web_search"}]
    # the continuation sends the paused turn back and nothing else
    assert [m["role"] for m in bodies[1]["messages"]] == ["user", "assistant"]
    assert bodies[1]["messages"][1]["content"][0]["type"] == "server_tool_use"


def test_openai_wire_format():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        response = {
            "id": "resp_1", "object": "response", "created_at": 0, "model": "gpt-6-astra",
            "status": "completed", "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
            "output": [{"type": "message", "id": "m", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": ANSWER, "annotations": []}]}],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2,
                      "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                      "output_tokens_details": {"reasoning_tokens": 0}}}
        events = [{"type": "response.created", "sequence_number": 0, "response": {**response, "status": "in_progress", "output": []}},
                  {"type": "response.completed", "sequence_number": 1, "response": response}]
        return httpx2.Response(200, text=_sse(events), headers={"content-type": "text/event-stream"})

    client = openai.AsyncOpenAI(
        api_key="test", http_client=openai.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler)))
    req = build_request(MODELS["gpt-5.6-sol"], STUDIES["headline"], "medium", "instr", "job")
    responses, text = asyncio.run(send(client, "openai", req))

    assert text == ANSWER and len(responses) == 1
    body = bodies[0]
    assert body["store"] is False and body["reasoning"]["effort"] == "medium" and body["stream"] is True
    assert body["max_output_tokens"] == 128_000
    assert body["text"]["format"]["type"] == "json_schema"
    assert "tools" not in body and "previous_response_id" not in body and "temperature" not in body
