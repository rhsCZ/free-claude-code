"""Native Messages reasoning reaches Gemini clients as readable text or replay."""

import pytest

from free_claude_code.core.gemini.messages_stream import GeminiMessagesStream
from free_claude_code.core.gemini.replay import decode_bundle
from free_claude_code.core.history_replay import (
    ReplayOrigin,
    ReplayRecord,
    encode_replay,
)
from free_claude_code.core.json_types import JsonObject, load_json_value
from free_claude_code.core.openai_tool_names import OpenAIToolNameCodec
from tests.wire_payload import object_array, object_payload

_ORIGIN = ReplayOrigin("provider", "messages", "endpoint", "key", "model")
_CARRIER = encode_replay(
    ReplayRecord(
        ReplayOrigin("provider", "chat", "endpoint", "key", "m"),
        {"reasoning_content": "Portable thought"},
    )
)


@pytest.mark.parametrize(
    ("block", "thought", "replayed"),
    [
        ({"type": "thinking", "thinking": "Find 17."}, "Find 17.", False),
        (
            {"type": "thinking", "thinking": "Find 17.", "extra_content": {"a": 1}},
            "Find 17.",
            True,
        ),
        (
            {"type": "thinking", "thinking": "Find 17.", "signature": "sig"},
            "Find 17.",
            True,
        ),
        ({"type": "redacted_thinking", "data": _CARRIER}, "Portable thought", False),
        (
            {"type": "thinking", "thinking": "Find 17.", "extension": {"a": 1}},
            "Find 17.",
            False,
        ),
    ],
    ids=[
        "unsigned",
        "unsigned-native-fields",
        "signed",
        "redacted-readable-carrier",
        "unsigned-unknown-field",
    ],
)
def test_completed_reasoning_replays_only_native_state(
    block: JsonObject, thought: str, replayed: bool
) -> None:
    stream = GeminiMessagesStream(
        response_id="response",
        model="public",
        include_thoughts=True,
        origin=_ORIGIN,
        tool_names=OpenAIToolNameCodec.from_names([]),
        declared_names=frozenset(),
        reserved_tool_ids=frozenset(),
    )
    events: list[JsonObject] = [
        {
            "type": "message_start",
            "message": {"id": "msg", "role": "assistant", "content": []},
        },
        {"type": "content_block_start", "index": 0, "content_block": block},
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "text", "text": "17"},
        },
        {"type": "content_block_stop", "index": 1},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
        {"type": "message_stop"},
    ]
    output = "".join(
        frame for event in events for frame in stream.feed(str(event["type"]), event)
    )

    parts = [
        part
        for line in output.splitlines()
        if line.startswith("data: ")
        for candidate in object_array(
            object_payload(load_json_value(line.removeprefix("data: ")))["candidates"]
        )
        for part in object_array(
            object_payload(candidate.get("content", {})).get("parts", [])
        )
    ]
    assert "".join(str(p["text"]) for p in parts if p.get("thought")) == thought
    assert (
        "".join(str(p.get("text", "")) for p in parts if not p.get("thought")) == "17"
    )
    signatures = [str(p["thoughtSignature"]) for p in parts if "thoughtSignature" in p]
    assert bool(signatures) == replayed
    # The next request must accept every replay this response wrote.
    assert all(decode_bundle(signature) is not None for signature in signatures)
