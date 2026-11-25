import pytest

from free_claude_code.core.chat_completions.messages_stream import ChatMessagesStream
from free_claude_code.core.chat_completions.request import (
    ChatRequest,
    chat_messages_body,
)
from free_claude_code.core.history_replay import (
    ReplayOrigin,
    ReplayRecord,
    encode_replay,
)
from free_claude_code.core.json_types import JsonObject, json_text, load_json_value
from tests.wire_payload import array_payload, object_array, object_payload

_ORIGIN = ReplayOrigin("provider", "messages", "endpoint", "key", "model")


def _chunks(events: list[JsonObject]) -> list[JsonObject]:
    stream = ChatMessagesStream(
        model="public",
        origin=_ORIGIN,
        reserved_tool_ids=frozenset(),
        include_usage=False,
        include_thoughts=True,
    )
    frames = "".join(
        frame for event in events for frame in stream.feed(str(event["type"]), event)
    )
    return [
        object_payload(load_json_value(line.removeprefix("data: ")))
        for line in frames.splitlines()
        if line.startswith("data: {")
    ]


_SIGNATURE: JsonObject = {
    "type": "content_block_delta",
    "index": 0,
    "delta": {"type": "signature_delta", "signature": "signed"},
}


@pytest.mark.parametrize(
    ("reasoning", "streamed", "replays"),
    [
        ([{"type": "thinking", "thinking": "Find 17."}], "Find 17.", 0),
        (
            [{"type": "thinking", "thinking": "Find 17.", "extra_content": {"a": 1}}],
            "Find 17.",
            1,
        ),
        ([{"type": "thinking", "thinking": "Find 17."}, _SIGNATURE], "Find 17.", 1),
        (
            [
                {
                    "type": "redacted_thinking",
                    "data": encode_replay(
                        ReplayRecord(
                            ReplayOrigin("provider", "chat", "endpoint", "key", "m"),
                            {"reasoning_content": "Find 17."},
                        )
                    ),
                }
            ],
            "Find 17.",
            0,
        ),
        (
            [{"type": "thinking", "thinking": "Find 17.", "extension": {"a": 1}}],
            "Find 17.",
            0,
        ),
    ],
    ids=[
        "unsigned",
        "unsigned-extra",
        "signed",
        "redacted-readable-carrier",
        "unsigned-unknown-field",
    ],
)
def test_completed_reasoning_replays_only_messages_replay_state(
    reasoning: list[JsonObject], streamed: str, replays: int
) -> None:
    block, *deltas = reasoning
    chunks = _chunks(
        [
            {
                "type": "message_start",
                "message": {"id": "msg", "role": "assistant", "content": []},
            },
            {"type": "content_block_start", "index": 0, "content_block": block},
            *deltas,
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
    )

    choices = [
        object_payload(choice)
        for chunk in chunks
        for choice in array_payload(chunk.get("choices", []))
    ]
    deltas_out = [object_payload(choice.get("delta", {})) for choice in choices]
    details = [
        object_payload(detail)
        for delta in deltas_out
        for detail in array_payload(delta.get("reasoning_details", []))
    ]
    assert "".join(str(d.get("text", "")) for d in details) == streamed
    assert "".join(str(d.get("content") or "") for d in deltas_out) == "17"
    assert sum(d["type"] == "reasoning.encrypted" for d in details) == replays
    assert choices[-1]["finish_reason"] == "stop"


_FIRST: JsonObject = {
    "type": "content_block_start",
    "index": 0,
    "content_block": {"type": "thinking", "thinking": "First."},
}
_SECOND: list[JsonObject] = [
    {
        "type": "content_block_start",
        "index": 1,
        "content_block": {"type": "thinking", "thinking": "Second."},
    },
    {"type": "content_block_stop", "index": 1},
]
_FIRST_STOP: JsonObject = {"type": "content_block_stop", "index": 0}


@pytest.mark.parametrize(
    "reasoning",
    [
        [_FIRST, _SIGNATURE, _FIRST_STOP, *_SECOND],
        [_FIRST, *_SECOND, _SIGNATURE, _FIRST_STOP],
    ],
    ids=["sequential", "overlapping"],
)
def test_echoed_reasoning_keeps_upstream_block_order(
    reasoning: list[JsonObject],
) -> None:
    chunks = _chunks(
        [
            {
                "type": "message_start",
                "message": {"id": "msg", "role": "assistant", "content": []},
            },
            *reasoning,
            {
                "type": "content_block_start",
                "index": 2,
                "content_block": {"type": "text", "text": "17"},
            },
            {"type": "content_block_stop", "index": 2},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
            {"type": "message_stop"},
        ]
    )
    details = [
        detail
        for chunk in chunks
        for choice in array_payload(chunk.get("choices", []))
        for detail in array_payload(
            object_payload(object_payload(choice).get("delta", {})).get(
                "reasoning_details", []
            )
        )
    ]
    body = chat_messages_body(
        ChatRequest.model_validate(
            {
                "model": "model",
                "messages": [
                    {"role": "user", "content": "Find it"},
                    {
                        "role": "assistant",
                        "content": "17",
                        "reasoning_details": details,
                    },
                    {"role": "user", "content": "Next"},
                ],
            }
        )
    )
    assistant = object_array(body["messages"])[1]
    thinking = [
        (json_text(block["thinking"]), bool(block.get("signature")))
        for block in object_array(assistant["content"])
        if block["type"] == "thinking"
    ]
    assert thinking == [("First.", True), ("Second.", False)]
