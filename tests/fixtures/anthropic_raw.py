"""Hand-built stand-ins for Anthropic's raw stream events.

The adapter only ever uses attribute access, so plain namespaces exercise the
normalizer faithfully without the SDK or a network call.
"""

from __future__ import annotations

from types import SimpleNamespace as NS
from typing import Any

ns = NS


def message_start(model: str = "claude-sonnet-5", input_tokens: int = 12) -> Any:
    return ns(
        type="message_start",
        message=ns(
            model=model,
            usage=ns(
                input_tokens=input_tokens,
                output_tokens=0,
                cache_read_input_tokens=3,
                cache_creation_input_tokens=0,
            ),
        ),
    )


def text_delta(text: str, index: int = 0) -> Any:
    return ns(type="content_block_delta", index=index, delta=ns(type="text_delta", text=text))


def thinking_delta(text: str, index: int = 0) -> Any:
    return ns(
        type="content_block_delta", index=index, delta=ns(type="thinking_delta", thinking=text)
    )


def signature_delta(signature: str, index: int = 0) -> Any:
    return ns(
        type="content_block_delta",
        index=index,
        delta=ns(type="signature_delta", signature=signature),
    )


def tool_block_start(index: int, tool_id: str, name: str) -> Any:
    return ns(
        type="content_block_start",
        index=index,
        content_block=ns(type="tool_use", id=tool_id, name=name),
    )


def input_json_delta(index: int, partial: str) -> Any:
    return ns(
        type="content_block_delta",
        index=index,
        delta=ns(type="input_json_delta", partial_json=partial),
    )


def block_stop(index: int = 0) -> Any:
    return ns(type="content_block_stop", index=index)


def message_delta(stop_reason: str = "end_turn", output_tokens: int = 7) -> Any:
    return ns(
        type="message_delta",
        delta=ns(stop_reason=stop_reason),
        usage=ns(output_tokens=output_tokens),
    )


def message_stop() -> Any:
    return ns(type="message_stop")


TEXT_TURN = [
    message_start(),
    ns(type="content_block_start", index=0, content_block=ns(type="text")),
    text_delta("Hello"),
    text_delta(" world"),
    block_stop(),
    message_delta(),
    message_stop(),
]

REASONING_TURN = [
    message_start(),
    thinking_delta("let me think"),
    signature_delta("sig-abc"),
    block_stop(),
    text_delta("answer", index=1),
    block_stop(1),
    message_delta(),
    message_stop(),
]

TOOL_TURN = [
    message_start(),
    tool_block_start(0, "toolu_1", "run_command"),
    input_json_delta(0, '{"cmd":'),
    input_json_delta(0, ' "kubectl get pods"}'),
    block_stop(0),
    message_delta(stop_reason="tool_use"),
    message_stop(),
]

MAX_TOKENS_TURN = [
    message_start(),
    text_delta("truncat"),
    block_stop(),
    message_delta(stop_reason="max_tokens"),
    message_stop(),
]
