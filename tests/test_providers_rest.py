"""Normalizer tests for the non-Anthropic adapters. No network, no SDK objects."""

from __future__ import annotations

from types import SimpleNamespace as ns
from typing import Any

import pytest

from wai.config.models import ProviderSettings
from wai.core.errors import ConfigError
from wai.core.events import (
    MessageEnd,
    ReasoningDelta,
    TextDelta,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
    UsageUpdate,
)
from wai.core.types import (
    ChatRequest,
    ImageBlock,
    Message,
    ReasoningBlock,
    Role,
    StopReason,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from wai.providers.azure_foundry import AzureFoundryProvider
from wai.providers.bedrock import BedrockProvider, _to_converse_messages
from wai.providers.deepseek import DeepSeekProvider
from wai.providers.gemini import GeminiProvider, _to_contents
from wai.providers.mistral import MistralProvider, _to_mistral_messages
from wai.providers.openai import OpenAIProvider, _to_openai_messages
from wai.providers.openrouter import OpenRouterProvider


class _AsyncList:
    def __init__(self, items: list[Any]) -> None:
        self.items = items

    def __aiter__(self) -> Any:
        async def gen() -> Any:
            for item in self.items:
                yield item

        return gen()


def _request(**kwargs: Any) -> ChatRequest:
    defaults: dict[str, Any] = {
        "model": "m",
        "messages": [Message.user("hi")],
        "max_tokens": 256,
    }
    return ChatRequest(**{**defaults, **kwargs})


async def _collect(provider: Any, request: ChatRequest | None = None) -> list[Any]:
    return [e async for e in provider.stream(request or _request())]


# ------------------------------------------------------------------ OpenAI family


def _oa_chunk(
    content: str | None = None,
    *,
    finish: str | None = None,
    reasoning: str | None = None,
    reasoning_field: str = "reasoning_content",
    tool_calls: list[Any] | None = None,
    usage: Any = None,
) -> Any:
    delta_kwargs: dict[str, Any] = {"content": content, "tool_calls": tool_calls}
    if reasoning is not None:
        delta_kwargs[reasoning_field] = reasoning
    return ns(
        model="m",
        usage=usage,
        choices=[ns(delta=ns(**delta_kwargs), finish_reason=finish)],
    )


def _oa_tool_call(index: int, *, call_id: str = "", name: str = "", args: str = "") -> Any:
    return ns(index=index, id=call_id, function=ns(name=name, arguments=args))


def _wire_openai(provider: OpenAIProvider, chunks: list[Any]) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    async def create(**body: Any) -> Any:
        captured.update(body)
        return _AsyncList(chunks)

    provider._get_client = lambda: ns(chat=ns(completions=ns(create=create)))  # type: ignore[method-assign]
    return captured


@pytest.fixture
def openai_provider() -> OpenAIProvider:
    return OpenAIProvider(api_key="k", settings=ProviderSettings())


async def test_openai_text_and_usage(openai_provider: OpenAIProvider) -> None:
    usage = ns(prompt_tokens=9, completion_tokens=4, prompt_tokens_details=ns(cached_tokens=2))
    _wire_openai(
        openai_provider,
        [_oa_chunk("Hel"), _oa_chunk("lo"), _oa_chunk(finish="stop"), _oa_chunk(usage=usage)],
    )
    events = await _collect(openai_provider)
    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "Hello"
    end = next(e for e in events if isinstance(e, MessageEnd))
    assert end.stop_reason is StopReason.END_TURN
    assert end.usage.input_tokens == 9
    assert end.usage.cache_read_tokens == 2


async def test_openai_length_finish_maps_to_max_tokens(openai_provider: OpenAIProvider) -> None:
    _wire_openai(openai_provider, [_oa_chunk("x"), _oa_chunk(finish="length")])
    events = await _collect(openai_provider)
    assert next(e for e in events if isinstance(e, MessageEnd)).stop_reason is StopReason.MAX_TOKENS


async def test_openai_tool_call_reassembly(openai_provider: OpenAIProvider) -> None:
    _wire_openai(
        openai_provider,
        [
            _oa_chunk(tool_calls=[_oa_tool_call(0, call_id="c1", name="deploy", args='{"env"')]),
            _oa_chunk(tool_calls=[_oa_tool_call(0, args=':"prod"}')]),
            _oa_chunk(finish="tool_calls"),
        ],
    )
    events = await _collect(openai_provider)
    start = next(e for e in events if isinstance(e, ToolCallStart))
    assert (start.id, start.name) == ("c1", "deploy")
    assert len([e for e in events if isinstance(e, ToolCallDelta)]) == 2
    end = next(e for e in events if isinstance(e, ToolCallEnd))
    assert end.input == {"env": "prod"}
    assert next(e for e in events if isinstance(e, MessageEnd)).stop_reason is StopReason.TOOL_USE


async def test_openai_tool_call_start_emitted_once(openai_provider: OpenAIProvider) -> None:
    _wire_openai(
        openai_provider,
        [
            _oa_chunk(tool_calls=[_oa_tool_call(0, call_id="c1", name="n", args="{")]),
            _oa_chunk(tool_calls=[_oa_tool_call(0, call_id="c1", name="n", args="}")]),
            _oa_chunk(finish="tool_calls"),
        ],
    )
    events = await _collect(openai_provider)
    assert len([e for e in events if isinstance(e, ToolCallStart)]) == 1


async def test_openai_system_is_an_inline_message(openai_provider: OpenAIProvider) -> None:
    body = _wire_openai(openai_provider, [_oa_chunk(finish="stop")])
    await _collect(openai_provider, _request(system="be terse"))
    assert body["messages"][0] == {"role": "system", "content": "be terse"}
    assert body["max_completion_tokens"] == 256
    assert body["stream_options"] == {"include_usage": True}


async def test_openai_empty_choices_chunk_is_tolerated(openai_provider: OpenAIProvider) -> None:
    """The usage-only final chunk has no choices."""
    usage = ns(prompt_tokens=1, completion_tokens=1, prompt_tokens_details=None)
    _wire_openai(openai_provider, [ns(model="m", usage=usage, choices=[])])
    events = await _collect(openai_provider)
    assert any(isinstance(e, UsageUpdate) for e in events)


async def test_deepseek_reasoning_content_is_mapped() -> None:
    provider = DeepSeekProvider(api_key="k", settings=ProviderSettings())
    _wire_openai(provider, [_oa_chunk(reasoning="pondering"), _oa_chunk(finish="stop")])
    events = await _collect(provider)
    assert next(e for e in events if isinstance(e, ReasoningDelta)).text == "pondering"


async def test_openrouter_reasoning_field_is_mapped() -> None:
    provider = OpenRouterProvider(api_key="k", settings=ProviderSettings())
    _wire_openai(
        provider,
        [_oa_chunk(reasoning="hmm", reasoning_field="reasoning"), _oa_chunk(finish="stop")],
    )
    events = await _collect(provider)
    assert next(e for e in events if isinstance(e, ReasoningDelta)).text == "hmm"


def test_compatible_endpoints_use_max_tokens_not_max_completion_tokens() -> None:
    assert DeepSeekProvider.max_tokens_field == "max_tokens"
    assert OpenRouterProvider.max_tokens_field == "max_tokens"
    assert OpenAIProvider.max_tokens_field == "max_completion_tokens"


def test_openrouter_sends_attribution_headers() -> None:
    provider = OpenRouterProvider(api_key="k", settings=ProviderSettings())
    assert "X-Title" in provider.default_headers()


def test_azure_requires_an_endpoint() -> None:
    provider = AzureFoundryProvider(api_key="k", settings=ProviderSettings())
    with pytest.raises(ConfigError, match="base_url"):
        provider._build_client()


def test_openai_message_translation() -> None:
    out = _to_openai_messages(
        Message(
            role=Role.ASSISTANT,
            content=[
                ReasoningBlock(text="dropped"),
                TextBlock(text="body"),
                ToolUseBlock(id="c1", name="n", input={"a": 1}),
            ],
        )
    )
    assert out[0]["content"] == "body"
    assert out[0]["tool_calls"][0]["function"]["arguments"] == '{"a": 1}'


def test_openai_tool_result_becomes_its_own_message() -> None:
    out = _to_openai_messages(
        Message(role=Role.USER, content=[ToolResultBlock(tool_use_id="c1", content="ok")])
    )
    assert out == [{"role": "tool", "tool_call_id": "c1", "content": "ok"}]


def test_openai_image_becomes_a_data_url() -> None:
    out = _to_openai_messages(
        Message(role=Role.USER, content=[ImageBlock(media_type="image/png", data="Zm9v")])
    )
    assert out[0]["content"][0]["image_url"]["url"] == "data:image/png;base64,Zm9v"


# ------------------------------------------------------------------------ Gemini


def _gemini_chunk(parts: list[Any], *, finish: Any = None, usage: Any = None) -> Any:
    return ns(
        usage_metadata=usage,
        candidates=[ns(finish_reason=finish, content=ns(parts=parts))],
    )


def _part(text: str | None = None, *, thought: bool = False, call: Any = None) -> Any:
    return ns(text=text, thought=thought, function_call=call)


def _wire_gemini(provider: GeminiProvider, chunks: list[Any]) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    async def generate(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return _AsyncList(chunks)

    provider._get_client = lambda: ns(aio=ns(models=ns(generate_content_stream=generate)))  # type: ignore[method-assign]
    return captured


@pytest.fixture
def gemini_provider() -> GeminiProvider:
    return GeminiProvider(api_key="k", settings=ProviderSettings())


async def test_gemini_text_reasoning_and_usage(gemini_provider: GeminiProvider) -> None:
    usage = ns(prompt_token_count=5, candidates_token_count=2, cached_content_token_count=1)
    _wire_gemini(
        gemini_provider,
        [
            _gemini_chunk([_part("think", thought=True), _part("Hi")]),
            _gemini_chunk([_part(" there")], finish="FinishReason.STOP", usage=usage),
        ],
    )
    events = await _collect(gemini_provider)
    assert next(e for e in events if isinstance(e, ReasoningDelta)).text == "think"
    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "Hi there"
    end = next(e for e in events if isinstance(e, MessageEnd))
    assert end.stop_reason is StopReason.END_TURN
    assert end.usage.input_tokens == 5


async def test_gemini_safety_finish_maps_to_content_filter(gemini_provider: GeminiProvider) -> None:
    _wire_gemini(gemini_provider, [_gemini_chunk([], finish="SAFETY")])
    events = await _collect(gemini_provider)
    assert (
        next(e for e in events if isinstance(e, MessageEnd)).stop_reason
        is StopReason.CONTENT_FILTER
    )


async def test_gemini_function_call(gemini_provider: GeminiProvider) -> None:
    call = ns(id="fc1", name="scale", args={"replicas": 3})
    _wire_gemini(gemini_provider, [_gemini_chunk([_part(call=call)], finish="STOP")])
    events = await _collect(gemini_provider)
    assert next(e for e in events if isinstance(e, ToolCallStart)).name == "scale"
    assert next(e for e in events if isinstance(e, ToolCallEnd)).input == {"replicas": 3}
    assert next(e for e in events if isinstance(e, MessageEnd)).stop_reason is StopReason.TOOL_USE


async def test_gemini_system_goes_into_config(gemini_provider: GeminiProvider) -> None:
    captured = _wire_gemini(gemini_provider, [_gemini_chunk([], finish="STOP")])
    await _collect(gemini_provider, _request(system="be terse"))
    assert captured["config"]["system_instruction"] == "be terse"
    assert captured["config"]["max_output_tokens"] == 256


def test_gemini_uses_model_role_and_drops_system() -> None:
    contents = _to_contents(
        [
            Message(role=Role.SYSTEM, content=[TextBlock(text="ignored")]),
            Message.user("hi"),
            Message.assistant("hello"),
        ]
    )
    assert [c["role"] for c in contents] == ["user", "model"]


def test_gemini_correlates_tool_results_by_name() -> None:
    """Gemini has no tool-call ids, so the id must be resolved back to a name."""
    contents = _to_contents(
        [
            Message(role=Role.ASSISTANT, content=[ToolUseBlock(id="u1", name="scale", input={})]),
            Message(role=Role.USER, content=[ToolResultBlock(tool_use_id="u1", content="done")]),
        ]
    )
    assert contents[1]["parts"][0]["function_response"]["name"] == "scale"


# ----------------------------------------------------------------------- Mistral


def _mistral_event(
    content: Any = None, *, finish: str | None = None, usage: Any = None, tool_calls: Any = None
) -> Any:
    return ns(
        data=ns(
            usage=usage,
            choices=[ns(delta=ns(content=content, tool_calls=tool_calls), finish_reason=finish)],
        )
    )


def _wire_mistral(provider: MistralProvider, events: list[Any]) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    async def stream_async(**body: Any) -> Any:
        captured.update(body)
        return _AsyncList(events)

    provider._get_client = lambda: ns(chat=ns(stream_async=stream_async))  # type: ignore[method-assign]
    return captured


@pytest.fixture
def mistral_provider() -> MistralProvider:
    return MistralProvider(api_key="k", settings=ProviderSettings())


async def test_mistral_text_and_usage(mistral_provider: MistralProvider) -> None:
    usage = ns(prompt_tokens=3, completion_tokens=2)
    _wire_mistral(
        mistral_provider,
        [_mistral_event("Bon"), _mistral_event("jour", finish="stop", usage=usage)],
    )
    events = await _collect(mistral_provider)
    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "Bonjour"
    assert next(e for e in events if isinstance(e, MessageEnd)).usage.output_tokens == 2


async def test_mistral_chunked_content_list(mistral_provider: MistralProvider) -> None:
    """Newer models return content as a list of typed chunks, not a string."""
    _wire_mistral(
        mistral_provider,
        [_mistral_event([ns(type="text", text="a"), ns(type="text", text="b")], finish="stop")],
    )
    events = await _collect(mistral_provider)
    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "ab"


async def test_mistral_tool_call(mistral_provider: MistralProvider) -> None:
    call = ns(index=0, id="c1", function=ns(name="rollout", arguments='{"n":1}'))
    _wire_mistral(mistral_provider, [_mistral_event(tool_calls=[call], finish="tool_calls")])
    events = await _collect(mistral_provider)
    assert next(e for e in events if isinstance(e, ToolCallStart)).name == "rollout"
    assert next(e for e in events if isinstance(e, ToolCallEnd)).input == {"n": 1}


def test_mistral_system_is_inline() -> None:
    messages = _to_mistral_messages(_request(system="be terse"))
    assert messages[0] == {"role": "system", "content": "be terse"}


# ----------------------------------------------------------------------- Bedrock


def _wire_bedrock(provider: BedrockProvider, events: list[dict[str, Any]]) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def converse_stream(**body: Any) -> dict[str, Any]:
        captured.update(body)
        return {"stream": iter(events)}

    provider._runtime = ns(converse_stream=converse_stream)
    return captured


@pytest.fixture
def bedrock_provider() -> BedrockProvider:
    return BedrockProvider(api_key=None, settings=ProviderSettings(region="us-east-1"))


async def test_bedrock_text_reasoning_and_usage(bedrock_provider: BedrockProvider) -> None:
    _wire_bedrock(
        bedrock_provider,
        [
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "Hel"}}},
            {
                "contentBlockDelta": {
                    "contentBlockIndex": 0,
                    "delta": {"reasoningContent": {"text": "why"}},
                }
            },
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "lo"}}},
            {"messageStop": {"stopReason": "end_turn"}},
            {"metadata": {"usage": {"inputTokens": 8, "outputTokens": 3}}},
        ],
    )
    events = await _collect(bedrock_provider)
    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "Hello"
    assert next(e for e in events if isinstance(e, ReasoningDelta)).text == "why"
    end = next(e for e in events if isinstance(e, MessageEnd))
    assert end.usage.input_tokens == 8
    assert end.stop_reason is StopReason.END_TURN


async def test_bedrock_tool_use(bedrock_provider: BedrockProvider) -> None:
    _wire_bedrock(
        bedrock_provider,
        [
            {
                "contentBlockStart": {
                    "contentBlockIndex": 1,
                    "start": {"toolUse": {"toolUseId": "t1", "name": "terraform_apply"}},
                }
            },
            {
                "contentBlockDelta": {
                    "contentBlockIndex": 1,
                    "delta": {"toolUse": {"input": '{"dir":'}},
                }
            },
            {
                "contentBlockDelta": {
                    "contentBlockIndex": 1,
                    "delta": {"toolUse": {"input": '"./infra"}'}},
                }
            },
            {"contentBlockStop": {"contentBlockIndex": 1}},
            {"messageStop": {"stopReason": "tool_use"}},
        ],
    )
    events = await _collect(bedrock_provider)
    assert next(e for e in events if isinstance(e, ToolCallStart)).name == "terraform_apply"
    assert next(e for e in events if isinstance(e, ToolCallEnd)).input == {"dir": "./infra"}
    assert next(e for e in events if isinstance(e, MessageEnd)).stop_reason is StopReason.TOOL_USE


async def test_bedrock_mid_stream_exception_is_raised(bedrock_provider: BedrockProvider) -> None:
    from wai.core.errors import RateLimitError

    _wire_bedrock(bedrock_provider, [{"throttlingException": {"message": "slow down"}}])
    with pytest.raises(RateLimitError):
        await _collect(bedrock_provider)


async def test_bedrock_request_shape(bedrock_provider: BedrockProvider) -> None:
    body = _wire_bedrock(bedrock_provider, [{"messageStop": {"stopReason": "end_turn"}}])
    await _collect(bedrock_provider, _request(system="be terse", temperature=0.3))
    assert body["system"] == [{"text": "be terse"}]
    assert body["inferenceConfig"]["maxTokens"] == 256
    assert body["inferenceConfig"]["temperature"] == 0.3


def test_bedrock_message_translation() -> None:
    out = _to_converse_messages(
        [
            Message(
                role=Role.ASSISTANT,
                content=[
                    ReasoningBlock(text="t", signature="s"),
                    TextBlock(text="body"),
                    ToolUseBlock(id="t1", name="n", input={"a": 1}),
                ],
            ),
            Message(
                role=Role.USER,
                content=[ToolResultBlock(tool_use_id="t1", content="bad", is_error=True)],
            ),
        ]
    )
    assert out[0]["content"][0]["reasoningContent"]["reasoningText"]["signature"] == "s"
    assert out[0]["content"][2]["toolUse"]["toolUseId"] == "t1"
    assert out[1]["content"][0]["toolResult"]["status"] == "error"


def test_bedrock_image_is_sent_as_raw_bytes() -> None:
    out = _to_converse_messages(
        [Message(role=Role.USER, content=[ImageBlock(media_type="image/png", data="Zm9v")])]
    )
    assert out[0]["content"][0]["image"] == {"format": "png", "source": {"bytes": b"foo"}}


def test_bedrock_declares_no_api_key_auth() -> None:
    from wai.config.secrets import USES_CREDENTIAL_CHAIN

    assert BedrockProvider.name in USES_CREDENTIAL_CHAIN


async def test_bedrock_stream_iteration_does_not_block_the_loop() -> None:
    """Regression guard: the boto3 iterator must be consumed off the loop."""
    import asyncio
    import threading

    main_thread = threading.get_ident()
    seen: list[int] = []

    class BlockingStream:
        def __init__(self) -> None:
            self.remaining = 3

        def __iter__(self) -> Any:
            return self

        def __next__(self) -> dict[str, Any]:
            seen.append(threading.get_ident())
            if self.remaining == 0:
                raise StopIteration
            self.remaining -= 1
            return {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "x"}}}

    provider = BedrockProvider(api_key=None, settings=ProviderSettings(region="us-east-1"))
    provider._runtime = ns(converse_stream=lambda **_: {"stream": BlockingStream()})

    ticks = 0

    async def heartbeat() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0)
            ticks += 1

    beat = asyncio.create_task(heartbeat())
    await _collect(provider)
    beat.cancel()

    assert seen and all(tid != main_thread for tid in seen), "boto3 iterated on the event loop"
    assert ticks > 0, "event loop was starved during streaming"
