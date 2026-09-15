"""OpenAI adapter, and the base for every OpenAI-compatible endpoint.

OpenRouter, DeepSeek and Azure AI Foundry subclass this. The differences that
actually matter are declared as class attributes rather than buried in
branches, so a new compatible endpoint is a handful of lines.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, ClassVar

from altus.core.errors import (
    AuthenticationError,
    ContentFilterError,
    ContextLengthError,
    InvalidRequestError,
    ModelNotFoundError,
    ProviderError,
    RateLimitError,
    TransientProviderError,
)
from altus.core.events import (
    MessageEnd,
    MessageStart,
    ReasoningDelta,
    StreamEvent,
    TextDelta,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
    UsageUpdate,
)
from altus.core.types import (
    ChatRequest,
    ImageBlock,
    Message,
    ModelInfo,
    ReasoningBlock,
    Role,
    StopReason,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)
from altus.providers.base import BaseProvider, ProviderCapabilities

if TYPE_CHECKING:
    from openai import AsyncOpenAI

_FINISH_REASONS = {
    "stop": StopReason.END_TURN,
    "length": StopReason.MAX_TOKENS,
    "tool_calls": StopReason.TOOL_USE,
    "function_call": StopReason.TOOL_USE,
    "content_filter": StopReason.CONTENT_FILTER,
}

_REASONING_FIELDS = ("reasoning_content", "reasoning")
"""DeepSeek uses the first, OpenRouter the second. Neither is standard."""


class OpenAIProvider(BaseProvider):
    name: ClassVar[str] = "openai"
    capabilities: ClassVar[ProviderCapabilities] = ProviderCapabilities(
        streaming=True,
        tools=True,
        parallel_tool_calls=True,
        reasoning=True,
        vision=True,
        system_as_field=False,
        requires_max_tokens=False,
    )

    default_base_url: ClassVar[str | None] = None
    max_tokens_field: ClassVar[str] = "max_completion_tokens"
    """Newer OpenAI models reject ``max_tokens``; compatible endpoints want it."""
    supports_usage_in_stream: ClassVar[bool] = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._client: AsyncOpenAI | None = None
        self._http: Any = None

    def _http_client(self) -> Any:
        """Own the transport rather than letting the SDK make one.

        With the SDK's own client, httpcore's connection-pool async generator
        is finalised at interpreter shutdown and raises "generator didn't stop
        after athrow()", printing a traceback after a perfectly good answer.
        Reproducible with the SDK alone, so this is a workaround, not a fix ---
        owning the client lets us close it deterministically.
        """
        import httpx

        if self._http is None:
            self._http = httpx.AsyncClient(timeout=self.settings.timeout)
        return self._http

    def _build_client(self) -> AsyncOpenAI:
        from openai import AsyncOpenAI

        from altus.config.secrets import require_api_key

        return AsyncOpenAI(
            api_key=self.api_key or require_api_key(self.name),
            base_url=self.settings.base_url or self.default_base_url,
            timeout=self.settings.timeout,
            max_retries=0,  # altus.core.retry owns retry policy
            default_headers=self.default_headers(),
            http_client=self._http_client(),
        )

    def default_headers(self) -> dict[str, str] | None:
        return self.settings.extra_headers or None

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = self._build_client()
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    async def list_models(self) -> list[ModelInfo]:
        from altus.providers.registry import catalog_for

        catalog = {m.id: m for m in catalog_for(self.name)}
        try:
            page = await self._get_client().models.list()
        except Exception as exc:
            raise self._map_error(exc) from exc
        return [
            catalog.get(item.id) or ModelInfo(id=item.id, provider=self.name) for item in page.data
        ]

    def build_body(self, request: ChatRequest) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": request.model,
            "messages": self.to_wire_messages(request),
            "stream": True,
        }
        if request.max_tokens:
            body[self.max_tokens_field] = request.max_tokens
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.top_p is not None:
            body["top_p"] = request.top_p
        if request.stop_sequences:
            body["stop"] = request.stop_sequences
        if request.tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    },
                }
                for t in request.tools
            ]
        if self.supports_usage_in_stream:
            body["stream_options"] = {"include_usage": True}
        body.update(request.extra)
        return body

    def to_wire_messages(self, request: ChatRequest) -> list[dict[str, Any]]:
        """System is an inline message here, not a separate field."""
        messages: list[dict[str, Any]] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        for message in request.messages:
            messages.extend(_to_openai_messages(message))
        return messages

    async def stream(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        client = self._get_client()
        try:
            raw_stream = await client.chat.completions.create(**self.build_body(request))
        except Exception as exc:
            raise self._map_error(exc) from exc

        usage = Usage()
        stop_reason = StopReason.END_TURN
        started = False
        calls: dict[int, dict[str, str]] = {}
        seen: set[int] = set()

        # Close the response explicitly. Left to interpreter shutdown, httpcore's
        # async generator raises "generator didn't stop after athrow()" and prints
        # a traceback after a perfectly good answer.
        try:
            async for chunk in raw_stream:
                if not started:
                    started = True
                    yield MessageStart(model=getattr(chunk, "model", request.model))

                if getattr(chunk, "usage", None):
                    usage = _to_usage(chunk.usage)
                    yield UsageUpdate(usage=usage)

                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                delta = choice.delta

                if delta is not None:
                    if delta.content:
                        yield TextDelta(text=delta.content)
                    for field in _REASONING_FIELDS:
                        text = getattr(delta, field, None)
                        if text:
                            yield ReasoningDelta(text=text)
                            break
                    for call in delta.tool_calls or []:
                        index = call.index
                        entry = calls.setdefault(index, {"id": "", "name": "", "args": ""})
                        if call.id:
                            entry["id"] = call.id
                        function = getattr(call, "function", None)
                        if function is not None and function.name:
                            entry["name"] = function.name
                        if index not in seen and entry["id"] and entry["name"]:
                            seen.add(index)
                            yield ToolCallStart(index=index, id=entry["id"], name=entry["name"])
                        if function is not None and function.arguments:
                            entry["args"] += function.arguments
                            yield ToolCallDelta(index=index, partial_json=function.arguments)

                if choice.finish_reason:
                    stop_reason = _FINISH_REASONS.get(choice.finish_reason, StopReason.END_TURN)
                    for index, entry in sorted(calls.items()):
                        yield ToolCallEnd(
                            index=index,
                            id=entry["id"],
                            name=entry["name"],
                            input=_loads(entry["args"]),
                        )
                    calls.clear()

        finally:
            await raw_stream.close()

        yield MessageEnd(stop_reason=stop_reason, usage=usage)

    def _map_error(self, exc: Exception) -> Exception:
        import openai

        if isinstance(exc, openai.APITimeoutError | openai.APIConnectionError):
            return TransientProviderError(str(exc), provider=self.name, cause=exc)
        if isinstance(exc, openai.RateLimitError):
            return RateLimitError(
                str(exc), provider=self.name, retry_after=_retry_after(exc), cause=exc
            )
        if isinstance(exc, openai.AuthenticationError | openai.PermissionDeniedError):
            return AuthenticationError(str(exc), provider=self.name, cause=exc)
        if isinstance(exc, openai.NotFoundError):
            return ModelNotFoundError(str(exc), provider=self.name, cause=exc)
        if isinstance(exc, openai.BadRequestError):
            text = str(exc).lower()
            if "context" in text or "too long" in text or "maximum" in text:
                return ContextLengthError(str(exc), provider=self.name, cause=exc)
            if "content" in text and "filter" in text:
                return ContentFilterError(str(exc), provider=self.name, cause=exc)
            return InvalidRequestError(str(exc), provider=self.name, cause=exc)
        if isinstance(exc, openai.APIStatusError):
            if exc.status_code >= 500:
                return TransientProviderError(str(exc), provider=self.name, cause=exc)
            return ProviderError(str(exc), provider=self.name, cause=exc)
        return exc


def _loads(text: str) -> dict[str, Any]:
    if not text.strip():
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _to_usage(raw: Any) -> Usage:
    details = getattr(raw, "prompt_tokens_details", None)
    return Usage(
        input_tokens=getattr(raw, "prompt_tokens", 0) or 0,
        output_tokens=getattr(raw, "completion_tokens", 0) or 0,
        cache_read_tokens=getattr(details, "cached_tokens", 0) or 0 if details else 0,
    )


def _to_openai_messages(message: Message) -> list[dict[str, Any]]:
    """One Altus message may become several: tool results are separate messages."""
    out: list[dict[str, Any]] = []
    parts: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []

    for block in message.content:
        if isinstance(block, TextBlock):
            if block.text:
                parts.append({"type": "text", "text": block.text})
        elif isinstance(block, ImageBlock):
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{block.media_type};base64,{block.data}"},
                }
            )
        elif isinstance(block, ToolUseBlock):
            tool_calls.append(
                {
                    "id": block.id,
                    "type": "function",
                    "function": {"name": block.name, "arguments": json.dumps(block.input)},
                }
            )
        elif isinstance(block, ToolResultBlock):
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": block.tool_use_id,
                    "content": block.content,
                }
            )
        elif isinstance(block, ReasoningBlock):
            # Not replayable: OpenAI-compatible APIs have no inbound reasoning field.
            continue

    if parts or tool_calls:
        entry: dict[str, Any] = {"role": _role(message.role)}
        if parts:
            entry["content"] = parts[0]["text"] if len(parts) == 1 and "text" in parts[0] else parts
        elif tool_calls:
            entry["content"] = None
        if tool_calls:
            entry["tool_calls"] = tool_calls
        out.insert(0, entry)
    return out


def _role(role: Role) -> str:
    return role.value


def _retry_after(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    header = getattr(response, "headers", {}).get("retry-after") if response else None
    try:
        return float(header) if header else None
    except TypeError, ValueError:
        return None
