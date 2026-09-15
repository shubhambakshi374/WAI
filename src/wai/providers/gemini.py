"""Google Gemini adapter.

Three shape differences from the OpenAI family: the assistant role is called
``model``, the system prompt is ``system_instruction`` inside a config object,
and message parts use ``functionCall``/``functionResponse`` rather than tool
call ids — so tool results are correlated by name, and we keep a local map of
id to name to bridge the two models.
"""

from __future__ import annotations

import base64
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, ClassVar, cast

from wai.core.errors import (
    AuthenticationError,
    ContextLengthError,
    InvalidRequestError,
    ModelNotFoundError,
    ProviderError,
    RateLimitError,
    TransientProviderError,
)
from wai.core.events import (
    MessageEnd,
    MessageStart,
    ReasoningDelta,
    StreamEvent,
    TextDelta,
    ToolCallEnd,
    ToolCallStart,
    UsageUpdate,
)
from wai.core.types import (
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
    new_id,
)
from wai.providers.base import BaseProvider, ProviderCapabilities

if TYPE_CHECKING:
    from google import genai

_FINISH_REASONS = {
    "STOP": StopReason.END_TURN,
    "MAX_TOKENS": StopReason.MAX_TOKENS,
    "SAFETY": StopReason.CONTENT_FILTER,
    "RECITATION": StopReason.CONTENT_FILTER,
    "PROHIBITED_CONTENT": StopReason.CONTENT_FILTER,
}


class GeminiProvider(BaseProvider):
    name: ClassVar[str] = "gemini"
    capabilities: ClassVar[ProviderCapabilities] = ProviderCapabilities(
        streaming=True,
        tools=True,
        parallel_tool_calls=True,
        reasoning=True,
        vision=True,
        system_as_field=True,
        requires_max_tokens=False,
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._client: genai.Client | None = None

    def _get_client(self) -> genai.Client:
        if self._client is None:
            from google import genai

            from wai.config.secrets import require_api_key

            self._client = genai.Client(api_key=self.api_key or require_api_key(self.name))
        return self._client

    async def list_models(self) -> list[ModelInfo]:
        from wai.providers.registry import catalog_for

        catalog = {m.id: m for m in catalog_for(self.name)}
        try:
            pager = await self._get_client().aio.models.list()
        except Exception as exc:
            raise self._map_error(exc) from exc
        models: list[ModelInfo] = []
        async for item in pager:
            model_id = (item.name or "").removeprefix("models/")
            if not model_id:
                continue
            models.append(
                catalog.get(model_id)
                or ModelInfo(
                    id=model_id,
                    provider=self.name,
                    display_name=getattr(item, "display_name", None),
                    context_window=getattr(item, "input_token_limit", 0) or 0,
                    max_output_tokens=getattr(item, "output_token_limit", 0) or 0,
                )
            )
        return models

    async def stream(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        from google.genai import types as gt

        client = self._get_client()
        config: dict[str, Any] = {}
        if request.system:
            config["system_instruction"] = request.system
        if request.max_tokens:
            config["max_output_tokens"] = request.max_tokens
        if request.temperature is not None:
            config["temperature"] = request.temperature
        if request.top_p is not None:
            config["top_p"] = request.top_p
        if request.stop_sequences:
            config["stop_sequences"] = request.stop_sequences
        if request.tools:
            config["tools"] = [
                gt.Tool(
                    function_declarations=[
                        gt.FunctionDeclaration(
                            name=t.name,
                            description=t.description,
                            parameters=cast("Any", t.input_schema),
                        )
                        for t in request.tools
                    ]
                )
            ]
        config.update(request.extra)

        try:
            raw_stream = await client.aio.models.generate_content_stream(
                model=request.model,
                # The SDK accepts a list of ContentDict, and `_to_contents`
                # builds exactly that shape --- but as `dict[str, Any]`, which
                # is not assignable to a TypedDict. Cast rather than restate
                # the SDK's types, which are a fourteen-member union.
                #
                # This only became visible once Pillow entered the environment
                # for the graphics extra: without it, `PIL.Image` in that union
                # was unresolvable and mypy checked nothing here at all.
                contents=cast("Any", _to_contents(request.messages)),
                config=cast("Any", config),
            )
        except Exception as exc:
            raise self._map_error(exc) from exc

        yield MessageStart(model=request.model)
        usage = Usage()
        stop_reason = StopReason.END_TURN
        tool_index = 0

        async for chunk in raw_stream:
            if getattr(chunk, "usage_metadata", None):
                usage = _to_usage(chunk.usage_metadata)
                yield UsageUpdate(usage=usage)
            for candidate in getattr(chunk, "candidates", None) or []:
                if candidate.finish_reason:
                    stop_reason = _FINISH_REASONS.get(
                        str(candidate.finish_reason).rsplit(".", 1)[-1], StopReason.END_TURN
                    )
                content = getattr(candidate, "content", None)
                for part in getattr(content, "parts", None) or []:
                    if getattr(part, "thought", False) and part.text:
                        yield ReasoningDelta(text=part.text)
                    elif part.text:
                        yield TextDelta(text=part.text)
                    call = getattr(part, "function_call", None)
                    if call is not None:
                        call_id = call.id or new_id("call_")
                        name = call.name or ""
                        yield ToolCallStart(index=tool_index, id=call_id, name=name)
                        yield ToolCallEnd(
                            index=tool_index,
                            id=call_id,
                            name=name,
                            input=dict(call.args or {}),
                        )
                        stop_reason = StopReason.TOOL_USE
                        tool_index += 1

        yield MessageEnd(stop_reason=stop_reason, usage=usage)

    def _map_error(self, exc: Exception) -> Exception:
        code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        text = str(exc)
        lowered = text.lower()
        if code in (429,) or "resource_exhausted" in lowered or "quota" in lowered:
            return RateLimitError(text, provider=self.name, cause=exc)
        if code in (401, 403) or "api key" in lowered or "permission" in lowered:
            return AuthenticationError(text, provider=self.name, cause=exc)
        if code == 404 or "not found" in lowered:
            return ModelNotFoundError(text, provider=self.name, cause=exc)
        if "token" in lowered and ("exceed" in lowered or "too large" in lowered):
            return ContextLengthError(text, provider=self.name, cause=exc)
        if isinstance(code, int) and code >= 500:
            return TransientProviderError(text, provider=self.name, cause=exc)
        if isinstance(exc, TimeoutError | ConnectionError):
            return TransientProviderError(text, provider=self.name, cause=exc)
        if code == 400:
            return InvalidRequestError(text, provider=self.name, cause=exc)
        return ProviderError(text, provider=self.name, cause=exc)


def _to_usage(raw: Any) -> Usage:
    return Usage(
        input_tokens=getattr(raw, "prompt_token_count", 0) or 0,
        output_tokens=getattr(raw, "candidates_token_count", 0) or 0,
        cache_read_tokens=getattr(raw, "cached_content_token_count", 0) or 0,
    )


def _to_contents(messages: list[Message]) -> list[dict[str, Any]]:
    """Gemini calls the assistant role ``model`` and has no system role here."""
    contents: list[dict[str, Any]] = []
    names: dict[str, str] = {}
    for message in messages:
        if message.role is Role.SYSTEM:
            continue
        role = "model" if message.role is Role.ASSISTANT else "user"
        parts: list[dict[str, Any]] = []
        for block in message.content:
            if isinstance(block, TextBlock):
                if block.text:
                    parts.append({"text": block.text})
            elif isinstance(block, ReasoningBlock):
                continue  # not replayable inbound
            elif isinstance(block, ImageBlock):
                parts.append(
                    {
                        "inline_data": {
                            "mime_type": block.media_type,
                            "data": base64.b64decode(block.data),
                        }
                    }
                )
            elif isinstance(block, ToolUseBlock):
                names[block.id] = block.name
                parts.append({"function_call": {"name": block.name, "args": block.input}})
            elif isinstance(block, ToolResultBlock):
                # Gemini correlates results by function name, not call id.
                parts.append(
                    {
                        "function_response": {
                            "name": names.get(block.tool_use_id, block.tool_use_id),
                            "response": {"output": block.content},
                        }
                    }
                )
        if parts:
            contents.append({"role": role, "parts": parts})
    return contents
