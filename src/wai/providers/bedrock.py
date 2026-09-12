"""AWS Bedrock adapter, via the Converse API.

Two things make this the odd one out:

* **Auth.** There is no API key. Credentials come from the standard boto3
  chain — env, shared profiles, instance/IRSA roles — because that is how a
  DevOps tool actually gets deployed. See ``config.secrets.USES_CREDENTIAL_CHAIN``.
* **Concurrency.** boto3 is synchronous. Every blocking call is pushed to a
  worker thread; blocking the event loop here would freeze the whole TUI.

Converse already normalizes across the model families Bedrock hosts, so one
adapter covers Anthropic, Llama, Mistral and the rest.
"""

from __future__ import annotations

import asyncio
import base64
from collections.abc import AsyncIterator, Iterator
from typing import Any, ClassVar

from wai.core.errors import (
    AuthenticationError,
    ContentFilterError,
    ContextLengthError,
    CredentialsError,
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
    ToolCallDelta,
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
)
from wai.providers.base import BaseProvider, ProviderCapabilities

_STOP_REASONS = {
    "end_turn": StopReason.END_TURN,
    "max_tokens": StopReason.MAX_TOKENS,
    "stop_sequence": StopReason.STOP_SEQUENCE,
    "tool_use": StopReason.TOOL_USE,
    "content_filtered": StopReason.CONTENT_FILTER,
    "guardrail_intervened": StopReason.CONTENT_FILTER,
}

_RETRYABLE_CODES = {
    "ThrottlingException",
    "TooManyRequestsException",
    "ServiceQuotaExceededException",
    "InternalServerException",
    "ServiceUnavailableException",
    "ModelStreamErrorException",
    "ModelTimeoutException",
    "ModelNotReadyException",
}

DEFAULT_REGION = "us-east-1"


class BedrockProvider(BaseProvider):
    name: ClassVar[str] = "bedrock"
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
        self._runtime: Any = None

    def _session(self) -> Any:
        import boto3

        return boto3.Session(
            profile_name=self.settings.aws_profile,
            region_name=self.settings.region or None,
        )

    def _build_runtime(self) -> Any:
        from botocore.config import Config as BotoConfig

        session = self._session()
        if session.get_credentials() is None:
            raise CredentialsError(
                "no AWS credentials found. Bedrock uses the standard AWS chain "
                "(AWS_PROFILE, environment, instance/IRSA roles), not an API key.",
                provider=self.name,
            )
        return session.client(
            "bedrock-runtime",
            region_name=self.settings.region or session.region_name or DEFAULT_REGION,
            config=BotoConfig(
                read_timeout=self.settings.timeout,
                retries={"max_attempts": 0, "mode": "standard"},  # core.retry owns this
            ),
        )

    async def _get_runtime(self) -> Any:
        if self._runtime is None:
            self._runtime = await asyncio.to_thread(self._build_runtime)
        return self._runtime

    async def close(self) -> None:
        self._runtime = None

    async def list_models(self) -> list[ModelInfo]:
        from wai.providers.registry import catalog_for

        catalog = {m.id: m for m in catalog_for(self.name)}

        def _fetch() -> list[dict[str, Any]]:
            session = self._session()
            client = session.client(
                "bedrock", region_name=self.settings.region or session.region_name or DEFAULT_REGION
            )
            response = client.list_foundation_models(byOutputModality="TEXT")
            summaries: list[dict[str, Any]] = response.get("modelSummaries", [])
            return summaries

        try:
            summaries = await asyncio.to_thread(_fetch)
        except Exception as exc:
            raise self._map_error(exc) from exc
        models: list[ModelInfo] = []
        for summary in summaries:
            model_id = summary.get("modelId", "")
            if not model_id:
                continue
            models.append(
                catalog.get(model_id)
                or ModelInfo(
                    id=model_id,
                    provider=self.name,
                    display_name=f"{summary.get('providerName', '')} {summary.get('modelName', '')}".strip(),
                    supports_tools="TOOL_USE" in (summary.get("inferenceTypesSupported") or []),
                )
            )
        return models

    async def stream(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        runtime = await self._get_runtime()

        body: dict[str, Any] = {
            "modelId": request.model,
            "messages": _to_converse_messages(request.messages),
        }
        if request.system:
            body["system"] = [{"text": request.system}]
        inference: dict[str, Any] = {}
        if request.max_tokens:
            inference["maxTokens"] = request.max_tokens
        if request.temperature is not None:
            inference["temperature"] = request.temperature
        if request.top_p is not None:
            inference["topP"] = request.top_p
        if request.stop_sequences:
            inference["stopSequences"] = request.stop_sequences
        if inference:
            body["inferenceConfig"] = inference
        if request.tools:
            body["toolConfig"] = {
                "tools": [
                    {
                        "toolSpec": {
                            "name": t.name,
                            "description": t.description,
                            "inputSchema": {"json": t.input_schema},
                        }
                    }
                    for t in request.tools
                ]
            }
        body.update(request.extra)

        try:
            response = await asyncio.to_thread(lambda: runtime.converse_stream(**body))
        except Exception as exc:
            raise self._map_error(exc) from exc

        yield MessageStart(model=request.model)
        usage = Usage()
        stop_reason = StopReason.END_TURN
        calls: dict[int, dict[str, str]] = {}

        try:
            async for event in _aiter_blocking(response["stream"]):
                if "contentBlockStart" in event:
                    chunk = event["contentBlockStart"]
                    index = chunk.get("contentBlockIndex", 0)
                    tool = chunk.get("start", {}).get("toolUse")
                    if tool:
                        calls[index] = {
                            "id": tool.get("toolUseId", ""),
                            "name": tool.get("name", ""),
                            "args": "",
                        }
                        yield ToolCallStart(
                            index=index, id=calls[index]["id"], name=calls[index]["name"]
                        )
                elif "contentBlockDelta" in event:
                    chunk = event["contentBlockDelta"]
                    index = chunk.get("contentBlockIndex", 0)
                    delta = chunk.get("delta", {})
                    if "text" in delta:
                        yield TextDelta(text=delta["text"])
                    if "reasoningContent" in delta:
                        reasoning = delta["reasoningContent"]
                        if "text" in reasoning:
                            yield ReasoningDelta(text=reasoning["text"])
                        elif "signature" in reasoning:
                            yield ReasoningDelta(text="", signature=reasoning["signature"])
                    if "toolUse" in delta:
                        fragment = delta["toolUse"].get("input", "")
                        entry = calls.setdefault(index, {"id": "", "name": "", "args": ""})
                        entry["args"] += fragment
                        yield ToolCallDelta(index=index, partial_json=fragment)
                elif "contentBlockStop" in event:
                    index = event["contentBlockStop"].get("contentBlockIndex", 0)
                    finished = calls.pop(index, None)
                    if finished is not None:
                        yield ToolCallEnd(
                            index=index,
                            id=finished["id"],
                            name=finished["name"],
                            input=_loads(finished["args"]),
                        )
                elif "messageStop" in event:
                    stop_reason = _STOP_REASONS.get(
                        event["messageStop"].get("stopReason", ""), StopReason.END_TURN
                    )
                elif "metadata" in event:
                    raw = event["metadata"].get("usage")
                    if raw:
                        usage = _to_usage(raw)
                        yield UsageUpdate(usage=usage)
                else:
                    error = _embedded_error(event)
                    if error is not None:
                        raise error
        except Exception as exc:
            mapped = self._map_error(exc)
            if isinstance(mapped, Exception) and mapped is not exc:
                raise mapped from exc
            raise

        yield MessageEnd(stop_reason=stop_reason, usage=usage)

    def _map_error(self, exc: Exception) -> Exception:
        from botocore.exceptions import (
            BotoCoreError,
            ClientError,
            NoCredentialsError,
            NoRegionError,
            ProfileNotFound,
        )

        if isinstance(exc, NoCredentialsError | ProfileNotFound):
            return CredentialsError(str(exc), provider=self.name, cause=exc)
        if isinstance(exc, NoRegionError):
            return InvalidRequestError(
                "no AWS region configured. Set AWS_REGION, or in config.toml:\n"
                '  [providers.bedrock]\n  region = "us-east-1"',
                provider=self.name,
                cause=exc,
            )
        if isinstance(exc, ClientError):
            code = exc.response.get("Error", {}).get("Code", "")
            text = str(exc)
            if code in _RETRYABLE_CODES:
                if "Throttling" in code or "TooManyRequests" in code:
                    return RateLimitError(text, provider=self.name, cause=exc)
                return TransientProviderError(text, provider=self.name, cause=exc)
            if code in (
                "AccessDeniedException",
                "UnrecognizedClientException",
                "ExpiredTokenException",
            ):
                return AuthenticationError(text, provider=self.name, cause=exc)
            if code == "ResourceNotFoundException":
                return ModelNotFoundError(text, provider=self.name, cause=exc)
            if code == "ValidationException":
                lowered = text.lower()
                if "too long" in lowered or ("token" in lowered and "exceed" in lowered):
                    return ContextLengthError(text, provider=self.name, cause=exc)
                return InvalidRequestError(text, provider=self.name, cause=exc)
            if code == "GuardrailException":
                return ContentFilterError(text, provider=self.name, cause=exc)
            return ProviderError(text, provider=self.name, cause=exc)
        if isinstance(exc, BotoCoreError | TimeoutError | ConnectionError):
            return TransientProviderError(str(exc), provider=self.name, cause=exc)
        return exc


async def _aiter_blocking(stream: Any) -> AsyncIterator[Any]:
    """Consume a blocking boto3 event stream without stalling the event loop.

    One thread hop per event. The cost is negligible next to the network wait,
    and unlike a pump thread plus queue it leaves nothing to deadlock or drain
    when the consumer is cancelled mid-stream.
    """
    iterator: Iterator[Any] = iter(stream)
    sentinel = object()
    while True:
        item = await asyncio.to_thread(next, iterator, sentinel)
        if item is sentinel:
            return
        yield item


def _embedded_error(event: dict[str, Any]) -> Exception | None:
    """Converse reports mid-stream failures as members of the event union."""
    for key, message in (
        ("internalServerException", TransientProviderError),
        ("modelStreamErrorException", TransientProviderError),
        ("throttlingException", RateLimitError),
        ("serviceUnavailableException", TransientProviderError),
        ("validationException", InvalidRequestError),
    ):
        if key in event:
            detail = event[key].get("message", key)
            return message(detail, provider=BedrockProvider.name)
    return None


def _loads(text: str) -> dict[str, Any]:
    import json

    if not text.strip():
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _to_usage(raw: dict[str, Any]) -> Usage:
    return Usage(
        input_tokens=raw.get("inputTokens", 0) or 0,
        output_tokens=raw.get("outputTokens", 0) or 0,
        cache_read_tokens=raw.get("cacheReadInputTokens", 0) or 0,
        cache_write_tokens=raw.get("cacheWriteInputTokens", 0) or 0,
    )


def _to_converse_messages(messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for message in messages:
        if message.role is Role.SYSTEM:
            continue
        blocks: list[dict[str, Any]] = []
        for block in message.content:
            if isinstance(block, TextBlock):
                if block.text:
                    blocks.append({"text": block.text})
            elif isinstance(block, ReasoningBlock):
                if block.text:
                    blocks.append(
                        {
                            "reasoningContent": {
                                "reasoningText": {
                                    "text": block.text,
                                    "signature": block.signature or "",
                                }
                            }
                        }
                    )
            elif isinstance(block, ImageBlock):
                blocks.append(
                    {
                        "image": {
                            "format": block.media_type.rsplit("/", 1)[-1],
                            "source": {"bytes": base64.b64decode(block.data)},
                        }
                    }
                )
            elif isinstance(block, ToolUseBlock):
                blocks.append(
                    {"toolUse": {"toolUseId": block.id, "name": block.name, "input": block.input}}
                )
            elif isinstance(block, ToolResultBlock):
                result: dict[str, Any] = {
                    "toolUseId": block.tool_use_id,
                    "content": [{"text": block.content}],
                }
                if block.is_error:
                    result["status"] = "error"
                blocks.append({"toolResult": result})
        if blocks:
            out.append({"role": message.role.value, "content": blocks})
    return out
