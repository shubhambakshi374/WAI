"""The provider contract.

Everything above this line is provider-agnostic; everything below it is
wire-format translation. The Phase 2 workflow engine talks to this Protocol,
which is why nothing here may know about the TUI.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import ClassVar, Protocol, runtime_checkable

from wai.config.models import ProviderSettings
from wai.core.events import StreamEvent
from wai.core.types import ChatRequest, ModelInfo


@dataclass(frozen=True)
class ProviderCapabilities:
    """The per-provider differences callers actually have to branch on."""

    streaming: bool = True
    tools: bool = True
    parallel_tool_calls: bool = True
    reasoning: bool = False
    vision: bool = False
    system_as_field: bool = True
    """False when the system prompt must be sent as an inline message."""
    requires_max_tokens: bool = False
    """True when the API rejects a request that omits max_tokens."""


@runtime_checkable
class Provider(Protocol):
    """What the TUI, the CLI and the Phase 2 engine are allowed to assume.

    ``name`` and ``capabilities`` are read-only so that implementations are
    free to declare them as ``ClassVar``, which they all do.
    """

    @property
    def name(self) -> str: ...

    @property
    def capabilities(self) -> ProviderCapabilities: ...

    def stream(self, request: ChatRequest) -> AsyncIterator[StreamEvent]: ...

    async def list_models(self) -> list[ModelInfo]: ...

    async def close(self) -> None: ...


class BaseProvider(ABC):
    """Shared construction and lifecycle for the concrete adapters."""

    name: ClassVar[str] = "base"
    capabilities: ClassVar[ProviderCapabilities] = ProviderCapabilities()

    def __init__(self, *, api_key: str | None, settings: ProviderSettings) -> None:
        self.api_key = api_key
        self.settings = settings

    @abstractmethod
    def stream(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        """Yield normalized events for one inference call.

        Implementations are async generators. Failures *before* the first
        yielded event must raise a ``WaiError`` so ``retry_stream`` can retry;
        failures after it must be yielded as ``StreamError``.
        """

    async def list_models(self) -> list[ModelInfo]:
        """Live model list. Defaults to the static catalog."""
        from wai.providers.registry import catalog_for

        return catalog_for(self.name)

    async def close(self) -> None:
        """Release any underlying HTTP client. Safe to call more than once."""
        return None

    async def __aenter__(self) -> BaseProvider:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()
