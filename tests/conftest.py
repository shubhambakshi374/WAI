from __future__ import annotations

import os
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import ClassVar

import pytest

from wai.config.models import Config, ProviderSettings
from wai.core.events import MessageEnd, MessageStart, StreamEvent, TextDelta, UsageUpdate
from wai.core.types import ChatRequest, ModelInfo, Usage
from wai.providers.base import BaseProvider, ProviderCapabilities


@pytest.fixture(autouse=True)
def isolated_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Never read or write the developer's real config, data or keyring."""
    monkeypatch.setenv("WAI_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("WAI_DATA_DIR", str(tmp_path / "data"))
    for key in list(os.environ):
        if key.endswith("_API_KEY"):
            monkeypatch.delenv(key, raising=False)
    # Block the OS keyring at the library boundary, not at our wrapper.
    # secrets.py does `import keyring` inside each function, so patching here
    # holds however the wrapper is imported --- and a test must never be able
    # to overwrite a real stored credential.
    import keyring

    vault: dict[tuple[str, str], str] = {}
    monkeypatch.setattr(keyring, "get_password", lambda s, u: vault.get((s, u)))
    monkeypatch.setattr(keyring, "set_password", lambda s, u, p: vault.__setitem__((s, u), p))
    monkeypatch.setattr(keyring, "delete_password", lambda s, u: vault.pop((s, u), None))
    monkeypatch.setattr("wai.config.secrets._keyring_get", lambda _provider: None)


class FakeProvider(BaseProvider):
    """Replays a scripted event sequence. No network, no SDK."""

    name: ClassVar[str] = "fake"
    capabilities: ClassVar[ProviderCapabilities] = ProviderCapabilities(reasoning=True)

    def __init__(
        self,
        events: Sequence[StreamEvent] | None = None,
        *,
        error: Exception | None = None,
        api_key: str | None = "test",
        settings: ProviderSettings | None = None,
    ) -> None:
        super().__init__(api_key=api_key, settings=settings or ProviderSettings())
        self.events = list(events) if events is not None else list(default_events())
        self.error = error
        self.calls: list[ChatRequest] = []
        self.closed = False

    async def stream(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        self.calls.append(request)
        if self.error is not None:
            raise self.error
        for event in self.events:
            yield event

    async def list_models(self) -> list[ModelInfo]:
        return [ModelInfo(id="fake-1", provider=self.name, display_name="Fake 1")]

    async def close(self) -> None:
        self.closed = True


def default_events(text: str = "Hello there.") -> list[StreamEvent]:
    usage = Usage(input_tokens=11, output_tokens=3)
    return [
        MessageStart(model="fake-1", usage=Usage(input_tokens=11)),
        *[TextDelta(text=chunk) for chunk in text.split(" ")],
        UsageUpdate(usage=usage),
        MessageEnd(usage=usage),
    ]


@pytest.fixture
def fake_provider() -> FakeProvider:
    return FakeProvider()


@pytest.fixture
def config() -> Config:
    return Config()
