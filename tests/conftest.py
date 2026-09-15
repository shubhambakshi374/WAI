from __future__ import annotations

import os
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import ClassVar

import pytest

from altus.config.models import Config, ProviderSettings
from altus.core.events import MessageEnd, MessageStart, StreamEvent, TextDelta, UsageUpdate
from altus.core.types import ChatRequest, ModelInfo, Usage
from altus.providers.base import BaseProvider, ProviderCapabilities


@pytest.fixture(autouse=True)
def isolated_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Never read or write the developer's real config, data or keyring."""
    monkeypatch.setenv("ALTUS_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("ALTUS_DATA_DIR", str(tmp_path / "data"))
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
    monkeypatch.setattr("altus.config.secrets._keyring_get", lambda _provider: None)
    # Bedrock authenticates through the AWS chain, and botocore reads ~/.aws
    # and AWS_* itself --- outside everything patched above. A developer with
    # working AWS credentials therefore had one provider configured and CI had
    # none, so the TUI opened the chat screen here and the first-run wizard
    # there, and fifty-two tests passed locally and failed on the runner.
    for key in list(os.environ):
        if key.startswith("AWS_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "no-aws-credentials"))
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "no-aws-config"))
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")


@pytest.fixture(autouse=True)
def one_provider_configured(
    isolated_dirs: None, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """Give the default provider a key, so the baseline is a usable app.

    Without this the isolation above leaves nothing configured, the chat
    screen opens the first-run wizard over itself, and every test that reaches
    for the Composer fails. Which state a test wants is now explicit: this is
    the default, and ``_no_credentials`` is how a test asks for the other one.
    """
    if "unconfigured" in request.keywords:
        return
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")


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
