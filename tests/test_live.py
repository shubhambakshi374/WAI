"""Live provider checks. Opt-in, and they cost money.

    uv run pytest -m live                    # every provider with credentials
    uv run pytest -m live -k anthropic       # just one

Providers without resolvable credentials are skipped, so this is safe to run
with only some keys configured. Never enabled in CI.
"""

from __future__ import annotations

import pytest

from altus.config.loader import load_config
from altus.config.secrets import credential_status
from altus.core.events import MessageEnd, TextDelta
from altus.core.types import ChatRequest, Message
from altus.providers import create_provider
from altus.providers.registry import catalog_for

pytestmark = pytest.mark.live

PROMPT = "Reply with exactly one word: pong"

# Cheapest sensible model per provider.
LIVE_MODELS = {
    "anthropic": "claude-haiku-4-5-20251001",
    "openai": "gpt-5-mini",
    "openrouter": "openai/gpt-5",
    "deepseek": "deepseek-chat",
    "gemini": "gemini-2.5-flash",
    "mistral": "mistral-large-latest",
    "bedrock": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "azure_foundry": "",  # deployment name is per-account; set via --model
}


@pytest.fixture(autouse=True)
def real_dirs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Live tests need the developer's real credentials, unlike every other test."""
    monkeypatch.delenv("ALTUS_CONFIG_DIR", raising=False)
    monkeypatch.delenv("ALTUS_DATA_DIR", raising=False)


@pytest.mark.parametrize("provider_name", sorted(LIVE_MODELS))
async def test_provider_streams(provider_name: str) -> None:
    if not credential_status(provider_name).available:
        pytest.skip(f"no credentials for {provider_name}")
    model = LIVE_MODELS[provider_name]
    if not model:
        pytest.skip(f"{provider_name} needs an account-specific deployment name")

    provider = create_provider(provider_name, load_config())
    try:
        events = [
            event
            async for event in provider.stream(
                ChatRequest(model=model, messages=[Message.user(PROMPT)], max_tokens=64)
            )
        ]
    finally:
        await provider.close()

    text = "".join(e.text for e in events if isinstance(e, TextDelta))
    assert text.strip(), "provider produced no text"
    end = next(e for e in events if isinstance(e, MessageEnd))
    assert end.usage.output_tokens > 0, "usage was not reported"


@pytest.mark.parametrize("provider_name", sorted(LIVE_MODELS))
async def test_list_models(provider_name: str) -> None:
    if not credential_status(provider_name).available:
        pytest.skip(f"no credentials for {provider_name}")
    provider = create_provider(provider_name, load_config())
    try:
        models = await provider.list_models()
    finally:
        await provider.close()
    if provider_name == "azure_foundry":
        assert models == []  # deployments are not enumerable
        return
    assert models, "provider returned no models"
    assert all(m.provider == provider_name for m in models) or catalog_for(provider_name)
