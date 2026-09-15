"""Provider lookup and the bootstrap model catalog.

The catalog exists so the model picker renders instantly and offline. It is a
hint, not a source of truth: every adapter that can enumerate models live
overrides ``list_models``.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from altus.config.models import Config, Profile
from altus.config.secrets import get_api_key
from altus.core.errors import ConfigError
from altus.core.types import ModelInfo

if TYPE_CHECKING:
    from altus.providers.base import BaseProvider

_MODULES: dict[str, tuple[str, str]] = {
    "anthropic": ("altus.providers.anthropic", "AnthropicProvider"),
    "openai": ("altus.providers.openai", "OpenAIProvider"),
    "openrouter": ("altus.providers.openrouter", "OpenRouterProvider"),
    "deepseek": ("altus.providers.deepseek", "DeepSeekProvider"),
    "azure_foundry": ("altus.providers.azure_foundry", "AzureFoundryProvider"),
    "gemini": ("altus.providers.gemini", "GeminiProvider"),
    "mistral": ("altus.providers.mistral", "MistralProvider"),
    "bedrock": ("altus.providers.bedrock", "BedrockProvider"),
    "local": ("altus.providers.local", "LocalProvider"),
}

PROVIDER_NAMES: tuple[str, ...] = tuple(_MODULES)


def _m(
    provider: str,
    model_id: str,
    display: str,
    context: int,
    max_out: int,
    *,
    vision: bool = False,
    reasoning: bool = False,
) -> ModelInfo:
    return ModelInfo(
        id=model_id,
        provider=provider,
        display_name=display,
        context_window=context,
        max_output_tokens=max_out,
        supports_vision=vision,
        supports_reasoning=reasoning,
    )


_CATALOG: dict[str, list[ModelInfo]] = {
    "anthropic": [
        _m(
            "anthropic",
            "claude-opus-5",
            "Claude Opus 5",
            200_000,
            64_000,
            vision=True,
            reasoning=True,
        ),
        _m(
            "anthropic",
            "claude-sonnet-5",
            "Claude Sonnet 5",
            200_000,
            64_000,
            vision=True,
            reasoning=True,
        ),
        _m(
            "anthropic",
            "claude-fable-5-1",
            "Claude Fable 5.1",
            200_000,
            64_000,
            vision=True,
            reasoning=True,
        ),
        _m(
            "anthropic",
            "claude-haiku-4-5-20251001",
            "Claude Haiku 4.5",
            200_000,
            32_000,
            vision=True,
        ),
    ],
    "openai": [
        _m("openai", "gpt-5", "GPT-5", 400_000, 128_000, vision=True, reasoning=True),
        _m("openai", "gpt-5-mini", "GPT-5 mini", 400_000, 128_000, vision=True, reasoning=True),
        _m("openai", "gpt-4.1", "GPT-4.1", 1_047_576, 32_768, vision=True),
    ],
    "openrouter": [
        _m(
            "openrouter",
            "anthropic/claude-sonnet-5",
            "Claude Sonnet 5",
            200_000,
            64_000,
            vision=True,
            reasoning=True,
        ),
        _m("openrouter", "openai/gpt-5", "GPT-5", 400_000, 128_000, vision=True, reasoning=True),
    ],
    "deepseek": [
        _m("deepseek", "deepseek-chat", "DeepSeek Chat", 128_000, 8_192),
        _m("deepseek", "deepseek-reasoner", "DeepSeek Reasoner", 128_000, 64_000, reasoning=True),
    ],
    "azure_foundry": [],
    "gemini": [
        _m(
            "gemini",
            "gemini-2.5-pro",
            "Gemini 2.5 Pro",
            1_048_576,
            65_536,
            vision=True,
            reasoning=True,
        ),
        _m(
            "gemini",
            "gemini-2.5-flash",
            "Gemini 2.5 Flash",
            1_048_576,
            65_536,
            vision=True,
            reasoning=True,
        ),
    ],
    "mistral": [
        _m("mistral", "mistral-large-latest", "Mistral Large", 128_000, 8_192),
        _m(
            "mistral",
            "magistral-medium-latest",
            "Magistral Medium",
            128_000,
            40_000,
            reasoning=True,
        ),
    ],
    "bedrock": [
        _m(
            "bedrock",
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "Claude Sonnet 4.5 (Bedrock)",
            200_000,
            64_000,
            vision=True,
            reasoning=True,
        ),
        _m(
            "bedrock",
            "us.meta.llama3-3-70b-instruct-v1:0",
            "Llama 3.3 70B (Bedrock)",
            128_000,
            8_192,
        ),
    ],
}


def catalog_for(provider: str) -> list[ModelInfo]:
    return list(_CATALOG.get(provider, []))


def known_models() -> list[ModelInfo]:
    return [model for provider in PROVIDER_NAMES for model in _CATALOG.get(provider, [])]


def create_provider(name: str, config: Config, *, profile: Profile | None = None) -> BaseProvider:
    """Import and construct a provider adapter. Raises if the name is unknown.

    A profile may override the endpoint and supply its own key variable, which
    is how one person holds an Ollama laptop and a vLLM cluster at once.
    """
    entry = _MODULES.get(name)
    if entry is None:
        raise ConfigError(f"unknown provider {name!r} (known: {', '.join(PROVIDER_NAMES)})")
    module_path, class_name = entry
    module = importlib.import_module(module_path)
    cls: type[BaseProvider] = getattr(module, class_name)

    settings = config.provider_settings(name)
    api_key = get_api_key(name)
    if profile is not None:
        if profile.base_url:
            settings = settings.model_copy(update={"base_url": profile.base_url})
        if profile.api_key_env:
            api_key = os.environ.get(profile.api_key_env) or api_key
    return cls(api_key=api_key, settings=settings)


async def live_models(
    name: str, config: Config, *, profile: Profile | None = None
) -> list[ModelInfo]:
    """Ask the provider what it actually serves.

    Doubles as the credential health check: it is a real authenticated call,
    read-only and cheap, and a bad key fails here with the provider's own
    error rather than on the user's first real prompt.
    """
    provider = create_provider(name, config, profile=profile)
    try:
        return await provider.list_models()
    finally:
        await provider.close()


def merge_models(*groups: Sequence[ModelInfo]) -> list[ModelInfo]:
    """Combine catalog and live results, keeping the first sighting of each id."""
    seen: dict[str, ModelInfo] = {}
    for group in groups:
        for model in group:
            seen.setdefault(model.id, model)
    return sorted(seen.values(), key=lambda m: m.id)


@dataclass
class ModelSearch:
    """Result of filtering models, and whether the provider was re-asked."""

    matches: list[ModelInfo]
    known: list[ModelInfo]
    requeried: bool = False
    error: str = ""

    @property
    def note(self) -> str:
        if self.error:
            return f"Live search failed: {self.error}"
        if self.requeried:
            return (
                f"{len(self.matches)} match after re-querying the provider."
                if self.matches
                else "The provider does not list anything matching that. "
                "You can still enter the id directly."
            )
        return ""


async def search_models(
    provider: str,
    config: Config,
    known: Sequence[ModelInfo],
    needle: str,
    *,
    profile: Profile | None = None,
) -> ModelSearch:
    """Filter locally; ask the provider again when nothing matches.

    A catalog goes stale faster than providers ship models, so "not in the
    list" must not mean "not available" --- it means look again.
    """
    wanted = needle.strip().casefold()
    pool = list(known)
    if not wanted:
        return ModelSearch(matches=pool, known=pool)

    matches = [m for m in pool if wanted in m.id.casefold()]
    if matches:
        return ModelSearch(matches=matches, known=pool)

    try:
        pool = merge_models(pool, await live_models(provider, config, profile=profile))
    except Exception as exc:
        first = str(exc).strip().splitlines()
        return ModelSearch(matches=[], known=list(known), error=first[0][:160] if first else "")
    return ModelSearch(
        matches=[m for m in pool if wanted in m.id.casefold()], known=pool, requeried=True
    )
