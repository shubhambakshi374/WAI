"""Provider lookup and the bootstrap model catalog.

The catalog exists so the model picker renders instantly and offline. It is a
hint, not a source of truth: every adapter that can enumerate models live
overrides ``list_models``.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from wai.config.models import Config
from wai.config.secrets import get_api_key
from wai.core.errors import ConfigError
from wai.core.types import ModelInfo

if TYPE_CHECKING:
    from wai.providers.base import BaseProvider

_MODULES: dict[str, tuple[str, str]] = {
    "anthropic": ("wai.providers.anthropic", "AnthropicProvider"),
    "openai": ("wai.providers.openai", "OpenAIProvider"),
    "openrouter": ("wai.providers.openrouter", "OpenRouterProvider"),
    "deepseek": ("wai.providers.deepseek", "DeepSeekProvider"),
    "azure_foundry": ("wai.providers.azure_foundry", "AzureFoundryProvider"),
    "gemini": ("wai.providers.gemini", "GeminiProvider"),
    "mistral": ("wai.providers.mistral", "MistralProvider"),
    "bedrock": ("wai.providers.bedrock", "BedrockProvider"),
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


def create_provider(name: str, config: Config) -> BaseProvider:
    """Import and construct a provider adapter. Raises if the name is unknown."""
    entry = _MODULES.get(name)
    if entry is None:
        raise ConfigError(f"unknown provider {name!r} (known: {', '.join(PROVIDER_NAMES)})")
    module_path, class_name = entry
    module = importlib.import_module(module_path)
    cls: type[BaseProvider] = getattr(module, class_name)
    return cls(api_key=get_api_key(name), settings=config.provider_settings(name))
