"""DeepSeek: OpenAI-compatible. ``deepseek-reasoner`` streams
``reasoning_content``, which the base class already maps to ReasoningDelta."""

from __future__ import annotations

from typing import ClassVar

from wai.providers.base import ProviderCapabilities
from wai.providers.openai import OpenAIProvider


class DeepSeekProvider(OpenAIProvider):
    name: ClassVar[str] = "deepseek"
    default_base_url: ClassVar[str] = "https://api.deepseek.com/v1"
    max_tokens_field: ClassVar[str] = "max_tokens"
    capabilities: ClassVar[ProviderCapabilities] = ProviderCapabilities(
        reasoning=True, vision=False, system_as_field=False
    )
