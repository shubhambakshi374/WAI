"""OpenRouter: OpenAI-compatible, plus attribution headers and `reasoning`."""

from __future__ import annotations

from typing import ClassVar

from wai.providers.base import ProviderCapabilities
from wai.providers.openai import OpenAIProvider


class OpenRouterProvider(OpenAIProvider):
    name: ClassVar[str] = "openrouter"
    default_base_url: ClassVar[str] = "https://openrouter.ai/api/v1"
    max_tokens_field: ClassVar[str] = "max_tokens"
    capabilities: ClassVar[ProviderCapabilities] = ProviderCapabilities(
        reasoning=True, vision=True, system_as_field=False
    )

    def default_headers(self) -> dict[str, str]:
        """OpenRouter attributes traffic by these; both are optional."""
        headers = {
            "HTTP-Referer": "https://github.com/shubhambakshi374/WAI",
            "X-Title": "WAI",
        }
        headers.update(self.settings.extra_headers)
        return headers
