"""Azure AI Foundry.

Two things differ from stock OpenAI: the "model" is a *deployment name* chosen
when the model was deployed, and the endpoint is per-resource, so `base_url`
is mandatory rather than optional.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from wai.core.errors import ConfigError
from wai.core.types import ModelInfo
from wai.providers.base import ProviderCapabilities
from wai.providers.openai import OpenAIProvider

if TYPE_CHECKING:
    from openai import AsyncOpenAI

DEFAULT_API_VERSION = "2024-10-21"


class AzureFoundryProvider(OpenAIProvider):
    name: ClassVar[str] = "azure_foundry"
    max_tokens_field: ClassVar[str] = "max_completion_tokens"
    capabilities: ClassVar[ProviderCapabilities] = ProviderCapabilities(
        reasoning=True, vision=True, system_as_field=False
    )

    def _build_client(self) -> AsyncOpenAI:
        from openai import AsyncAzureOpenAI

        from wai.config.secrets import require_api_key

        if not self.settings.base_url:
            raise ConfigError(
                "azure_foundry needs an endpoint. Set it in config.toml:\n"
                "  [providers.azure_foundry]\n"
                '  base_url = "https://<resource>.openai.azure.com"',
                provider=self.name,
            )
        return AsyncAzureOpenAI(
            api_key=self.api_key or require_api_key(self.name),
            azure_endpoint=self.settings.base_url,
            api_version=self.settings.api_version or DEFAULT_API_VERSION,
            timeout=self.settings.timeout,
            max_retries=0,
            default_headers=self.settings.extra_headers or None,
        )

    async def list_models(self) -> list[ModelInfo]:  # type: ignore[override]
        """Azure exposes deployments, not models; there is nothing to enumerate."""
        return []
