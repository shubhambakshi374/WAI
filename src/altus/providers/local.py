"""Models you run yourself: Ollama, LM Studio, vLLM, llama.cpp.

Almost nothing here is a new wire format. Every one of these servers speaks
the OpenAI chat-completions shape, and ``OpenAIProvider`` was written with
``base_url`` and ``max_tokens_field`` overridable, so streaming, usage and
tool calling already work against them unchanged.

What is genuinely different is everything around the request: there is no API
key, the endpoint has to be discovered or supplied, the model may or may not
be able to call tools, and a large mixture-of-experts model can take a minute
to load before it emits a single token.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, ClassVar

from altus.core.errors import ConfigError
from altus.core.types import ModelInfo
from altus.providers.base import ProviderCapabilities
from altus.providers.openai import OpenAIProvider

if TYPE_CHECKING:
    from altus.config.models import Config

PLACEHOLDER_KEY = "local"
"""Most local servers ignore the key, but the OpenAI SDK refuses an empty one."""

LOCAL_TIMEOUT = 1800.0
"""Half an hour. A cold MoE load happens *before* the first token, so a short
timeout turns "still loading" into a spurious failure."""

MAX_ENRICHED = 40
"""Cap the per-model capability lookups; a big Ollama library would otherwise
mean one request per model on every listing."""


class LocalProvider(OpenAIProvider):
    name: ClassVar[str] = "local"
    max_tokens_field: ClassVar[str] = "max_tokens"
    """Compat servers want max_tokens; max_completion_tokens is OpenAI-only."""
    capabilities: ClassVar[ProviderCapabilities] = ProviderCapabilities(
        streaming=True,
        tools=True,
        parallel_tool_calls=False,
        reasoning=False,
        vision=False,
        system_as_field=False,
        requires_max_tokens=False,
    )

    def _build_client(self) -> Any:
        from openai import AsyncOpenAI

        base_url = self.settings.base_url
        if not base_url:
            raise ConfigError(
                "the local provider needs an endpoint. Run /setup local to find one, "
                'or set base_url on the profile:\n  [profiles.laptop]\n  provider = "local"\n'
                '  base_url = "http://127.0.0.1:11434/v1"',
                provider=self.name,
            )
        return AsyncOpenAI(
            api_key=self.api_key or PLACEHOLDER_KEY,
            base_url=base_url,
            timeout=max(self.settings.timeout, LOCAL_TIMEOUT),
            max_retries=0,
            default_headers=self.settings.extra_headers or None,
            http_client=self._http_client(),
        )

    async def list_models(self) -> list[ModelInfo]:
        """What the endpoint serves, enriched with capabilities where possible."""
        try:
            page = await self._get_client().models.list()
        except Exception as exc:
            raise self._map_error(exc) from exc

        models = [
            ModelInfo(id=item.id, provider=self.name, display_name=item.id) for item in page.data
        ]
        base = self.settings.base_url or ""
        if await _is_ollama(base):
            models = await _enrich_from_ollama(base, models)
        return models


# --------------------------------------------------------------- Ollama extras


def ollama_root(base_url: str) -> str:
    """`http://host:11434/v1` -> `http://host:11434`, for the native API."""
    return base_url.rstrip("/").removesuffix("/v1")


async def _is_ollama(base_url: str) -> bool:
    if not base_url:
        return False
    payload = await _get_json(f"{ollama_root(base_url)}/api/version", seconds=2.0)
    return isinstance(payload, dict) and "version" in payload


async def _enrich_from_ollama(base_url: str, models: list[ModelInfo]) -> list[ModelInfo]:
    """Ask Ollama what each model can do.

    Tool support is the difference between Altus being a DevOps harness and a
    chat box, and Ollama will simply tell us --- so read it rather than guess.
    """
    root = ollama_root(base_url)
    targets = models[:MAX_ENRICHED]
    details = await asyncio.gather(*(_show(root, m.id) for m in targets), return_exceptions=True)
    out: list[ModelInfo] = []
    for model, detail in zip(targets, details, strict=True):
        if not isinstance(detail, dict):
            out.append(model)
            continue
        capabilities = detail.get("capabilities") or []
        info = detail.get("model_info") or {}
        context = next((int(v) for k, v in info.items() if k.endswith("context_length") and v), 0)
        out.append(
            model.model_copy(
                update={
                    # A missing capabilities key means an older Ollama, not a
                    # model without tools --- stay optimistic there.
                    "supports_tools": "tools" in capabilities if capabilities else True,
                    "supports_vision": "vision" in capabilities,
                    "context_window": context,
                    "display_name": _describe(model.id, detail),
                }
            )
        )
    return out + models[MAX_ENRICHED:]


def _describe(model_id: str, detail: dict[str, Any]) -> str:
    facts = detail.get("details") or {}
    parts = [p for p in (facts.get("parameter_size"), facts.get("quantization_level")) if p]
    if "tools" not in (detail.get("capabilities") or ["tools"]):
        parts.append("no tools")
    return f"{model_id}  ({', '.join(parts)})" if parts else model_id


async def _show(root: str, model: str) -> dict[str, Any] | None:
    payload = await _post_json(f"{root}/api/show", {"model": model}, seconds=5.0)
    return payload if isinstance(payload, dict) else None


async def _get_json(url: str, *, seconds: float) -> Any:
    import httpx

    try:
        async with httpx.AsyncClient(timeout=seconds) as client:
            response = await client.get(url)
        return response.json() if response.status_code == 200 else None
    except Exception:
        return None


async def _post_json(url: str, body: dict[str, Any], *, seconds: float) -> Any:
    import httpx

    try:
        async with httpx.AsyncClient(timeout=seconds) as client:
            response = await client.post(url, json=body)
        return response.json() if response.status_code == 200 else None
    except Exception:
        return None


# ------------------------------------------------------------------- profiles


def configured_endpoints(config: Config) -> list[tuple[str, str]]:
    """(profile name, base_url) for every profile pointing at a local endpoint.

    ``credential_status`` cannot answer "is local set up?" --- there is no key
    to look for --- so readiness is "does a profile name an endpoint".
    """
    return [
        (name, profile.base_url)
        for name, profile in config.profiles.items()
        if profile.provider == LocalProvider.name and profile.base_url
    ]
