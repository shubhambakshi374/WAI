"""Find local inference servers.

A reachable port is not an inference server. On the machine this was written
on, port 5000 is open and answers 403 --- it is macOS AirPlay Receiver. So an
endpoint only counts when ``GET /v1/models`` returns 200 *and* parses to a
JSON object with a ``data`` list. Anything else is reported with its reason,
which is more useful than silently omitting it.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field

PROBE_TIMEOUT = 1.5
"""Short: this runs while someone waits at a setup screen."""

#: Where these servers listen by default.
KNOWN_PORTS: tuple[tuple[int, str], ...] = (
    (11434, "Ollama"),
    (1234, "LM Studio"),
    (8000, "vLLM"),
    (8080, "llama.cpp"),
    (1337, "Jan"),
    (4891, "GPT4All"),
    (5001, "text-generation-webui"),
)


@dataclass
class LocalEndpoint:
    url: str
    kind: str = ""
    models: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and bool(self.url)

    @property
    def label(self) -> str:
        if not self.ok:
            return f"{self.url} — {self.error}"
        count = f"{len(self.models)} model{'s' if len(self.models) != 1 else ''}"
        return f"{self.kind or 'OpenAI-compatible'} · {self.url} · {count}"


def _normalise(url: str) -> str:
    """Accept `localhost:11434`, `http://host:8000` or a full `/v1` URL."""
    url = url.strip().rstrip("/")
    if not url:
        return ""
    if "://" not in url:
        url = f"http://{url}"
    return url if url.endswith("/v1") else f"{url}/v1"


async def probe(url: str, *, seconds: float = PROBE_TIMEOUT) -> LocalEndpoint:
    """Check one endpoint. Never raises; the reason lands in ``error``."""
    import httpx

    base = _normalise(url)
    if not base:
        return LocalEndpoint(url=url, error="not a URL")

    try:
        async with httpx.AsyncClient(timeout=seconds) as client:
            response = await client.get(f"{base}/models")
    except Exception as exc:
        return LocalEndpoint(
            url=base, error=type(exc).__name__.replace("Error", "").lower() or "unreachable"
        )

    if response.status_code != 200:
        # The AirPlay case: open, answering, not an inference server.
        return LocalEndpoint(url=base, error=f"HTTP {response.status_code}")
    try:
        payload = response.json()
    except Exception:
        return LocalEndpoint(url=base, error="not JSON")
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        return LocalEndpoint(url=base, error="no model list")

    models = [str(m.get("id", "")) for m in payload["data"] if isinstance(m, dict)]
    return LocalEndpoint(url=base, kind=await _identify(base), models=[m for m in models if m])


async def _identify(base: str) -> str:
    """Name the server where it will tell us. Only Ollama reliably does."""
    import httpx

    from altus.providers.local import ollama_root

    try:
        async with httpx.AsyncClient(timeout=1.0) as client:
            response = await client.get(f"{ollama_root(base)}/api/version")
        if response.status_code == 200 and "version" in response.json():
            return "Ollama"
    except Exception:
        pass
    return ""


async def discover(
    extra_urls: Sequence[str] = (), *, host: str = "127.0.0.1"
) -> list[LocalEndpoint]:
    """Every local server that answers properly, best first.

    Only loopback by default: scanning someone's network unprompted is not a
    thing a developer tool should do. Remote endpoints are supplied by URL.
    """
    targets = [f"http://{host}:{port}" for port, _ in KNOWN_PORTS]
    targets += [u for u in extra_urls if u]
    results = await asyncio.gather(*(probe(t) for t in targets), return_exceptions=True)

    found: list[LocalEndpoint] = []
    for target, result in zip(targets, results, strict=True):
        if isinstance(result, BaseException):
            found.append(LocalEndpoint(url=target, error=str(result)[:60]))
        else:
            found.append(result)
    # Working endpoints first, then most models, then a stable order.
    return sorted(found, key=lambda e: (not e.ok, -len(e.models), e.url))


async def discover_working(extra_urls: Sequence[str] = ()) -> list[LocalEndpoint]:
    return [e for e in await discover(extra_urls) if e.ok]
