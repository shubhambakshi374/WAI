"""Local and self-hosted models: discovery, capabilities, endpoints.

No network. Discovery is driven through a patched httpx so the awkward cases
--- an open port that is not an inference server, a 403, HTML --- can be
reproduced deterministically.
"""

from __future__ import annotations

from typing import Any

import pytest

from altus.config.models import Config, Profile
from altus.core.types import ModelInfo
from altus.providers import discovery
from altus.providers.local import LocalProvider, configured_endpoints, ollama_root
from altus.providers.registry import create_provider


class FakeResponse:
    def __init__(self, status: int, body: Any = None, *, text: str = "") -> None:
        self.status_code = status
        self._body = body
        self._text = text

    def json(self) -> Any:
        if self._text:
            raise ValueError("not JSON")
        return self._body


class FakeClient:
    """Stands in for httpx.AsyncClient. Routes are exact URLs."""

    def __init__(self, routes: dict[str, FakeResponse], **_: Any) -> None:
        self.routes = routes

    async def __aenter__(self) -> FakeClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def get(self, url: str) -> FakeResponse:
        if url not in self.routes:
            raise ConnectionRefusedError(url)
        return self.routes[url]

    async def post(self, url: str, json: Any = None) -> FakeResponse:
        if url not in self.routes:
            raise ConnectionRefusedError(url)
        return self.routes[url]


def wire(monkeypatch: pytest.MonkeyPatch, routes: dict[str, FakeResponse]) -> None:
    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: FakeClient(routes, **kw))


def models_payload(*ids: str) -> FakeResponse:
    return FakeResponse(200, {"object": "list", "data": [{"id": i} for i in ids]})


# ------------------------------------------------------------------ discovery


async def test_a_validated_endpoint_is_found(monkeypatch: pytest.MonkeyPatch) -> None:
    wire(
        monkeypatch,
        {
            "http://127.0.0.1:11434/v1/models": models_payload("llama3.2", "qwen3"),
            "http://127.0.0.1:11434/api/version": FakeResponse(200, {"version": "0.5.0"}),
        },
    )
    found = await discovery.probe("http://127.0.0.1:11434")
    assert found.ok
    assert found.kind == "Ollama"
    assert found.models == ["llama3.2", "qwen3"]


async def test_an_open_port_that_is_not_an_inference_server_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real case this exists for: port 5000 on macOS is AirPlay Receiver,
    open and answering 403. Connecting proves nothing."""
    wire(monkeypatch, {"http://127.0.0.1:5000/v1/models": FakeResponse(403)})
    found = await discovery.probe("http://127.0.0.1:5000")
    assert not found.ok
    assert found.error == "HTTP 403"


async def test_html_on_an_open_port_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    wire(
        monkeypatch,
        {"http://127.0.0.1:8080/v1/models": FakeResponse(200, text="<html>hello</html>")},
    )
    found = await discovery.probe("http://127.0.0.1:8080")
    assert not found.ok and found.error == "not JSON"


async def test_json_without_a_model_list_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    wire(monkeypatch, {"http://127.0.0.1:8000/v1/models": FakeResponse(200, {"ok": True})})
    found = await discovery.probe("http://127.0.0.1:8000")
    assert not found.ok and found.error == "no model list"


async def test_a_closed_port_is_reported_not_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    wire(monkeypatch, {})
    found = await discovery.probe("http://127.0.0.1:9")
    assert not found.ok and found.error


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("localhost:11434", "http://localhost:11434/v1"),
        ("http://host:8000", "http://host:8000/v1"),
        ("https://vllm.internal/v1", "https://vllm.internal/v1"),
        ("http://host:8000/v1/", "http://host:8000/v1"),
    ],
)
def test_urls_are_normalised(given: str, expected: str) -> None:
    assert discovery._normalise(given) == expected


async def test_discover_puts_working_endpoints_first(monkeypatch: pytest.MonkeyPatch) -> None:
    wire(
        monkeypatch,
        {
            "http://127.0.0.1:11434/v1/models": models_payload("a"),
            "http://127.0.0.1:11434/api/version": FakeResponse(200, {"version": "0.5"}),
            "http://127.0.0.1:8000/v1/models": models_payload("x", "y", "z"),
            "http://127.0.0.1:1234/v1/models": FakeResponse(403),
        },
    )
    found = await discovery.discover()
    assert found[0].url == "http://127.0.0.1:8000/v1", "most models first"
    assert found[1].url == "http://127.0.0.1:11434/v1"
    assert all(not e.ok for e in found[2:])
    assert [e.url for e in await discovery.discover_working()] == [
        "http://127.0.0.1:8000/v1",
        "http://127.0.0.1:11434/v1",
    ]


async def test_discovery_only_touches_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scanning someone's network unprompted is not a thing a dev tool does."""
    seen: list[str] = []

    async def record(url: str, **kw: Any) -> discovery.LocalEndpoint:
        seen.append(url)
        return discovery.LocalEndpoint(url=url, error="stub")

    monkeypatch.setattr(discovery, "probe", record)
    await discovery.discover()
    assert all(u.startswith("http://127.0.0.1:") for u in seen)


# --------------------------------------------------------------- capabilities


def ollama_routes(capabilities: list[str] | None, context: int = 8192) -> dict[str, FakeResponse]:
    show: dict[str, Any] = {
        "details": {"parameter_size": "3.2B", "quantization_level": "Q4_K_M"},
        "model_info": {"llama.context_length": context},
    }
    if capabilities is not None:
        show["capabilities"] = capabilities
    return {
        "http://127.0.0.1:11434/api/version": FakeResponse(200, {"version": "0.5.0"}),
        "http://127.0.0.1:11434/api/show": FakeResponse(200, show),
    }


class FakeModels:
    def __init__(self, ids: list[str]) -> None:
        self.ids = ids

    async def list(self) -> Any:
        return type("Page", (), {"data": [type("M", (), {"id": i})() for i in self.ids]})()


def local_provider(
    monkeypatch: pytest.MonkeyPatch, ids: list[str], routes: dict[str, FakeResponse]
):  # type: ignore[no-untyped-def]
    from altus.config.models import ProviderSettings

    wire(monkeypatch, routes)
    provider = LocalProvider(
        api_key=None, settings=ProviderSettings(base_url="http://127.0.0.1:11434/v1")
    )
    monkeypatch.setattr(
        provider, "_get_client", lambda: type("C", (), {"models": FakeModels(ids)})()
    )
    return provider


async def test_ollama_capabilities_become_tool_support(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = local_provider(monkeypatch, ["llama3.2"], ollama_routes(["completion", "tools"]))
    model = (await provider.list_models())[0]
    assert model.supports_tools is True
    assert model.context_window == 8192
    assert "3.2B" in (model.display_name or "")


async def test_a_model_without_tools_is_marked_and_labelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = local_provider(monkeypatch, ["tinyllama"], ollama_routes(["completion"]))
    model = (await provider.list_models())[0]
    assert model.supports_tools is False
    assert "no tools" in (model.display_name or "")


async def test_missing_capabilities_stays_optimistic(monkeypatch: pytest.MonkeyPatch) -> None:
    """An older Ollama omits the key; that is not evidence of no tool support."""
    provider = local_provider(monkeypatch, ["oldmodel"], ollama_routes(None))
    assert (await provider.list_models())[0].supports_tools is True


async def test_a_non_ollama_endpoint_is_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """vLLM and llama.cpp report nothing, so assume tools and let config say otherwise."""
    provider = local_provider(monkeypatch, ["Qwen/Qwen3-235B"], {})
    model = (await provider.list_models())[0]
    assert model.supports_tools is True
    assert model.display_name == "Qwen/Qwen3-235B"


def test_ollama_root_strips_the_openai_suffix() -> None:
    assert ollama_root("http://h:11434/v1") == "http://h:11434"
    assert ollama_root("http://h:11434") == "http://h:11434"


# ------------------------------------------------------------------ endpoints


def test_a_local_provider_without_an_endpoint_says_which_field(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from altus.core.errors import ConfigError

    provider = create_provider("local", Config())
    with pytest.raises(ConfigError, match="base_url"):
        provider._build_client()


def test_a_profile_supplies_the_endpoint() -> None:
    profile = Profile(provider="local", model="m", base_url="http://box:8000/v1")
    provider = create_provider("local", Config(), profile=profile)
    assert provider.settings.base_url == "http://box:8000/v1"


def test_a_profile_can_name_an_env_var_for_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_TOKEN", "s3cret")
    profile = Profile(
        provider="local", model="m", base_url="https://vllm.internal/v1", api_key_env="VLLM_TOKEN"
    )
    assert create_provider("local", Config(), profile=profile).api_key == "s3cret"


def test_local_never_reads_the_keyring(monkeypatch: pytest.MonkeyPatch) -> None:
    """A self-hosted endpoint has no business in the OS keychain."""
    import keyring

    from altus.config.secrets import get_api_key

    keyring.set_password("altus", "local", "should-not-be-used")
    assert get_api_key("local") is None
    monkeypatch.setenv("ALTUS_LOCAL_API_KEY", "explicit")
    assert get_api_key("local") == "explicit"


def test_configured_endpoints_reads_profiles() -> None:
    config = Config()
    config.profiles["laptop"] = Profile(
        provider="local", model="a", base_url="http://127.0.0.1:11434/v1"
    )
    config.profiles["cluster"] = Profile(
        provider="local", model="b", base_url="https://vllm.internal/v1"
    )
    config.profiles["cloud"] = Profile(provider="anthropic", model="c")
    assert dict(configured_endpoints(config)) == {
        "laptop": "http://127.0.0.1:11434/v1",
        "cluster": "https://vllm.internal/v1",
    }


def test_endpoint_survives_a_config_round_trip() -> None:
    from altus.config.loader import load_config, save_config

    config = Config()
    config.profiles["laptop"] = Profile(
        provider="local",
        model="llama3.2",
        base_url="http://127.0.0.1:11434/v1",
        api_key_env="TOK",
        supports_tools=False,
    )
    save_config(config)
    reloaded = load_config().profiles["laptop"]
    assert reloaded.base_url == "http://127.0.0.1:11434/v1"
    assert reloaded.api_key_env == "TOK"
    assert reloaded.supports_tools is False


# ------------------------------------------------------------- no-tools path


def test_a_model_without_tools_gets_none_declared() -> None:
    from altus.agent import build_system_prompt
    from altus.core.session import Session
    from altus.tools import default_registry

    registry = default_registry(kubernetes=False)
    capable = Session(provider="local", model="qwen3", workspace_root="/w")
    incapable = Session(
        provider="local", model="tiny", workspace_root="/w", model_supports_tools=False
    )

    assert "read_file" in (build_system_prompt(capable, registry) or "")
    prompt = build_system_prompt(incapable, registry) or ""
    assert "read_file" not in prompt
    assert "does not support tool calling" in prompt, "the reason must be stated"


async def test_the_loop_declares_no_tools_for_an_incapable_model(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from altus.agent import run_agent
    from altus.core.session import Session
    from altus.core.types import Message
    from altus.tools import ToolContext, default_registry
    from altus.workspace import Workspace
    from tests.test_agent import ScriptedProvider, text_turn

    session = Session(
        provider="local", model="tiny", workspace_root=str(tmp_path), model_supports_tools=False
    )
    session.append(Message.user("hello"))
    provider = ScriptedProvider([text_turn("hi")])
    ctx = ToolContext(workspace=Workspace(root=tmp_path))
    async for _ in run_agent(provider, session, default_registry(kubernetes=False), ctx):
        pass
    assert provider.requests[0].tools == [], "no tools may be offered"


def test_session_remembers_the_endpoint_and_capability() -> None:
    from altus.config.loader import sessions_dir
    from altus.core.session import Session
    from altus.storage.sessions import SessionStore

    store = SessionStore(sessions_dir())
    session = Session(
        provider="local",
        model="llama3.2",
        base_url="http://127.0.0.1:11434/v1",
        model_supports_tools=False,
    )
    store.create(session)
    store.update_header(session)
    reloaded = store.load(session.id)
    assert reloaded.base_url == "http://127.0.0.1:11434/v1"
    assert reloaded.model_supports_tools is False


# ------------------------------------------------------------------ live


@pytest.mark.live
async def test_against_a_real_local_server() -> None:
    """Runs entirely on this machine --- no egress, no cost. Skips if nothing
    is listening."""
    working = await discovery.discover_working()
    if not working:
        pytest.skip("no local inference server running")
    endpoint = working[0]

    profile = Profile(provider="local", model="", base_url=endpoint.url)
    provider = create_provider("local", Config(), profile=profile)
    try:
        models = await provider.list_models()
        assert models, "the endpoint listed no models"
        assert all(m.provider == "local" for m in models)
    finally:
        await provider.close()


def test_model_info_carries_capability_by_default() -> None:
    assert ModelInfo(id="x", provider="local").supports_tools is True


def test_local_is_not_ready_until_an_endpoint_exists() -> None:
    """Reporting `local` as configured out of the box suppressed the first-run
    wizard entirely --- there is no key to find, so readiness is a profile."""
    from altus.config.secrets import credential_status

    status = credential_status("local")
    assert status.available is False
    assert "endpoint" in status.detail


async def test_first_run_still_opens_the_wizard_with_local_present(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from altus.tui.widgets.setup import SetupWizard
    from tests.test_tui import _no_credentials, make_app

    _no_credentials(monkeypatch)
    app = make_app()
    async with app.run_test() as pilot:
        assert app.configured_providers == [], "local must not count as configured"
        for _ in range(40):
            await pilot.pause()
            if isinstance(pilot.app.screen, SetupWizard):
                break
        assert isinstance(pilot.app.screen, SetupWizard)
        await pilot.press("escape")
        await pilot.pause()


async def test_a_configured_endpoint_makes_local_ready(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from tests.test_tui import _no_credentials, make_app

    _no_credentials(monkeypatch)
    app = make_app()
    app.config.profiles["laptop"] = Profile(
        provider="local", model="m", base_url="http://127.0.0.1:11434/v1"
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.configured_providers == ["local"]
