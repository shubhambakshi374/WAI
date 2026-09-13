"""The Textual application shell."""

from __future__ import annotations

from textual.app import App

from wai import __version__
from wai.agent import build_tool_context, build_workspace
from wai.config import load_config, resolve_profile, save_config, sessions_dir
from wai.config.models import Config
from wai.core.session import Session
from wai.core.types import ModelInfo
from wai.providers import create_provider
from wai.providers.base import BaseProvider
from wai.storage.sessions import SessionStore
from wai.tools import ToolContext, ToolRegistry, default_registry
from wai.tools.approval import AllowAll, SessionApprovals
from wai.tui.commands import CommandRegistry, build_registry
from wai.tui.screens.chat import ChatScreen
from wai.tui.widgets.approval import InteractiveApproval


class WaiApp(App[None]):
    CSS_PATH = "wai.tcss"
    TITLE = "WAI"
    # Ctrl+P is the model picker. Phase 1 has no commands worth a palette;
    # revisit once the workflow designer has actions to expose.
    ENABLE_COMMAND_PALETTE = False

    def __init__(
        self,
        *,
        profile_name: str | None = None,
        resume: str | None = None,
        model_override: str | None = None,
        provider_override: str | None = None,
        no_tools: bool = False,
        extra_roots: tuple[str, ...] = (),
        auto_approve: bool = False,
        config: Config | None = None,
        provider: BaseProvider | None = None,
        workspace_root: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config or load_config()
        _, profile = resolve_profile(self.config, profile_name)
        if provider_override:
            profile = profile.model_copy(update={"provider": provider_override})
        if model_override:
            profile = profile.model_copy(update={"model": model_override})
        self.profile = profile
        self.workspace = build_workspace(self.config, root=workspace_root, extra_roots=extra_roots)
        self.registry: ToolRegistry = default_registry()
        self.approvals = SessionApprovals(AllowAll() if auto_approve else InteractiveApproval(self))
        self.tool_ctx: ToolContext = build_tool_context(
            self.config, self.workspace, approvals=self.approvals
        )
        self.tools_enabled = self.config.tools.enabled and not no_tools
        self.commands: CommandRegistry = build_registry()
        self.store = SessionStore(sessions_dir())
        self.session = self._load_or_create(resume)
        # An injected provider keeps the app testable without any network.
        self._injected_provider = provider
        self.provider: BaseProvider = provider or create_provider(profile.provider, self.config)
        self.sub_title = f"v{__version__}"

    def _load_or_create(self, resume: str | None) -> Session:
        if resume:
            session_id = self.store.latest_id() if resume == "last" else resume
            if session_id:
                try:
                    return self.store.load(session_id)
                except (FileNotFoundError, ValueError):
                    pass
        return self._new_session()

    def _new_session(self) -> Session:
        session = Session(
            provider=self.profile.provider,
            model=self.profile.model,
            system=self.profile.system,
            max_tokens=self.profile.max_tokens,
            temperature=self.profile.temperature,
            workspace_root=str(self.workspace.root),
            tools_enabled=self.tools_enabled,
        )
        self.store.create(session)
        return session

    def start_new_session(self) -> None:
        self.session = self._new_session()

    async def switch_model(self, model: ModelInfo) -> None:
        """Swap provider and/or model, keeping the current transcript."""
        if model.provider != self.session.provider:
            if self._injected_provider is None:
                await self.provider.close()
                self.provider = create_provider(model.provider, self.config)
            self.session.provider = model.provider
        self.session.model = model.id
        if model.max_output_tokens:
            self.session.max_tokens = min(self.session.max_tokens, model.max_output_tokens)
        self.store.update_header(self.session)

    # ------------------------------------------------ slash-command surface

    async def switch_provider(self, name: str) -> None:
        """Swap provider, keeping the transcript, and pick a sensible model."""
        from wai.providers.registry import catalog_for

        if self._injected_provider is None:
            await self.provider.close()
            self.provider = create_provider(name, self.config)
        self.session.provider = name
        catalog = catalog_for(name)
        if catalog and not any(m.id == self.session.model for m in catalog):
            self.session.model = catalog[0].id
        self.store.update_header(self.session)

    def set_kube_context(self, name: str) -> None:
        """Record the context for WAI only; ~/.kube/config is never touched."""
        self.config.cloud.kube_context = name
        save_config(self.config)

    def add_kubeconfig(self, path: str) -> None:
        if path not in self.config.cloud.kubeconfigs:
            self.config.cloud.kubeconfigs.append(path)
            save_config(self.config)

    async def apply_setup(self, provider: str, model: str) -> None:
        """Adopt what the wizard chose, for this session and the next."""
        from wai.config.loader import resolve_profile as _resolve

        await self.switch_provider(provider)
        self.session.model = model
        name, profile = _resolve(self.config, None)
        self.config.profiles[name] = profile.model_copy(
            update={"provider": provider, "model": model}
        )
        save_config(self.config)
        self.store.update_header(self.session)

    @property
    def configured_providers(self) -> list[str]:
        from wai.config.secrets import credential_status
        from wai.providers import PROVIDER_NAMES

        return [n for n in PROVIDER_NAMES if credential_status(n).available]

    def open_setup(self, provider: str | None = None) -> None:
        from wai.tui.widgets.setup import SetupWizard

        self.push_screen(SetupWizard(provider))

    def open_model_picker(self) -> None:
        screen = self.screen
        if isinstance(screen, ChatScreen):
            screen.action_pick_model()

    async def new_session_from_command(self) -> None:
        screen = self.screen
        if isinstance(screen, ChatScreen):
            await screen.action_new_session()

    def on_mount(self) -> None:
        self.theme = self.config.ui.theme
        self.push_screen(ChatScreen())

    async def on_unmount(self) -> None:
        if self._injected_provider is None:
            await self.provider.close()
