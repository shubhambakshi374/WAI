"""The built-in slash commands.

These are a new front door onto machinery that already exists ---
``config.secrets``, ``providers.registry``, ``cloud.auth``, ``cloud.kube`` and
the existing ``ModelPicker`` --- rather than new machinery.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from wai.tui.commands.registry import Command, CommandRegistry, CommandResult

if TYPE_CHECKING:
    from wai.tui.app import WaiApp

CLOUDS = ("k8s", "aws", "azure", "gcp")


# ------------------------------------------------------------------------ help


async def cmd_help(app: WaiApp, args: list[str]) -> CommandResult:
    lines = ["Commands:"]
    for command in app.commands.unique:
        lines.append(f"  /{command.usage or command.name:<26} {command.summary}")
    return CommandResult("\n".join(lines), title="Help")


# ------------------------------------------------------------------- providers


async def cmd_provider(app: WaiApp, args: list[str]) -> CommandResult:
    from wai.config.secrets import credential_status
    from wai.providers import PROVIDER_NAMES
    from wai.tui.widgets.provider_picker import ProviderPicker

    if not args:
        # A list you cannot act on is a dead end. Choosing a configured
        # provider switches; choosing an unconfigured one opens its setup.
        chosen = await app.push_screen_wait(ProviderPicker())
        if not chosen:
            return CommandResult.silent()
        if not credential_status(chosen).available:
            app.open_setup(chosen)
            return CommandResult.silent()
        await app.switch_provider(chosen)
        return CommandResult(f"Switched to {chosen} ({app.session.model}).")

    if args and args[0] == "use":
        if len(args) < 2:
            return CommandResult.error("usage: /provider use <name>")
        name = args[1]
        if name not in PROVIDER_NAMES:
            return CommandResult.error(
                f"Unknown provider {name!r}. Known: {', '.join(PROVIDER_NAMES)}"
            )
        if not credential_status(name).available:
            return CommandResult.error(f"No credentials for {name}. Run /key {name} first.")
        await app.switch_provider(name)
        return CommandResult(f"Switched to {name} ({app.session.model}).")

    return CommandResult.error(f"usage: /provider  or  /provider use <name>. Got: {args}")


async def cmd_key(app: WaiApp, args: list[str]) -> CommandResult:
    from wai.config.secrets import USES_CREDENTIAL_CHAIN, delete_api_key
    from wai.providers import PROVIDER_NAMES
    from wai.tui.widgets.key_prompt import KeyPrompt

    if not args:
        return CommandResult.error("usage: /key <provider>   or   /key rm <provider>")

    if args[0] == "rm":
        if len(args) < 2:
            return CommandResult.error("usage: /key rm <provider>")
        removed = delete_api_key(args[1])
        return CommandResult(
            f"Removed the stored key for {args[1]}." if removed else f"No stored key for {args[1]}."
        )

    name = args[0]
    if name not in PROVIDER_NAMES:
        return CommandResult.error(f"Unknown provider {name!r}. Known: {', '.join(PROVIDER_NAMES)}")
    if name in USES_CREDENTIAL_CHAIN:
        return CommandResult.warn(
            f"{name} authenticates through its cloud credential chain, not an API key."
        )
    app.push_screen(KeyPrompt(name))
    return CommandResult.silent()


async def cmd_profile(app: WaiApp, args: list[str]) -> CommandResult:
    """Profiles are how a self-hosted endpoint is named, so they need a door."""
    profiles = app.config.profiles
    if args and args[0] == "use":
        if len(args) < 2:
            return CommandResult.error("usage: /profile use <name>")
        try:
            await app.use_profile(args[1])
        except KeyError:
            return CommandResult.error(
                f"No profile named {args[1]!r}. Known: {', '.join(sorted(profiles))}"
            )
        session = app.session
        where = f" at {session.base_url}" if session.base_url else ""
        return CommandResult(f"Using profile {args[1]}: {session.provider}/{session.model}{where}.")

    rows = ["Profiles:"]
    for name, profile in sorted(profiles.items()):
        mark = "→" if name == app.config.default_profile else " "
        where = f"  {profile.base_url}" if profile.base_url else ""
        rows.append(f" {mark} {name:<16} {profile.provider}/{profile.model}{where}")
    rows.append("\n  /profile use <name> to switch")
    return CommandResult("\n".join(rows), title="Profiles")


async def cmd_setup(app: WaiApp, args: list[str]) -> CommandResult:
    from wai.providers import PROVIDER_NAMES

    provider = args[0] if args else None
    if provider and provider not in PROVIDER_NAMES:
        return CommandResult.error(
            f"Unknown provider {provider!r}. Known: {', '.join(PROVIDER_NAMES)}"
        )
    app.open_setup(provider)
    return CommandResult.silent()


async def cmd_model(app: WaiApp, args: list[str]) -> CommandResult:
    from wai.core.types import ModelInfo
    from wai.providers.registry import known_models
    from wai.tui.widgets.model_picker import ModelPicker

    if not args:
        chosen = await app.push_screen_wait(ModelPicker())
        if chosen is None:
            return CommandResult.silent()
        await app.switch_model(chosen)
        return CommandResult(f"Model set to {chosen.id} ({chosen.provider}).")

    wanted = args[0]
    match = next((m for m in known_models() if m.id == wanted), None)
    if match is None:
        # Not in the catalog: accept it against the current provider anyway,
        # since catalogs go stale faster than providers ship models.
        match = ModelInfo(id=wanted, provider=app.session.provider)
    await app.switch_model(match)
    return CommandResult(f"Model set to {match.id} ({match.provider}).")


# ----------------------------------------------------------------------- cloud


async def cmd_login(app: WaiApp, args: list[str]) -> CommandResult:
    from wai.cloud.auth import all_status, login, status

    extra = tuple(app.config.cloud.kubeconfigs)
    if not args:
        rows = ["Cloud authentication:"]
        for entry in await asyncio.to_thread(all_status, extra):
            mark = "●" if entry.authenticated else ("○" if entry.available else "·")
            rows.append(f"  {mark} {entry.cloud:<6} {entry.state:<16} {entry.detail or entry.hint}")
        rows.append("\n  /login <cloud> to sign in")
        return CommandResult("\n".join(rows), title="Cloud auth")

    cloud = args[0].casefold()
    if cloud not in CLOUDS:
        return CommandResult.error(f"Unknown cloud {cloud!r}. Known: {', '.join(CLOUDS)}")
    current = await asyncio.to_thread(status, cloud, extra_kubeconfigs=extra)
    if not current.available:
        return CommandResult.error(f"{cloud} support is not installed: {current.hint}")

    profile = args[args.index("--profile") + 1] if "--profile" in args else None
    ok, message = await login(cloud, profile=profile)
    return CommandResult(message, severity="information" if ok else "error", title=f"{cloud} login")


async def cmd_kube(app: WaiApp, args: list[str]) -> CommandResult:
    from wai.cloud.base import CloudTarget, ProtectionRules
    from wai.cloud.kube import list_contexts

    settings = app.config.cloud
    extra = tuple(settings.kubeconfigs)
    rules = ProtectionRules.build(
        settings.protected.patterns, settings.protected.accounts, settings.protected.mode
    )

    if args and args[0] == "use":
        if len(args) < 2:
            return CommandResult.error("usage: /kube use <context>")
        wanted = args[1]
        contexts, _ = await asyncio.to_thread(list_contexts, extra)
        if not any(c.name == wanted for c in contexts):
            return CommandResult.error(f"No context named {wanted!r}. Run /kube to list them.")

        scope = settings.kube_context_scope
        if "--global" in args:
            scope = "global"
        elif "--local" in args:
            scope = "wai"

        app.set_kube_context(wanted)
        written = ""
        if scope == "global":
            from wai.cloud.kube import set_current_context

            try:
                path = await asyncio.to_thread(set_current_context, wanted, extra)
            except Exception as exc:
                return CommandResult.warn(
                    f"Using context {wanted} in WAI, but the kubeconfig could not be updated: {exc}"
                )
            written = f"  Also set current-context in {path} — other terminals will follow."

        protected = rules.matches(CloudTarget("k8s", wanted))
        note = "  ⚠ protected: changes here need extra confirmation" if protected else ""
        return CommandResult(f"Using context {wanted}.{note}{written}")

    if args and args[0] == "add":
        if len(args) < 2:
            return CommandResult.error("usage: /kube add <path-to-kubeconfig>")
        resolved = await asyncio.to_thread(_resolve_file, args[1])
        if resolved is None:
            return CommandResult.error(f"No such file: {args[1]}")
        app.add_kubeconfig(resolved)
        return CommandResult(f"Registered {resolved}.")

    contexts, active = await asyncio.to_thread(list_contexts, extra)
    if not contexts:
        return CommandResult.warn(
            "No kubeconfig found. Add one with /kube add <path>, or set KUBECONFIG."
        )
    selected = settings.kube_context or active
    rows = ["Kubernetes contexts:"]
    for context in contexts:
        mark = "→" if context.name == selected else " "
        flag = " ⚠ protected" if rules.matches(context.target()) else ""
        rows.append(f" {mark} {context.name:<46} ns={context.namespace}{flag}")
    rows.append("\n  /kube use <context> to switch (your ~/.kube/config is never modified)")
    return CommandResult("\n".join(rows), title="Kubernetes")


def _resolve_file(raw: str) -> str | None:
    """Expand and stat off the event loop; None when it is not a file."""
    path = Path(raw).expanduser()
    return str(path) if path.is_file() else None


# ----------------------------------------------------------------------- misc


async def cmd_tools(app: WaiApp, args: list[str]) -> CommandResult:
    from wai.cloud.base import INTEGRATIONS
    from wai.cloud.kube import classify

    rows = [f"Workspace: {app.workspace.root}", "", "Tools:"]
    for tool in sorted(app.registry, key=lambda t: t.name):
        if tool.read_only:
            access = "read-only"
        elif classify(
            getattr(tool, "verb", "update"), "", getattr(tool, "subresource", "")
        ).needs_challenge:
            access = "type to confirm"
        else:
            access = "needs approval"
        rows.append(f"  {tool.name:<18} [{access:^15}]")

    from wai.tools.k8s import disabled_classes

    off = disabled_classes(app.config.cloud.k8s)
    if off:
        rows.append("\nSwitched off in [cloud.k8s] — not offered to the model:")
        rows += [f"  {text}" for text in off]
    missing = [i for i in INTEGRATIONS if not i.available]
    if missing:
        rows.append("\nNot installed:")
        rows += [f"  {i.name:<6} {i.summary:<24} {i.install_hint}" for i in missing]
    granted = app.approvals.always_allowed
    if granted:
        rows.append(f"\n⚠ standing approval this session: {', '.join(sorted(granted))}")
    return CommandResult("\n".join(rows), title="Tools")


async def cmd_new(app: WaiApp, args: list[str]) -> CommandResult:
    await app.new_session_from_command()
    return CommandResult.silent()


def build_registry() -> CommandRegistry:
    registry = CommandRegistry()
    for command in (
        Command("help", "Show this list", "help", cmd_help, aliases=("?",)),
        Command(
            "setup",
            "Add a provider: key, health check, default model",
            "setup [<provider>]",
            cmd_setup,
        ),
        Command(
            "provider",
            "Switch provider (opens setup if it needs a key)",
            "provider [use <name>]",
            cmd_provider,
            aliases=("providers",),
        ),
        Command("key", "Store or remove an API key", "key <provider> | rm <provider>", cmd_key),
        Command(
            "model",
            "Pick a model — searches the provider for unlisted ids",
            "model [<id>]",
            cmd_model,
            aliases=("models",),
        ),
        Command("login", "Cloud auth status, or sign in", "login [<cloud>]", cmd_login),
        Command("kube", "Kubernetes contexts", "kube [use <ctx> | add <path>]", cmd_kube),
        Command("tools", "Tools and installed integrations", "tools", cmd_tools),
        Command("new", "Start a new session", "new", cmd_new, aliases=("clear",)),
    ):
        registry.register(command)
    return registry
