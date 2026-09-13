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

    rows = ["Providers:"]
    for name in PROVIDER_NAMES:
        status = credential_status(name)
        mark = "●" if status.available else "○"
        current = "  ← current" if name == app.session.provider else ""
        rows.append(f"  {mark} {name:<15} {status.source:<12}{current}")
    rows.append("\n  ● configured   ○ no credentials — /key <provider> to add one")
    return CommandResult("\n".join(rows), title="Providers")


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

    if not args:
        app.open_model_picker()
        return CommandResult.silent()

    wanted = args[0]
    match = next((m for m in known_models() if m.id == wanted), None)
    if match is None:
        # Not in the catalog: accept it against the current provider anyway,
        # since catalogs go stale faster than providers ship models.
        match = ModelInfo(id=wanted, provider=app.session.provider)
    await app.switch_model(match)
    return CommandResult(f"Model set to {match.id} ({match.provider}).")


async def cmd_models(app: WaiApp, args: list[str]) -> CommandResult:
    from wai.providers.registry import catalog_for

    provider = args[0] if args else app.session.provider
    models = catalog_for(provider)
    if not models:
        return CommandResult.warn(
            f"No catalog entries for {provider}. Try: wai providers models {provider} --live"
        )
    rows = [f"Models for {provider}:"]
    rows += [f"  {m.id:<48} {m.label}" for m in models]
    return CommandResult("\n".join(rows), title="Models")


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

    rows = [f"Workspace: {app.workspace.root}", "", "Tools:"]
    for tool in sorted(app.registry, key=lambda t: t.name):
        access = "read-only" if tool.read_only else "needs approval"
        rows.append(f"  {tool.name:<14} [{access:^14}]")
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
        Command("provider", "List or switch LLM provider", "provider [use <name>]", cmd_provider),
        Command("key", "Store or remove an API key", "key <provider> | rm <provider>", cmd_key),
        Command("model", "Pick a model, or set one directly", "model [<id>]", cmd_model),
        Command("models", "List models for a provider", "models [<provider>]", cmd_models),
        Command("login", "Cloud auth status, or sign in", "login [<cloud>]", cmd_login),
        Command("kube", "Kubernetes contexts", "kube [use <ctx> | add <path>]", cmd_kube),
        Command("tools", "Tools and installed integrations", "tools", cmd_tools),
        Command("new", "Start a new session", "new", cmd_new, aliases=("clear",)),
    ):
        registry.register(command)
    return registry
