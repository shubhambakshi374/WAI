"""Command line entry point.

``wai`` with no subcommand launches the TUI. Everything else is deliberately
usable headlessly, so the provider layer can be exercised without a terminal
UI — and so CI can too.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import Annotated

import typer

from wai import __version__
from wai.agent import build_tool_context, build_workspace, run_agent
from wai.config import (
    config_path,
    load_config,
    resolve_profile,
    sessions_dir,
    write_starter_config,
)
from wai.config.models import Profile
from wai.config.secrets import (
    PROVIDER_NAMES_HINT,
    credential_status,
    delete_api_key,
    set_api_key,
)
from wai.core.errors import WaiError
from wai.core.events import StreamError, TextDelta, ToolFinished, ToolStarted
from wai.core.session import Session
from wai.core.types import Message, ModelInfo
from wai.providers import PROVIDER_NAMES, create_provider
from wai.storage.sessions import SessionStore
from wai.tools import default_registry

app = typer.Typer(
    name="wai",
    help="A TUI coding and DevOps harness with BYOK multi-provider LLM support.",
    no_args_is_help=False,
    add_completion=False,
)
config_app = typer.Typer(help="Inspect configuration and manage API keys.")
providers_app = typer.Typer(help="Inspect providers and their models.")
sessions_app = typer.Typer(help="Browse saved sessions.")
tools_app = typer.Typer(help="Inspect the tools the model can call.")
app.add_typer(config_app, name="config")
app.add_typer(providers_app, name="providers")
app.add_typer(sessions_app, name="sessions")
app.add_typer(tools_app, name="tools")


def _fail(message: str) -> None:
    typer.secho(f"error: {message}", fg=typer.colors.RED, err=True)
    raise typer.Exit(1)


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    version: Annotated[bool, typer.Option("--version", help="Show version and exit.")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging.")] = False,
) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if version:
        typer.echo(f"wai {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        _launch_tui(profile=None, resume=None)


@app.command()
def chat(
    profile: Annotated[str | None, typer.Option("--profile", "-p")] = None,
    model: Annotated[str | None, typer.Option("--model", "-m")] = None,
    provider: Annotated[str | None, typer.Option("--provider")] = None,
    resume: Annotated[str | None, typer.Option("--resume", help="Session id, or 'last'.")] = None,
    once: Annotated[
        str | None,
        typer.Option("--once", help="Send one prompt, stream to stdout, exit. No TUI."),
    ] = None,
    no_tools: Annotated[
        bool, typer.Option("--no-tools", help="Disable filesystem tools; plain chat.")
    ] = False,
    allow_path: Annotated[
        list[str] | None,
        typer.Option("--allow-path", help="Extra readable root. Repeatable."),
    ] = None,
) -> None:
    """Start the chat TUI, or run a single headless turn with --once."""
    extra = tuple(allow_path or ())
    if once is None:
        _launch_tui(
            profile=profile,
            resume=resume,
            model=model,
            provider=provider,
            no_tools=no_tools,
            extra_roots=extra,
        )
        return
    try:
        raise SystemExit(asyncio.run(_chat_once(once, profile, model, provider, no_tools, extra)))
    except WaiError as exc:
        _fail(str(exc))
    except KeyboardInterrupt:
        raise typer.Exit(130) from None


async def _chat_once(
    prompt: str,
    profile_name: str | None,
    model: str | None,
    provider_name: str | None,
    no_tools: bool,
    extra_roots: tuple[str, ...],
) -> int:
    """One prompt through the full agent loop.

    Text goes to stdout and tool activity to stderr, so the answer stays
    pipeable while the tool trace stays visible.
    """
    config = load_config()
    _, prof = resolve_profile(config, profile_name)
    prof = _override(prof, model=model, provider=provider_name)

    workspace = build_workspace(config, extra_roots=extra_roots)
    registry = default_registry()
    ctx = build_tool_context(config, workspace)
    tools_on = config.tools.enabled and not no_tools

    session = Session(
        provider=prof.provider,
        model=prof.model,
        system=prof.system,
        max_tokens=prof.max_tokens,
        temperature=prof.temperature,
        workspace_root=str(workspace.root),
        tools_enabled=tools_on,
    )
    session.append(Message.user(prompt))

    adapter = create_provider(prof.provider, config)
    failed = False
    mid_line = False
    try:
        async for event in run_agent(
            adapter, session, registry, ctx, max_iterations=config.tools.max_iterations
        ):
            if isinstance(event, TextDelta):
                sys.stdout.write(event.text)
                sys.stdout.flush()
                mid_line = not event.text.endswith("\n")
            elif isinstance(event, ToolStarted):
                if mid_line:
                    # stdout and stderr share a terminal; don't splice the tool
                    # trace into the middle of a sentence.
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                    mid_line = False
                typer.secho(
                    f"  → {event.name}({_brief(event.args)})",
                    fg=typer.colors.BRIGHT_BLACK,
                    err=True,
                )
            elif isinstance(event, ToolFinished):
                colour = typer.colors.RED if event.is_error else typer.colors.BRIGHT_BLACK
                typer.secho(f"    {event.summary}", fg=colour, err=True)
                failed = failed or event.is_error
            elif isinstance(event, StreamError):
                typer.secho(f"\nstream error: {event.message}", fg=typer.colors.RED, err=True)
                failed = True
    finally:
        await adapter.close()

    if mid_line:
        sys.stdout.write("\n")
    typer.secho(
        f"[{prof.provider}/{prof.model}] "
        f"{session.usage.input_tokens} in / {session.usage.output_tokens} out"
        + (f" · workspace {workspace.root}" if tools_on else " · tools off"),
        fg=typer.colors.BRIGHT_BLACK,
        err=True,
    )
    return 1 if failed else 0


def _brief(args: dict[str, object], limit: int = 60) -> str:
    rendered = ", ".join(f"{k}={v!r}" for k, v in args.items())
    return rendered if len(rendered) <= limit else rendered[: limit - 1] + "…"


def _override(profile: Profile, *, model: str | None, provider: str | None) -> Profile:
    updates: dict[str, str] = {}
    if model:
        updates["model"] = model
    if provider:
        if provider not in PROVIDER_NAMES:
            _fail(f"unknown provider {provider!r} (known: {PROVIDER_NAMES_HINT})")
        updates["provider"] = provider
    return profile.model_copy(update=updates) if updates else profile


def _launch_tui(
    *,
    profile: str | None,
    resume: str | None,
    model: str | None = None,
    provider: str | None = None,
    no_tools: bool = False,
    extra_roots: tuple[str, ...] = (),
) -> None:
    from wai.tui.app import WaiApp

    try:
        WaiApp(
            profile_name=profile,
            resume=resume,
            model_override=model,
            provider_override=provider,
            no_tools=no_tools,
            extra_roots=extra_roots,
        ).run()
    except WaiError as exc:
        _fail(str(exc))


# --------------------------------------------------------------------------- config


@config_app.command("path")
def config_path_cmd() -> None:
    """Print the config file path."""
    typer.echo(str(config_path()))


@config_app.command("show")
def config_show() -> None:
    """Print the effective configuration. Contains no secrets."""
    import tomli_w

    config = load_config()
    typer.echo(tomli_w.dumps(config.model_dump(mode="json", exclude_none=True)).rstrip())


@config_app.command("init")
def config_init(
    force: Annotated[bool, typer.Option("--force", help="Overwrite an existing file.")] = False,
) -> None:
    """Write a starter config file."""
    path = config_path()
    if path.exists() and not force:
        _fail(f"{path} already exists (use --force to overwrite)")
    written = write_starter_config()
    typer.echo(f"wrote {written}")


@config_app.command("set-key")
def config_set_key(
    provider: Annotated[str, typer.Argument(help=f"One of: {', '.join(PROVIDER_NAMES)}")],
) -> None:
    """Store an API key in the OS keyring. Never written to disk in plaintext."""
    if provider not in PROVIDER_NAMES:
        _fail(f"unknown provider {provider!r} (known: {PROVIDER_NAMES_HINT})")
    from wai.config.secrets import USES_CREDENTIAL_CHAIN

    if provider in USES_CREDENTIAL_CHAIN:
        _fail(
            f"{provider} authenticates through its cloud SDK credential chain "
            "(AWS_PROFILE, instance/IRSA roles), not an API key"
        )
    key = typer.prompt(f"{provider} API key", hide_input=True)
    if not key.strip():
        _fail("empty key")
    try:
        set_api_key(provider, key.strip())
    except Exception as exc:
        _fail(f"keyring unavailable: {exc}")
    typer.secho(f"stored key for {provider} in the OS keyring", fg=typer.colors.GREEN)


@config_app.command("delete-key")
def config_delete_key(provider: str) -> None:
    """Remove a stored API key from the OS keyring."""
    typer.echo(
        f"deleted key for {provider}"
        if delete_api_key(provider)
        else f"no stored key for {provider}"
    )


@config_app.command("doctor")
def config_doctor() -> None:
    """Report which providers can authenticate. Never prints key material."""
    config = load_config()
    typer.echo(f"config:   {config_path()}")
    typer.echo(f"sessions: {sessions_dir()}")
    typer.echo(f"default profile: {config.default_profile}\n")
    ok = 0
    for name in PROVIDER_NAMES:
        status = credential_status(name)
        if status.available:
            ok += 1
        typer.secho(
            f"  {'ok  ' if status.available else 'none'}  {name:<14} "
            f"{status.source:<10} {status.detail}",
            fg=typer.colors.GREEN if status.available else typer.colors.YELLOW,
        )
    typer.echo(f"\n{ok}/{len(PROVIDER_NAMES)} providers configured")


# ------------------------------------------------------------------------ providers


@providers_app.command("list")
def providers_list() -> None:
    """List supported providers and their credential status."""
    for name in PROVIDER_NAMES:
        status = credential_status(name)
        typer.echo(f"  {name:<14} {'configured' if status.available else '-'}")


@providers_app.command("models")
def providers_models(
    provider: str,
    live: Annotated[
        bool, typer.Option("--live", help="Query the provider instead of the local catalog.")
    ] = False,
) -> None:
    """List a provider's models."""
    if provider not in PROVIDER_NAMES:
        _fail(f"unknown provider {provider!r} (known: {PROVIDER_NAMES_HINT})")
    from wai.providers.registry import catalog_for

    if not live:
        models = catalog_for(provider)
    else:
        config = load_config()
        adapter = create_provider(provider, config)

        async def _go() -> list[ModelInfo]:
            try:
                return await adapter.list_models()
            finally:
                await adapter.close()

        try:
            models = asyncio.run(_go())
        except WaiError as exc:
            _fail(str(exc))
    if not models:
        typer.echo("no models known; try --live")
        return
    for model in models:
        typer.echo(f"  {model.id:<50} {model.label}")


# ------------------------------------------------------------------------- sessions


@sessions_app.command("list")
def sessions_list(
    limit: Annotated[int, typer.Option("--limit", "-n")] = 20,
) -> None:
    """List saved sessions, newest first."""
    store = SessionStore(sessions_dir())
    found = store.list_sessions(limit=limit)
    if not found:
        typer.echo("no sessions yet")
        return
    for session in found:
        stamp = session.updated_at.astimezone().strftime("%Y-%m-%d %H:%M")
        typer.echo(f"  {session.id}  {stamp}  {session.model:<28} {session.title or '-'}")


@sessions_app.command("show")
def sessions_show(session_id: str) -> None:
    """Print a session transcript."""
    store = SessionStore(sessions_dir())
    if session_id == "last":
        latest = store.latest_id()
        if latest is None:
            _fail("no sessions yet")
        session_id = latest or ""
    try:
        session = store.load(session_id)
    except (FileNotFoundError, ValueError) as exc:
        _fail(str(exc))
        return
    for message in session.messages:
        typer.secho(f"\n{message.role.value}:", fg=typer.colors.CYAN, bold=True)
        typer.echo(message.text)


@sessions_app.command("rm")
def sessions_rm(session_id: str) -> None:
    """Delete a saved session."""
    store = SessionStore(sessions_dir())
    typer.echo(
        f"deleted {session_id}" if store.delete(session_id) else f"no such session: {session_id}"
    )


@tools_app.command("list")
def tools_list() -> None:
    """List the tools the model can call, and the workspace they operate in."""
    config = load_config()
    workspace = build_workspace(config)
    registry = default_registry()
    state = "enabled" if config.tools.enabled else "disabled"
    typer.echo(f"workspace: {workspace.root}")
    for extra in workspace.extra_roots:
        typer.echo(f"  + {extra}")
    typer.echo(f"tools:     {state} (max {config.tools.max_iterations} iterations/turn)\n")
    for tool in sorted(registry, key=lambda t: t.name):
        access = "read-only" if tool.read_only else "WRITES"
        typer.echo(f"  {tool.name:<12} [{access}]  {tool.description.split('.')[0]}.")
    if config.workspace.deny_secrets:
        typer.echo(
            "\ncredential-shaped files (.env, private keys) are blocked inside the workspace"
        )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
