"""Running things inside containers, and tunnels out of the cluster.

Every tool here is PRIVILEGED, and the class can be switched off entirely in
``[cloud.k8s]``. These are the operations where the sandbox stops being about
Kubernetes objects: an exec can reach whatever the container can reach, and a
port-forward puts cluster-internal services on this machine's loopback.

Two rules hold throughout. Commands are argv lists, never strings, so a shell
metacharacter in a model-generated argument is inert rather than a second
command. And nothing runs unbounded: an exec has a deadline, a port-forward is
session-scoped and named so it can be stopped.
"""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

from wai.cloud.redact import redact_text
from wai.tools.base import ToolContext, ToolOutcome
from wai.tools.k8s.base import K8sMutatingTool

MAX_OUTPUT = 40_000


def _argv(raw: Any) -> list[str] | None:
    """A command is a list. A string would have to be split, and splitting is
    where `sh -c` creeps back in."""
    if not isinstance(raw, list) or not raw:
        return None
    if not all(isinstance(part, str) for part in raw):
        return None
    return [str(part) for part in raw]


class K8sExecTool(K8sMutatingTool):
    name: ClassVar[str] = "k8s_exec"
    action: ClassVar[str] = "exec in"
    verb: ClassVar[str] = "get"
    subresource: ClassVar[str] = "exec"
    description: ClassVar[str] = (
        "Run a command inside a running container and return what it printed. "
        "The command is a list of arguments, not a shell line: ['cat', "
        "'/etc/nginx/nginx.conf'], not 'cat /etc/nginx/nginx.conf | grep x'. "
        "There is no shell, no pipes and no TTY. Prefer k8s_logs, k8s_get and "
        "k8s_events — reach for this only when the answer is genuinely inside "
        "the container's filesystem or process table. Requires a typed "
        "confirmation every time."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "pod": {"type": "string"},
            "command": {
                "type": "array",
                "items": {"type": "string"},
                "description": 'argv, e.g. ["ls", "-la", "/data"].',
            },
            "namespace": {"type": "string"},
            "container": {"type": "string", "description": "Required if the pod has several."},
            "stdin": {"type": "string", "description": "Fed to the command's standard input."},
            "reason": {
                "type": "string",
                "description": "Why a native tool cannot answer this. Shown to the user.",
            },
        },
        "required": ["pod", "command"],
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        pod = str(args.get("pod", "")).strip()
        command = _argv(args.get("command"))
        if not pod:
            return ToolOutcome.error("pod is required")
        if command is None:
            return ToolOutcome.error(
                "command must be a non-empty array of strings — there is no shell here, "
                'so ["sh", "-c", "..."] is the only way to get one and it is discouraged'
            )

        resolved = await self.client(ctx)
        if isinstance(resolved, ToolOutcome):
            return resolved
        client, context_name = resolved
        namespace = str(args.get("namespace") or "default")
        container = str(args.get("container") or "").strip() or None
        shown = " ".join(command)
        reason = str(args.get("reason") or "").strip()

        refused = await self.confirm(
            ctx,
            client=client,
            context_name=context_name,
            kind="Pod",
            name=pod,
            namespace=namespace,
            diff=(
                f"run in {pod}{f' ({container})' if container else ''}:\n"
                f"  {shown}\n" + (f"\nreason given: {reason}" if reason else "\n(no reason given)")
            ),
            dry_run="an exec cannot be dry-run; it either runs or it does not",
            recoverability=("whatever this command does inside the container is not undone by WAI"),
        )
        if refused is not None:
            return refused

        timeout = float(getattr(ctx.cloud, "exec_timeout", 60) or 60)
        try:
            out, err = await client.exec_pod(
                pod,
                namespace,
                command,
                container=container,
                stdin=args.get("stdin"),
                timeout_seconds=timeout,
            )
        except Exception as exc:
            return ToolOutcome.error(f"exec failed: {exc}", summary="failed")

        body = redact_text(out, enabled=ctx.cloud.redact_secrets)
        errors = redact_text(err, enabled=ctx.cloud.redact_secrets)
        if len(body) > MAX_OUTPUT:
            body = body[:MAX_OUTPUT] + "\n[truncated]"
        parts = [f"$ {shown}", body or "(no output)"]
        if errors.strip():
            parts.append(f"stderr:\n{errors}")
        return ToolOutcome(content="\n".join(parts), summary=f"exec in {pod}")


class K8sCpTool(K8sMutatingTool):
    name: ClassVar[str] = "k8s_cp"
    action: ClassVar[str] = "copy"
    verb: ClassVar[str] = "get"
    subresource: ClassVar[str] = "exec"
    description: ClassVar[str] = (
        "Copy a single file out of a container into the workspace, or from the "
        "workspace into a container. Implemented with exec and tar, exactly as "
        "kubectl does, so the container needs tar on its PATH. Requires a typed "
        "confirmation."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "pod": {"type": "string"},
            "namespace": {"type": "string"},
            "container": {"type": "string"},
            "direction": {"type": "string", "enum": ["from_pod", "to_pod"]},
            "remote_path": {"type": "string", "description": "Absolute path in the container."},
            "local_path": {"type": "string", "description": "Workspace-relative path."},
        },
        "required": ["pod", "direction", "remote_path", "local_path"],
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        pod = str(args.get("pod", "")).strip()
        direction = str(args.get("direction", "")).strip()
        remote = str(args.get("remote_path", "")).strip()
        local = str(args.get("local_path", "")).strip()
        if not pod or not remote or not local:
            return ToolOutcome.error("pod, remote_path and local_path are required")
        if direction not in {"from_pod", "to_pod"}:
            return ToolOutcome.error("direction must be from_pod or to_pod")
        if not remote.startswith("/"):
            return ToolOutcome.error("remote_path must be absolute")

        # The workspace guard owns every local path, so a copy out of a
        # container cannot write outside the workspace or over a denylisted
        # name --- the same rule that stops write_file touching ~/.ssh.
        try:
            target = ctx.workspace.resolve(local)
        except Exception as exc:
            return ToolOutcome.error(str(exc), summary="denied")
        if direction == "to_pod" and not target.is_file():
            return ToolOutcome.error(f"{local} is not a file in the workspace")

        resolved = await self.client(ctx)
        if isinstance(resolved, ToolOutcome):
            return resolved
        client, context_name = resolved
        namespace = str(args.get("namespace") or "default")
        container = str(args.get("container") or "").strip() or None

        arrow = (
            f"{pod}:{remote} -> {target}"
            if direction == "from_pod"
            else (f"{target} -> {pod}:{remote}")
        )
        refused = await self.confirm(
            ctx,
            client=client,
            context_name=context_name,
            kind="Pod",
            name=pod,
            namespace=namespace,
            diff=f"copy {arrow}",
            dry_run="a copy cannot be dry-run",
            recoverability=(
                f"writes {target} in your workspace"
                if direction == "from_pod"
                else "overwrites the path inside the container"
            ),
        )
        if refused is not None:
            return refused

        try:
            if direction == "from_pod":
                message = await self._out(client, pod, namespace, container, remote, target, ctx)
            else:
                message = await self._in(client, pod, namespace, container, remote, target, ctx)
        except Exception as exc:
            return ToolOutcome.error(f"copy failed: {exc}", summary="failed")
        return ToolOutcome(content=f"{message} ({context_name})", summary="copied")

    async def _out(
        self,
        client: Any,
        pod: str,
        namespace: str,
        container: str | None,
        remote: str,
        target: Any,
        ctx: ToolContext,
    ) -> str:
        import base64

        # base64 rather than raw tar: the exec channel is text-framed, and a
        # binary tar stream through it arrives corrupted.
        out, err = await client.exec_pod(
            pod,
            namespace,
            ["sh", "-c", f"base64 < {_quote(remote)}"],
            container=container,
            timeout_seconds=float(getattr(ctx.cloud, "exec_timeout", 60) or 60),
        )
        if not out.strip():
            raise RuntimeError(err.strip() or f"{remote} produced nothing")
        data = base64.b64decode("".join(out.split()))
        await asyncio.to_thread(target.write_bytes, data)
        return f"copied {len(data)} bytes from {pod}:{remote} to {target}"

    async def _in(
        self,
        client: Any,
        pod: str,
        namespace: str,
        container: str | None,
        remote: str,
        target: Any,
        ctx: ToolContext,
    ) -> str:
        import base64

        data = await asyncio.to_thread(target.read_bytes)
        encoded = base64.b64encode(data).decode("ascii")
        _out, err = await client.exec_pod(
            pod,
            namespace,
            ["sh", "-c", f"base64 -d > {_quote(remote)}"],
            container=container,
            stdin=encoded,
            timeout_seconds=float(getattr(ctx.cloud, "exec_timeout", 60) or 60),
        )
        if err.strip():
            raise RuntimeError(err.strip())
        return f"copied {len(data)} bytes from {target} to {pod}:{remote}"


def _quote(path: str) -> str:
    """The one place a shell string is unavoidable --- `sh -c` is how tar and
    base64 are reached at all. So the path is quoted properly rather than
    interpolated, and a single quote inside it cannot end the quoting."""
    return "'" + path.replace("'", "'\\''") + "'"


class PortForwards:
    """Session-scoped registry of open tunnels.

    Held here rather than in a tool instance because the tools are stateless
    singletons, and because a tunnel that outlives the session is a hole
    nobody remembers opening. ``close_all`` runs on session teardown.
    """

    def __init__(self) -> None:
        self._open: dict[str, tuple[Any, str]] = {}

    def add(self, key: str, handle: Any, describe: str) -> None:
        self._open[key] = (handle, describe)

    def describe(self) -> list[str]:
        return [f"{key}: {text}" for key, (_handle, text) in sorted(self._open.items())]

    def close(self, key: str) -> bool:
        entry = self._open.pop(key, None)
        if entry is None:
            return False
        self._shut(entry[0])
        return True

    def close_all(self) -> int:
        count = len(self._open)
        for handle, _ in self._open.values():
            self._shut(handle)
        self._open.clear()
        return count

    @staticmethod
    def _shut(handle: Any) -> None:
        import contextlib

        with contextlib.suppress(Exception):
            handle.close()

    def __contains__(self, key: object) -> bool:
        return key in self._open

    def __len__(self) -> int:
        return len(self._open)


class K8sPortForwardTool(K8sMutatingTool):
    name: ClassVar[str] = "k8s_port_forward"
    action: ClassVar[str] = "forward a port from"
    verb: ClassVar[str] = "get"
    subresource: ClassVar[str] = "portforward"
    description: ClassVar[str] = (
        "Open a tunnel from this machine to a port on a pod, so a local client "
        "can reach a cluster-internal service. action=start returns an id; "
        "action=stop closes it; action=list shows what is open. Every tunnel "
        "closes when the session ends. Requires a typed confirmation."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["start", "stop", "list"]},
            "pod": {"type": "string"},
            "namespace": {"type": "string"},
            "remote_port": {"type": "integer"},
            "local_port": {"type": "integer", "description": "Defaults to the remote port."},
            "id": {"type": "string", "description": "For action=stop."},
        },
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        action = str(args.get("action") or "start").casefold()
        forwards = getattr(ctx.cloud, "port_forwards", None)
        if forwards is None:
            return ToolOutcome.error(
                "this session cannot hold port forwards open", summary="unsupported"
            )

        if action == "list":
            open_now = forwards.describe()
            return ToolOutcome(
                content="\n".join(open_now) or "no port forwards are open",
                summary=f"{len(open_now)} open",
            )
        if action == "stop":
            key = str(args.get("id", "")).strip()
            if not key:
                return ToolOutcome.error("id is required to stop a forward")
            closed = forwards.close(key)
            return ToolOutcome(
                content=f"closed {key}" if closed else f"no forward called {key}",
                summary="closed" if closed else "not found",
                is_error=not closed,
            )
        if action != "start":
            return ToolOutcome.error(f"unknown action {action!r}")

        pod = str(args.get("pod", "")).strip()
        if not pod:
            return ToolOutcome.error("pod is required")
        try:
            remote_port = int(args.get("remote_port") or 0)
        except (TypeError, ValueError):
            return ToolOutcome.error("remote_port must be a whole number")
        if not 1 <= remote_port <= 65535:
            return ToolOutcome.error("remote_port must be between 1 and 65535")
        local_port = int(args.get("local_port") or remote_port)

        resolved = await self.client(ctx)
        if isinstance(resolved, ToolOutcome):
            return resolved
        client, context_name = resolved
        namespace = str(args.get("namespace") or "default")
        key = f"{namespace}/{pod}:{remote_port}"
        if key in forwards:
            return ToolOutcome(content=f"{key} is already open", summary="already open")

        refused = await self.confirm(
            ctx,
            client=client,
            context_name=context_name,
            kind="Pod",
            name=pod,
            namespace=namespace,
            diff=(
                f"open localhost:{local_port} -> {pod}:{remote_port}\n"
                "anything on this machine that can reach that port can then reach "
                "the cluster-internal service behind it"
            ),
            dry_run="a tunnel cannot be dry-run",
            recoverability=(f"closes when the session ends, or with action=stop id={key}"),
        )
        if refused is not None:
            return refused

        try:
            handle = await client.port_forward(pod, namespace, remote_port, local_port)
        except Exception as exc:
            return ToolOutcome.error(f"could not open the tunnel: {exc}", summary="failed")

        forwards.add(key, handle, f"localhost:{local_port} -> {pod}:{remote_port} ({context_name})")
        return ToolOutcome(
            content=f"forwarding localhost:{local_port} to {pod}:{remote_port} in "
            f"{namespace} ({context_name})\nstop it with action=stop id={key}",
            summary=f"forwarding {local_port}",
        )
