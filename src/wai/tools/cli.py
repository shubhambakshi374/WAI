"""The CLI fallback: kubectl, helm, kustomize.

Last resort, not the front door. Everything the native tools can express should
go through them --- they return structured objects, dry-run before mutating,
redact secrets, and classify what they are about to do. A shelled-out command
gives up all four, so the tool description asks the model to justify reaching
for it and the justification is shown to the user in the prompt.

The security property that matters here is simple and absolute: **a command is
an argv list handed to execve, never a string handed to a shell.** There is no
``sh -c`` anywhere in this module. A model-generated argument containing
``; rm -rf /`` arrives at kubectl as one literal argument and does nothing.
"""

from __future__ import annotations

import asyncio
import shutil
from typing import Any, ClassVar

from wai.cloud.base import Sensitivity
from wai.tools.approval import ApprovalRequest, Decision
from wai.tools.base import BaseTool, ToolContext, ToolOutcome

MAX_OUTPUT = 40_000
DEFAULT_TIMEOUT = 120.0

#: Subcommands that only read. Everything not listed is treated as a change,
#: so a subcommand we have never heard of is gated rather than waved through.
READ_SUBCOMMANDS: dict[str, frozenset[str]] = {
    "kubectl": frozenset(
        {
            "get",
            "describe",
            "logs",
            "explain",
            "api-resources",
            "api-versions",
            "version",
            "cluster-info",
            "top",
            "diff",
            "events",
            "auth",
        }
    ),
    "helm": frozenset({"list", "ls", "status", "get", "history", "show", "search", "version"}),
    "kustomize": frozenset({"build", "version", "cfg"}),
}

#: Subcommands that are privileged whatever else they look like.
PRIVILEGED_SUBCOMMANDS: dict[str, frozenset[str]] = {
    "kubectl": frozenset(
        {"exec", "attach", "port-forward", "proxy", "cp", "drain", "cordon", "uncordon", "taint"}
    ),
    "helm": frozenset({"plugin"}),
    "kustomize": frozenset(),
}

#: Flags we set ourselves from session state. A model supplying its own is
#: either confused or retargeting the command at a cluster the user did not
#: approve, and the prompt would then name the wrong blast radius.
RESERVED_FLAGS = (
    "--kubeconfig",
    "--context",
    "--kube-context",
    "--as",
    "--as-group",
    "--token",
    # AWS: these retarget the command at another account or identity, which
    # would make the approval prompt name the wrong blast radius.
    "--profile",
    "--region",
    "--endpoint-url",
    "--ca-bundle",
)


class CliTool(BaseTool):
    """One allowlisted binary."""

    binary: ClassVar[str] = ""
    read_only: ClassVar[bool] = False
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "args": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Arguments, one per element. Not a shell line.",
            },
            "reason": {
                "type": "string",
                "description": "Why the native k8s_* tools cannot do this. Shown to the user.",
            },
        },
        "required": ["args"],
    }

    def classify(self, args: list[str]) -> Sensitivity:
        positional = [a for a in args if not a.startswith("-")]
        subcommand = positional[0] if positional else ""
        if subcommand in PRIVILEGED_SUBCOMMANDS.get(self.binary, frozenset()):
            return Sensitivity.PRIVILEGED
        if self.binary == "aws":
            # Ask the same classifier the SDK path uses, rather than keeping a
            # second table that would drift from it. Two answers for
            # `terminate-instances` depending on which door it came through is
            # exactly the kind of gap a gate is supposed not to have.
            return _classify_aws(positional)
        if any(a == "--raw" for a in args):
            # kubectl --raw reaches any API path with any verb, unclassified.
            return Sensitivity.PRIVILEGED
        if subcommand in READ_SUBCOMMANDS.get(self.binary, frozenset()):
            return Sensitivity.READ
        return Sensitivity.MUTATE

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        raw = args.get("args")
        if not isinstance(raw, list) or not raw or not all(isinstance(a, str) for a in raw):
            return ToolOutcome.error(
                "args must be a non-empty array of strings, one argument per element. "
                "There is no shell, so pipes and redirection do not work here."
            )
        argv = [str(a) for a in raw]

        # Read from the context, not from a `cli` attribute that never existed
        # --- the old check was reading None and passing every time, so the
        # allowlist was enforced only by registration.
        allowlist = getattr(ctx.cloud, "cli_allowlist", ()) or ()
        if allowlist and self.binary not in allowlist:
            return ToolOutcome.error(
                f"{self.binary} is not in [cloud] cli_allowlist", summary="not allowed"
            )
        offenders = [a for a in argv if a.split("=")[0] in RESERVED_FLAGS]
        if offenders:
            return ToolOutcome.error(
                f"{', '.join(offenders)} is set by WAI from the session's context — "
                "remove it. Switch cluster with k8s_use_context instead, which asks first.",
                summary="reserved flag",
            )
        path = shutil.which(self.binary)
        if path is None:
            return ToolOutcome.error(
                f"{self.binary} is not installed or not on PATH", summary="not found"
            )

        context_name = ctx.cloud.kube_context or ""
        full = [*argv]
        if context_name and self.binary in {"kubectl", "helm"}:
            full = [*argv, "--context", context_name]
        elif self.binary == "aws":
            region = getattr(ctx.cloud, "aws_region", "") or ""
            if region:
                full = [*argv, "--region", region]

        sensitivity = self.classify(argv)
        if sensitivity.needs_approval:
            reason = str(args.get("reason") or "").strip()
            decision = await ctx.approvals.request(
                ApprovalRequest(
                    tool=self.name,
                    action="run",
                    path=f"{self.binary} {' '.join(argv)}",
                    target=f"cluster {context_name or '(kubeconfig default)'}",
                    diff=f"$ {self.binary} {' '.join(full)}\n"
                    + (
                        f"\nreason given: {reason}"
                        if reason
                        else "\n(no reason given for using the CLI over the native tools)"
                    ),
                    dry_run="a shelled-out command is not dry-run; WAI cannot see what it will do",
                    recoverability="unknown — WAI does not model what this command changes",
                    destructive=True,
                    sensitivity=sensitivity,
                )
            )
            if decision is Decision.DENY:
                return ToolOutcome.rejected("The user rejected this command.")

        try:
            code, out, err = await _execute(path, full, timeout_seconds=DEFAULT_TIMEOUT)
        except Exception as exc:
            return ToolOutcome.error(f"{self.binary} failed to start: {exc}", summary="failed")

        from wai.cloud.redact import redact_text

        body = redact_text(out, enabled=ctx.cloud.redact_secrets)
        errors = redact_text(err, enabled=ctx.cloud.redact_secrets)
        if len(body) > MAX_OUTPUT:
            body = body[:MAX_OUTPUT] + "\n[truncated]"
        parts = [f"$ {self.binary} {' '.join(full)}", body or "(no output)"]
        if errors.strip():
            parts.append(f"stderr:\n{errors}")
        return ToolOutcome(
            content="\n".join(parts),
            is_error=code != 0,
            summary=f"{self.binary} exit {code}",
        )


#: `aws s3 ls` and friends, whose verbs are not the API operation name.
AWS_SHORTHAND: dict[tuple[str, str], str] = {
    ("s3", "ls"): "ListBuckets",
    ("s3", "cp"): "PutObject",
    ("s3", "mv"): "PutObject",
    ("s3", "rm"): "DeleteObject",
    ("s3", "rb"): "DeleteBucket",
    ("s3", "mb"): "CreateBucket",
    ("s3", "sync"): "PutObject",
}


def _classify_aws(positional: list[str]) -> Sensitivity:
    """`aws <service> <verb>` through the SDK's own classifier.

    The CLI's kebab-case verb is the API operation name with hyphens, so the
    two can share one table --- which is the point. An unrecognised shape falls
    through to the classifier's own fail-closed answer.
    """
    from wai.cloud.aws import classify

    if len(positional) < 2:
        return Sensitivity.MUTATE
    service, verb = positional[0], positional[1]
    operation = AWS_SHORTHAND.get((service, verb)) or "".join(
        part.capitalize() for part in verb.split("-")
    )
    return classify(service, operation)


async def _execute(path: str, argv: list[str], *, timeout_seconds: float) -> tuple[int, str, str]:
    """create_subprocess_exec, never create_subprocess_shell.

    The distinction is the whole security model of this module: exec takes an
    argument vector, so no part of ``argv`` is ever parsed as syntax.
    """
    proc = await asyncio.create_subprocess_exec(
        path,
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout_seconds)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return 124, "", f"timed out after {timeout_seconds:g}s"
    return (
        proc.returncode or 0,
        out.decode("utf-8", errors="replace"),
        err.decode("utf-8", errors="replace"),
    )


class KubectlTool(CliTool):
    name: ClassVar[str] = "k8s_kubectl"
    binary: ClassVar[str] = "kubectl"
    description: ClassVar[str] = (
        "Run kubectl when — and only when — the native k8s_* tools cannot "
        "express what you need. They are better: structured output, a server "
        "dry run before every change, secret redaction, and an approval prompt "
        "that names what will happen. Say in `reason` why they are not enough; "
        "the user sees it. Arguments are a list, not a shell line, and "
        "--context is supplied by WAI."
    )


class HelmTool(CliTool):
    name: ClassVar[str] = "helm"
    binary: ClassVar[str] = "helm"
    description: ClassVar[str] = (
        "Run helm. A release install or upgrade changes many objects at once "
        "and WAI cannot dry-run it for you, so prefer `helm template` piped "
        "into k8s_apply when you want the change reviewed object by object."
    )


class AwsCliTool(CliTool):
    name: ClassVar[str] = "aws_cli"
    binary: ClassVar[str] = "aws"
    description: ClassVar[str] = (
        "Run the AWS CLI when — and only when — aws_call cannot express what "
        "you need. It almost always can: aws_explain gives you the exact "
        "parameter names, the response comes back structured and redacted, and "
        "the approval prompt says what was checked. The CLI gives up all four. "
        "Say in `reason` why it is necessary; the user sees it. --profile and "
        "--region are supplied by WAI."
    )


class KustomizeTool(CliTool):
    name: ClassVar[str] = "kustomize"
    binary: ClassVar[str] = "kustomize"
    description: ClassVar[str] = (
        "Run kustomize, usually `build` to render an overlay. Rendering is a "
        "read; feed the result to k8s_apply so the change is reviewed."
    )


def cli_tools() -> list[CliTool]:
    return [KubectlTool(), HelmTool(), KustomizeTool(), AwsCliTool()]
