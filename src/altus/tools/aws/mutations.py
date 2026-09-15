"""Changes to AWS. Every one is gated.

The Kubernetes gate can promise something this one cannot: there, every
mutation is dry-run against the API server and the prompt carries the server's
verdict. AWS offers that for 819 of 19,189 operations --- 4.3%, almost all EC2.

So the preflight assembles whatever *could* be checked and the prompt says
which of those it was. Never "dry run succeeded" when nothing of the kind
happened: a prompt that buys false confidence at the moment of consent is
worse than one that admits the gap.
"""

from __future__ import annotations

from typing import Any, ClassVar

from altus.cloud import aws as aws_api
from altus.cloud.base import Sensitivity
from altus.tools.aws.base import AwsMutatingTool
from altus.tools.base import ToolContext, ToolOutcome

#: What to read to show the current state before changing something. A "before"
#: is the closest AWS gets to a diff, and it is worth having where it is cheap.
BEFORE_READS: dict[str, tuple[str, str]] = {
    "ec2": ("ec2", "DescribeInstances"),
    "rds": ("rds", "DescribeDBInstances"),
    "s3": ("s3", "ListBuckets"),
    "lambda": ("lambda", "ListFunctions"),
    "iam": ("iam", "ListRoles"),
}

RECOVERABILITY: dict[Sensitivity, str] = {
    Sensitivity.PRIVILEGED: "this cannot be undone from here, and may not be undoable at all",
    Sensitivity.MUTATE: "AWS does not model an undo; reversing this is your own work",
}


class AwsWriteTool(AwsMutatingTool):
    name: ClassVar[str] = "aws_write"
    action: ClassVar[str] = "call"
    description: ClassVar[str] = (
        "Perform an AWS operation that changes something. Requires user "
        "approval every time. Call aws_explain first for the exact parameter "
        "names — they are case-sensitive and the SDK rejects unknown ones, and "
        "a rejected call still costs you a prompt. EC2 operations are dry-run "
        "against AWS first; for everything else AWS offers no preview, so the "
        "prompt shows a permission check and the resource's current state "
        "instead, and says so."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "service": {"type": "string"},
            "operation": {"type": "string", "description": "e.g. StopInstances"},
            "params": {"type": "object", "description": "Operation parameters, exactly named."},
            "region": {"type": "string"},
            "reason": {
                "type": "string",
                "description": "Why this change is wanted. Shown to the user.",
            },
        },
        "required": ["service", "operation"],
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        service = str(args.get("service", "")).strip()
        operation = str(args.get("operation", "")).strip()
        if not service or not operation:
            return ToolOutcome.error("service and operation are required")
        params = args.get("params") or {}
        if not isinstance(params, dict):
            return ToolOutcome.error("params must be an object")

        sensitivity = aws_api.classify(service, operation)
        if sensitivity in (Sensitivity.READ, Sensitivity.SENSITIVE_READ):
            return ToolOutcome.error(
                f"{service}:{operation} only reads. Use aws_call, which does not prompt.",
                summary="wrong tool",
            )

        provider = await self.provider(ctx)
        if isinstance(provider, ToolOutcome):
            return provider
        identity = await self.identity(ctx, provider)
        if isinstance(identity, ToolOutcome):
            return identity
        region = self.region_for(args, ctx, provider)

        permitted, preflight = await self.preflight(provider, service, operation, params, region)
        if not permitted:
            # Refused before anyone is asked. Prompting for something the
            # preflight already says will fail spends the user's attention on
            # a decision that does not exist.
            return ToolOutcome.error(
                f"{service}:{operation} would not succeed — {preflight}", summary="refused"
            )

        before = await self._before(provider, service, region)
        reason = str(args.get("reason") or "").strip()
        summary = f"{service}:{operation}\n" + _render_params(params)
        if reason:
            summary += f"\nreason given: {reason}"
        else:
            summary += "\n(no reason given)"

        refused = await self.confirm(
            ctx,
            service=service,
            operation=operation,
            account=identity["account"],
            region=region,
            summary=summary,
            preflight=preflight,
            recoverability=RECOVERABILITY.get(sensitivity, ""),
            before=before,
        )
        if refused is not None:
            return refused

        try:
            payload = await provider.call(service, operation, params, region=region)
        except Exception as exc:
            return ToolOutcome.error(f"{service}:{operation} failed: {exc}", summary="failed")

        payload = self.scrub(payload, ctx)
        return ToolOutcome(
            content=f"{service}:{operation} in {region} ({identity['account']})\n"
            + _render_params(payload),
            summary=f"{service}:{operation}",
        )

    async def _before(self, provider: Any, service: str, region: str) -> str:
        """A cheap read of what exists now.

        Best-effort by design: a failure here is not worth blocking a change
        the user may still want to make, so it comes back empty and the prompt
        simply has one less thing in it.
        """
        read = BEFORE_READS.get(service)
        if read is None:
            return ""
        try:
            payload = await provider.call(read[0], read[1], region=region)
        except Exception:
            return ""
        counts = [
            f"{len(value)} {key}" for key, value in payload.items() if isinstance(value, list)
        ]
        return ", ".join(counts[:4])


def _render_params(payload: dict[str, Any]) -> str:
    import yaml

    if not payload:
        return "  (no parameters)"
    text = yaml.safe_dump(payload, default_flow_style=False, sort_keys=True, allow_unicode=True)
    return "\n".join(f"  {line}" for line in str(text).splitlines()[:40])
