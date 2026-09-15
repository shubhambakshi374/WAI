"""Reads that need no approval: identity, introspection, and any read call.

AWS is 431 services and 19,189 operations, so there is no tool per operation.
There is one generic call, and the service models are what make it usable: the
model can be handed an operation's exact parameter contract instead of guessing
at it, which is also the argument for preferring this over the CLI.
"""

from __future__ import annotations

from typing import Any, ClassVar

from wai.cloud import aws as aws_api
from wai.cloud.base import Sensitivity
from wai.core.visuals import Table
from wai.tools.aws.base import AwsTool
from wai.tools.base import ToolContext, ToolOutcome

MAX_CONTENT = 24_000


def _as_yaml(payload: Any) -> str:
    import yaml

    return str(
        yaml.safe_dump(payload, default_flow_style=False, sort_keys=False, allow_unicode=True)
    )


class AwsWhoamiTool(AwsTool):
    name: ClassVar[str] = "aws_whoami"
    description: ClassVar[str] = (
        "Which AWS account and identity this session is using, and the default "
        "region. Call this before anything else if unsure where you are — every "
        "other AWS tool acts in this account."
    )
    input_schema: ClassVar[dict[str, Any]] = {"type": "object", "properties": {}}

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        provider = await self.provider(ctx)
        if isinstance(provider, ToolOutcome):
            return provider
        identity = await self.identity(ctx, provider)
        if isinstance(identity, ToolOutcome):
            return identity

        region = self.region_for(args, ctx, provider)
        rules = ctx.cloud.protection
        target = aws_api.target_for(identity["account"], region)
        protected = bool(rules and rules.matches(target))
        lines = [
            f"account  {identity['account']}",
            f"arn      {identity['arn']}",
            f"region   {region}",
        ]
        if protected:
            lines.append("⚠ PROTECTED: changes here need a typed confirmation")
        return ToolOutcome(content="\n".join(lines), summary=identity["account"])


class AwsServicesTool(AwsTool):
    name: ClassVar[str] = "aws_services"
    description: ClassVar[str] = (
        "List the AWS services this SDK can call, optionally filtered. Use it "
        "to find the right service name before aws_explain or aws_call."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {"filter": {"type": "string", "description": "Substring to match."}},
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        needle = str(args.get("filter", "")).casefold()
        names = [s for s in aws_api.available_services() if not needle or needle in s]
        shown = names[:200]
        body = "\n".join(shown) or "(none matched)"
        if len(names) > len(shown):
            body += f"\n[{len(names) - len(shown)} more; narrow the filter]"
        return ToolOutcome(content=body, summary=f"{len(names)} services")


class AwsExplainTool(AwsTool):
    name: ClassVar[str] = "aws_explain"
    description: ClassVar[str] = (
        "The exact parameter contract for an operation, from the SDK's own "
        "service model: which members it takes, which are required, what they "
        "mean, and how sensitive the operation is. Call this BEFORE aws_call — "
        "it is authoritative, and it is why using the SDK beats guessing CLI "
        "flag names. Omit `operation` to list a service's operations."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "service": {"type": "string", "description": "e.g. ec2, s3, rds"},
            "operation": {"type": "string", "description": "e.g. DescribeInstances"},
            "filter": {"type": "string", "description": "Substring, when listing operations."},
        },
        "required": ["service"],
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        service = str(args.get("service", "")).strip()
        if not service:
            return ToolOutcome.error("service is required")
        operation = str(args.get("operation", "")).strip()

        try:
            if not operation:
                return self._list(service, str(args.get("filter", "")))
            described = aws_api.describe_operation(service, operation)
        except Exception as exc:
            return ToolOutcome.error(f"no such service or operation: {exc}", summary="unknown")

        lines = [
            f"{service}:{operation}  [{described['sensitivity']}]",
            f"  {described['documentation']}",
            "",
        ]
        for name, info in sorted(described["parameters"].items()):
            mark = "*" if info["required"] else " "
            lines.append(f"  {mark} {name:<28} {info['type']:<12} {info['documentation'][:90]}")
        lines.append("\n  * = required")
        return ToolOutcome(content="\n".join(lines), summary=f"{service}:{operation}")

    def _list(self, service: str, needle: str) -> ToolOutcome:
        found = aws_api.operations(service)
        lowered = needle.casefold()
        if lowered:
            found = [op for op in found if lowered in op.casefold()]
        rows = [[op, aws_api.classify(service, op).value] for op in found[:120]]
        table = Table(
            title=f"{service} operations", columns=["operation", "sensitivity"], rows=rows
        )
        body = table.to_text(max_rows=120)
        if len(found) > 120:
            body += f"\n[{len(found) - 120} more; narrow the filter]"
        return ToolOutcome(content=body, summary=f"{len(found)} operations", visual=table)


class AwsCallTool(AwsTool):
    name: ClassVar[str] = "aws_call"
    description: ClassVar[str] = (
        "Call any read-only AWS operation and get the response back. Paginated "
        "and capped automatically. Call aws_explain first for the parameter "
        "names — they are case-sensitive and the SDK rejects unknown ones. "
        "Anything that changes something goes through aws_write, which asks."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "service": {"type": "string"},
            "operation": {"type": "string", "description": "e.g. DescribeInstances"},
            "params": {"type": "object", "description": "Operation parameters, exactly named."},
            "region": {"type": "string", "description": "Defaults to the session's region."},
        },
        "required": ["service", "operation"],
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        service = str(args.get("service", "")).strip()
        operation = str(args.get("operation", "")).strip()
        if not service or not operation:
            return ToolOutcome.error("service and operation are required")

        sensitivity = aws_api.classify(service, operation)
        if sensitivity not in (Sensitivity.READ, Sensitivity.SENSITIVE_READ):
            return ToolOutcome.error(
                f"{service}:{operation} classifies as {sensitivity.value} — it changes "
                "something. Use aws_write, which shows you what it can check first.",
                summary="wrong tool",
            )

        provider = await self.provider(ctx)
        if isinstance(provider, ToolOutcome):
            return provider
        region = self.region_for(args, ctx, provider)
        params = args.get("params") or {}
        if not isinstance(params, dict):
            return ToolOutcome.error("params must be an object")

        try:
            payload = await provider.call(service, operation, params, region=region)
        except Exception as exc:
            return ToolOutcome.error(f"{service}:{operation} failed: {exc}", summary="failed")

        payload = self.scrub(payload, ctx)
        body = _as_yaml(payload)
        if len(body) > MAX_CONTENT:
            body = body[:MAX_CONTENT] + "\n[truncated]"
        return ToolOutcome(
            content=f"# {service}:{operation} in {region}\n{body}",
            summary=f"{service}:{operation}",
        )


class AwsCanITool(AwsTool):
    name: ClassVar[str] = "aws_can_i"
    description: ClassVar[str] = (
        "Ask IAM whether this identity may perform an action, BEFORE attempting "
        "it. Cheaper and clearer than discovering an AccessDenied halfway "
        "through a plan. Needs iam:SimulatePrincipalPolicy, and says so when "
        "that is itself denied."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "actions": {
                "type": "array",
                "items": {"type": "string"},
                "description": 'e.g. ["s3:DeleteBucket", "ec2:TerminateInstances"]',
            },
            "resources": {
                "type": "array",
                "items": {"type": "string"},
                "description": "ARNs the actions would target. Omit for any.",
            },
            "region": {"type": "string"},
        },
        "required": ["actions"],
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        actions = args.get("actions")
        if not isinstance(actions, list) or not actions:
            return ToolOutcome.error("actions must be a non-empty array of service:Operation")

        provider = await self.provider(ctx)
        if isinstance(provider, ToolOutcome):
            return provider
        identity = await self.identity(ctx, provider)
        if isinstance(identity, ToolOutcome):
            return identity
        region = self.region_for(args, ctx, provider)

        params: dict[str, Any] = {
            "PolicySourceArn": identity["arn"],
            "ActionNames": [str(a) for a in actions],
        }
        if args.get("resources"):
            params["ResourceArns"] = [str(r) for r in args["resources"]]

        try:
            result = await provider.call("iam", "SimulatePrincipalPolicy", params, region=region)
        except Exception as exc:
            return ToolOutcome.error(
                f"the simulation could not run: {exc}. This needs "
                "iam:SimulatePrincipalPolicy, which plenty of working roles lack.",
                summary="cannot check",
            )

        rows = [
            [
                str(entry.get("EvalActionName", "")),
                str(entry.get("EvalDecision", "")),
                str(entry.get("EvalResourceName", "*")),
            ]
            for entry in result.get("EvaluationResults") or []
        ]
        if not rows:
            return ToolOutcome.error("the simulation returned no decisions", summary="no answer")
        allowed = sum(1 for row in rows if row[1] == "allowed")
        table = Table(
            title=f"permissions for {identity['arn']}",
            columns=["action", "decision", "resource"],
            rows=rows,
        )
        return ToolOutcome(
            content=table.to_text(max_rows=60),
            summary=f"{allowed}/{len(rows)} allowed",
            visual=table,
        )


class AwsRegionsTool(AwsTool):
    name: ClassVar[str] = "aws_regions"
    description: ClassVar[str] = "Regions available for a service. Defaults to ec2."
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {"service": {"type": "string"}},
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        provider = await self.provider(ctx)
        if isinstance(provider, ToolOutcome):
            return provider
        service = str(args.get("service") or "ec2")
        try:
            found = await provider.regions(service)
        except Exception as exc:
            return ToolOutcome.error(f"could not list regions: {exc}", summary="failed")
        return ToolOutcome(
            content="\n".join(found) or "(none)",
            summary=f"{len(found)} regions",
        )
