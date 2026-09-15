"""Changes to Azure. Every one is gated.

Kubernetes can promise something AWS could not: every mutation is dry-run
against the API server and the prompt carries the server's verdict. AWS offers
that for 4.3% of its operations, so its prompt had to admit the gap.

Azure sits between the two, and which of the three mechanisms ran decides what
the prompt may claim:

  PUT/PATCH   What-If. A real server-side, property-level diff --- the closest
              any cloud gets to `kubectl diff`.
  DELETE      No preview exists. A lock check, which is Azure's own answer to
              "would this actually work", plus an RBAC check.
  POST        No preview exists. An RBAC check.

Never "what-if says no changes" when no What-If ran: a prompt that buys false
confidence at the moment of consent is worse than one that admits the gap.
"""

from __future__ import annotations

from typing import Any, ClassVar

from wai.cloud import azure as arm
from wai.cloud.base import Sensitivity
from wai.tools.azure.base import AzureMutatingTool
from wai.tools.base import ToolContext, ToolOutcome

RECOVERABILITY: dict[Sensitivity, str] = {
    Sensitivity.PRIVILEGED: "this cannot be undone from here, and may not be undoable at all",
    Sensitivity.MUTATE: "ARM does not model an undo; reversing this is your own work",
}


class _ArmChange(AzureMutatingTool):
    """Everything the three share: resolve the id, classify, ask, then send."""

    verb: ClassVar[str] = ""
    method: ClassVar[str] = ""

    async def prepare(
        self, args: dict[str, Any], ctx: ToolContext
    ) -> tuple[Any, dict[str, str], str, str] | ToolOutcome:
        """Provider, the parsed id, the operation string and the api-version."""
        resource_id = str(args.get("id", "")).strip()
        if not resource_id:
            return ToolOutcome.error("id is required: the full ARM resource id")
        parts = arm.parse_resource_id(resource_id)
        if not parts["namespace"] or not parts["type"]:
            return ToolOutcome.error(
                f"{resource_id!r} is not an ARM resource id. It looks like "
                "/subscriptions/<id>/resourceGroups/<rg>/providers/<namespace>/<type>/<name>.",
                summary="bad id",
            )

        provider = await self.provider(ctx)
        if isinstance(provider, ToolOutcome):
            return provider

        operation = arm.operation_for(resource_id, self.verb, str(args.get("action", "")))
        version = str(args.get("api_version", "")).strip()
        if not version:
            try:
                version = await provider.api_version_for(parts["namespace"], parts["type"])
            except Exception as exc:
                return ToolOutcome.error(str(exc), summary="no api-version")
        return provider, parts, operation, version

    async def gate(
        self,
        args: dict[str, Any],
        ctx: ToolContext,
        *,
        resource_id: str,
        parts: dict[str, str],
        operation: str,
        summary: str,
        preflight: str,
    ) -> ToolOutcome | None:
        reason = str(args.get("reason") or "").strip()
        summary += f"\nreason given: {reason}" if reason else "\n(no reason given)"
        return await self.confirm(
            ctx,
            operation=operation,
            scope=resource_id,
            subscription=parts["subscription"],
            region="",
            group=parts["group"],
            summary=summary,
            preflight=preflight,
            recoverability=RECOVERABILITY.get(arm.classify(operation, resource_id), ""),
        )


_SHARED_PROPERTIES: dict[str, Any] = {
    "id": {"type": "string", "description": "Full ARM resource id."},
    "api_version": {"type": "string", "description": "Only if you must override."},
    "reason": {
        "type": "string",
        "description": "Why this change is wanted. Shown to the user.",
    },
}


class AzureWriteTool(_ArmChange):
    name: ClassVar[str] = "azure_write"
    action: ClassVar[str] = "write"
    verb: ClassVar[str] = "write"
    method: ClassVar[str] = "PUT"
    description: ClassVar[str] = (
        "Create or update an Azure resource. Requires user approval every "
        "time. Before asking, WAI runs an ARM What-If against your "
        "subscription and puts the real property-level diff in the prompt — "
        "so the user sees exactly what would change. Call azure_explain first "
        "for the api-version and the shape of `body`."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            **_SHARED_PROPERTIES,
            "body": {
                "type": "object",
                "description": "The ARM resource body: location, properties, tags, sku…",
            },
        },
        "required": ["id", "body"],
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        prepared = await self.prepare(args, ctx)
        if isinstance(prepared, ToolOutcome):
            return prepared
        provider, parts, operation, version = prepared
        body = args.get("body")
        if not isinstance(body, dict) or not body:
            return ToolOutcome.error("body must be a non-empty object")
        resource_id = str(args["id"]).strip()

        changes, preflight = await self._what_if(provider, resource_id, parts, version, body)
        refused = await self.gate(
            args,
            ctx,
            resource_id=resource_id,
            parts=parts,
            operation=operation,
            summary=f"PUT {resource_id}\n(api-version {version})\n\n{changes}",
            preflight=preflight,
        )
        if refused is not None:
            return refused

        try:
            payload = await provider.call("PUT", resource_id, api_version=version, body=body)
        except Exception as exc:
            return ToolOutcome.error(f"{resource_id} failed: {exc}", summary="failed")
        return ToolOutcome(
            content=f"wrote {resource_id}\n" + _render(self.scrub(payload, ctx)),
            summary=f"{parts['type']}/{parts['name']}",
        )

    async def _what_if(
        self,
        provider: Any,
        resource_id: str,
        parts: dict[str, str],
        version: str,
        body: dict[str, Any],
    ) -> tuple[str, str]:
        """The diff, and the sentence describing how it was obtained.

        Best-effort: What-If needs Microsoft.Resources/deployments/whatIf at
        the scope, and plenty of identities that can write a resource cannot
        ask what writing it would do. That comes back as "could not run"
        rather than as a refusal --- and crucially never as "no changes".
        """
        if not parts["group"]:
            return "", (
                "not previewed: What-If needs a resource-group scope, and this id "
                "is not inside one."
            )
        resource = {
            "type": f"{parts['namespace']}/{parts['type']}",
            "apiVersion": version,
            "name": parts["name"],
            **body,
        }
        scope = f"/subscriptions/{parts['subscription']}/resourceGroups/{parts['group']}"
        try:
            result = await provider.what_if(scope, resource)
        except Exception as exc:
            return "", (
                "not previewed: the What-If could not run "
                f"({str(exc)[:120]}). What this changes is not known in advance."
            )

        changes = ((result.get("properties") or {}).get("changes")) or []
        if not changes:
            return "", (
                "What-If ran and reported no changes: this resource already "
                "matches what you are about to send."
            )
        return _render_changes(changes), (
            f"What-If ran against your subscription: {_summarise(changes)}. "
            "This is a real server-side diff, not a guess."
        )


class AzureDeleteTool(_ArmChange):
    name: ClassVar[str] = "azure_delete"
    action: ClassVar[str] = "delete"
    verb: ClassVar[str] = "delete"
    method: ClassVar[str] = "DELETE"
    description: ClassVar[str] = (
        "Delete an Azure resource. Requires user approval every time. ARM "
        "offers no preview for a delete, so WAI checks for a resource lock "
        "and asks RBAC whether the call is permitted, and the prompt says "
        "that is what it checked. A CanNotDelete lock refuses outright."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": dict(_SHARED_PROPERTIES),
        "required": ["id"],
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        prepared = await self.prepare(args, ctx)
        if isinstance(prepared, ToolOutcome):
            return prepared
        provider, parts, operation, version = prepared
        resource_id = str(args["id"]).strip()

        # A lock means the call will fail, so this is a refusal rather than a
        # warning. Prompting for a decision that does not exist spends the
        # user's attention on nothing.
        unlocked, lock = await self.check_lock(provider, resource_id)
        if not unlocked:
            return ToolOutcome.error(
                f"{resource_id} cannot be deleted: {lock}. Remove the lock first — "
                "which is itself a privileged change.",
                summary="locked",
            )

        allowed, verdict = await self.check_access(provider, resource_id, operation)
        if allowed is False:
            return ToolOutcome.error(
                f"RBAC refuses this: {verdict}. The call would fail.", summary="refused"
            )
        access = f"RBAC check: {verdict}" if allowed else f"RBAC check could not run ({verdict})"
        preflight = (
            f"no preview exists: ARM cannot say what a delete removes. {lock}. "
            f"{access}. What this takes with it is not known in advance."
        )

        refused = await self.gate(
            args,
            ctx,
            resource_id=resource_id,
            parts=parts,
            operation=operation,
            summary=f"DELETE {resource_id}\n(api-version {version})",
            preflight=preflight,
        )
        if refused is not None:
            return refused

        try:
            payload = await provider.call("DELETE", resource_id, api_version=version)
        except Exception as exc:
            return ToolOutcome.error(f"{resource_id} failed: {exc}", summary="failed")
        return ToolOutcome(
            content=f"deleted {resource_id}\n" + _render(self.scrub(payload, ctx)),
            summary=f"deleted {parts['name']}",
        )


class AzureActionTool(_ArmChange):
    name: ClassVar[str] = "azure_action"
    action: ClassVar[str] = "action"
    verb: ClassVar[str] = "action"
    method: ClassVar[str] = "POST"
    description: ClassVar[str] = (
        "Invoke an action on a resource — start, restart, failover, listKeys "
        "and the rest of the POST operations. Requires user approval every "
        "time. ARM offers no preview for an action, so WAI asks RBAC whether "
        "it is permitted and the prompt says that is all it checked. Call "
        "azure_explain to find the action name."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            **_SHARED_PROPERTIES,
            "action": {"type": "string", "description": "e.g. start, restart, listKeys"},
            "body": {"type": "object", "description": "Action parameters, if it takes any."},
        },
        "required": ["id", "action"],
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        action = str(args.get("action", "")).strip()
        if not action:
            return ToolOutcome.error("action is required, e.g. start")
        prepared = await self.prepare(args, ctx)
        if isinstance(prepared, ToolOutcome):
            return prepared
        provider, parts, operation, version = prepared
        resource_id = str(args["id"]).strip()
        body = args.get("body") if isinstance(args.get("body"), dict) else None

        allowed, verdict = await self.check_access(provider, resource_id, operation)
        if allowed is False:
            return ToolOutcome.error(
                f"RBAC refuses this: {verdict}. The call would fail.", summary="refused"
            )
        access = f"RBAC check: {verdict}" if allowed else f"RBAC check could not run ({verdict})"
        preflight = (
            f"no preview exists: ARM cannot preview an action. {access}. "
            "What it does is not known in advance."
        )

        summary = f"POST {resource_id}/{action}\n(api-version {version})"
        if body:
            summary += "\n" + _render(body)
        refused = await self.gate(
            args,
            ctx,
            resource_id=resource_id,
            parts=parts,
            operation=operation,
            summary=summary,
            preflight=preflight,
        )
        if refused is not None:
            return refused

        try:
            payload = await provider.call(
                "POST", f"{resource_id}/{action}", api_version=version, body=body
            )
        except Exception as exc:
            return ToolOutcome.error(f"{resource_id}/{action} failed: {exc}", summary="failed")
        return ToolOutcome(
            content=f"{action} on {resource_id}\n" + _render(self.scrub(payload, ctx)),
            summary=f"{action} {parts['name']}",
        )


#: What-If's own vocabulary, kept rather than translated: these are the words
#: Azure's own tooling uses, so a user who has seen `az deployment what-if`
#: reads the same thing here.
CHANGE_ORDER = ("Delete", "Create", "Modify", "Deploy", "NoChange", "Ignore")


def _summarise(changes: list[dict[str, Any]]) -> str:
    counts: dict[str, int] = {}
    for change in changes:
        counts[str(change.get("changeType", "?"))] = (
            counts.get(str(change.get("changeType", "?")), 0) + 1
        )
    ordered = sorted(
        counts.items(),
        key=lambda kv: CHANGE_ORDER.index(kv[0]) if kv[0] in CHANGE_ORDER else len(CHANGE_ORDER),
    )
    return ", ".join(f"{count} {kind}" for kind, count in ordered)


def _render_changes(changes: list[dict[str, Any]]) -> str:
    """The diff, in the shape a person reads rather than the shape ARM sends."""
    lines: list[str] = []
    for change in changes:
        kind = str(change.get("changeType", "?"))
        target = str(change.get("resourceId", ""))
        lines.append(f"{kind}: {target}")
        for delta in change.get("delta") or []:
            path = str(delta.get("path", ""))
            before = delta.get("before")
            after = delta.get("after")
            if delta.get("propertyChangeType") == "Modify":
                lines.append(f"  ~ {path}: {before!r} → {after!r}")
            elif delta.get("propertyChangeType") == "Delete":
                lines.append(f"  - {path}: {before!r}")
            else:
                lines.append(f"  + {path}: {after!r}")
    return "\n".join(lines[:60])


def _render(payload: Any) -> str:
    import yaml

    if not payload:
        return "  (no response body)"
    text = yaml.safe_dump(payload, default_flow_style=False, sort_keys=True, allow_unicode=True)
    return "\n".join(f"  {line}" for line in str(text).splitlines()[:40])
