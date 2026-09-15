"""Reads that need no approval: identity, introspection, and any ARM read.

``azure_get`` issues one HTTP method and only one: GET. That is the whole
read-only argument, and it is structural rather than a policy the tool has to
remember --- ARM's read verb *is* GET, and the operations that look like reads
but are not (``listKeys`` and its relatives) are POSTs, so they cannot arrive
here at all. They go through ``azure_action``, which asks.
"""

from __future__ import annotations

from typing import Any, ClassVar

from wai.cloud import azure as arm
from wai.cloud.base import Sensitivity
from wai.core.visuals import Table
from wai.tools.azure.base import AzureTool
from wai.tools.base import ToolContext, ToolOutcome

MAX_CONTENT = 24_000


def _as_yaml(payload: Any) -> str:
    import yaml

    return str(
        yaml.safe_dump(payload, default_flow_style=False, sort_keys=False, allow_unicode=True)
    )


class AzureWhoamiTool(AzureTool):
    name: ClassVar[str] = "azure_whoami"
    description: ClassVar[str] = (
        "Which Azure tenant, subscription and principal this session is using. "
        "Call this before anything else if unsure where you are — every other "
        "Azure tool acts in exactly one subscription, and this names it."
    )
    input_schema: ClassVar[dict[str, Any]] = {"type": "object", "properties": {}}

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        provider = await self.provider(ctx)
        if isinstance(provider, ToolOutcome):
            return provider
        identity = await self.identity(ctx, provider)
        if isinstance(identity, ToolOutcome):
            return identity

        subscription = self.subscription_for(args, ctx, provider)
        rules = ctx.cloud.protection
        protected = bool(rules and rules.matches(arm.target_for(subscription)))
        lines = [
            f"tenant        {identity['tenant']}",
            f"principal     {identity['principal'] or identity['object_id']}",
            f"subscription  {subscription or '(none selected — call azure_subscriptions)'}",
        ]
        if protected:
            lines.append("⚠ PROTECTED: changes here need a typed confirmation")
        return ToolOutcome(content="\n".join(lines), summary=subscription or "no subscription")


class AzureSubscriptionsTool(AzureTool):
    name: ClassVar[str] = "azure_subscriptions"
    description: ClassVar[str] = (
        "Every subscription this credential can see. One Azure login commonly "
        "reaches many, and WAI acts in exactly one at a time — the user "
        "switches it with /azure sub <id>."
    )
    input_schema: ClassVar[dict[str, Any]] = {"type": "object", "properties": {}}

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        provider = await self.provider(ctx)
        if isinstance(provider, ToolOutcome):
            return provider
        try:
            found = await provider.subscriptions()
        except Exception as exc:
            return ToolOutcome.error(f"could not list subscriptions: {exc}", summary="failed")

        active = self.subscription_for(args, ctx, provider)
        rows = [
            [
                "→" if entry["id"] == active else "",
                entry["name"],
                entry["id"],
                entry["state"],
            ]
            for entry in found
        ]
        if not rows:
            return ToolOutcome(content="this credential sees no subscriptions", summary="none")
        table = Table(title="subscriptions", columns=["", "name", "id", "state"], rows=rows)
        return ToolOutcome(
            content=table.to_text(max_rows=60), summary=f"{len(rows)} subscriptions", visual=table
        )


class AzureProvidersTool(AzureTool):
    name: ClassVar[str] = "azure_providers"
    description: ClassVar[str] = (
        "Resource providers registered in this subscription, and for one "
        "namespace its resource types with the api-versions and regions each "
        "supports. api-version is mandatory on every ARM call and differs per "
        "type, so this is how you find it — do not guess one."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "namespace": {"type": "string", "description": "e.g. Microsoft.Compute"},
            "filter": {"type": "string", "description": "Substring, when listing namespaces."},
        },
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        provider = await self.provider(ctx)
        if isinstance(provider, ToolOutcome):
            return provider
        namespace = str(args.get("namespace", "")).strip()

        try:
            payload = await provider.providers(namespace)
        except Exception as exc:
            return ToolOutcome.error(f"could not read providers: {exc}", summary="failed")

        if not namespace:
            needle = str(args.get("filter", "")).casefold()
            names = sorted(
                str(entry.get("namespace", ""))
                for entry in payload.get("value") or []
                if not needle or needle in str(entry.get("namespace", "")).casefold()
            )
            body = "\n".join(names[:200]) or "(none matched)"
            if len(names) > 200:
                body += f"\n[{len(names) - 200} more; narrow the filter]"
            return ToolOutcome(content=body, summary=f"{len(names)} namespaces")

        rows = [
            [
                str(entry.get("resourceType", "")),
                (list(entry.get("apiVersions") or []) or [""])[0],
                str(len(entry.get("locations") or [])),
            ]
            for entry in payload.get("resourceTypes") or []
        ]
        table = Table(
            title=f"{namespace} resource types",
            columns=["type", "latest api-version", "regions"],
            rows=sorted(rows),
        )
        return ToolOutcome(
            content=table.to_text(max_rows=120), summary=f"{len(rows)} types", visual=table
        )


class AzureExplainTool(AzureTool):
    name: ClassVar[str] = "azure_explain"
    description: ClassVar[str] = (
        "The exact contract for a resource type: the api-version to send, and "
        "every RBAC operation it defines with what each one does and how "
        "sensitive it is. Call this BEFORE azure_get or azure_write — it is "
        "authoritative, and it is why using ARM directly beats guessing `az` "
        "flag names."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "namespace": {"type": "string", "description": "e.g. Microsoft.Compute"},
            "type": {"type": "string", "description": "e.g. virtualMachines"},
            "filter": {"type": "string", "description": "Substring to match operation names."},
        },
        "required": ["namespace"],
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        provider = await self.provider(ctx)
        if isinstance(provider, ToolOutcome):
            return provider
        namespace = str(args.get("namespace", "")).strip()
        if not namespace:
            return ToolOutcome.error("namespace is required, e.g. Microsoft.Compute")
        wanted = str(args.get("type", "")).strip().casefold()
        needle = str(args.get("filter", "")).casefold()

        try:
            catalog = await provider.provider_operations(namespace)
        except Exception as exc:
            return ToolOutcome.error(
                f"could not read the operation catalog for {namespace}: {exc}", summary="unknown"
            )

        rows: list[list[str]] = []
        for entry in catalog.get("resourceTypes") or []:
            type_name = str(entry.get("name", ""))
            if wanted and wanted not in type_name.casefold():
                continue
            for operation in entry.get("operations") or []:
                name = str(operation.get("name", ""))
                if needle and needle not in name.casefold():
                    continue
                rows.append(
                    [
                        name,
                        arm.classify(name, data_action=bool(operation.get("isDataAction"))).value,
                        str(operation.get("description") or operation.get("displayName") or "")[
                            :90
                        ],
                    ]
                )

        if not rows:
            return ToolOutcome.error(
                f"no operations matched under {namespace}. Call azure_providers "
                "to see which types it offers.",
                summary="none",
            )

        head = []
        if wanted:
            try:
                version = await provider.api_version_for(namespace, args["type"])
            except Exception as exc:
                head.append(f"api-version: could not resolve ({exc})")
            else:
                head.append(f"api-version: {version}")

        table = Table(
            title=f"{namespace}{'/' + str(args.get('type')) if wanted else ''}",
            columns=["operation", "sensitivity", "what it does"],
            rows=sorted(rows)[:150],
            caption="; ".join(head),
        )
        body = "\n".join([*head, "", table.to_text(max_rows=150)])
        if len(rows) > 150:
            body += f"\n[{len(rows) - 150} more; narrow with `filter`]"
        return ToolOutcome(content=body, summary=f"{len(rows)} operations", visual=table)


class AzureGetTool(AzureTool):
    name: ClassVar[str] = "azure_get"
    description: ClassVar[str] = (
        "Read anything in Azure Resource Manager. Give `id` for one resource, "
        "or `namespace` + `type` to list a collection. Paginated and capped "
        "automatically, and the api-version is resolved for you. Operations "
        "that look like reads but hand back credentials — listKeys and its "
        "relatives — are POSTs and go through azure_action, which asks."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "Full ARM resource id, e.g. /subscriptions/…/virtualMachines/web1",
            },
            "namespace": {"type": "string", "description": "e.g. Microsoft.Compute"},
            "type": {"type": "string", "description": "e.g. virtualMachines"},
            "group": {"type": "string", "description": "Resource group, to narrow a collection."},
            "api_version": {"type": "string", "description": "Only if you must override."},
        },
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        provider = await self.provider(ctx)
        if isinstance(provider, ToolOutcome):
            return provider
        subscription = self.subscription_for(args, ctx, provider)

        resource_id = str(args.get("id", "")).strip()
        namespace = str(args.get("namespace", "")).strip()
        resource_type = str(args.get("type", "")).strip()

        if resource_id:
            parts = arm.parse_resource_id(resource_id)
            if not parts["namespace"]:
                return ToolOutcome.error(
                    f"{resource_id!r} is not an ARM resource id. It looks like "
                    "/subscriptions/<id>/resourceGroups/<rg>/providers/<namespace>/<type>/<name>.",
                    summary="bad id",
                )
            namespace, resource_type = parts["namespace"], parts["type"]
            path = resource_id
        elif namespace and resource_type:
            if not subscription:
                return ToolOutcome.error(
                    "no subscription is selected. Call azure_subscriptions first.",
                    summary="no subscription",
                )
            group = str(args.get("group", "")).strip()
            path = f"/subscriptions/{subscription}"
            if group:
                path += f"/resourceGroups/{group}"
            path += f"/providers/{namespace}/{resource_type}"
        else:
            return ToolOutcome.error("give either `id`, or both `namespace` and `type`")

        # Structurally a read: this tool issues GET and nothing else, so the
        # classification is a cross-check rather than the control.
        sensitivity = arm.classify(f"{namespace}/{resource_type}/read", path)
        if sensitivity not in (Sensitivity.READ, Sensitivity.SENSITIVE_READ):
            return ToolOutcome.error(
                f"{namespace}/{resource_type} reads classify as {sensitivity.value}",
                summary="refused",
            )

        version = str(args.get("api_version", "")).strip()
        if not version:
            try:
                version = await provider.api_version_for(namespace, resource_type)
            except Exception as exc:
                return ToolOutcome.error(str(exc), summary="no api-version")

        try:
            payload = await provider.call("GET", path, api_version=version)
        except Exception as exc:
            return ToolOutcome.error(f"{path} failed: {exc}", summary="failed")

        body = _as_yaml(self.scrub(payload, ctx))
        if len(body) > MAX_CONTENT:
            body = body[:MAX_CONTENT] + "\n[truncated]"
        return ToolOutcome(
            content=f"# GET {path}  (api-version {version})\n{body}",
            summary=f"{namespace}/{resource_type}",
        )


class AzureQueryTool(AzureTool):
    name: ClassVar[str] = "azure_query"
    description: ClassVar[str] = (
        "Run a KQL query across every resource in the subscription at once, "
        "through Azure Resource Graph. Free, and far faster than listing types "
        "one at a time — this is the right first move for 'what have we got' "
        "or 'find everything with tag X'. Example: "
        "resources | where type =~ 'microsoft.compute/virtualmachines' | "
        "project name, location, resourceGroup"
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "A KQL query."},
            "subscriptions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Widen beyond the active subscription. Omit for just this one.",
            },
        },
        "required": ["query"],
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        query = str(args.get("query", "")).strip()
        if not query:
            return ToolOutcome.error("query is required")
        provider = await self.provider(ctx)
        if isinstance(provider, ToolOutcome):
            return provider

        widened = [str(s) for s in (args.get("subscriptions") or [])]
        subscription = self.subscription_for(args, ctx, provider)
        if not widened and not subscription:
            return ToolOutcome.error(
                "no subscription is selected. Call azure_subscriptions first.",
                summary="no subscription",
            )

        try:
            payload = await provider.graph(query, widened or [subscription])
        except Exception as exc:
            return ToolOutcome.error(f"the query failed: {exc}", summary="failed")

        rows = payload.get("data") or []
        if not isinstance(rows, list) or not rows:
            return ToolOutcome(content="the query matched nothing", summary="no rows")

        rows = self.scrub(rows, ctx)
        columns = list(rows[0].keys()) if isinstance(rows[0], dict) else ["value"]
        table = Table(
            title="resource graph",
            columns=columns,
            rows=[[str(row.get(c, "")) for c in columns] for row in rows if isinstance(row, dict)],
            caption=f"across {len(widened or [subscription])} subscription(s)",
        )
        return ToolOutcome(
            content=table.to_text(max_rows=80), summary=f"{len(rows)} rows", visual=table
        )


class AzureCanITool(AzureTool):
    name: ClassVar[str] = "azure_can_i"
    description: ClassVar[str] = (
        "Ask Azure RBAC whether this identity may perform operations at a "
        "scope, BEFORE attempting them. Cheaper and clearer than discovering "
        "an AuthorizationFailed halfway through a plan. Unlike AWS's "
        "equivalent, asking about yourself needs no extra permission."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "actions": {
                "type": "array",
                "items": {"type": "string"},
                "description": 'e.g. ["Microsoft.Compute/virtualMachines/delete"]',
            },
            "scope": {
                "type": "string",
                "description": "Resource id, resource group or subscription path. "
                "Defaults to the whole subscription.",
            },
        },
        "required": ["actions"],
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        actions = args.get("actions")
        if not isinstance(actions, list) or not actions:
            return ToolOutcome.error("actions must be a non-empty array of RBAC operation strings")

        provider = await self.provider(ctx)
        if isinstance(provider, ToolOutcome):
            return provider
        subscription = self.subscription_for(args, ctx, provider)
        scope = str(args.get("scope", "")).strip() or f"/subscriptions/{subscription}"
        if not subscription and not args.get("scope"):
            return ToolOutcome.error(
                "no subscription is selected and no scope was given.", summary="no scope"
            )

        try:
            decisions = await provider.check_access(scope, [str(a) for a in actions])
        except Exception as exc:
            return ToolOutcome.error(f"the RBAC check could not run: {exc}", summary="cannot check")

        rows = [
            [
                entry["action"],
                entry["decision"],
                arm.classify(entry["action"], scope).value,
            ]
            for entry in decisions
        ]
        allowed = sum(1 for row in rows if row[1].casefold() == "allowed")
        table = Table(
            title=f"permissions at {scope}",
            columns=["operation", "decision", "sensitivity"],
            rows=rows,
        )
        return ToolOutcome(
            content=table.to_text(max_rows=60),
            summary=f"{allowed}/{len(rows)} allowed",
            visual=table,
        )
