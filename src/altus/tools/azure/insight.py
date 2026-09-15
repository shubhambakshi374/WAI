"""The derived views: inventory, topology, cost and quota.

Where the terminal-graphics work pays off. These return a ``Visual`` beside a
compact text summary, so the human gets a picture and the model pays for a
paragraph --- and both renderers already exist and need nothing new.

Two of them are better than their AWS counterparts for reasons that belong to
Azure rather than to this code. Inventory and topology are single Resource
Graph queries instead of a loop of per-service describes. And ``azure_quotas``
gets real current usage alongside the ceiling, so ``Bar.limit`` finally means
what it was built to mean --- the AWS version could only plot the quota and had
to say so in its caption.
"""

from __future__ import annotations

from typing import Any, ClassVar

from altus.cloud import azure as arm
from altus.core.visuals import Bar, Bars, Chart, GraphEdge, GraphNode, ResourceGraph, Series, Table
from altus.tools.azure.base import AzureTool
from altus.tools.base import ToolContext, ToolOutcome


#: Node identities stay Kind/scope/name --- the same three-part shape
#: Kubernetes and AWS use --- so ``split_node_id`` and the drill-down path work
#: unchanged. The scope is the resource group here, as it is the namespace
#: there and the region in AWS.
def node_id(kind: str, group: str, name: str) -> str:
    return f"{kind}/{group}/{name}"


def _short_type(full: str) -> str:
    """``microsoft.compute/virtualmachines`` -> ``virtualMachines``."""
    return full.rsplit("/", 1)[-1] if full else ""


class AzureInventoryTool(AzureTool):
    name: ClassVar[str] = "azure_inventory"
    description: ClassVar[str] = (
        "What exists in this subscription: every resource, its type, group and "
        "region, in one table. The fastest answer to 'what have we got' — it "
        "is a single Resource Graph query rather than a walk of each service."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "group": {"type": "string", "description": "Limit to one resource group."},
            "type": {"type": "string", "description": "Limit to one type, e.g. virtualMachines."},
        },
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        provider = await self.provider(ctx)
        if isinstance(provider, ToolOutcome):
            return provider
        subscription = self.subscription_for(args, ctx, provider)
        if not subscription:
            return ToolOutcome.error(
                "no subscription is selected. Call azure_subscriptions first.",
                summary="no subscription",
            )

        query = "resources"
        if args.get("group"):
            query += f" | where resourceGroup =~ '{_kql(str(args['group']))}'"
        if args.get("type"):
            query += f" | where type contains '{_kql(str(args['type'])).casefold()}'"
        query += " | project name, type, resourceGroup, location | order by type asc, name asc"

        try:
            payload = await provider.graph(query, [subscription])
        except Exception as exc:
            return ToolOutcome.error(f"the inventory query failed: {exc}", summary="failed")

        rows = [
            [
                _short_type(str(row.get("type", ""))),
                str(row.get("name", "")),
                str(row.get("resourceGroup", "")),
                str(row.get("location", "")),
            ]
            for row in payload.get("data") or []
            if isinstance(row, dict)
        ]
        if not rows:
            return ToolOutcome(content=f"nothing found in {subscription}", summary="empty")

        kinds = len({row[0] for row in rows})
        table = Table(
            title=f"inventory — {subscription}",
            columns=["type", "name", "group", "region"],
            rows=rows,
            caption=f"{len(rows)} resources across {kinds} types",
        )
        return ToolOutcome(
            content=table.to_text(max_rows=80), summary=f"{len(rows)} resources", visual=table
        )


#: The network objects a topology is drawn from. Fetched in one query, because
#: Resource Graph can do that and a per-type walk would be five round trips.
TOPOLOGY_TYPES = (
    "microsoft.network/virtualnetworks",
    "microsoft.network/networkinterfaces",
    "microsoft.network/networksecuritygroups",
    "microsoft.network/publicipaddresses",
    "microsoft.compute/virtualmachines",
)


class AzureTopologyTool(AzureTool):
    name: ClassVar[str] = "azure_topology"
    description: ClassVar[str] = (
        "A birds-eye view of the subscription's network: which virtual "
        "networks hold which subnets, what is attached to them, and which "
        "security groups and public IPs reach them. Use this to understand how "
        "something is exposed, or what a change would touch."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "group": {"type": "string", "description": "Limit to one resource group."},
        },
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        provider = await self.provider(ctx)
        if isinstance(provider, ToolOutcome):
            return provider
        subscription = self.subscription_for(args, ctx, provider)
        if not subscription:
            return ToolOutcome.error(
                "no subscription is selected. Call azure_subscriptions first.",
                summary="no subscription",
            )

        types = ", ".join(f"'{t}'" for t in TOPOLOGY_TYPES)
        query = f"resources | where type in~ ({types})"
        if args.get("group"):
            query += f" | where resourceGroup =~ '{_kql(str(args['group']))}'"
        query += " | project id, name, type, location, resourceGroup, properties"

        try:
            payload = await provider.graph(query, [subscription])
        except Exception as exc:
            return ToolOutcome.error(f"the topology query failed: {exc}", summary="failed")

        rows = [row for row in payload.get("data") or [] if isinstance(row, dict)]
        by_type: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_type.setdefault(str(row.get("type", "")).casefold(), []).append(row)

        nodes: list[GraphNode] = []
        edges: list[GraphEdge] = []
        #: ARM id (lowercased) -> node id, so a property that references
        #: another resource by id can be turned into an edge.
        by_id: dict[str, str] = {}

        def add(kind: str, row: dict[str, Any], name: str = "", detail: str = "") -> str:
            group = str(row.get("resourceGroup", ""))
            label = name or str(row.get("name", ""))
            ident = node_id(kind, group, label)
            nodes.append(
                GraphNode(
                    id=ident,
                    kind=kind,
                    name=label,
                    namespace=group,
                    status=str(row.get("location", "")),
                    detail=detail,
                    reader="azure_get",
                )
            )
            return ident

        for vnet in by_type.get("microsoft.network/virtualnetworks", []):
            properties = vnet.get("properties") or {}
            prefixes = ((properties.get("addressSpace") or {}).get("addressPrefixes")) or []
            ident = add("VirtualNetwork", vnet, detail=", ".join(str(p) for p in prefixes[:2]))
            by_id[str(vnet.get("id", "")).casefold()] = ident
            # Subnets are not resources of their own in Resource Graph; they
            # live on the parent, which is why they are read from here.
            for subnet in properties.get("subnets") or []:
                child = add(
                    "Subnet",
                    vnet,
                    name=str(subnet.get("name", "")),
                    detail=str((subnet.get("properties") or {}).get("addressPrefix", "")),
                )
                by_id[str(subnet.get("id", "")).casefold()] = child
                edges.append(GraphEdge(source=ident, target=child, relation="owns"))

        nic_ids: dict[str, str] = {}
        #: (public IP id, NIC node) pairs, drawn once the addresses exist as
        #: nodes --- Resource Graph returns them in no particular order.
        public_links: list[tuple[str, str]] = []
        for nic in by_type.get("microsoft.network/networkinterfaces", []):
            ident = add("NetworkInterface", nic)
            nic_ids[str(nic.get("id", "")).casefold()] = ident
            by_id[str(nic.get("id", "")).casefold()] = ident
            for configuration in (nic.get("properties") or {}).get("ipConfigurations") or []:
                properties = configuration.get("properties") or {}
                subnet = by_id.get(str((properties.get("subnet") or {}).get("id", "")).casefold())
                if subnet:
                    edges.append(GraphEdge(source=subnet, target=ident, relation="owns"))
                public = str((properties.get("publicIPAddress") or {}).get("id", "")).casefold()
                if public:
                    public_links.append((public, ident))

        for address in by_type.get("microsoft.network/publicipaddresses", []):
            properties = address.get("properties") or {}
            ident = add("PublicIP", address, detail=str(properties.get("ipAddress", "")))
            by_id[str(address.get("id", "")).casefold()] = ident

        # A public address is what makes something reachable from outside, so
        # it is drawn as a relation rather than folded into the interface.
        for public, attached_to in public_links:
            if public in by_id:
                edges.append(
                    GraphEdge(source=by_id[public], target=attached_to, relation="exposes")
                )

        for machine in by_type.get("microsoft.compute/virtualmachines", []):
            properties = machine.get("properties") or {}
            size = ((properties.get("hardwareProfile") or {}).get("vmSize")) or ""
            ident = add("VirtualMachine", machine, detail=str(size))
            attached = (properties.get("networkProfile") or {}).get("networkInterfaces") or []
            parent = next(
                (
                    nic_ids[key]
                    for key in (str(n.get("id", "")).casefold() for n in attached)
                    if nic_ids.get(key)
                ),
                "",
            )
            if parent:
                edges.append(GraphEdge(source=parent, target=ident, relation="owns"))

        # Security groups are the part that makes this a graph rather than a
        # tree: an NSG associates across the virtual-network hierarchy.
        for group in by_type.get("microsoft.network/networksecuritygroups", []):
            properties = group.get("properties") or {}
            associated = [
                *(properties.get("subnets") or []),
                *(properties.get("networkInterfaces") or []),
            ]
            targets = [
                by_id[key]
                for key in (str(item.get("id", "")).casefold() for item in associated)
                if key in by_id
            ]
            if not targets:
                continue  # an unattached NSG is noise on a topology map
            ident = add("NetworkSecurityGroup", group)
            for target in targets:
                edges.append(GraphEdge(source=ident, target=target, relation="secures"))

        if not nodes:
            return ToolOutcome(
                content=f"no network resources found in {subscription}", summary="empty"
            )

        counts = {kind: sum(1 for n in nodes if n.kind == kind) for kind in {n.kind for n in nodes}}
        graph = ResourceGraph(
            title=f"{subscription} — network",
            nodes=nodes,
            edges=edges,
            caption=" · ".join(f"{count} {kind}" for kind, count in sorted(counts.items())),
        )
        return ToolOutcome(content=graph.to_text(), summary=f"{len(nodes)} resources", visual=graph)


class AzureCostTool(AzureTool):
    name: ClassVar[str] = "azure_cost"
    description: ClassVar[str] = (
        "Spend over time, broken down by service, as a chart. Defaults to the "
        "last 30 days by day. Azure's Cost Management query API is free, "
        "unlike AWS Cost Explorer — there is nothing to warn the user about "
        "before calling it."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "days": {"type": "integer", "description": "How far back. Default 30."},
            "granularity": {"type": "string", "enum": ["Daily", "Monthly"]},
            "top": {"type": "integer", "description": "How many services to plot. Default 5."},
        },
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        from datetime import UTC, datetime, timedelta

        provider = await self.provider(ctx)
        if isinstance(provider, ToolOutcome):
            return provider
        subscription = self.subscription_for(args, ctx, provider)
        if not subscription:
            return ToolOutcome.error(
                "no subscription is selected. Call azure_subscriptions first.",
                summary="no subscription",
            )

        days = max(1, min(int(args.get("days") or 30), 365))
        granularity = (
            "Monthly" if str(args.get("granularity", "")).casefold() == "monthly" else "Daily"
        )
        end = datetime.now(UTC).date()
        start = end - timedelta(days=days)

        try:
            payload = await provider.call(
                "POST",
                f"/subscriptions/{subscription}/providers/Microsoft.CostManagement/query",
                api_version=arm.COST_API,
                body={
                    "type": "ActualCost",
                    "timeframe": "Custom",
                    "timePeriod": {
                        "from": f"{start.isoformat()}T00:00:00Z",
                        "to": f"{end.isoformat()}T23:59:59Z",
                    },
                    "dataset": {
                        "granularity": granularity,
                        "aggregation": {"totalCost": {"name": "Cost", "function": "Sum"}},
                        "grouping": [{"type": "Dimension", "name": "ServiceName"}],
                    },
                },
            )
        except Exception as exc:
            return ToolOutcome.error(f"the cost query failed: {exc}", summary="failed")

        properties = payload.get("properties") or {}
        columns = [str((c or {}).get("name", "")) for c in properties.get("columns") or []]
        rows = properties.get("rows") or []
        if not columns or not rows:
            return ToolOutcome(content="no cost data for that period", summary="no data")

        # Column order is not contractual, so every field is looked up by name
        # rather than by position.
        index = {name.casefold(): position for position, name in enumerate(columns)}
        cost_at = index.get("cost", index.get("costusd", 0))
        date_at = index.get("usagedate", index.get("billingmonth"))
        service_at = index.get("servicename")
        currency = ""
        if "currency" in index:
            currency = str(rows[0][index["currency"]])

        by_service: dict[str, list[tuple[float, float]]] = {}
        for row in rows:
            try:
                amount = float(row[cost_at])
            except TypeError, ValueError, IndexError:
                continue
            stamp = _stamp(row[date_at]) if date_at is not None else 0.0
            service = str(row[service_at]) if service_at is not None else "all services"
            by_service.setdefault(service, []).append((stamp, amount))

        if not by_service:
            return ToolOutcome(content="no cost data for that period", summary="no data")

        ranked = sorted(by_service.items(), key=lambda kv: -sum(a for _s, a in kv[1]))
        top = max(1, min(int(args.get("top") or 5), 8))
        unit = currency or ""
        series = [
            Series(
                label=name,
                points=[amount for _stamp, amount in sorted(points)],
                at=[stamp for stamp, _amount in sorted(points)],
                unit=unit,
            )
            for name, points in ranked[:top]
        ]
        total = sum(amount for _name, points in ranked for _s, amount in points)
        chart = Chart(
            title=f"spend, last {days} days",
            series=series,
            unit=unit,
            caption=f"{total:,.2f} {unit} total across {len(ranked)} services".strip(),
        )
        return ToolOutcome(
            content=chart.to_text(),
            summary=f"{total:,.2f} {unit} over {days}d".strip(),
            visual=chart,
        )


def _stamp(value: Any) -> float:
    """Cost Management dates are integers: 20260115 daily, 2026-01 monthly."""
    from datetime import UTC, datetime

    text = str(value).strip()
    try:
        if len(text) == 8 and text.isdigit():
            return datetime.strptime(text, "%Y%m%d").replace(tzinfo=UTC).timestamp()
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


class AzureQuotasTool(AzureTool):
    name: ClassVar[str] = "azure_quotas"
    description: ClassVar[str] = (
        "Compute quotas in a region and how close this subscription is to "
        "them, with real current usage against each ceiling. The answer to "
        "'why did that deployment fail' when nothing looks wrong."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "region": {"type": "string", "description": "e.g. westeurope"},
            "used_only": {
                "type": "boolean",
                "description": "Only quotas with something consumed. Default true.",
            },
        },
        "required": ["region"],
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        provider = await self.provider(ctx)
        if isinstance(provider, ToolOutcome):
            return provider
        subscription = self.subscription_for(args, ctx, provider)
        region = str(args.get("region", "")).strip()
        if not region:
            return ToolOutcome.error("region is required, e.g. westeurope")
        if not subscription:
            return ToolOutcome.error(
                "no subscription is selected. Call azure_subscriptions first.",
                summary="no subscription",
            )

        try:
            version = await provider.api_version_for("Microsoft.Compute", "locations/usages")
            payload = await provider.call(
                "GET",
                f"/subscriptions/{subscription}/providers/Microsoft.Compute"
                f"/locations/{region}/usages",
                api_version=version,
            )
        except Exception as exc:
            return ToolOutcome.error(f"could not read quotas: {exc}", summary="failed")

        used_only = args.get("used_only", True)
        bars: list[Bar] = []
        for entry in payload.get("value") or []:
            limit = float(entry.get("limit") or 0)
            current = float(entry.get("currentValue") or 0)
            if limit <= 0 or (used_only and current <= 0):
                continue
            bars.append(
                Bar(
                    label=str((entry.get("name") or {}).get("localizedValue", ""))[:40],
                    value=current,
                    limit=limit,
                    unit=""
                    if str(entry.get("unit", "")) == "Count"
                    else str(entry.get("unit", "")),
                )
            )

        if not bars:
            return ToolOutcome(
                content=f"nothing is consuming a compute quota in {region}", summary="none"
            )

        bars.sort(key=lambda b: -(b.value / (b.limit or 1)))
        tightest = bars[0]
        chart = Bars(
            title=f"compute quotas — {region}",
            bars=bars[:25],
            caption=f"tightest: {tightest.label} at "
            f"{tightest.value / (tightest.limit or 1):.0%} of its ceiling",
        )
        return ToolOutcome(content=chart.to_text(), summary=f"{len(bars)} quotas", visual=chart)


def _kql(value: str) -> str:
    """Escape a value going into a KQL string literal.

    Resource Graph is read-only and scoped to the subscription, so the worst a
    crafted value could do is widen a query --- but a quote that terminates the
    literal early would still let a model-supplied group name change what the
    query means, and there is no reason to allow that.
    """
    return value.replace("\\", "\\\\").replace("'", "\\'")
