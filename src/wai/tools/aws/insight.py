"""The derived views: inventory, topology, cost and quota.

Where the terminal-graphics work pays off. These return a ``Visual`` beside a
compact text summary, so the human gets a picture and the model pays for a
paragraph --- and the renderers, both of them, already exist.
"""

from __future__ import annotations

from typing import Any, ClassVar

from wai.core.visuals import Bar, Bars, Chart, GraphEdge, GraphNode, ResourceGraph, Series, Table
from wai.tools.aws.base import AwsTool
from wai.tools.base import ToolContext, ToolOutcome


#: Node identities are Kind/region/name, the same three-part shape Kubernetes
#: uses, so ``split_node_id`` and the drill-down path work unchanged.
def node_id(kind: str, region: str, name: str) -> str:
    return f"{kind}/{region}/{name}"


def _tag(item: dict[str, Any], key: str = "Name") -> str:
    for tag in item.get("Tags") or []:
        if tag.get("Key") == key:
            return str(tag.get("Value", ""))
    return ""


class AwsInventoryTool(AwsTool):
    name: ClassVar[str] = "aws_inventory"
    description: ClassVar[str] = (
        "What is running in a region: EC2 instances, RDS databases and Lambda "
        "functions, in one table. The fastest answer to 'what have we got'. "
        "A service that fails is reported rather than taking the view down — "
        "IAM routinely permits some of these and not others."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "region": {"type": "string"},
            "services": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Subset of ec2, rds, lambda. Omit for all.",
            },
        },
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        provider = await self.provider(ctx)
        if isinstance(provider, ToolOutcome):
            return provider
        region = self.region_for(args, ctx, provider)
        wanted = {str(s).casefold() for s in (args.get("services") or ["ec2", "rds", "lambda"])}

        rows: list[list[str]] = []
        notes: list[str] = []
        for service, gather in (
            ("ec2", self._instances),
            ("rds", self._databases),
            ("lambda", self._functions),
        ):
            if service not in wanted:
                continue
            try:
                rows.extend(await gather(provider, region))
            except Exception as exc:
                notes.append(f"{service}: {exc}")

        if not rows:
            body = f"nothing found in {region}"
            return ToolOutcome(
                content=body + ("\n" + "\n".join(notes) if notes else ""), summary="empty"
            )

        table = Table(
            title=f"inventory — {region}",
            columns=["type", "name", "state", "detail"],
            rows=rows,
            caption="; ".join(notes),
        )
        return ToolOutcome(
            content=table.to_text(max_rows=60), summary=f"{len(rows)} resources", visual=table
        )

    async def _instances(self, provider: Any, region: str) -> list[list[str]]:
        payload = await provider.call("ec2", "DescribeInstances", region=region)
        out = []
        for reservation in payload.get("Reservations") or []:
            for item in reservation.get("Instances") or []:
                out.append(
                    [
                        "EC2",
                        _tag(item) or str(item.get("InstanceId", "")),
                        str((item.get("State") or {}).get("Name", "")),
                        str(item.get("InstanceType", "")),
                    ]
                )
        return out

    async def _databases(self, provider: Any, region: str) -> list[list[str]]:
        payload = await provider.call("rds", "DescribeDBInstances", region=region)
        return [
            [
                "RDS",
                str(item.get("DBInstanceIdentifier", "")),
                str(item.get("DBInstanceStatus", "")),
                f"{item.get('Engine', '')} {item.get('DBInstanceClass', '')}".strip(),
            ]
            for item in payload.get("DBInstances") or []
        ]

    async def _functions(self, provider: Any, region: str) -> list[list[str]]:
        payload = await provider.call("lambda", "ListFunctions", region=region)
        return [
            [
                "Lambda",
                str(item.get("FunctionName", "")),
                str(item.get("State", "Active")),
                str(item.get("Runtime", "")),
            ]
            for item in payload.get("Functions") or []
        ]


class AwsTopologyTool(AwsTool):
    name: ClassVar[str] = "aws_topology"
    description: ClassVar[str] = (
        "A birds-eye view of a region's network: which VPCs hold which subnets, "
        "what runs in them, and which security groups and load balancers reach "
        "them. Use this to understand how something is exposed, or what a "
        "change would touch."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "region": {"type": "string"},
            "vpc_id": {"type": "string", "description": "Limit to one VPC."},
        },
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        provider = await self.provider(ctx)
        if isinstance(provider, ToolOutcome):
            return provider
        region = self.region_for(args, ctx, provider)
        wanted = str(args.get("vpc_id") or "").strip()

        async def fetch(service: str, operation: str, key: str) -> list[dict[str, Any]]:
            try:
                payload = await provider.call(service, operation, region=region)
            except Exception:
                return []  # partial permissions are normal; draw what we can
            return list(payload.get(key) or [])

        vpcs = await fetch("ec2", "DescribeVpcs", "Vpcs")
        subnets = await fetch("ec2", "DescribeSubnets", "Subnets")
        groups = await fetch("ec2", "DescribeSecurityGroups", "SecurityGroups")
        reservations = await fetch("ec2", "DescribeInstances", "Reservations")
        instances = [i for r in reservations for i in (r.get("Instances") or [])]

        if wanted:
            vpcs = [v for v in vpcs if v.get("VpcId") == wanted]
            subnets = [s for s in subnets if s.get("VpcId") == wanted]
            instances = [i for i in instances if i.get("VpcId") == wanted]
            groups = [g for g in groups if g.get("VpcId") == wanted]

        nodes: list[GraphNode] = []
        edges: list[GraphEdge] = []

        def add(kind: str, name: str, status: str = "", detail: str = "") -> str:
            ident = node_id(kind, region, name)
            nodes.append(
                GraphNode(
                    id=ident,
                    kind=kind,
                    name=name,
                    namespace=region,
                    status=status,
                    detail=detail,
                    reader="aws_call",
                )
            )
            return ident

        for vpc in vpcs:
            vpc_id = str(vpc.get("VpcId", ""))
            add(
                "Vpc", _tag(vpc) or vpc_id, str(vpc.get("State", "")), str(vpc.get("CidrBlock", ""))
            )
        vpc_ids = {
            str(v.get("VpcId")): node_id("Vpc", region, _tag(v) or str(v.get("VpcId", "")))
            for v in vpcs
        }

        subnet_ids: dict[str, str] = {}
        for subnet in subnets:
            name = _tag(subnet) or str(subnet.get("SubnetId", ""))
            ident = add(
                "Subnet",
                name,
                str(subnet.get("State", "")),
                str(subnet.get("AvailabilityZone", "")),
            )
            subnet_ids[str(subnet.get("SubnetId"))] = ident
            parent = vpc_ids.get(str(subnet.get("VpcId")))
            if parent:
                edges.append(GraphEdge(source=parent, target=ident, relation="owns"))

        instance_ids: dict[str, str] = {}
        for item in instances:
            name = _tag(item) or str(item.get("InstanceId", ""))
            ident = add(
                "Instance",
                name,
                str((item.get("State") or {}).get("Name", "")),
                str(item.get("InstanceType", "")),
            )
            instance_ids[str(item.get("InstanceId"))] = ident
            parent = subnet_ids.get(str(item.get("SubnetId"))) or vpc_ids.get(
                str(item.get("VpcId"))
            )
            if parent:
                edges.append(GraphEdge(source=parent, target=ident, relation="owns"))

        # Security groups are the part that makes this a graph rather than a
        # tree: membership cuts across the VPC hierarchy entirely.
        attached = {
            str(group.get("GroupId"))
            for item in instances
            for group in item.get("SecurityGroups") or []
        }
        for group in groups:
            group_id = str(group.get("GroupId", ""))
            if group_id not in attached:
                continue  # an unused group is noise on a topology map
            ident = add("SecurityGroup", str(group.get("GroupName", group_id)))
            for item in instances:
                if any(g.get("GroupId") == group_id for g in item.get("SecurityGroups") or []):
                    target = instance_ids.get(str(item.get("InstanceId")))
                    if target:
                        edges.append(GraphEdge(source=ident, target=target, relation="secures"))

        if not nodes:
            return ToolOutcome(content=f"no VPC resources found in {region}", summary="empty")

        graph = ResourceGraph(
            title=f"{region} — network",
            nodes=nodes,
            edges=edges,
            caption=f"{len(vpcs)} VPCs · {len(subnets)} subnets · {len(instances)} instances",
        )
        return ToolOutcome(content=graph.to_text(), summary=f"{len(nodes)} resources", visual=graph)


class AwsCostTool(AwsTool):
    name: ClassVar[str] = "aws_cost"
    description: ClassVar[str] = (
        "COSTS MONEY: Cost Explorer bills roughly $0.01 per request, so say so "
        "before calling it. Spend over time, broken down by service, as a "
        "chart. Defaults to the last 30 days by day."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "days": {"type": "integer", "description": "How far back. Default 30."},
            "granularity": {"type": "string", "enum": ["DAILY", "MONTHLY"]},
            "top": {"type": "integer", "description": "How many services to plot. Default 5."},
        },
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        from datetime import UTC, datetime, timedelta

        settings = getattr(ctx.cloud, "aws_settings", None)
        if settings is not None and not getattr(settings, "allow_cost_explorer", True):
            return ToolOutcome.error(
                "Cost Explorer is disabled by [cloud.aws] allow_cost_explorer.",
                summary="disabled",
            )

        provider = await self.provider(ctx)
        if isinstance(provider, ToolOutcome):
            return provider

        days = max(1, min(int(args.get("days") or 30), 365))
        granularity = str(args.get("granularity") or "DAILY").upper()
        end = datetime.now(UTC).date()
        start = end - timedelta(days=days)

        try:
            payload = await provider.call(
                "ce",
                "GetCostAndUsage",
                {
                    "TimePeriod": {"Start": start.isoformat(), "End": end.isoformat()},
                    "Granularity": granularity,
                    "Metrics": ["UnblendedCost"],
                    "GroupBy": [{"Type": "DIMENSION", "Key": "SERVICE"}],
                },
                region="us-east-1",  # Cost Explorer is global, served from here
            )
        except Exception as exc:
            return ToolOutcome.error(f"Cost Explorer failed: {exc}", summary="failed")

        by_service: dict[str, list[tuple[float, float]]] = {}
        for period in payload.get("ResultsByTime") or []:
            stamp = _epoch(str((period.get("TimePeriod") or {}).get("Start", "")))
            for group in period.get("Groups") or []:
                name = (group.get("Keys") or ["unknown"])[0]
                amount = float(
                    ((group.get("Metrics") or {}).get("UnblendedCost") or {}).get("Amount", 0) or 0
                )
                by_service.setdefault(name, []).append((stamp, amount))

        if not by_service:
            return ToolOutcome(content="no cost data for that period", summary="no data")

        ranked = sorted(by_service.items(), key=lambda kv: -sum(a for _s, a in kv[1]))
        top = max(1, min(int(args.get("top") or 5), 8))
        series = [
            Series(
                label=name,
                points=[amount for _stamp, amount in sorted(points)],
                at=[stamp for stamp, _amount in sorted(points)],
                unit="$",
            )
            for name, points in ranked[:top]
        ]
        total = sum(amount for _name, points in ranked for _s, amount in points)
        chart = Chart(
            title=f"spend, last {days} days",
            series=series,
            unit="$",
            caption=f"${total:,.2f} total across {len(ranked)} services",
        )
        return ToolOutcome(
            content=chart.to_text(), summary=f"${total:,.2f} over {days}d", visual=chart
        )


def _epoch(iso_date: str) -> float:
    from datetime import UTC, datetime

    try:
        return datetime.fromisoformat(iso_date).replace(tzinfo=UTC).timestamp()
    except ValueError:
        return 0.0


class AwsQuotasTool(AwsTool):
    name: ClassVar[str] = "aws_quotas"
    description: ClassVar[str] = (
        "Service quotas and how close this account is to them. The answer to "
        "'why did that fail to launch' when nothing looks wrong."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "service": {"type": "string", "description": "Quota service code, e.g. ec2."},
            "region": {"type": "string"},
        },
        "required": ["service"],
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        provider = await self.provider(ctx)
        if isinstance(provider, ToolOutcome):
            return provider
        region = self.region_for(args, ctx, provider)
        service = str(args.get("service", "")).strip()
        if not service:
            return ToolOutcome.error("service is required, e.g. ec2")

        try:
            payload = await provider.call(
                "service-quotas", "ListServiceQuotas", {"ServiceCode": service}, region=region
            )
        except Exception as exc:
            return ToolOutcome.error(f"could not read quotas: {exc}", summary="failed")

        quotas = payload.get("Quotas") or []
        bars = [
            Bar(
                label=str(q.get("QuotaName", ""))[:40],
                value=float(q.get("Value") or 0),
                limit=float(q.get("Value") or 0),
                unit=str(q.get("Unit", "")).replace("None", ""),
            )
            for q in quotas
            if q.get("Value")
        ][:25]
        if not bars:
            return ToolOutcome(content=f"no quotas reported for {service}", summary="none")

        chart = Bars(
            title=f"{service} quotas — {region}",
            bars=bars,
            caption="applied quota values; current usage needs CloudWatch",
        )
        return ToolOutcome(content=chart.to_text(), summary=f"{len(bars)} quotas", visual=chart)
