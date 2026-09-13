"""Kubernetes tools.

Every one returns a compact text summary for the model and, where it helps, a
``Visual`` for the human. The model never sees the chart --- that is what keeps
a whole-cluster topology affordable in context.

Read-only in this commit; the mutating tools land next.
"""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

from wai.cloud import k8s as k8s_api
from wai.cloud.base import Sensitivity
from wai.cloud.kube import classify
from wai.cloud.redact import redact
from wai.core.visuals import Bar, Bars, Table, VisualGroup
from wai.tools.base import BaseTool, ToolContext, ToolOutcome

MAX_LOG_LINES = 400


class K8sTool(BaseTool):
    """Shared plumbing: resolve a client, and refuse clearly when we cannot."""

    read_only: ClassVar[bool] = True

    async def client(self, ctx: ToolContext) -> tuple[k8s_api.K8sClient, str] | ToolOutcome:
        provider = ctx.cloud.k8s
        if provider is None:
            return ToolOutcome.error(
                "no Kubernetes client is configured for this session", summary="unavailable"
            )
        try:
            got: tuple[k8s_api.K8sClient, str] = await provider.get()
            return got
        except Exception as exc:
            return ToolOutcome.error(
                f"could not reach the cluster: {exc}. Check /kube and /login.",
                summary="unreachable",
            )

    def scrub(self, payload: Any, ctx: ToolContext) -> Any:
        return redact(payload, enabled=ctx.cloud.redact_secrets)


def _rows(items: list[dict[str, Any]], columns: list[str]) -> list[list[str]]:
    out: list[list[str]] = []
    for item in items:
        meta = item.get("metadata") or {}
        out.append(
            [
                str(meta.get("name", "")),
                str(meta.get("namespace", "")),
                k8s_api.status_for(item.get("kind", ""), item),
                str(meta.get("creationTimestamp", ""))[:19].replace("T", " "),
            ][: len(columns)]
        )
    return out


class K8sContextsTool(K8sTool):
    name: ClassVar[str] = "k8s_contexts"
    description: ClassVar[str] = (
        "List the Kubernetes contexts available, which one is active, and which "
        "are marked protected. Use this before anything else if unsure where you are."
    )
    input_schema: ClassVar[dict[str, Any]] = {"type": "object", "properties": {}}

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        from wai.cloud.kube import list_contexts

        extra = ctx.cloud.kubeconfigs
        contexts, active = await asyncio.to_thread(list_contexts, extra)
        if not contexts:
            return ToolOutcome.error("no kubeconfig found", summary="no contexts")
        selected = ctx.cloud.kube_context or active
        rules = ctx.cloud.protection
        lines = []
        for context in contexts:
            marks = []
            if context.name == selected:
                marks.append("active")
            if rules is not None and rules.matches(context.target()):
                marks.append("PROTECTED")
            suffix = f"  [{', '.join(marks)}]" if marks else ""
            lines.append(f"{context.name} (ns={context.namespace}){suffix}")
        return ToolOutcome(content="\n".join(lines), summary=f"{len(contexts)} contexts")


class K8sApiResourcesTool(K8sTool):
    name: ClassVar[str] = "k8s_api_resources"
    description: ClassVar[str] = (
        "List the API resource kinds this cluster serves, custom resources included. "
        "Use it to discover what is available before k8s_list."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {"filter": {"type": "string", "description": "Substring to match."}},
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        resolved = await self.client(ctx)
        if isinstance(resolved, ToolOutcome):
            return resolved
        client, _ = resolved
        groups = sorted(await client.api_groups())
        needle = str(args.get("filter", "")).casefold()
        if needle:
            groups = [g for g in groups if needle in g.casefold()]
        return ToolOutcome(
            content="\n".join(groups) or "(none matched)",
            summary=f"{len(groups)} api groups",
        )


class K8sListTool(K8sTool):
    name: ClassVar[str] = "k8s_list"
    description: ClassVar[str] = (
        "List Kubernetes objects of any kind. Prefer this over shelling out to "
        "kubectl. Secrets are returned with their values redacted."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "description": "Pod, Deployment, Service, a CRD kind…"},
            "api_version": {"type": "string", "description": "Defaults to v1 or apps/v1."},
            "namespace": {"type": "string", "description": "Omit for the default namespace."},
            "all_namespaces": {"type": "boolean"},
            "label_selector": {"type": "string", "description": "e.g. app=web,tier=front"},
        },
        "required": ["kind"],
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        kind = str(args.get("kind", "")).strip()
        if not kind:
            return ToolOutcome.error("kind is required")
        resolved = await self.client(ctx)
        if isinstance(resolved, ToolOutcome):
            return resolved
        client, context_name = resolved

        namespace = None if args.get("all_namespaces") else (args.get("namespace") or "default")
        api_version = await client.resolve_kind(kind, str(args.get("api_version") or ""))
        try:
            items = await client.list_kind(
                api_version, kind, namespace, label_selector=args.get("label_selector")
            )
        except Exception as exc:
            return ToolOutcome.error(f"could not list {kind}: {exc}", summary="failed")

        items = self.scrub(items, ctx)
        if not items:
            where = "all namespaces" if namespace is None else f"namespace {namespace}"
            return ToolOutcome(content=f"no {kind} found in {where}", summary=f"0 {kind}")

        for item in items:
            item.setdefault("kind", kind)
        columns = ["name", "namespace", "status", "created"]
        table = Table(title=f"{kind} — {context_name}", columns=columns, rows=_rows(items, columns))
        return ToolOutcome(
            content=table.to_text(max_rows=60),
            summary=f"{len(items)} {kind}",
            visual=table,
        )


class K8sEventsTool(K8sTool):
    name: ClassVar[str] = "k8s_events"
    description: ClassVar[str] = (
        "Recent cluster events, warnings first. The fastest way to find out why "
        "something is not working."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "namespace": {"type": "string"},
            "all_namespaces": {"type": "boolean"},
        },
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        resolved = await self.client(ctx)
        if isinstance(resolved, ToolOutcome):
            return resolved
        client, _ = resolved
        namespace = None if args.get("all_namespaces") else (args.get("namespace") or "default")
        try:
            events = await client.list_kind("v1", "Event", namespace)
        except Exception as exc:
            return ToolOutcome.error(f"could not read events: {exc}", summary="failed")
        if not events:
            return ToolOutcome(content="no recent events", summary="0 events")

        events.sort(
            key=lambda e: (e.get("type") != "Warning", e.get("lastTimestamp") or ""), reverse=False
        )
        columns = ["type", "reason", "object", "message"]
        rows = [
            [
                str(e.get("type", "")),
                str(e.get("reason", "")),
                f"{(e.get('involvedObject') or {}).get('kind', '')}/"
                f"{(e.get('involvedObject') or {}).get('name', '')}",
                str(e.get("message", ""))[:90],
            ]
            for e in events[:60]
        ]
        warnings = sum(1 for e in events if e.get("type") == "Warning")
        table = Table(title="Events", columns=columns, rows=rows)
        return ToolOutcome(
            content=table.to_text(max_rows=40),
            summary=f"{len(events)} events, {warnings} warnings",
            visual=table,
        )


class K8sLogsTool(K8sTool):
    name: ClassVar[str] = "k8s_logs"
    description: ClassVar[str] = "Container logs for a pod, most recent lines first capped."
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "pod": {"type": "string"},
            "namespace": {"type": "string"},
            "container": {"type": "string"},
            "tail": {"type": "integer", "description": f"Lines, capped at {MAX_LOG_LINES}."},
            "previous": {"type": "boolean", "description": "Logs from the previous crash."},
        },
        "required": ["pod"],
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        pod = str(args.get("pod", "")).strip()
        if not pod:
            return ToolOutcome.error("pod is required")
        resolved = await self.client(ctx)
        if isinstance(resolved, ToolOutcome):
            return resolved
        client, _ = resolved
        namespace = str(args.get("namespace") or "default")
        tail = min(int(args.get("tail") or 200), MAX_LOG_LINES)

        def _fetch() -> str:
            from kubernetes import client as kube_client

            core = kube_client.CoreV1Api(client.dynamic.client)
            return str(
                core.read_namespaced_pod_log(
                    name=pod,
                    namespace=namespace,
                    container=args.get("container"),
                    tail_lines=tail,
                    previous=bool(args.get("previous")),
                )
            )

        try:
            text = await asyncio.to_thread(_fetch)
        except Exception as exc:
            return ToolOutcome.error(f"could not read logs for {pod}: {exc}", summary="failed")

        from wai.cloud.redact import redact_text

        text = redact_text(text, enabled=ctx.cloud.redact_secrets)
        lines = text.splitlines()
        return ToolOutcome(
            content="\n".join(lines) or "(no output)", summary=f"{len(lines)} log lines"
        )


class K8sTopTool(K8sTool):
    name: ClassVar[str] = "k8s_top"
    description: ClassVar[str] = (
        "Live CPU and memory for pods or nodes, from metrics-server. Says so "
        "plainly if metrics-server is not installed."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": ["pods", "nodes"]},
            "namespace": {"type": "string"},
        },
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        resolved = await self.client(ctx)
        if isinstance(resolved, ToolOutcome):
            return resolved
        client, context_name = resolved
        nodes = str(args.get("kind", "pods")) == "nodes"
        namespace = None if nodes else str(args.get("namespace") or "default")
        try:
            items = await client.metrics("NodeMetrics" if nodes else "PodMetrics", namespace)
        except k8s_api.MetricsUnavailable as exc:
            return ToolOutcome.error(str(exc), summary="no metrics-server")
        except Exception as exc:
            return ToolOutcome.error(f"could not read metrics: {exc}", summary="failed")
        if not items:
            return ToolOutcome(content="no metrics returned", summary="0 metrics")

        cpu_bars: list[Bar] = []
        mem_bars: list[Bar] = []
        for item in items:
            name = (item.get("metadata") or {}).get("name", "")
            usage = item.get("usage") or {}
            containers = item.get("containers") or []
            if containers:
                cpu = sum(k8s_api.parse_cpu((c.get("usage") or {}).get("cpu")) for c in containers)
                mem = sum(
                    k8s_api.parse_memory((c.get("usage") or {}).get("memory")) for c in containers
                )
            else:
                cpu = k8s_api.parse_cpu(usage.get("cpu"))
                mem = k8s_api.parse_memory(usage.get("memory"))
            cpu_bars.append(Bar(label=name, value=round(cpu, 1), unit="m"))
            mem_bars.append(Bar(label=name, value=round(mem, 1), unit="Mi"))

        cpu_bars.sort(key=lambda b: b.value, reverse=True)
        mem_bars.sort(key=lambda b: b.value, reverse=True)
        visual = VisualGroup(
            title=f"{'Nodes' if nodes else 'Pods'} — {context_name}",
            items=[
                Bars(title="CPU", bars=cpu_bars[:25], caption="live, from metrics-server"),
                Bars(title="Memory", bars=mem_bars[:25]),
            ],
        )
        return ToolOutcome(
            content=visual.to_text(),
            summary=f"{len(items)} {'nodes' if nodes else 'pods'}",
            visual=visual,
        )


class K8sUsageTool(K8sTool):
    name: ClassVar[str] = "k8s_usage"
    description: ClassVar[str] = (
        "Requests versus limits versus actual usage for a namespace, and quota "
        "headroom. The best single view of whether a namespace is sized right."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {"namespace": {"type": "string"}},
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        resolved = await self.client(ctx)
        if isinstance(resolved, ToolOutcome):
            return resolved
        client, context_name = resolved
        namespace = str(args.get("namespace") or "default")

        try:
            pods = await client.list_kind("v1", "Pod", namespace)
        except Exception as exc:
            return ToolOutcome.error(f"could not list pods: {exc}", summary="failed")
        if not pods:
            return ToolOutcome(content=f"no pods in {namespace}", summary="empty namespace")

        actual: dict[str, tuple[float, float]] = {}
        note = ""
        try:
            for entry in await client.metrics("PodMetrics", namespace):
                name = (entry.get("metadata") or {}).get("name", "")
                containers = entry.get("containers") or []
                actual[name] = (
                    sum(k8s_api.parse_cpu((c.get("usage") or {}).get("cpu")) for c in containers),
                    sum(
                        k8s_api.parse_memory((c.get("usage") or {}).get("memory"))
                        for c in containers
                    ),
                )
        except k8s_api.MetricsUnavailable:
            note = "actual usage unavailable — metrics-server is not installed"
        except Exception:
            note = "actual usage unavailable"

        cpu_bars: list[Bar] = []
        mem_bars: list[Bar] = []
        for pod in pods:
            name = (pod.get("metadata") or {}).get("name", "")
            totals = k8s_api.container_resources(pod)
            used_cpu, used_mem = actual.get(name, (0.0, 0.0))
            cpu_bars.append(
                Bar(
                    label=name,
                    value=round(used_cpu or totals["cpu_request"], 1),
                    limit=totals["cpu_limit"] or None,
                    request=totals["cpu_request"] or None,
                    unit="m",
                    note="" if actual else "(requested; no live metrics)",
                )
            )
            mem_bars.append(
                Bar(
                    label=name,
                    value=round(used_mem or totals["mem_request"], 1),
                    limit=totals["mem_limit"] or None,
                    request=totals["mem_request"] or None,
                    unit="Mi",
                )
            )

        visual = VisualGroup(
            title=f"{namespace} — {context_name}",
            items=[
                Bars(title="CPU: actual vs limit (req marked)", bars=cpu_bars[:25], caption=note),
                Bars(title="Memory: actual vs limit (req marked)", bars=mem_bars[:25]),
            ],
        )
        unlimited = sum(1 for p in pods if not k8s_api.container_resources(p)["cpu_limit"])
        summary = f"{len(pods)} pods"
        content = visual.to_text()
        if unlimited:
            content += f"\n\n{unlimited} pod(s) have no CPU limit set."
        return ToolOutcome(content=content, summary=summary, visual=visual)


class K8sStorageTool(K8sTool):
    name: ClassVar[str] = "k8s_storage"
    description: ClassVar[str] = (
        "Persistent volume claims, their capacity and bound state. Note that "
        "actual fill level is NOT available from metrics-server; it needs Prometheus."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "namespace": {"type": "string"},
            "all_namespaces": {"type": "boolean"},
        },
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        resolved = await self.client(ctx)
        if isinstance(resolved, ToolOutcome):
            return resolved
        client, context_name = resolved
        namespace = None if args.get("all_namespaces") else (args.get("namespace") or "default")
        try:
            claims = await client.list_kind("v1", "PersistentVolumeClaim", namespace)
        except Exception as exc:
            return ToolOutcome.error(f"could not list PVCs: {exc}", summary="failed")
        if not claims:
            return ToolOutcome(content="no persistent volume claims", summary="0 PVCs")

        sized: list[tuple[str, float, str]] = []
        rows: list[list[str]] = []
        for claim in claims:
            meta = claim.get("metadata") or {}
            spec = claim.get("spec") or {}
            status = claim.get("status") or {}
            capacity = k8s_api.parse_memory(
                (status.get("capacity") or {}).get("storage")
            ) or k8s_api.parse_memory(
                ((spec.get("resources") or {}).get("requests") or {}).get("storage")
            )
            label = f"{meta.get('namespace', '')}/{meta.get('name', '')}".strip("/")
            sized.append((label, capacity / 1024, str(status.get("phase", ""))))
            rows.append(
                [
                    str(meta.get("name", "")),
                    str(meta.get("namespace", "")),
                    str(status.get("phase", "")),
                    str(spec.get("storageClassName", "")),
                    ",".join(spec.get("accessModes") or []),
                ]
            )

        # Chart provisioned size relative to the largest claim. Charting
        # requested-against-capacity would show every bound volume at 100%,
        # which reads as "full" and means nothing --- and actual fill level is
        # simply not available from metrics-server.
        sized.sort(key=lambda item: item[1], reverse=True)
        largest = sized[0][1] if sized else 0.0
        bars = [
            Bar(
                label=label,
                value=round(size, 1),
                scale=round(largest, 1) or None,
                unit="Gi",
                note=phase if phase != "Bound" else "",
            )
            for label, size, phase in sized
        ]
        total = sum(size for _, size, _ in sized)
        caption = (
            f"provisioned size, largest first — {total:,.0f}Gi total. "
            "Fill level is unavailable: metrics-server exposes no volume stats, "
            "that needs Prometheus."
        )
        visual = VisualGroup(
            title=f"Storage — {context_name}",
            items=[
                Bars(title="Claims", bars=bars[:25], caption=caption, label_width=36),
                Table(
                    columns=["name", "namespace", "phase", "class", "access"],
                    rows=rows[:40],
                ),
            ],
        )
        pending = sum(1 for c in claims if (c.get("status") or {}).get("phase") != "Bound")
        return ToolOutcome(
            content=visual.to_text(),
            summary=f"{len(claims)} PVCs" + (f", {pending} not bound" if pending else ""),
            visual=visual,
        )


class K8sTopologyTool(K8sTool):
    name: ClassVar[str] = "k8s_topology"
    description: ClassVar[str] = (
        "A birds-eye view of a namespace: what owns what, which Services select "
        "which Pods, what Ingresses route to, and which volumes and ConfigMaps "
        "are in use. Use this to understand how an application hangs together."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {"namespace": {"type": "string"}},
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        resolved = await self.client(ctx)
        if isinstance(resolved, ToolOutcome):
            return resolved
        client, context_name = resolved
        namespace = str(args.get("namespace") or "default")
        try:
            objects = await client.list_many(k8s_api.TOPOLOGY_KINDS, namespace)
        except Exception as exc:
            return ToolOutcome.error(f"could not build topology: {exc}", summary="failed")

        objects = self.scrub(objects, ctx)
        graph = k8s_api.build_graph(objects, namespace, title=f"{namespace} — {context_name}")
        if not graph.nodes:
            return ToolOutcome(content=f"nothing found in {namespace}", summary="empty")

        return ToolOutcome(
            content=graph.to_text(),
            summary=f"{len(graph.nodes)} resources, {len(graph.edges)} links",
            visual=graph,
        )


class K8sExplainTool(K8sTool):
    name: ClassVar[str] = "k8s_explain"
    description: ClassVar[str] = (
        "The schema for a resource kind or one of its fields, read from this "
        "cluster's own OpenAPI. Use it BEFORE writing a manifest: it is "
        "authoritative for this cluster's version and covers custom resources, "
        "so it prevents apply failures rather than reacting to them. "
        "Field paths are dotted, e.g. kind=Deployment field=spec.template.spec."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "description": "Deployment, Certificate, any CRD kind."},
            "field": {"type": "string", "description": "Dotted path, e.g. spec.strategy."},
            "api_version": {"type": "string", "description": "Only if the kind is ambiguous."},
        },
        "required": ["kind"],
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        kind = str(args.get("kind", "")).strip()
        if not kind:
            return ToolOutcome.error("kind is required")
        resolved = await self.client(ctx)
        if isinstance(resolved, ToolOutcome):
            return resolved
        client, _ = resolved

        try:
            api_version = await client.resolve_kind(kind, str(args.get("api_version") or ""))
            document = await client.schema_document(api_version)
            described = k8s_api.explain(document, kind, str(args.get("field") or ""))
        except KeyError as exc:
            return ToolOutcome.error(str(exc).strip("'\""), summary="unknown")
        except Exception as exc:
            return ToolOutcome.error(f"could not read the schema: {exc}", summary="failed")

        lines = [f"{described['path']}  ({api_version})", f"  {described['description']}"]
        if described["required"]:
            lines.append(f"  REQUIRED: {', '.join(described['required'])}")
        lines.append("")
        for name, info in sorted(described["fields"].items()):
            mark = "*" if info["required"] else " "
            lines.append(f"  {mark} {name:<26} {info['type']:<28} {info['description']}")
        if not described["fields"]:
            lines.append("  (a scalar; no sub-fields)")
        table = Table(
            title=described["path"],
            columns=["field", "type", "required", "description"],
            rows=[
                [n, i["type"], "yes" if i["required"] else "", i["description"]]
                for n, i in sorted(described["fields"].items())
            ],
            caption=f"{api_version} — from the cluster's own OpenAPI schema",
        )
        return ToolOutcome(
            content="\n".join(lines),
            summary=f"{described['path']} ({len(described['fields'])} fields)",
            visual=table if described["fields"] else None,
        )


def k8s_tools() -> list[BaseTool]:
    return [
        K8sContextsTool(),
        K8sApiResourcesTool(),
        K8sExplainTool(),
        K8sListTool(),
        K8sEventsTool(),
        K8sLogsTool(),
        K8sTopTool(),
        K8sUsageTool(),
        K8sStorageTool(),
        K8sTopologyTool(),
    ]


def sensitivity_for(kind: str) -> Sensitivity:
    return classify("get", kind)


__all__ = ["k8s_tools", "sensitivity_for"]
