"""Reads that need no approval: discovery, objects, events, logs.

Every one returns a compact text summary for the model and, where it helps,
a ``Visual`` for the human. The model never sees the chart --- that is what
keeps a whole-cluster topology affordable in context."""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

from wai.core.visuals import Table
from wai.tools.base import ToolContext, ToolOutcome
from wai.tools.k8s.base import MAX_LOG_LINES, K8sTool, _rows


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
