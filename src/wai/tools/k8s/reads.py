"""Reads that need no approval: discovery, objects, events, logs.

Every one returns a compact text summary for the model and, where it helps,
a ``Visual`` for the human. The model never sees the chart --- that is what
keeps a whole-cluster topology affordable in context."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, ClassVar

from wai.cloud.kube import PRIVILEGED_SUBRESOURCES
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


#: Fields the API server maintains and nobody reads on purpose. Stripping them
#: from a full object is most of the difference between a readable manifest and
#: a wall of bookkeeping.
NOISE_FIELDS = ("managedFields", "generation", "resourceVersion", "uid", "selfLink")


def _quiet(obj: dict[str, Any]) -> dict[str, Any]:
    meta = {k: v for k, v in (obj.get("metadata") or {}).items() if k not in NOISE_FIELDS}
    annotations = meta.get("annotations") or {}
    if "kubectl.kubernetes.io/last-applied-configuration" in annotations:
        # A verbatim copy of the whole object, inside the object. It doubles
        # the token cost of every `kubectl apply`-managed resource.
        meta["annotations"] = {
            k: v
            for k, v in annotations.items()
            if k != "kubectl.kubernetes.io/last-applied-configuration"
        }
    return {**obj, "metadata": meta}


def _as_yaml(obj: Any) -> str:
    import yaml

    # allow_unicode, or safe_dump escapes anything non-ASCII: the redaction
    # marker comes out as "\xABredacted by wai\xBB", and so does every label
    # or annotation with an accent in it.
    return str(
        yaml.safe_dump(
            obj, default_flow_style=False, sort_keys=False, width=100, allow_unicode=True
        )
    )


class K8sGetTool(K8sTool):
    name: ClassVar[str] = "k8s_get"
    description: ClassVar[str] = (
        "The full object, as YAML — every field, not the summary k8s_list gives. "
        "Use it when you need the spec you are about to change, a status "
        "condition, or an annotation. Also reads subresources: status, scale. "
        "Secrets come back redacted."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "kind": {"type": "string"},
            "name": {"type": "string"},
            "namespace": {"type": "string"},
            "api_version": {"type": "string", "description": "Only if the kind is ambiguous."},
            "subresource": {"type": "string", "description": "status or scale."},
            "quiet": {
                "type": "boolean",
                "description": "Strip managedFields and other server bookkeeping. Default true.",
            },
        },
        "required": ["kind", "name"],
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        kind, name = str(args.get("kind", "")).strip(), str(args.get("name", "")).strip()
        if not kind or not name:
            return ToolOutcome.error("kind and name are required")
        subresource = str(args.get("subresource") or "").strip()
        if subresource in PRIVILEGED_SUBRESOURCES:
            return ToolOutcome.error(
                f"{kind}/{name}/{subresource} is not a read. "
                f"Use {_TOOL_FOR.get(subresource, 'the dedicated tool')} instead.",
                summary="wrong tool",
            )
        resolved = await self.client(ctx)
        if isinstance(resolved, ToolOutcome):
            return resolved
        client, context_name = resolved
        namespace = str(args.get("namespace") or "default")
        api_version = await client.resolve_kind(kind, str(args.get("api_version") or ""))

        try:
            obj = await client.get_one(api_version, kind, name, namespace, subresource=subresource)
        except Exception as exc:
            return ToolOutcome.error(f"could not get {kind}/{name}: {exc}", summary="failed")

        obj = self.scrub(obj, ctx)
        if args.get("quiet", True):
            obj = _quiet(obj)
        where = f"{kind}/{name}" + (f"/{subresource}" if subresource else "")
        return ToolOutcome(
            content=f"# {where} in {namespace} ({context_name})\n{_as_yaml(obj)}",
            summary=where,
        )


#: Reached through k8s_get or k8s_raw, these would be a read that is not one.
#: Naming the right tool is worth more than a refusal.
_TOOL_FOR = {
    "exec": "k8s_exec",
    "attach": "k8s_attach",
    "portforward": "k8s_port_forward",
    "eviction": "k8s_drain",
    "token": "k8s_create",
    "approval": "k8s_patch",
}


class K8sRawTool(K8sTool):
    name: ClassVar[str] = "k8s_raw"
    description: ClassVar[str] = (
        "GET any path the API server serves, for the endpoints that are not "
        "resources: /healthz, /readyz, /livez, /version, /metrics, /apis, and "
        "aggregated APIs. Prefer the typed tools for anything that is an object "
        "— this is the escape hatch, not the front door."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "e.g. /healthz, /version, /apis"},
        },
        "required": ["path"],
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        path = str(args.get("path", "")).strip()
        if not path.startswith("/"):
            return ToolOutcome.error("path must be absolute, e.g. /healthz")
        tail = path.rstrip("/").rsplit("/", 1)[-1].casefold()
        if tail in PRIVILEGED_SUBRESOURCES:
            return ToolOutcome.error(
                f"/{tail} is a privileged operation, not a read. "
                f"Use {_TOOL_FOR.get(tail, 'the dedicated tool')}, which asks first.",
                summary="wrong tool",
            )
        resolved = await self.client(ctx)
        if isinstance(resolved, ToolOutcome):
            return resolved
        client, context_name = resolved

        try:
            payload = await client.raw("GET", path)
        except Exception as exc:
            return ToolOutcome.error(f"GET {path} failed: {exc}", summary="failed")

        payload = self.scrub(payload, ctx)
        body = payload["raw"] if set(payload) == {"raw"} else _as_yaml(payload)
        text = str(body)
        if len(text) > 8000:
            text = text[:8000] + "\n[truncated]"
        return ToolOutcome(content=f"# GET {path} ({context_name})\n{text}", summary=path)


class K8sCanITool(K8sTool):
    name: ClassVar[str] = "k8s_can_i"
    description: ClassVar[str] = (
        "Ask the cluster's RBAC whether these credentials may do something, "
        "BEFORE attempting it. Cheaper and clearer than discovering a 403 "
        "halfway through a plan. Omit verb and resource to list everything "
        "you are allowed to do in a namespace."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "verb": {"type": "string", "description": "get, list, create, delete, patch…"},
            "resource": {"type": "string", "description": "Plural, e.g. pods, deployments."},
            "namespace": {"type": "string"},
            "subresource": {"type": "string", "description": "e.g. exec, log."},
            "name": {"type": "string", "description": "A specific object, if it matters."},
            "group": {"type": "string", "description": "API group, e.g. apps. Empty for core."},
        },
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        resolved = await self.client(ctx)
        if isinstance(resolved, ToolOutcome):
            return resolved
        client, context_name = resolved
        namespace = str(args.get("namespace") or "default")
        verb = str(args.get("verb") or "").strip()
        resource = str(args.get("resource") or "").strip()

        if not verb or not resource:
            return await self._rules(client, context_name, namespace)

        attributes = {
            "namespace": namespace,
            "verb": verb,
            "resource": resource,
            "group": str(args.get("group") or ""),
        }
        for key in ("subresource", "name"):
            if args.get(key):
                attributes[key] = str(args[key])
        body = {
            "apiVersion": "authorization.k8s.io/v1",
            "kind": "SelfSubjectAccessReview",
            "spec": {"resourceAttributes": attributes},
        }
        try:
            result = await client.create("authorization.k8s.io/v1", "SelfSubjectAccessReview", body)
        except Exception as exc:
            return ToolOutcome.error(f"the access review failed: {exc}", summary="failed")

        status = result.get("status") or {}
        allowed = bool(status.get("allowed"))
        what = f"{verb} {resource}"
        if args.get("subresource"):
            what += f"/{args['subresource']}"
        verdict = "yes" if allowed else "no"
        reason = status.get("reason") or ""
        detail = f" — {reason}" if reason else ""
        return ToolOutcome(
            content=f"{verdict}: you may{'' if allowed else ' not'} {what} "
            f"in {namespace} ({context_name}){detail}",
            summary=f"{verdict}: {what}",
        )

    async def _rules(self, client: Any, context_name: str, namespace: str) -> ToolOutcome:
        """Everything the credentials may do here. The broad question."""
        body = {
            "apiVersion": "authorization.k8s.io/v1",
            "kind": "SelfSubjectRulesReview",
            "spec": {"namespace": namespace},
        }
        try:
            result = await client.create("authorization.k8s.io/v1", "SelfSubjectRulesReview", body)
        except Exception as exc:
            return ToolOutcome.error(f"the rules review failed: {exc}", summary="failed")

        rules = (result.get("status") or {}).get("resourceRules") or []
        lines = []
        for rule in rules[:80]:
            verbs = ",".join(rule.get("verbs") or [])
            resources = ",".join(rule.get("resources") or [])
            if verbs and resources:
                lines.append(f"  {verbs:<40} {resources}")
        head = f"what you may do in {namespace} ({context_name}):"
        return ToolOutcome(
            content="\n".join([head, *lines]) or f"{head}\n  (nothing)",
            summary=f"{len(rules)} rules",
        )


def _condition_met(obj: dict[str, Any], condition: str, want: str) -> bool:
    for entry in (obj.get("status") or {}).get("conditions") or []:
        if str(entry.get("type", "")).casefold() == condition.casefold():
            return str(entry.get("status", "")).casefold() == want.casefold()
    return False


def _predicate(
    *, deleted: bool, condition: str, want: str, field: str, value: str
) -> Callable[[dict[str, Any]], bool]:
    """What counts as done. Deletion is the odd one: no state of the object
    satisfies it, so the watch runs to its bound and absence is read from the
    transition list instead."""
    if deleted:
        return lambda _obj: False
    if condition:
        return lambda obj: _condition_met(obj, condition, want)
    return lambda obj: _field_equals(obj, field, value)


def _field_equals(obj: dict[str, Any], path: str, want: str) -> bool:
    cursor: Any = obj
    for part in path.split("."):
        if not isinstance(cursor, dict):
            return False
        cursor = cursor.get(part)
    return str(cursor) == want


class K8sWaitTool(K8sTool):
    name: ClassVar[str] = "k8s_wait"
    description: ClassVar[str] = (
        "Watch an object until a condition holds, or give up. Use this after a "
        "change instead of listing repeatedly — it reports the transitions it "
        "saw, so a timeout tells you WHY (image pull, then CrashLoopBackOff) "
        "rather than just that it did not happen. Always bounded."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "kind": {"type": "string"},
            "name": {"type": "string", "description": "Omit to watch by label."},
            "namespace": {"type": "string"},
            "api_version": {"type": "string"},
            "label_selector": {"type": "string"},
            "condition": {
                "type": "string",
                "description": "Condition type, e.g. Ready, Available, Complete.",
            },
            "status": {"type": "string", "description": "Expected value. Default True."},
            "field": {"type": "string", "description": "Dotted path, e.g. status.phase."},
            "value": {"type": "string", "description": "Expected value for `field`."},
            "deleted": {"type": "boolean", "description": "Wait for the object to go away."},
            "timeout": {"type": "integer", "description": "Seconds, capped at 600. Default 120."},
        },
        "required": ["kind"],
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        kind = str(args.get("kind", "")).strip()
        if not kind:
            return ToolOutcome.error("kind is required")

        condition = str(args.get("condition") or "").strip()
        field = str(args.get("field") or "").strip()
        deleted = bool(args.get("deleted"))
        if not (condition or field or deleted):
            return ToolOutcome.error(
                "say what to wait for: condition, field and value, or deleted",
                summary="no condition",
            )

        resolved = await self.client(ctx)
        if isinstance(resolved, ToolOutcome):
            return resolved
        client, context_name = resolved
        namespace = str(args.get("namespace") or "default")
        name = str(args.get("name") or "").strip() or None
        api_version = await client.resolve_kind(kind, str(args.get("api_version") or ""))

        ready = _predicate(
            deleted=deleted,
            condition=condition,
            want=str(args.get("status") or "True"),
            field=field,
            value=str(args.get("value") or ""),
        )
        try:
            satisfied, transitions, why = await client.watch_until(
                api_version,
                kind,
                namespace,
                ready,
                name=name,
                label_selector=args.get("label_selector"),
                timeout_seconds=float(args.get("timeout") or 120),
            )
        except Exception as exc:
            return ToolOutcome.error(f"the watch failed: {exc}", summary="failed")

        if deleted:
            satisfied = any(t.get("type") == "DELETED" for t in transitions)
            why = "the object was deleted" if satisfied else why

        seen = [f"  {t['type']:<10} {t['name']:<40} {t['status']}" for t in transitions[-20:]]
        target = f"{kind}/{name}" if name else f"{kind} ({args.get('label_selector') or 'all'})"
        head = (
            f"{'met' if satisfied else 'NOT met'}: {target} in {namespace} ({context_name}) — {why}"
        )
        return ToolOutcome(
            content="\n".join([head, *seen]) if seen else f"{head}\n  (no events seen)",
            summary="condition met" if satisfied else "timed out",
            is_error=not satisfied,
        )


class K8sRolloutStatusTool(K8sTool):
    name: ClassVar[str] = "k8s_rollout_status"
    description: ClassVar[str] = (
        "Where a rollout has got to, and the revisions available to roll back "
        "to. Reading this needs no approval, which is why it is separate from "
        "k8s_rollout — checking on a deploy should never prompt."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "description": "Deployment, StatefulSet or DaemonSet."},
            "name": {"type": "string"},
            "namespace": {"type": "string"},
            "history": {"type": "boolean", "description": "Also list the retained revisions."},
        },
        "required": ["kind", "name"],
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        kind, name = str(args.get("kind", "")).strip(), str(args.get("name", "")).strip()
        if not kind or not name:
            return ToolOutcome.error("kind and name are required")
        resolved = await self.client(ctx)
        if isinstance(resolved, ToolOutcome):
            return resolved
        client, context_name = resolved
        namespace = str(args.get("namespace") or "default")
        api_version = await client.resolve_kind(kind, "")

        try:
            live = await client.get_one(api_version, kind, name, namespace)
        except Exception as exc:
            return ToolOutcome.error(f"{kind}/{name} not found: {exc}", summary="not found")

        spec, status = live.get("spec") or {}, live.get("status") or {}
        wanted = spec.get("replicas", 1)
        lines = [f"{kind}/{name} in {namespace} ({context_name})"]
        if spec.get("paused"):
            lines.append("  PAUSED — resume it with k8s_rollout action=resume")
        for label, key in (
            ("desired", None),
            ("updated", "updatedReplicas"),
            ("ready", "readyReplicas"),
            ("available", "availableReplicas"),
            ("unavailable", "unavailableReplicas"),
        ):
            value = wanted if key is None else status.get(key, 0)
            lines.append(f"  {label:<12} {value}")

        observed = status.get("observedGeneration")
        generation = (live.get("metadata") or {}).get("generation")
        if observed is not None and generation is not None and observed < generation:
            lines.append("  the controller has not yet observed the latest change")
        complete = status.get("updatedReplicas") == wanted and status.get("readyReplicas") == wanted
        lines.append(f"  {'rollout complete' if complete else 'rollout in progress'}")

        if args.get("history"):
            lines.extend(await self._history(client, kind, name, namespace))
        return ToolOutcome(
            content="\n".join(lines),
            summary="complete" if complete else "in progress",
        )

    async def _history(self, client: Any, kind: str, name: str, namespace: str) -> list[str]:
        """Revisions are ReplicaSets, which is also how `kubectl rollout undo`
        finds them --- there is no history API to ask."""
        if kind != "Deployment":
            return ["  (revision history is only tracked for Deployments)"]
        try:
            sets = await client.list_kind("apps/v1", "ReplicaSet", namespace)
        except Exception as exc:
            return [f"  (could not read revisions: {exc})"]

        rows = []
        for entry in sets:
            meta = entry.get("metadata") or {}
            owners = meta.get("ownerReferences") or []
            if not any(o.get("kind") == "Deployment" and o.get("name") == name for o in owners):
                continue
            revision = (meta.get("annotations") or {}).get("deployment.kubernetes.io/revision", "?")
            images = [
                str(c.get("image", ""))
                for c in (((entry.get("spec") or {}).get("template") or {}).get("spec") or {}).get(
                    "containers"
                )
                or []
            ]
            rows.append((revision, meta.get("name", ""), ", ".join(images)))
        if not rows:
            return ["  (no revisions retained)"]
        rows.sort(key=lambda r: str(r[0]), reverse=True)
        return ["  revisions:", *(f"    {rev:<4} {rs:<34} {img}" for rev, rs, img in rows)]
