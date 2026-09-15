"""Node lifecycle: cordon, taint, and drain.

Everything here is PRIVILEGED, because it takes capacity out of service rather
than changing a workload. Draining in particular is not one API call --- it is
an algorithm, and the parts that make it safe are the parts that are easy to
leave out.
"""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

from altus.tools.base import ToolContext, ToolOutcome
from altus.tools.k8s.base import K8sMutatingTool

#: Effects the API accepts. A typo here is silently a different policy.
TAINT_EFFECTS = ("NoSchedule", "PreferNoSchedule", "NoExecute")

#: A mirror pod is a static pod's read-only shadow. Evicting it does nothing:
#: the kubelet recreates it from the file on disk, and the eviction call just
#: fails. Skipping it is correct, not a compromise.
MIRROR_ANNOTATION = "kubernetes.io/config.mirror"


def _owner_kinds(pod: dict[str, Any]) -> set[str]:
    return {
        str(ref.get("kind", "")) for ref in (pod.get("metadata") or {}).get("ownerReferences") or []
    }


def _has_local_storage(pod: dict[str, Any]) -> bool:
    """emptyDir is lost when the pod moves. The data may not matter, but that
    is the user's call to make, not ours."""
    return any("emptyDir" in volume for volume in ((pod.get("spec") or {}).get("volumes") or []))


class K8sNodeTool(K8sMutatingTool):
    name: ClassVar[str] = "k8s_node"
    action: ClassVar[str] = "change"
    verb: ClassVar[str] = "patch"
    description: ClassVar[str] = (
        "Control whether a node accepts work: cordon stops new pods being "
        "scheduled there, uncordon allows it again, taint and untaint express "
        "finer policy. None of these move pods that are already running — use "
        "k8s_drain for that. Requires a typed confirmation."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "node": {"type": "string"},
            "action": {
                "type": "string",
                "enum": ["cordon", "uncordon", "taint", "untaint"],
            },
            "key": {"type": "string", "description": "Taint key, for taint/untaint."},
            "value": {"type": "string"},
            "effect": {
                "type": "string",
                "enum": list(TAINT_EFFECTS),
                "description": "NoExecute also evicts pods that do not tolerate it.",
            },
        },
        "required": ["node", "action"],
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        node = str(args.get("node", "")).strip()
        action = str(args.get("action", "")).strip().casefold()
        if not node:
            return ToolOutcome.error("node is required")
        if action not in {"cordon", "uncordon", "taint", "untaint"}:
            return ToolOutcome.error(f"unknown action {action!r}")

        resolved = await self.client(ctx)
        if isinstance(resolved, ToolOutcome):
            return resolved
        client, context_name = resolved

        try:
            live = await client.get_one("v1", "Node", node)
        except Exception as exc:
            return ToolOutcome.error(f"node {node} not found: {exc}", summary="not found")
        spec = live.get("spec") or {}

        if action in {"cordon", "uncordon"}:
            wanted = action == "cordon"
            if bool(spec.get("unschedulable")) is wanted:
                return ToolOutcome(
                    content=f"{node} is already {'cordoned' if wanted else 'schedulable'}",
                    summary="no change",
                )
            patch: dict[str, Any] = {"spec": {"unschedulable": wanted}}
            summary = f"{action} {node}"
            recover = f"reversible: {'uncordon' if wanted else 'cordon'} {node}"
            danger = (
                "new pods will not be scheduled here; running pods stay put"
                if wanted
                else "the node accepts work again"
            )
        else:
            outcome = self._taint_patch(action, spec, args)
            if isinstance(outcome, ToolOutcome):
                return outcome
            patch, summary, recover, danger = outcome

        try:
            await client.patch("v1", "Node", node, patch, dry_run=True)
            dry = "server-side dry run succeeded"
        except Exception as exc:
            return ToolOutcome.error(f"the API server refused it: {exc}", summary="refused")

        refused = await self.confirm(
            ctx,
            client=client,
            context_name=context_name,
            kind="Node",
            name=node,
            namespace="",
            diff=f"{summary}\n{danger}",
            dry_run=dry,
            recoverability=recover,
        )
        if refused is not None:
            return refused

        try:
            await client.patch("v1", "Node", node, patch)
        except Exception as exc:
            return ToolOutcome.error(f"{action} failed: {exc}", summary="failed")
        return ToolOutcome(content=f"{summary} ({context_name})", summary=summary)

    def _taint_patch(
        self, action: str, spec: dict[str, Any], args: dict[str, Any]
    ) -> tuple[dict[str, Any], str, str, str] | ToolOutcome:
        key = str(args.get("key", "")).strip()
        if not key:
            return ToolOutcome.error("key is required to taint or untaint")
        effect = str(args.get("effect") or "NoSchedule")
        if effect not in TAINT_EFFECTS:
            return ToolOutcome.error(f"effect must be one of {', '.join(TAINT_EFFECTS)}")

        taints = list(spec.get("taints") or [])
        if action == "taint":
            if any(t.get("key") == key and t.get("effect") == effect for t in taints):
                return ToolOutcome(content=f"already tainted {key}:{effect}", summary="no change")
            entry = {"key": key, "effect": effect}
            if args.get("value"):
                entry["value"] = str(args["value"])
            taints.append(entry)
            danger = (
                "NoExecute also evicts running pods that do not tolerate it"
                if effect == "NoExecute"
                else "only affects scheduling of new pods"
            )
            return (
                {"spec": {"taints": taints}},
                f"taint {key}={args.get('value', '')}:{effect}",
                f"reversible: untaint {key} {effect}",
                danger,
            )

        remaining = [t for t in taints if not (t.get("key") == key and t.get("effect") == effect)]
        if len(remaining) == len(taints):
            return ToolOutcome(content=f"no taint {key}:{effect} to remove", summary="no change")
        return (
            {"spec": {"taints": remaining}},
            f"untaint {key}:{effect}",
            "reversible: apply the taint again",
            "workloads that were kept off this node may now schedule onto it",
        )


class K8sDrainTool(K8sMutatingTool):
    name: ClassVar[str] = "k8s_drain"
    action: ClassVar[str] = "drain"
    verb: ClassVar[str] = "delete"
    description: ClassVar[str] = (
        "Take a node out of service: cordon it, then evict everything running "
        "on it. Evictions go through the eviction API, so PodDisruptionBudgets "
        "are enforced by the API server. Reports what it skipped and why. "
        "Requires a typed confirmation."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "node": {"type": "string"},
            "delete_emptydir_data": {
                "type": "boolean",
                "description": "Allow evicting pods with emptyDir volumes. That data is lost.",
            },
            "force": {
                "type": "boolean",
                "description": "Also evict pods no controller will recreate. They just go.",
            },
            "timeout": {"type": "integer", "description": "Seconds to wait. Default 120."},
        },
        "required": ["node"],
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        node = str(args.get("node", "")).strip()
        if not node:
            return ToolOutcome.error("node is required")

        resolved = await self.client(ctx)
        if isinstance(resolved, ToolOutcome):
            return resolved
        client, context_name = resolved

        try:
            pods = await client.list_kind("v1", "Pod", None, field_selector=f"spec.nodeName={node}")
        except Exception as exc:
            return ToolOutcome.error(f"could not list pods on {node}: {exc}", summary="failed")

        evict, skipped = self._triage(
            pods,
            allow_emptydir=bool(args.get("delete_emptydir_data")),
            force=bool(args.get("force")),
        )
        blocked = {n: why for n, why in skipped.items() if why.startswith("refusing")}
        if blocked:
            return ToolOutcome.error(
                "refusing to drain:\n"
                + "\n".join(f"  {name}: {why}" for name, why in blocked.items())
                + "\nPass force or delete_emptydir_data if that is what you intend.",
                summary="refused",
            )
        if not evict:
            return ToolOutcome(
                content=f"nothing to evict from {node}"
                + (f"\nskipped: {len(skipped)}" if skipped else ""),
                summary="nothing to do",
            )

        budgets = await self._budgets(client, evict)
        listing = "\n".join(f"  - {ns}/{name}" for ns, name in evict[:40])
        if len(evict) > 40:
            listing += f"\n  ... and {len(evict) - 40} more"
        skipped_note = (
            "\nskipped:\n" + "\n".join(f"  - {n}: {why}" for n, why in list(skipped.items())[:10])
            if skipped
            else ""
        )

        refused = await self.confirm(
            ctx,
            client=client,
            context_name=context_name,
            kind="Node",
            name=node,
            namespace="",
            diff=f"cordon {node}, then evict {len(evict)} pods:\n{listing}{skipped_note}",
            dry_run=budgets,
            recoverability=(
                "the node stays cordoned afterwards; uncordon it to return it to service"
            ),
            verb="deletecollection",
        )
        if refused is not None:
            return refused

        try:
            await client.patch("v1", "Node", node, {"spec": {"unschedulable": True}})
        except Exception as exc:
            return ToolOutcome.error(f"could not cordon {node}: {exc}", summary="failed")

        evicted, failed = await self._evict_all(client, evict)
        lines = [f"drained {node} ({context_name})", f"  cordoned, evicted {len(evicted)} pods"]
        if failed:
            lines.append(f"  {len(failed)} refused by the API server:")
            lines.extend(f"    {name}: {why}" for name, why in failed[:10])
            lines.append("  a PodDisruptionBudget usually means retry once replacements are ready")
        if skipped:
            lines.append(f"  {len(skipped)} skipped (DaemonSet, mirror or standalone pods)")
        return ToolOutcome(
            content="\n".join(lines),
            summary=f"drained {node}",
            is_error=bool(failed),
        )

    def _triage(
        self, pods: list[dict[str, Any]], *, allow_emptydir: bool, force: bool
    ) -> tuple[list[tuple[str, str]], dict[str, str]]:
        """Which pods to evict, and why each of the others is left alone.

        The skips are not an optimisation. A DaemonSet pod is recreated on the
        same node immediately, so evicting it loops; a mirror pod cannot be
        evicted at all.
        """
        evict: list[tuple[str, str]] = []
        skipped: dict[str, str] = {}
        for entry in pods:
            meta = entry.get("metadata") or {}
            name = str(meta.get("name", ""))
            namespace = str(meta.get("namespace", ""))
            label = f"{namespace}/{name}"
            owners = _owner_kinds(entry)

            if MIRROR_ANNOTATION in (meta.get("annotations") or {}):
                skipped[label] = "mirror pod — the kubelet owns it"
            elif "DaemonSet" in owners:
                skipped[label] = "DaemonSet — it would be recreated here immediately"
            elif not owners and not force:
                skipped[label] = "refusing: no controller would recreate it"
            elif _has_local_storage(entry) and not allow_emptydir:
                skipped[label] = "refusing: has emptyDir data that would be lost"
            else:
                evict.append((namespace, name))
        return evict, skipped

    async def _budgets(self, client: Any, evict: list[tuple[str, str]]) -> str:
        """Name the PDBs in play. "evict 12 pods" is not a decision anyone can
        make without knowing what is protecting them."""
        namespaces = {ns for ns, _ in evict}
        found: list[str] = []
        for namespace in sorted(namespaces):
            try:
                budgets = await client.list_kind("policy/v1", "PodDisruptionBudget", namespace)
            except Exception:
                continue
            found.extend(
                f"{namespace}/{(b.get('metadata') or {}).get('name', '')}" for b in budgets
            )
        if not found:
            return (
                "no PodDisruptionBudgets cover these namespaces — evictions will not be held back"
            )
        return f"{len(found)} PodDisruptionBudget(s) will be enforced: {', '.join(found[:6])}"

    async def _evict_all(
        self, client: Any, evict: list[tuple[str, str]]
    ) -> tuple[list[str], list[tuple[str, str]]]:
        """Evictions, not deletes: the API server applies PodDisruptionBudgets
        to an eviction and ignores them for a delete. A 429 here is the budget
        doing its job, so it is reported rather than retried into submission.
        """

        async def evict_one(namespace: str, name: str) -> tuple[str, str | None]:
            body = {
                "apiVersion": "policy/v1",
                "kind": "Eviction",
                "metadata": {"name": name, "namespace": namespace},
            }
            try:
                await client.subresource(
                    "POST", "v1", "Pod", name, "eviction", namespace=namespace, body=body
                )
            except Exception as exc:
                return f"{namespace}/{name}", str(exc)
            return f"{namespace}/{name}", None

        results = await asyncio.gather(*(evict_one(ns, n) for ns, n in evict))
        evicted = [label for label, error in results if error is None]
        failed = [(label, error) for label, error in results if error is not None]
        return evicted, failed
