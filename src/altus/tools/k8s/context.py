"""Switching which cluster the session is pointed at.

Not an object mutation, so it does not go through ``K8sMutatingTool``: there
is nothing to dry-run. It still asks every time, because a context switch
changes the blast radius of every command after it --- and the protection
rules are keyed on the context name, so a silent switch is exactly how a
change meant for staging lands in production.

By default the choice is Altus's alone. Writing ``current-context`` into the
kubeconfig would retarget every other terminal the user has open, so that
stays behind ``[cloud] kube_context_scope = "global"``.
"""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

from altus.tools.approval import ApprovalRequest, Decision
from altus.tools.base import ToolContext, ToolOutcome
from altus.tools.k8s.base import K8sTool


class K8sUseContextTool(K8sTool):
    name: ClassVar[str] = "k8s_use_context"
    description: ClassVar[str] = (
        "Point this session at a different Kubernetes context. Everything after "
        "it lands in the new cluster, so it always asks first. List what is "
        "available with k8s_contexts. This does not touch your kubeconfig, so "
        "other terminals are unaffected."
    )
    read_only: ClassVar[bool] = False
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "context": {"type": "string", "description": "The context name, exactly."},
            "namespace": {"type": "string", "description": "Default namespace to use with it."},
        },
        "required": ["context"],
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        from altus.cloud.kube import list_contexts

        wanted = str(args.get("context", "")).strip()
        if not wanted:
            return ToolOutcome.error("context is required")

        contexts, active = await asyncio.to_thread(list_contexts, ctx.cloud.kubeconfigs)
        if not contexts:
            return ToolOutcome.error("no kubeconfig found", summary="no contexts")
        target = next((c for c in contexts if c.name == wanted), None)
        if target is None:
            known = ", ".join(sorted(c.name for c in contexts)[:12])
            return ToolOutcome.error(
                f"no context named {wanted!r}. Available: {known}", summary="unknown context"
            )

        current = ctx.cloud.kube_context or active or "(none)"
        if wanted == current:
            return ToolOutcome(content=f"already using {wanted}", summary="no change")

        namespace = str(args.get("namespace") or target.namespace)
        rules = ctx.cloud.protection
        protected = bool(rules and rules.matches(target.target(namespace)))

        decision = await ctx.approvals.request(
            ApprovalRequest(
                tool=self.name,
                action="switch to",
                path=wanted,
                target=f"cluster {target.cluster or wanted} · namespace {namespace}",
                diff=f"context: {current} -> {wanted}\nnamespace: {namespace}",
                recoverability=f"reversible: switch back to {current}",
                dry_run="your kubeconfig is not modified; this session only",
                destructive=False,
                protected=protected,
            )
        )
        if decision is Decision.DENY:
            return ToolOutcome.rejected("The user declined the context switch.")

        switch = getattr(ctx.cloud, "switch_context", None)
        if switch is None:
            return ToolOutcome.error(
                "this session cannot switch context; use /kube use " + wanted,
                summary="unsupported",
            )
        switch(wanted, namespace)
        note = "  ⚠ protected: changes here need a typed confirmation" if protected else ""
        return ToolOutcome(
            content=f"now using {wanted} (namespace {namespace}){note}",
            summary=f"using {wanted}",
        )
