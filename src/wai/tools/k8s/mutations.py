"""Changes to objects. Every one is gated.

Nothing here writes before the approval returns, and the dry-run result is
what goes in the prompt --- what the API server says will happen, rather
than what the model claims will happen.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, ClassVar

from wai.cloud import k8s as k8s_api
from wai.tools.base import ToolContext, ToolOutcome
from wai.tools.k8s.base import K8sMutatingTool


class K8sApplyTool(K8sMutatingTool):
    name: ClassVar[str] = "k8s_apply"
    action: ClassVar[str] = "apply"
    verb: ClassVar[str] = "apply"
    description: ClassVar[str] = (
        "Create or update a resource from a manifest, via server-side apply. "
        "Call k8s_explain first if unsure of the schema: the API server "
        "validates before anything is written, and a schema error here costs a "
        "round trip. Requires user approval."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "manifest": {
                "type": "object",
                "description": "A complete object with apiVersion, kind, metadata and spec.",
            },
            "namespace": {"type": "string", "description": "Overrides metadata.namespace."},
        },
        "required": ["manifest"],
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        manifest = args.get("manifest")
        if not isinstance(manifest, dict):
            return ToolOutcome.error("manifest must be an object")
        kind = str(manifest.get("kind", "")).strip()
        api_version = str(manifest.get("apiVersion", "")).strip()
        meta = manifest.get("metadata") or {}
        name = str(meta.get("name", "")).strip()
        if not kind or not name:
            return ToolOutcome.error("the manifest needs kind and metadata.name")

        resolved = await self.client(ctx)
        if isinstance(resolved, ToolOutcome):
            return resolved
        client, context_name = resolved
        namespace = str(args.get("namespace") or meta.get("namespace") or "default")
        api_version = await client.resolve_kind(kind, api_version)

        try:
            preview = await client.apply(api_version, kind, manifest, namespace, dry_run=True)
        except Exception as exc:
            return ToolOutcome.error(
                f"the API server rejected this manifest: {exc}", summary="invalid"
            )

        try:
            live: dict[str, Any] | None = await client.get_one(api_version, kind, name, namespace)
        except Exception:
            live = None

        refused = await self.confirm(
            ctx,
            client=client,
            context_name=context_name,
            kind=kind,
            name=name,
            namespace=namespace,
            diff=k8s_api.summarise_change(live, preview),
            dry_run="server-side dry run succeeded: the manifest is valid",
            recoverability=(
                "this object already exists and would be updated in place"
                if live
                else "this object does not exist yet and would be created"
            ),
        )
        if refused is not None:
            return refused

        try:
            await client.apply(api_version, kind, manifest, namespace)
        except Exception as exc:
            return ToolOutcome.error(f"apply failed: {exc}", summary="failed")
        verb = "updated" if live else "created"
        return ToolOutcome(
            content=f"{verb} {kind}/{name} in {namespace} ({context_name})",
            summary=f"{verb} {kind}/{name}",
        )


class K8sDeleteTool(K8sMutatingTool):
    name: ClassVar[str] = "k8s_delete"
    action: ClassVar[str] = "delete"
    verb: ClassVar[str] = "delete"
    description: ClassVar[str] = (
        "Delete a resource. Requires user approval and cannot be undone unless "
        "the manifest is stored elsewhere."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "kind": {"type": "string"},
            "name": {"type": "string"},
            "namespace": {"type": "string"},
            "api_version": {"type": "string"},
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
        api_version = await client.resolve_kind(kind, str(args.get("api_version") or ""))

        try:
            live = await client.get_one(api_version, kind, name, namespace)
        except Exception as exc:
            return ToolOutcome.error(
                f"{kind}/{name} not found in {namespace}: {exc}", summary="not found"
            )

        owned = len((live.get("metadata") or {}).get("ownerReferences") or []) > 0
        note = (
            "this object is owned by a controller and will be recreated"
            if owned
            else "nothing here recreates it; deletion is permanent"
        )
        try:
            await client.delete(api_version, kind, name, namespace, dry_run=True)
            dry = "server-side dry run succeeded: the delete is permitted"
        except Exception as exc:
            return ToolOutcome.error(f"the API server refused the delete: {exc}", summary="refused")

        refused = await self.confirm(
            ctx,
            client=client,
            context_name=context_name,
            kind=kind,
            name=name,
            namespace=namespace,
            diff=k8s_api.summarise_change(live, {}),
            dry_run=dry,
            recoverability=note,
        )
        if refused is not None:
            return refused

        try:
            await client.delete(api_version, kind, name, namespace)
        except Exception as exc:
            return ToolOutcome.error(f"delete failed: {exc}", summary="failed")
        return ToolOutcome(
            content=f"deleted {kind}/{name} from {namespace} ({context_name})",
            summary=f"deleted {kind}/{name}",
        )


class K8sScaleTool(K8sMutatingTool):
    name: ClassVar[str] = "k8s_scale"
    action: ClassVar[str] = "scale"
    verb: ClassVar[str] = "scale"
    description: ClassVar[str] = (
        "Change the replica count of a Deployment, StatefulSet or ReplicaSet. "
        "Requires user approval."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "description": "Deployment, StatefulSet or ReplicaSet."},
            "name": {"type": "string"},
            "replicas": {"type": "integer"},
            "namespace": {"type": "string"},
        },
        "required": ["kind", "name", "replicas"],
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        kind, name = str(args.get("kind", "")).strip(), str(args.get("name", "")).strip()
        if "replicas" not in args:
            return ToolOutcome.error("replicas is required")
        try:
            replicas = int(args["replicas"])
        except (TypeError, ValueError):
            return ToolOutcome.error("replicas must be a whole number")
        if replicas < 0:
            return ToolOutcome.error("replicas cannot be negative")

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
        current = ((live.get("spec") or {}).get("replicas")) or 0
        if current == replicas:
            return ToolOutcome(
                content=f"{kind}/{name} is already at {replicas} replicas", summary="no change"
            )

        patch = {"spec": {"replicas": replicas}}
        try:
            await client.patch(api_version, kind, name, patch, namespace, dry_run=True)
            dry = f"server-side dry run succeeded: {current} -> {replicas} replicas"
        except Exception as exc:
            return ToolOutcome.error(f"the API server refused the scale: {exc}", summary="refused")

        refused = await self.confirm(
            ctx,
            client=client,
            context_name=context_name,
            kind=kind,
            name=name,
            namespace=namespace,
            diff=f"replicas: {current} -> {replicas}",
            dry_run=dry,
            recoverability=f"reversible: scale back to {current}",
        )
        if refused is not None:
            return refused

        try:
            await client.patch(api_version, kind, name, patch, namespace)
        except Exception as exc:
            return ToolOutcome.error(f"scale failed: {exc}", summary="failed")
        return ToolOutcome(
            content=f"scaled {kind}/{name} from {current} to {replicas} in {namespace}",
            summary=f"scaled {kind}/{name} to {replicas}",
        )


class K8sRolloutTool(K8sMutatingTool):
    name: ClassVar[str] = "k8s_rollout"
    action: ClassVar[str] = "restart"
    verb: ClassVar[str] = "patch"
    description: ClassVar[str] = (
        "Trigger a rolling restart of a Deployment, StatefulSet or DaemonSet, "
        "the same way `kubectl rollout restart` does. Requires user approval."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "kind": {"type": "string"},
            "name": {"type": "string"},
            "namespace": {"type": "string"},
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

        stamp = datetime.now(UTC).isoformat()
        patch = {
            "spec": {
                "template": {
                    "metadata": {"annotations": {"kubectl.kubernetes.io/restartedAt": stamp}}
                }
            }
        }
        try:
            await client.patch(api_version, kind, name, patch, namespace, dry_run=True)
            dry = "server-side dry run succeeded"
        except Exception as exc:
            return ToolOutcome.error(
                f"the API server refused the restart: {exc}", summary="refused"
            )

        refused = await self.confirm(
            ctx,
            client=client,
            context_name=context_name,
            kind=kind,
            name=name,
            namespace=namespace,
            diff=f"annotate pod template restartedAt={stamp}\n(rolls every pod, one at a time)",
            dry_run=dry,
            recoverability="pods are replaced; the previous ReplicaSet is retained for rollback",
        )
        if refused is not None:
            return refused

        try:
            await client.patch(api_version, kind, name, patch, namespace)
        except Exception as exc:
            return ToolOutcome.error(f"restart failed: {exc}", summary="failed")
        return ToolOutcome(
            content=f"restarting {kind}/{name} in {namespace} ({context_name})",
            summary=f"restarted {kind}/{name}",
        )
