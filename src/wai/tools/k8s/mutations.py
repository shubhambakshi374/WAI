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
        "Delete a resource by name, or a whole set of them with label_selector. "
        "Requires user approval and cannot be undone unless the manifest is "
        "stored elsewhere. A selector delete is shown to the user as the actual "
        "list of objects it would remove, not as the selector."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "kind": {"type": "string"},
            "name": {"type": "string", "description": "One object. Omit if using label_selector."},
            "label_selector": {
                "type": "string",
                "description": "Delete every match, e.g. app=web. Never matches everything.",
            },
            "namespace": {"type": "string"},
            "api_version": {"type": "string"},
        },
        "required": ["kind"],
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        kind, name = str(args.get("kind", "")).strip(), str(args.get("name", "")).strip()
        selector = str(args.get("label_selector") or "").strip()
        if not kind:
            return ToolOutcome.error("kind is required")
        if name and selector:
            return ToolOutcome.error("give a name or a label_selector, not both")
        if not name and not selector:
            return ToolOutcome.error(
                "give a name, or a label_selector to delete a set. "
                "There is deliberately no way to ask for everything of a kind."
            )
        if selector:
            return await self._delete_collection(kind, selector, args, ctx)
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

    async def _delete_collection(
        self, kind: str, selector: str, args: dict[str, Any], ctx: ToolContext
    ) -> ToolOutcome:
        """One call, an unbounded number of objects, and no per-object prompt.

        So the prompt has to carry the weight: list what actually matches before
        asking, because approving `app=web` is not consent to delete whatever
        happens to wear that label today.
        """
        resolved = await self.client(ctx)
        if isinstance(resolved, ToolOutcome):
            return resolved
        client, context_name = resolved
        namespace = str(args.get("namespace") or "default")
        api_version = await client.resolve_kind(kind, str(args.get("api_version") or ""))

        try:
            matches = await client.list_kind(api_version, kind, namespace, label_selector=selector)
        except Exception as exc:
            return ToolOutcome.error(f"could not list {kind}: {exc}", summary="failed")
        if not matches:
            return ToolOutcome(
                content=f"nothing matches {selector} in {namespace}; nothing deleted",
                summary="no matches",
            )

        names = [str((m.get("metadata") or {}).get("name", "")) for m in matches]
        listing = "\n".join(f"  - {kind}/{n}" for n in names[:50])
        if len(names) > 50:
            listing += f"\n  ... and {len(names) - 50} more"

        try:
            await client.delete_collection(
                api_version, kind, namespace, label_selector=selector, dry_run=True
            )
            dry = f"server-side dry run succeeded: {len(names)} objects would be deleted"
        except Exception as exc:
            return ToolOutcome.error(f"the API server refused the delete: {exc}", summary="refused")

        refused = await self.confirm(
            ctx,
            client=client,
            context_name=context_name,
            kind=kind,
            name=f"{len(names)} matching {selector}",
            namespace=namespace,
            diff=f"delete {len(names)} {kind}:\n{listing}",
            dry_run=dry,
            recoverability="permanent unless the manifests are stored elsewhere",
            verb="deletecollection",
        )
        if refused is not None:
            return refused

        try:
            await client.delete_collection(api_version, kind, namespace, label_selector=selector)
        except Exception as exc:
            return ToolOutcome.error(f"delete failed: {exc}", summary="failed")
        return ToolOutcome(
            content=f"deleted {len(names)} {kind} matching {selector} "
            f"from {namespace} ({context_name})",
            summary=f"deleted {len(names)} {kind}",
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
        "Drive a rollout: restart, pause, resume, or undo it back to the "
        "previous revision. Read the current state with k8s_rollout_status "
        "instead — that one asks nothing. Requires user approval."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "kind": {"type": "string"},
            "name": {"type": "string"},
            "namespace": {"type": "string"},
            "action": {
                "type": "string",
                "enum": ["restart", "pause", "resume", "undo"],
                "description": "Default restart.",
            },
        },
        "required": ["kind", "name"],
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        kind, name = str(args.get("kind", "")).strip(), str(args.get("name", "")).strip()
        if not kind or not name:
            return ToolOutcome.error("kind and name are required")
        action = str(args.get("action") or "restart").casefold()
        if action not in {"restart", "pause", "resume", "undo"}:
            return ToolOutcome.error(f"unknown action {action!r}")
        if action != "restart":
            return await self._other(action, kind, name, args, ctx)
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

    async def _other(
        self, action: str, kind: str, name: str, args: dict[str, Any], ctx: ToolContext
    ) -> ToolOutcome:
        """pause, resume and undo. All three are a patch underneath."""
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

        if action in {"pause", "resume"}:
            paused = action == "pause"
            if bool((live.get("spec") or {}).get("paused")) is paused:
                return ToolOutcome(
                    content=f"{kind}/{name} is already {'paused' if paused else 'running'}",
                    summary="no change",
                )
            patch: dict[str, Any] = {"spec": {"paused": paused}}
            summary = f"{action} the rollout of {kind}/{name}"
            recover = f"reversible: {'resume' if paused else 'pause'}"
        else:
            previous = await self._previous_template(client, api_version, kind, name, namespace)
            if previous is None:
                return ToolOutcome.error(
                    f"no previous revision of {kind}/{name} is retained, so there is "
                    "nothing to roll back to",
                    summary="no history",
                )
            patch = {"spec": {"template": previous}}
            summary = f"roll {kind}/{name} back to its previous revision"
            recover = "the current revision is retained, so this can be undone again"

        try:
            await client.patch(api_version, kind, name, patch, namespace, dry_run=True)
            dry = "server-side dry run succeeded"
        except Exception as exc:
            return ToolOutcome.error(f"the API server refused it: {exc}", summary="refused")

        refused = await self.confirm(
            ctx,
            client=client,
            context_name=context_name,
            kind=kind,
            name=name,
            namespace=namespace,
            diff=summary,
            dry_run=dry,
            recoverability=recover,
        )
        if refused is not None:
            return refused

        try:
            await client.patch(api_version, kind, name, patch, namespace)
        except Exception as exc:
            return ToolOutcome.error(f"{action} failed: {exc}", summary="failed")
        return ToolOutcome(
            content=f"{summary} in {namespace} ({context_name})", summary=f"{action} {kind}/{name}"
        )

    async def _previous_template(
        self, client: Any, api_version: str, kind: str, name: str, namespace: str
    ) -> dict[str, Any] | None:
        """The pod template of the revision before the current one.

        Kubernetes keeps no rollback API --- `kubectl rollout undo` reads the
        old ReplicaSet and patches its template back, so that is what this does.
        """
        if kind != "Deployment":
            return None
        try:
            sets = await client.list_kind("apps/v1", "ReplicaSet", namespace)
        except Exception:
            return None

        owned = [
            rs
            for rs in sets
            if any(
                ref.get("kind") == "Deployment" and ref.get("name") == name
                for ref in (rs.get("metadata") or {}).get("ownerReferences") or []
            )
        ]

        def revision(rs: dict[str, Any]) -> int:
            raw = ((rs.get("metadata") or {}).get("annotations") or {}).get(
                "deployment.kubernetes.io/revision", "0"
            )
            try:
                return int(raw)
            except (TypeError, ValueError):
                return 0

        owned.sort(key=revision, reverse=True)
        if len(owned) < 2:
            return None
        template = (owned[1].get("spec") or {}).get("template")
        if not isinstance(template, dict):
            return None
        # The hash label belongs to the old ReplicaSet, not to the Deployment's
        # template; carrying it over makes the new ReplicaSet un-selectable.
        labels = {
            k: v
            for k, v in ((template.get("metadata") or {}).get("labels") or {}).items()
            if k != "pod-template-hash"
        }
        return {**template, "metadata": {**(template.get("metadata") or {}), "labels": labels}}


class K8sPatchTool(K8sMutatingTool):
    name: ClassVar[str] = "k8s_patch"
    action: ClassVar[str] = "patch"
    verb: ClassVar[str] = "patch"
    description: ClassVar[str] = (
        "Change part of an existing object: labels, annotations, a single field, "
        "a subresource. Prefer this over k8s_apply when you are editing something "
        "you did not author, because apply takes ownership of every field it "
        "sends. patch_type matters: `strategic` edits one entry of a keyed list "
        "such as spec.containers, `merge` replaces the whole list, `json` is an "
        "RFC6902 operation array. Requires user approval."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "kind": {"type": "string"},
            "name": {"type": "string"},
            "namespace": {"type": "string"},
            "api_version": {"type": "string"},
            "patch": {
                "description": "An object, or an array of operations for patch_type=json.",
            },
            "patch_type": {
                "type": "string",
                "enum": ["strategic", "merge", "json"],
                "description": "Default strategic, which is what kubectl edit uses.",
            },
            "subresource": {"type": "string", "description": "status or scale."},
        },
        "required": ["kind", "name", "patch"],
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        kind, name = str(args.get("kind", "")).strip(), str(args.get("name", "")).strip()
        if not kind or not name:
            return ToolOutcome.error("kind and name are required")
        patch = args.get("patch")
        patch_type = str(args.get("patch_type") or "strategic")
        if patch_type == "json":
            if not isinstance(patch, list):
                return ToolOutcome.error("a json patch must be an array of operations")
        elif not isinstance(patch, dict):
            return ToolOutcome.error("patch must be an object")
        if patch_type not in k8s_api.PATCH_TYPES:
            return ToolOutcome.error(
                f"unknown patch_type {patch_type!r}; use {', '.join(k8s_api.PATCH_TYPES)}"
            )
        subresource = str(args.get("subresource") or "").strip()

        resolved = await self.client(ctx)
        if isinstance(resolved, ToolOutcome):
            return resolved
        client, context_name = resolved
        namespace = str(args.get("namespace") or "default")
        api_version = await client.resolve_kind(kind, str(args.get("api_version") or ""))

        try:
            live = await client.get_one(api_version, kind, name, namespace)
        except Exception as exc:
            return ToolOutcome.error(f"{kind}/{name} not found: {exc}", summary="not found")

        try:
            preview = await client.patch(
                api_version,
                kind,
                name,
                patch,
                namespace,
                dry_run=True,
                patch_type=patch_type,
            )
            dry = "server-side dry run succeeded"
        except Exception as exc:
            return ToolOutcome.error(f"the API server refused the patch: {exc}", summary="refused")

        refused = await self.confirm(
            ctx,
            client=client,
            context_name=context_name,
            kind=kind,
            name=name,
            namespace=namespace,
            diff=k8s_api.summarise_change(live, preview),
            dry_run=dry,
            recoverability="reversible by patching the previous values back",
            subresource=subresource,
        )
        if refused is not None:
            return refused

        try:
            await client.patch(api_version, kind, name, patch, namespace, patch_type=patch_type)
        except Exception as exc:
            return ToolOutcome.error(f"patch failed: {exc}", summary="failed")
        return ToolOutcome(
            content=f"patched {kind}/{name} in {namespace} ({context_name})",
            summary=f"patched {kind}/{name}",
        )


class K8sCreateTool(K8sMutatingTool):
    name: ClassVar[str] = "k8s_create"
    action: ClassVar[str] = "create"
    verb: ClassVar[str] = "create"
    description: ClassVar[str] = (
        "Create an object that must not already exist, or create a subresource. "
        "Unlike k8s_apply this fails rather than adopting something already "
        "there, which is what you want for generateName and for subresource "
        "creates: evictions, ServiceAccount tokens, CSR approvals, bindings. "
        "Requires user approval."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "manifest": {"type": "object", "description": "A complete object."},
            "namespace": {"type": "string"},
            "on": {
                "type": "string",
                "description": "Create a subresource ON this object, by name.",
            },
            "subresource": {"type": "string", "description": "e.g. token, eviction, approval."},
        },
        "required": ["manifest"],
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        manifest = args.get("manifest")
        if not isinstance(manifest, dict):
            return ToolOutcome.error("manifest must be an object")
        kind = str(manifest.get("kind", "")).strip()
        if not kind:
            return ToolOutcome.error("the manifest needs a kind")
        meta = manifest.get("metadata") or {}
        name = str(meta.get("name") or meta.get("generateName") or "").strip()
        subresource = str(args.get("subresource") or "").strip()
        on = str(args.get("on") or "").strip()
        if subresource and not on:
            return ToolOutcome.error("say which object to create the subresource on, with `on`")

        resolved = await self.client(ctx)
        if isinstance(resolved, ToolOutcome):
            return resolved
        client, context_name = resolved
        namespace = str(args.get("namespace") or meta.get("namespace") or "default")
        api_version = str(manifest.get("apiVersion") or "") or await client.resolve_kind(kind, "")

        target = f"{kind}/{on}/{subresource}" if subresource else f"{kind}/{name or '(generated)'}"
        if subresource:
            dry = "subresource creates cannot be dry-run; the server acts when called"
        else:
            try:
                await client.create(api_version, kind, manifest, namespace, dry_run=True)
                dry = "server-side dry run succeeded: the create is permitted"
            except Exception as exc:
                return ToolOutcome.error(
                    f"the API server refused the create: {exc}", summary="refused"
                )

        refused = await self.confirm(
            ctx,
            client=client,
            context_name=context_name,
            kind=kind,
            name=on or name,
            namespace=namespace,
            diff=k8s_api.summarise_change(None, manifest),
            dry_run=dry,
            recoverability=(
                "a credential created here outlives this session"
                if subresource in {"token", "approval"}
                else f"reversible by deleting {target}"
            ),
            subresource=subresource,
        )
        if refused is not None:
            return refused

        try:
            if subresource:
                result = await client.subresource(
                    "POST",
                    api_version,
                    kind,
                    on,
                    subresource,
                    namespace=namespace,
                    body=manifest,
                )
            else:
                result = await client.create(api_version, kind, manifest, namespace)
        except Exception as exc:
            return ToolOutcome.error(f"create failed: {exc}", summary="failed")

        created = (result.get("metadata") or {}).get("name") or name
        return ToolOutcome(
            content=f"created {kind}/{created} in {namespace} ({context_name})",
            summary=f"created {kind}/{created}",
        )


class K8sReplaceTool(K8sMutatingTool):
    name: ClassVar[str] = "k8s_replace"
    action: ClassVar[str] = "replace"
    verb: ClassVar[str] = "replace"
    description: ClassVar[str] = (
        "Replace an object wholesale with the manifest given. Needs "
        "metadata.resourceVersion from a fresh k8s_get, so it fails rather than "
        "clobbering somebody else's concurrent edit. Use k8s_apply or k8s_patch "
        "unless you specifically need to remove fields. Requires user approval."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "manifest": {"type": "object"},
            "namespace": {"type": "string"},
        },
        "required": ["manifest"],
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        manifest = args.get("manifest")
        if not isinstance(manifest, dict):
            return ToolOutcome.error("manifest must be an object")
        kind = str(manifest.get("kind", "")).strip()
        meta = manifest.get("metadata") or {}
        name = str(meta.get("name", "")).strip()
        if not kind or not name:
            return ToolOutcome.error("the manifest needs a kind and metadata.name")
        if not meta.get("resourceVersion"):
            return ToolOutcome.error(
                "replace needs metadata.resourceVersion — read the object with k8s_get "
                "first, so a concurrent edit fails loudly instead of being overwritten",
                summary="no resourceVersion",
            )

        resolved = await self.client(ctx)
        if isinstance(resolved, ToolOutcome):
            return resolved
        client, context_name = resolved
        namespace = str(args.get("namespace") or meta.get("namespace") or "default")
        api_version = str(manifest.get("apiVersion") or "") or await client.resolve_kind(kind, "")

        try:
            live = await client.get_one(api_version, kind, name, namespace)
        except Exception as exc:
            return ToolOutcome.error(f"{kind}/{name} not found: {exc}", summary="not found")
        try:
            await client.replace(api_version, kind, name, manifest, namespace, dry_run=True)
            dry = "server-side dry run succeeded"
        except Exception as exc:
            return ToolOutcome.error(
                f"the API server refused the replace: {exc}", summary="refused"
            )

        refused = await self.confirm(
            ctx,
            client=client,
            context_name=context_name,
            kind=kind,
            name=name,
            namespace=namespace,
            diff=k8s_api.summarise_change(live, manifest),
            dry_run=dry,
            recoverability="fields absent from the manifest are removed, not left alone",
        )
        if refused is not None:
            return refused

        try:
            await client.replace(api_version, kind, name, manifest, namespace)
        except Exception as exc:
            return ToolOutcome.error(f"replace failed: {exc}", summary="failed")
        return ToolOutcome(
            content=f"replaced {kind}/{name} in {namespace} ({context_name})",
            summary=f"replaced {kind}/{name}",
        )
