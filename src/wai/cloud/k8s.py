"""Kubernetes reads, and the relationship model behind the topology view.

Headless. The dynamic client is injected, so every test here drives recorded
API payloads rather than a cluster.

The SDK is synchronous, so anything touching the network goes through
``asyncio.to_thread``; topology needs a dozen resource types and fetching them
serially is a visible stall.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any

from wai.core.visuals import GraphEdge, GraphNode, ResourceGraph

METRICS_API = "metrics.k8s.io/v1beta1"
METRICS_MISSING = (
    "metrics-server is not installed on this cluster, so live CPU and memory "
    "are unavailable. Requests, limits and object counts still work. "
    "Install it with: kubectl apply -f "
    "https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/"
    "components.yaml"
)

#: What topology walks. Ordered cheapest-first so a partial failure still
#: yields something useful.
TOPOLOGY_KINDS: tuple[tuple[str, str], ...] = (
    ("apps/v1", "Deployment"),
    ("apps/v1", "StatefulSet"),
    ("apps/v1", "DaemonSet"),
    ("apps/v1", "ReplicaSet"),
    ("batch/v1", "CronJob"),
    ("batch/v1", "Job"),
    ("v1", "Pod"),
    ("v1", "Service"),
    ("networking.k8s.io/v1", "Ingress"),
    ("v1", "PersistentVolumeClaim"),
    ("v1", "ConfigMap"),
    ("autoscaling/v2", "HorizontalPodAutoscaler"),
)

DEFAULT_LIMIT = 500
DISCOVERY_TIMEOUT = 30.0


class MetricsUnavailable(RuntimeError):
    """metrics-server is absent. Carries the install hint, not a bare failure."""

    def __init__(self) -> None:
        super().__init__(METRICS_MISSING)


@dataclass
class K8sClient:
    """A thin async face over the synchronous dynamic client."""

    dynamic: Any
    context: str = ""
    _apis: set[str] | None = field(default=None, repr=False)
    _openapi_index: dict[str, Any] | None = field(default=None, repr=False)
    _schema_cache: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)

    async def api_groups(self) -> set[str]:
        if self._apis is None:

            def _fetch() -> set[str]:
                groups: set[str] = set()
                for resource in self.dynamic.resources.search():
                    version = getattr(resource, "group_version", None)
                    if version:
                        groups.add(version)
                return groups

            self._apis = await asyncio.to_thread(_fetch)
        return self._apis

    async def api_available(self, group_version: str) -> bool:
        return group_version in await self.api_groups()

    async def list_kind(
        self,
        api_version: str,
        kind: str,
        namespace: str | None = None,
        *,
        label_selector: str | None = None,
        limit: int = DEFAULT_LIMIT,
    ) -> list[dict[str, Any]]:
        """One kind. Returns plain dicts --- no SDK objects escape this module.

        Deliberately not called ``list``: a method of that name shadows the
        builtin inside the class body, so every ``list[...]`` annotation here
        would silently resolve to the method instead.
        """

        def _fetch() -> list[dict[str, Any]]:
            resource = self.dynamic.resources.get(api_version=api_version, kind=kind)
            kwargs: dict[str, Any] = {"limit": limit}
            if namespace:
                kwargs["namespace"] = namespace
            if label_selector:
                kwargs["label_selector"] = label_selector
            result = resource.get(**kwargs)
            raw = result.to_dict() if hasattr(result, "to_dict") else dict(result)
            return list(raw.get("items") or [])

        return await asyncio.to_thread(_fetch)

    async def list_many(
        self,
        kinds: tuple[tuple[str, str], ...],
        namespace: str | None = None,
        *,
        limit: int = DEFAULT_LIMIT,
    ) -> dict[str, list[dict[str, Any]]]:
        """Several kinds concurrently. A kind that fails yields [] rather than
        taking the whole view down --- RBAC often permits some and not others."""
        results = await asyncio.gather(
            *(self.list_kind(api, kind, namespace, limit=limit) for api, kind in kinds),
            return_exceptions=True,
        )
        out: dict[str, list[dict[str, Any]]] = {}
        for (_, kind), result in zip(kinds, results, strict=True):
            out[kind] = [] if isinstance(result, BaseException) else result
        return out

    # ------------------------------------------------------------- schema

    async def resolve_kind(self, kind: str, api_version: str = "") -> str:
        """The apiVersion this cluster actually serves for a kind.

        Discovery knows; guessing does not. A hardcoded table covers the dozen
        built-in kinds and is wrong for every CRD --- and this cluster serves
        107 non-core API groups.
        """
        if api_version:
            return api_version

        def _search() -> str:
            found = list(self.dynamic.resources.search(kind=kind))
            if not found:
                return ""

            # Prefer a stable version over alpha/beta, then the shortest group.
            def rank(resource: Any) -> tuple[int, int]:
                version = str(getattr(resource, "group_version", ""))
                unstable = 1 if ("alpha" in version or "beta" in version) else 0
                return (unstable, len(version))

            return str(getattr(sorted(found, key=rank)[0], "group_version", ""))

        return await asyncio.to_thread(_search) or "v1"

    async def openapi_index(self) -> dict[str, Any]:
        if self._openapi_index is None:
            self._openapi_index = await asyncio.to_thread(self._get_json, "/openapi/v3")
        return self._openapi_index

    async def schema_document(self, group_version: str) -> dict[str, Any]:
        """The OpenAPI document for one API group, cached."""
        if group_version in self._schema_cache:
            return self._schema_cache[group_version]
        index = await self.openapi_index()
        key = f"apis/{group_version}" if "/" in group_version else f"api/{group_version}"
        entry = (index.get("paths") or {}).get(key)
        if entry is None:
            raise KeyError(f"the cluster serves no schema for {group_version}")
        document = await asyncio.to_thread(self._get_json, entry["serverRelativeURL"])
        self._schema_cache[group_version] = document
        return document

    def _get_json(self, path: str) -> dict[str, Any]:
        import json

        response = self.dynamic.client.call_api(
            path,
            "GET",
            auth_settings=["BearerToken"],
            _preload_content=False,
            _return_http_data_only=True,
        )
        parsed: dict[str, Any] = json.loads(response.data)
        return parsed

    # ------------------------------------------------------------ mutation

    async def apply(
        self,
        api_version: str,
        kind: str,
        body: dict[str, Any],
        namespace: str | None = None,
        *,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Server-side apply. With dry_run the API server validates and reports
        what would change without persisting anything."""

        def _do() -> dict[str, Any]:
            resource = self.dynamic.resources.get(api_version=api_version, kind=kind)
            kwargs: dict[str, Any] = {
                "body": body,
                "field_manager": "wai",
                "force_conflicts": True,
            }
            if namespace:
                kwargs["namespace"] = namespace
            if dry_run:
                kwargs["query_params"] = [("dryRun", "All")]
            result = resource.server_side_apply(**kwargs)
            return dict(result.to_dict() if hasattr(result, "to_dict") else result)

        return await asyncio.to_thread(_do)

    async def delete(
        self,
        api_version: str,
        kind: str,
        name: str,
        namespace: str | None = None,
        *,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        def _do() -> dict[str, Any]:
            resource = self.dynamic.resources.get(api_version=api_version, kind=kind)
            kwargs: dict[str, Any] = {"name": name}
            if namespace:
                kwargs["namespace"] = namespace
            if dry_run:
                kwargs["query_params"] = [("dryRun", "All")]
            result = resource.delete(**kwargs)
            return dict(result.to_dict() if hasattr(result, "to_dict") else result or {})

        return await asyncio.to_thread(_do)

    async def patch(
        self,
        api_version: str,
        kind: str,
        name: str,
        patch: dict[str, Any],
        namespace: str | None = None,
        *,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        def _do() -> dict[str, Any]:
            resource = self.dynamic.resources.get(api_version=api_version, kind=kind)
            kwargs: dict[str, Any] = {
                "name": name,
                "body": patch,
                "content_type": "application/merge-patch+json",
            }
            if namespace:
                kwargs["namespace"] = namespace
            if dry_run:
                kwargs["query_params"] = [("dryRun", "All")]
            result = resource.patch(**kwargs)
            return dict(result.to_dict() if hasattr(result, "to_dict") else result)

        return await asyncio.to_thread(_do)

    async def get_one(
        self, api_version: str, kind: str, name: str, namespace: str | None = None
    ) -> dict[str, Any]:
        def _do() -> dict[str, Any]:
            resource = self.dynamic.resources.get(api_version=api_version, kind=kind)
            kwargs: dict[str, Any] = {"name": name}
            if namespace:
                kwargs["namespace"] = namespace
            result = resource.get(**kwargs)
            return dict(result.to_dict() if hasattr(result, "to_dict") else result)

        return await asyncio.to_thread(_do)

    async def metrics(self, kind: str, namespace: str | None = None) -> list[dict[str, Any]]:
        if not await self.api_available(METRICS_API):
            raise MetricsUnavailable
        return await self.list_kind(METRICS_API, kind, namespace)


# ------------------------------------------------------------- quantity maths


def parse_cpu(value: str | None) -> float:
    """Kubernetes CPU quantity to millicores. `100m` -> 100, `2` -> 2000."""
    if not value:
        return 0.0
    text = str(value).strip()
    if text.endswith("m"):
        return float(text[:-1])
    if text.endswith("n"):  # nanocores, what metrics-server reports
        return float(text[:-1]) / 1_000_000
    if text.endswith("u"):
        return float(text[:-1]) / 1_000
    return float(text) * 1000


_MEMORY_UNITS = {
    "Ki": 1024,
    "Mi": 1024**2,
    "Gi": 1024**3,
    "Ti": 1024**4,
    "Pi": 1024**5,
    "K": 1000,
    "M": 1000**2,
    "G": 1000**3,
    "T": 1000**4,
    "P": 1000**5,
    "k": 1000,
}


def parse_memory(value: str | None) -> float:
    """Kubernetes memory quantity to mebibytes."""
    if not value:
        return 0.0
    text = str(value).strip()
    match = re.fullmatch(r"([0-9.]+)([A-Za-z]*)", text)
    if not match:
        return 0.0
    amount, unit = float(match.group(1)), match.group(2)
    return amount * _MEMORY_UNITS.get(unit, 1) / (1024**2)


def container_resources(pod: dict[str, Any]) -> dict[str, float]:
    """Summed requests and limits across a pod's containers, in m and MiB."""
    totals = {"cpu_request": 0.0, "cpu_limit": 0.0, "mem_request": 0.0, "mem_limit": 0.0}
    for container in (pod.get("spec") or {}).get("containers") or []:
        resources = container.get("resources") or {}
        requests, limits = resources.get("requests") or {}, resources.get("limits") or {}
        totals["cpu_request"] += parse_cpu(requests.get("cpu"))
        totals["cpu_limit"] += parse_cpu(limits.get("cpu"))
        totals["mem_request"] += parse_memory(requests.get("memory"))
        totals["mem_limit"] += parse_memory(limits.get("memory"))
    return totals


def pod_status(pod: dict[str, Any]) -> str:
    """What kubectl would print in the STATUS column, near enough."""
    status = pod.get("status") or {}
    phase = status.get("phase", "Unknown")
    for container in status.get("containerStatuses") or []:
        waiting = (container.get("state") or {}).get("waiting") or {}
        if waiting.get("reason"):
            return str(waiting["reason"])
    ready = sum(1 for c in status.get("containerStatuses") or [] if c.get("ready"))
    total = len(status.get("containerStatuses") or [])
    return f"{phase} {ready}/{total}" if total else str(phase)


# ------------------------------------------------------------ the edge model


def node_id(kind: str, namespace: str, name: str) -> str:
    return f"{kind}/{namespace}/{name}"


def _meta(obj: dict[str, Any]) -> dict[str, Any]:
    return obj.get("metadata") or {}


def build_graph(
    objects: dict[str, list[dict[str, Any]]], namespace: str = "", title: str = ""
) -> ResourceGraph:
    """Turn a set of listed objects into nodes and the six relationship kinds.

    Ownership comes from ownerReferences, which is authoritative. Everything
    else is inferred from spec fields, which is what makes this a graph rather
    than the tree ``kubectl get`` already gives you.
    """
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    index: dict[str, dict[str, Any]] = {}
    pods_by_id: dict[str, dict[str, Any]] = {}

    for kind, items in objects.items():
        for obj in items:
            meta = _meta(obj)
            name, ns = meta.get("name", ""), meta.get("namespace", namespace)
            if not name:
                continue
            ident = node_id(kind, ns, name)
            index[ident] = obj
            if kind == "Pod":
                pods_by_id[ident] = obj
            nodes.append(
                GraphNode(
                    id=ident,
                    kind=kind,
                    name=name,
                    namespace=ns,
                    status=status_for(kind, obj),
                    detail=detail_for(kind, obj),
                )
            )

    known = {n.id for n in nodes}

    # owns --- from ownerReferences
    for ident, obj in index.items():
        for owner in _meta(obj).get("ownerReferences") or []:
            owner_id = node_id(
                owner.get("kind", ""), _meta(obj).get("namespace", namespace), owner.get("name", "")
            )
            if owner_id in known:
                edges.append(GraphEdge(source=owner_id, target=ident, relation="owns"))

    # selects --- Service spec.selector against pod labels
    for service in objects.get("Service", []):
        selector = (service.get("spec") or {}).get("selector") or {}
        if not selector:
            continue
        svc_id = node_id(
            "Service", _meta(service).get("namespace", namespace), _meta(service).get("name", "")
        )
        for pod_id, pod in pods_by_id.items():
            labels = _meta(pod).get("labels") or {}
            if all(labels.get(k) == v for k, v in selector.items()):
                edges.append(GraphEdge(source=svc_id, target=pod_id, relation="selects"))

    # routes-to --- Ingress backends
    for ingress in objects.get("Ingress", []):
        ns = _meta(ingress).get("namespace", namespace)
        ing_id = node_id("Ingress", ns, _meta(ingress).get("name", ""))
        for service_name in _ingress_services(ingress):
            target = node_id("Service", ns, service_name)
            if target in known:
                edges.append(GraphEdge(source=ing_id, target=target, relation="routes-to"))

    # mounts / uses --- pod volumes and envFrom
    for pod_id, pod in pods_by_id.items():
        ns = _meta(pod).get("namespace", namespace)
        for volume in (pod.get("spec") or {}).get("volumes") or []:
            if claim := (volume.get("persistentVolumeClaim") or {}).get("claimName"):
                target = node_id("PersistentVolumeClaim", ns, claim)
                if target in known:
                    edges.append(GraphEdge(source=pod_id, target=target, relation="mounts"))
            if name := (volume.get("configMap") or {}).get("name"):
                target = node_id("ConfigMap", ns, name)
                if target in known:
                    edges.append(GraphEdge(source=pod_id, target=target, relation="uses"))
        for container in (pod.get("spec") or {}).get("containers") or []:
            for source in container.get("envFrom") or []:
                if name := (source.get("configMapRef") or {}).get("name"):
                    target = node_id("ConfigMap", ns, name)
                    if target in known:
                        edges.append(GraphEdge(source=pod_id, target=target, relation="uses"))

    # scales --- HPA scaleTargetRef
    for hpa in objects.get("HorizontalPodAutoscaler", []):
        ns = _meta(hpa).get("namespace", namespace)
        ref = (hpa.get("spec") or {}).get("scaleTargetRef") or {}
        target = node_id(ref.get("kind", ""), ns, ref.get("name", ""))
        if target in known:
            hpa_id = node_id("HorizontalPodAutoscaler", ns, _meta(hpa).get("name", ""))
            edges.append(GraphEdge(source=hpa_id, target=target, relation="scales"))

    seen: set[tuple[str, str, str]] = set()
    unique: list[GraphEdge] = []
    for edge in edges:
        key = (edge.source, edge.target, edge.relation)
        if key not in seen:
            seen.add(key)
            unique.append(edge)
    return ResourceGraph(
        title=title or f"namespace: {namespace or 'all'}", nodes=nodes, edges=unique
    )


def _ingress_services(ingress: dict[str, Any]) -> list[str]:
    names: list[str] = []
    spec = ingress.get("spec") or {}
    if default := (spec.get("defaultBackend") or {}).get("service", {}).get("name"):
        names.append(default)
    for rule in spec.get("rules") or []:
        for path in (rule.get("http") or {}).get("paths") or []:
            if name := ((path.get("backend") or {}).get("service") or {}).get("name"):
                names.append(name)
    return names


def status_for(kind: str, obj: dict[str, Any]) -> str:
    status = obj.get("status") or {}
    if kind == "Pod":
        return pod_status(obj)
    if kind in ("Deployment", "StatefulSet", "ReplicaSet"):
        ready = status.get("readyReplicas") or 0
        want = (obj.get("spec") or {}).get("replicas", status.get("replicas") or 0)
        return f"{ready}/{want} ready"
    if kind == "DaemonSet":
        return f"{status.get('numberReady', 0)}/{status.get('desiredNumberScheduled', 0)} ready"
    if kind == "PersistentVolumeClaim":
        return str(status.get("phase", ""))
    if kind == "Job":
        return "complete" if status.get("succeeded") else "running"
    return ""


def detail_for(kind: str, obj: dict[str, Any]) -> str:
    spec = obj.get("spec") or {}
    if kind == "Service":
        return str(spec.get("type", ""))
    if kind == "PersistentVolumeClaim":
        return str(((spec.get("resources") or {}).get("requests") or {}).get("storage", ""))
    if kind == "HorizontalPodAutoscaler":
        return f"{spec.get('minReplicas', '?')}-{spec.get('maxReplicas', '?')} replicas"
    return ""


@dataclass
class K8sProvider:
    """Builds a client on first use and caches it.

    Connecting is a network call, so a session that never touches Kubernetes
    should never pay for one --- which is why this is lazy rather than
    constructed alongside the workspace.
    """

    context: str | None = None
    kubeconfigs: tuple[str, ...] = ()
    _client: K8sClient | None = field(default=None, repr=False)

    async def get(self) -> tuple[K8sClient, str]:
        if self._client is None:
            from wai.cloud.kube import build_client, resolve_context

            resolved = await asyncio.to_thread(resolve_context, self.context, self.kubeconfigs)
            name = resolved.name if resolved else (self.context or "")
            dynamic = await asyncio.to_thread(build_client, name or None, self.kubeconfigs)
            self._client = K8sClient(dynamic=dynamic, context=name)
        return self._client, self._client.context


# ----------------------------------------------------------------- explain


def _deref(schema: dict[str, Any], schemas: dict[str, Any]) -> dict[str, Any]:
    """Follow $ref, including the allOf wrapper Kubernetes uses for objects."""
    seen = 0
    while seen < 10:
        seen += 1
        ref = schema.get("$ref")
        if not ref and len(schema.get("allOf") or []) == 1:
            ref = (schema["allOf"][0] or {}).get("$ref")
        if not ref:
            return schema
        target = schemas.get(ref.split("/")[-1])
        if target is None:
            return schema
        # Keep the outer description: it is the field-specific one.
        merged = dict(target)
        if schema.get("description"):
            merged["description"] = schema["description"]
        schema = merged
    return schema


def _type_of(schema: dict[str, Any]) -> str:
    if ref := (schema.get("$ref") or ((schema.get("allOf") or [{}])[0] or {}).get("$ref")):
        return str(ref.split(".")[-1])
    kind = schema.get("type", "object")
    if kind == "array":
        item = schema.get("items") or {}
        return f"[]{_type_of(item)}"
    return str(kind)


def find_schema(document: dict[str, Any], kind: str) -> tuple[str, dict[str, Any]] | None:
    """Locate a kind's schema by its authoritative group-version-kind marker."""
    schemas = (document.get("components") or {}).get("schemas") or {}
    for name, schema in schemas.items():
        for gvk in schema.get("x-kubernetes-group-version-kind") or []:
            if gvk.get("kind") == kind:
                return name, schema
    # Fall back to a name match for schemas without the marker.
    for name, schema in schemas.items():
        if name.rsplit(".", 1)[-1] == kind:
            return name, schema
    return None


def explain(document: dict[str, Any], kind: str, field_path: str = "") -> dict[str, Any]:
    """`kubectl explain`, from the cluster's own schema.

    Authoritative for this cluster at this version, and it covers custom
    resources for free --- which no static reference can.
    """
    schemas = (document.get("components") or {}).get("schemas") or {}
    found = find_schema(document, kind)
    if found is None:
        raise KeyError(f"no schema for kind {kind!r}")
    name, schema = found
    schema = _deref(schema, schemas)

    walked = [kind]
    for part in [p for p in field_path.split(".") if p]:
        properties = schema.get("properties") or {}
        if part not in properties:
            options = ", ".join(sorted(properties)[:15]) or "none"
            raise KeyError(f"{'.'.join(walked)} has no field {part!r}. Available: {options}")
        schema = _deref(properties[part], schemas)
        if schema.get("type") == "array":
            schema = _deref(schema.get("items") or {}, schemas)
        walked.append(part)

    fields = {
        child: {
            "type": _type_of(value),
            "required": child in (schema.get("required") or []),
            "description": (_deref(value, schemas).get("description") or "").split(". ")[0][:200],
        }
        for child, value in (schema.get("properties") or {}).items()
    }
    return {
        "path": ".".join(walked),
        "schema": name,
        "type": _type_of(schema),
        "description": (schema.get("description") or "")[:400],
        "required": schema.get("required") or [],
        "fields": fields,
    }


def summarise_change(before: dict[str, Any] | None, after: dict[str, Any]) -> str:
    """A short diff of the fields a human cares about, for the approval prompt.

    Not a full object diff: server-side apply rewrites managedFields,
    resourceVersion and timestamps on every call, and burying the real change
    in that noise defeats the point of showing a diff at all.
    """
    import difflib

    interesting = ("spec", "data", "stringData", "rules", "subjects", "roleRef")

    def slim(obj: dict[str, Any]) -> str:
        import json

        kept = {k: v for k, v in obj.items() if k in interesting}
        meta = obj.get("metadata") or {}
        trimmed = {k: meta[k] for k in ("name", "namespace", "labels", "annotations") if k in meta}
        if trimmed:
            kept["metadata"] = trimmed
        return json.dumps(kept, indent=2, sort_keys=True, default=str)

    if before is None:
        return "(new object)\n" + slim(after)
    lines = list(
        difflib.unified_diff(
            slim(before).splitlines(keepends=True),
            slim(after).splitlines(keepends=True),
            fromfile="live",
            tofile="proposed",
            n=2,
        )
    )
    if not lines:
        return "(no change to spec, data or metadata)"
    return "".join(lines[:160])
