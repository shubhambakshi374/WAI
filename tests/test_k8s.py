"""Kubernetes tools and the topology model. No cluster is contacted.

The dynamic client is injected, so everything here drives recorded API
payloads. The graph tests are the load-bearing ones: a wrong edge set means a
misleading picture of someone's production namespace.
"""

from __future__ import annotations

from typing import Any

import pytest

from altus.cloud import k8s as k8s_api
from altus.cloud.base import ProtectionRules, Sensitivity
from altus.cloud.k8s import MetricsUnavailable, build_graph, parse_cpu, parse_memory
from altus.cloud.redact import MARKER
from altus.core.visuals import Bars, ResourceGraph, Table, VisualGroup
from altus.tools.approval import Decision
from altus.tools.base import CloudContext, ToolContext
from altus.tools.k8s import k8s_tools
from altus.workspace import Workspace

# ------------------------------------------------------------------- fixtures


def pod(
    name: str,
    labels: dict[str, str] | None = None,
    *,
    phase: str = "Running",
    owner: str | None = "web-7d9",
    claim: str | None = "data",
    cpu_req: str = "250m",
    cpu_lim: str = "500m",
) -> dict[str, Any]:
    meta: dict[str, Any] = {"name": name, "namespace": "shop", "labels": labels or {"app": "web"}}
    if owner:
        meta["ownerReferences"] = [{"kind": "ReplicaSet", "name": owner}]
    volumes: list[dict[str, Any]] = [{"name": "cfg", "configMap": {"name": "app-config"}}]
    if claim:
        volumes.append({"name": "d", "persistentVolumeClaim": {"claimName": claim}})
    return {
        "kind": "Pod",
        "metadata": meta,
        "spec": {
            "containers": [
                {
                    "name": "c",
                    "resources": {
                        "requests": {"cpu": cpu_req, "memory": "256Mi"},
                        "limits": {"cpu": cpu_lim, "memory": "512Mi"},
                    },
                }
            ],
            "volumes": volumes,
        },
        "status": {"phase": phase, "containerStatuses": [{"ready": phase == "Running"}]},
    }


CLUSTER: dict[str, list[dict[str, Any]]] = {
    "Deployment": [
        {
            "metadata": {"name": "web", "namespace": "shop"},
            "spec": {"replicas": 2},
            "status": {"readyReplicas": 2},
        }
    ],
    "ReplicaSet": [
        {
            "metadata": {
                "name": "web-7d9",
                "namespace": "shop",
                "ownerReferences": [{"kind": "Deployment", "name": "web"}],
            },
            "spec": {"replicas": 2},
            "status": {"readyReplicas": 2},
        }
    ],
    "Pod": [pod("web-7d9-aaa"), pod("web-7d9-bbb", phase="CrashLoopBackOff")],
    "Service": [
        {
            "metadata": {"name": "web", "namespace": "shop"},
            "spec": {"selector": {"app": "web"}, "type": "ClusterIP"},
        }
    ],
    "Ingress": [
        {
            "metadata": {"name": "shop", "namespace": "shop"},
            "spec": {"rules": [{"http": {"paths": [{"backend": {"service": {"name": "web"}}}]}}]},
        }
    ],
    "PersistentVolumeClaim": [
        {
            "metadata": {"name": "data", "namespace": "shop"},
            "spec": {
                "resources": {"requests": {"storage": "10Gi"}},
                "storageClassName": "gp3",
                "accessModes": ["ReadWriteOnce"],
            },
            "status": {"phase": "Bound", "capacity": {"storage": "10Gi"}},
        }
    ],
    "ConfigMap": [{"metadata": {"name": "app-config", "namespace": "shop"}}],
    "HorizontalPodAutoscaler": [
        {
            "metadata": {"name": "web", "namespace": "shop"},
            "spec": {
                "scaleTargetRef": {"kind": "Deployment", "name": "web"},
                "minReplicas": 2,
                "maxReplicas": 10,
            },
        }
    ],
}


class FakeClient:
    """Stands in for K8sClient. Records what was asked for."""

    def __init__(
        self,
        objects: dict[str, list[dict[str, Any]]] | None = None,
        *,
        metrics: list[dict[str, Any]] | None = None,
        has_metrics: bool = True,
        groups: set[str] | None = None,
    ) -> None:
        self.objects = objects if objects is not None else CLUSTER
        self._metrics = metrics or []
        self.has_metrics = has_metrics
        self.context = "AKS_QAM"
        self.calls: list[tuple[str, str]] = []
        self._groups = groups or {"v1", "apps/v1", "networking.k8s.io/v1"}
        self.created: list[tuple[str, dict[str, Any], bool]] = []
        self.raw_calls: list[tuple[str, str]] = []
        self.raw_responses: dict[str, Any] = {}
        self.reviews: dict[str, Any] = {}
        self.watches: list[tuple[str, dict[str, Any]]] = []
        self.watch_result: tuple[bool, list[dict[str, Any]], str] = (True, [], "met")

    async def api_groups(self) -> set[str]:
        return self._groups | ({k8s_api.METRICS_API} if self.has_metrics else set())

    async def api_available(self, group_version: str) -> bool:
        return group_version in await self.api_groups()

    async def resolve_kind(self, kind, api_version=""):  # type: ignore[no-untyped-def]
        if api_version:
            return api_version
        return {"Deployment": "apps/v1", "Ingress": "networking.k8s.io/v1"}.get(kind, "v1")

    async def list_kind(self, api_version, kind, namespace=None, **kw):  # type: ignore[no-untyped-def]
        self.calls.append((api_version, kind))
        return list(self.objects.get(kind, []))

    async def list_many(self, kinds, namespace=None, **kw):  # type: ignore[no-untyped-def]
        return {kind: list(self.objects.get(kind, [])) for _, kind in kinds}

    async def metrics(self, kind, namespace=None):  # type: ignore[no-untyped-def]
        if not self.has_metrics:
            raise MetricsUnavailable
        return self._metrics

    # --- the rest of the API surface -------------------------------------

    async def get_one(self, api_version, kind, name, namespace=None, *, subresource=""):  # type: ignore[no-untyped-def]
        self.calls.append(
            (api_version, f"{kind}/{name}" + (f"/{subresource}" if subresource else ""))
        )
        for obj in self.objects.get(kind, []):
            if (obj.get("metadata") or {}).get("name") == name:
                return obj
        raise KeyError(f"no {kind}/{name}")

    async def create(self, api_version, kind, body, namespace=None, *, dry_run=False):  # type: ignore[no-untyped-def]
        self.created.append((kind, body, dry_run))
        return self.reviews.get(kind, {"status": {"allowed": False}})

    async def raw(self, method, path, *, body=None, **params):  # type: ignore[no-untyped-def]
        self.raw_calls.append((method, path))
        return self.raw_responses.get(path, {"raw": "ok"})

    async def watch_until(self, api_version, kind, namespace, ready, **kw):  # type: ignore[no-untyped-def]
        self.watches.append((kind, kw))
        return self.watch_result


class FakeProvider:
    def __init__(self, client: FakeClient) -> None:
        self.client = client

    async def get(self) -> tuple[Any, str]:
        return self.client, self.client.context


def context_for(client: FakeClient | None, tmp_path, **cloud: Any) -> ToolContext:  # type: ignore[no-untyped-def]
    return ToolContext(
        workspace=Workspace(root=tmp_path),
        cloud=CloudContext(
            k8s=FakeProvider(client) if client else None,
            protection=ProtectionRules(patterns=("*prod*",)),
            **cloud,
        ),
    )


def tool(name: str):  # type: ignore[no-untyped-def]
    return next(t for t in k8s_tools() if t.name == name)


# ------------------------------------------------------------ quantity maths


@pytest.mark.parametrize(
    ("value", "expected"),
    [("100m", 100.0), ("2", 2000.0), ("1500000000n", 1500.0), ("500u", 0.5), (None, 0.0)],
)
def test_cpu_quantities(value: str | None, expected: float) -> None:
    assert parse_cpu(value) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("value", "expected"),
    [("512Mi", 512.0), ("2Gi", 2048.0), ("1Ki", 1 / 1024), ("1G", 953.674316), (None, 0.0)],
)
def test_memory_quantities(value: str | None, expected: float) -> None:
    assert parse_memory(value) == pytest.approx(expected, rel=1e-4)


def test_pod_status_prefers_the_waiting_reason() -> None:
    crashing = {
        "status": {
            "phase": "Running",
            "containerStatuses": [
                {"ready": False, "state": {"waiting": {"reason": "CrashLoopBackOff"}}}
            ],
        }
    }
    assert k8s_api.pod_status(crashing) == "CrashLoopBackOff"
    assert (
        k8s_api.pod_status({"status": {"phase": "Running", "containerStatuses": [{"ready": True}]}})
        == "Running 1/1"
    )


# ----------------------------------------------------------------- the graph


def test_graph_finds_every_relationship() -> None:
    graph = build_graph(CLUSTER, "shop")
    edges = {(e.relation, e.source.split("/")[0], e.target.split("/")[0]) for e in graph.edges}
    assert ("owns", "Deployment", "ReplicaSet") in edges
    assert ("owns", "ReplicaSet", "Pod") in edges
    assert ("selects", "Service", "Pod") in edges
    assert ("routes-to", "Ingress", "Service") in edges
    assert ("mounts", "Pod", "PersistentVolumeClaim") in edges
    assert ("uses", "Pod", "ConfigMap") in edges
    assert ("scales", "HorizontalPodAutoscaler", "Deployment") in edges


def test_graph_has_no_duplicate_edges() -> None:
    graph = build_graph(CLUSTER, "shop")
    keys = [(e.source, e.target, e.relation) for e in graph.edges]
    assert len(keys) == len(set(keys))


def test_graph_roots_are_the_unowned() -> None:
    graph = build_graph(CLUSTER, "shop")
    roots = {n.kind for n in graph.roots()}
    assert "Deployment" in roots
    assert "ReplicaSet" not in roots, "an owned resource is never a root"


def test_orphan_still_appears() -> None:
    objects = {"Pod": [pod("lonely", owner=None, claim=None)]}
    graph = build_graph(objects, "shop")
    assert "lonely" in graph.to_text()


def test_owner_cycle_terminates() -> None:
    """Illegal, but it happens after a botched restore --- and unbounded
    recursion would take the whole session down."""
    objects = {
        "Pod": [
            {
                "metadata": {
                    "name": "a",
                    "namespace": "n",
                    "ownerReferences": [{"kind": "Pod", "name": "b"}],
                }
            },
            {
                "metadata": {
                    "name": "b",
                    "namespace": "n",
                    "ownerReferences": [{"kind": "Pod", "name": "a"}],
                }
            },
        ]
    }
    text = build_graph(objects, "n").to_text()
    assert "Pod/a" in text and "Pod/b" in text
    assert "shown above" in text, "the cycle is reported, not silently truncated"


def test_tree_connectors_are_well_formed() -> None:
    rows = build_graph(CLUSTER, "shop").layout()
    depth_one = [r for r in rows if r.prefix in ("└─ ", "├─ ")]
    assert depth_one, "children must carry a connector"
    assert all("├─ ├─" not in r.prefix for r in rows), "connectors must not double up"


def test_edges_are_grouped_not_repeated() -> None:
    """A Service fronting many pods is one line, not one per pod."""
    text = build_graph(CLUSTER, "shop").to_text()
    assert "selects 2 Pods" in text


def test_graph_survives_a_missing_target() -> None:
    """An Ingress pointing at a Service that was not listed must not crash."""
    objects = {"Ingress": CLUSTER["Ingress"]}
    graph = build_graph(objects, "shop")
    assert graph.nodes and graph.edges == []


# ------------------------------------------------------------------- tooling


async def test_list_returns_a_table(tmp_path) -> None:  # type: ignore[no-untyped-def]
    out = await tool("k8s_list").run(
        {"kind": "Pod", "namespace": "shop"}, context_for(FakeClient(), tmp_path)
    )
    assert not out.is_error
    assert isinstance(out.visual, Table)
    assert out.summary == "2 Pod"
    assert "web-7d9-aaa" in out.content


async def test_list_without_a_client_says_so(tmp_path) -> None:  # type: ignore[no-untyped-def]
    out = await tool("k8s_list").run({"kind": "Pod"}, context_for(None, tmp_path))
    assert out.is_error and out.summary == "unavailable"


async def test_topology_returns_a_graph(tmp_path) -> None:  # type: ignore[no-untyped-def]
    out = await tool("k8s_topology").run({"namespace": "shop"}, context_for(FakeClient(), tmp_path))
    assert isinstance(out.visual, ResourceGraph)
    assert "Deployment/web" in out.content
    assert "9 resources" in out.summary  # 1 deploy, 1 rs, 2 pods, svc, ing, pvc, cm, hpa


async def test_top_without_metrics_server_names_the_cause(tmp_path) -> None:  # type: ignore[no-untyped-def]
    out = await tool("k8s_top").run({}, context_for(FakeClient(has_metrics=False), tmp_path))
    assert out.is_error
    assert out.summary == "no metrics-server"
    assert "metrics-server is not installed" in out.content
    assert "kubectl apply" in out.content, "tell the user how to fix it"


async def test_usage_still_works_without_metrics_server(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Requests and limits come from the core API and must survive."""
    out = await tool("k8s_usage").run(
        {"namespace": "shop"}, context_for(FakeClient(has_metrics=False), tmp_path)
    )
    assert not out.is_error
    assert isinstance(out.visual, VisualGroup)
    assert "metrics-server is not installed" in out.content
    assert "500m" in out.content, "the limit is still charted"


async def test_usage_uses_live_metrics_when_present(tmp_path) -> None:  # type: ignore[no-untyped-def]
    metrics = [
        {
            "metadata": {"name": "web-7d9-aaa"},
            "containers": [{"usage": {"cpu": "420m", "memory": "300Mi"}}],
        },
        {
            "metadata": {"name": "web-7d9-bbb"},
            "containers": [{"usage": {"cpu": "80m", "memory": "100Mi"}}],
        },
    ]
    out = await tool("k8s_usage").run(
        {"namespace": "shop"}, context_for(FakeClient(metrics=metrics), tmp_path)
    )
    assert "420" in out.content
    assert "metrics-server is not installed" not in out.content


async def test_top_charts_and_sorts_by_usage(tmp_path) -> None:  # type: ignore[no-untyped-def]
    metrics = [
        {
            "metadata": {"name": "small"},
            "containers": [{"usage": {"cpu": "10m", "memory": "10Mi"}}],
        },
        {
            "metadata": {"name": "big"},
            "containers": [{"usage": {"cpu": "900m", "memory": "900Mi"}}],
        },
    ]
    out = await tool("k8s_top").run({}, context_for(FakeClient(metrics=metrics), tmp_path))
    assert isinstance(out.visual, VisualGroup)
    cpu = out.visual.items[0]
    assert isinstance(cpu, Bars)
    assert [b.label for b in cpu.bars] == ["big", "small"], "largest first"


async def test_storage_is_honest_about_fill_level(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """metrics-server has no volume stats; claiming a utilisation would be a lie."""
    out = await tool("k8s_storage").run({"namespace": "shop"}, context_for(FakeClient(), tmp_path))
    assert not out.is_error
    assert "Prometheus" in out.content
    assert "1 PVCs" in out.summary


async def test_events_put_warnings_first(tmp_path) -> None:  # type: ignore[no-untyped-def]
    objects = {
        "Event": [
            {
                "type": "Normal",
                "reason": "Started",
                "involvedObject": {"kind": "Pod", "name": "a"},
                "message": "ok",
            },
            {
                "type": "Warning",
                "reason": "BackOff",
                "involvedObject": {"kind": "Pod", "name": "b"},
                "message": "bad",
            },
        ]
    }
    out = await tool("k8s_events").run({}, context_for(FakeClient(objects), tmp_path))
    assert "1 warnings" in out.summary
    assert out.content.index("BackOff") < out.content.index("Started")


async def test_contexts_flags_protected(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    kubeconfig = tmp_path / "config"
    kubeconfig.write_text(
        "apiVersion: v1\nkind: Config\ncurrent-context: staging\n"
        "clusters:\n- name: c\n  cluster: {server: https://x}\n"
        "contexts:\n- name: AKS_EU_PROD\n  context: {cluster: c, user: u}\n"
        "- name: staging\n  context: {cluster: c, user: u}\n"
        "users:\n- name: u\n  user: {}\n"
    )
    monkeypatch.setenv("KUBECONFIG", str(kubeconfig))
    out = await tool("k8s_contexts").run({}, context_for(None, tmp_path))
    assert "AKS_EU_PROD" in out.content
    assert "PROTECTED" in out.content


# ----------------------------------------------------------------- redaction


async def test_secrets_are_redacted_before_reaching_the_model(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The single most important assertion here: tool content is transmitted
    to the LLM provider, so a Secret value must never survive this far."""
    objects = {
        "Secret": [
            {
                "kind": "Secret",
                "metadata": {"name": "db-creds", "namespace": "shop"},
                "data": {"password": "aHVudGVyMg==", "username": "YWRtaW4="},
            }
        ]
    }
    ctx = context_for(FakeClient(objects), tmp_path)
    out = await tool("k8s_list").run({"kind": "Secret", "namespace": "shop"}, ctx)
    assert "aHVudGVyMg==" not in out.content
    assert "aHVudGVyMg==" not in str(out.visual)


async def test_redaction_can_be_turned_off(tmp_path) -> None:  # type: ignore[no-untyped-def]
    objects = {"ConfigMap": [{"kind": "ConfigMap", "metadata": {"name": "c", "namespace": "shop"}}]}
    ctx = context_for(FakeClient(objects), tmp_path, redact_secrets=False)
    out = await tool("k8s_list").run({"kind": "ConfigMap"}, ctx)
    assert not out.is_error


# ------------------------------------------------------- concurrency & errors


async def test_a_forbidden_kind_does_not_sink_the_whole_topology() -> None:
    """RBAC routinely allows some kinds and not others."""

    class Partial(FakeClient):
        async def list_kind(self, api_version, kind, namespace=None, **kw):  # type: ignore[no-untyped-def]
            if kind == "Ingress":
                raise PermissionError("forbidden")
            return list(self.objects.get(kind, []))

    client = Partial()
    real = k8s_api.K8sClient.list_many
    result = await real(client, k8s_api.TOPOLOGY_KINDS, "shop")  # type: ignore[arg-type]
    assert result["Ingress"] == []
    assert result["Pod"], "the rest still came back"


async def test_all_visuals_render_headlessly(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """to_text() is what `altus chat --once` and the Phase 3 engine will use."""
    metrics = [
        {"metadata": {"name": "p"}, "containers": [{"usage": {"cpu": "1m", "memory": "1Mi"}}]}
    ]
    ctx = context_for(FakeClient(metrics=metrics), tmp_path)
    for name, args in [
        ("k8s_list", {"kind": "Pod"}),
        ("k8s_topology", {}),
        ("k8s_top", {}),
        ("k8s_usage", {}),
        ("k8s_storage", {}),
    ]:
        out = await tool(name).run(args, ctx)
        assert out.visual is not None, f"{name} should carry a visual"
        assert out.visual.to_text().strip(), f"{name} rendered empty headlessly"


async def test_visual_never_reaches_the_model(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The whole economy of this design: charts cost no context."""
    from altus.core.types import ToolResultBlock

    out = await tool("k8s_topology").run({}, context_for(FakeClient(), tmp_path))
    block = ToolResultBlock(tool_use_id="x", content=out.content)
    assert "visual" not in ToolResultBlock.model_fields
    assert len(block.content) < 4000, "the summary must stay compact"


def test_scale_sizes_the_bar_without_inventing_a_percentage() -> None:
    """`limit` means a real ceiling and shows a percentage. `scale` is only
    bar length --- a percentage there would be read as a fill level."""
    from altus.core.visuals import Bar

    limited = Bar(label="cpu", value=500, limit=1000, unit="m").render()
    assert "500m / 1,000m" in limited and "50%" in limited

    scaled = Bar(label="vol", value=250, scale=300, unit="Gi").render()
    assert "250Gi" in scaled
    assert "%" not in scaled, "relative size must not be presented as a percentage"
    assert "/" not in scaled.split("Gi")[0]


def test_bar_labels_truncate_in_the_middle() -> None:
    """Elasticsearch PVCs share a long prefix; cutting the tail hides which is which."""
    from altus.core.visuals import Bar

    a = Bar(label="sandbox/elasticsearch-data-esmain-es-esmain-0", value=1).render(label_width=30)
    b = Bar(label="sandbox/elasticsearch-data-esmain-es-esmain-1", value=1).render(label_width=30)
    assert a != b
    assert a.strip().startswith("sandbox/")
    assert "-0" in a and "-1" in b


async def test_storage_does_not_claim_a_fill_level(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A bound PVC has requested == capacity, so charting one against the
    other shows every volume at 100% and reads as 'full'."""
    from altus.core.visuals import Bars

    objects = {
        "PersistentVolumeClaim": [
            {
                "metadata": {"name": f"vol-{i}", "namespace": "shop"},
                "spec": {"resources": {"requests": {"storage": f"{size}Gi"}}},
                "status": {"phase": "Bound", "capacity": {"storage": f"{size}Gi"}},
            }
            for i, size in enumerate((300, 100, 10))
        ]
    }
    out = await tool("k8s_storage").run(
        {"all_namespaces": True}, context_for(FakeClient(objects), tmp_path)
    )
    assert isinstance(out.visual, VisualGroup)
    bars = out.visual.items[0]
    assert isinstance(bars, Bars)
    assert [b.value for b in bars.bars] == [300.0, 100.0, 10.0], "largest first"
    assert all(b.limit is None for b in bars.bars), "no fake ceiling"
    assert all(b.scale == 300.0 for b in bars.bars), "scaled against the largest"
    assert "100%" not in out.content
    assert "Prometheus" in out.content


# ------------------------------------------------------------------- explain

DEPLOYMENT_DOC: dict[str, Any] = {
    "components": {
        "schemas": {
            "io.k8s.api.apps.v1.Deployment": {
                "description": "Deployment enables declarative updates.",
                "type": "object",
                "x-kubernetes-group-version-kind": [
                    {"group": "apps", "kind": "Deployment", "version": "v1"}
                ],
                "properties": {
                    "spec": {
                        "description": "Specification of the desired behavior.",
                        "allOf": [
                            {"$ref": "#/components/schemas/io.k8s.api.apps.v1.DeploymentSpec"}
                        ],
                    }
                },
            },
            "io.k8s.api.apps.v1.DeploymentSpec": {
                "type": "object",
                "description": "DeploymentSpec is the specification.",
                "required": ["selector", "template"],
                "properties": {
                    "replicas": {"type": "integer", "description": "Number of desired pods."},
                    "selector": {"type": "object", "description": "Label selector."},
                    "strategy": {
                        "description": "The deployment strategy.",
                        "allOf": [
                            {"$ref": "#/components/schemas/io.k8s.api.apps.v1.DeploymentStrategy"}
                        ],
                    },
                    "containers": {
                        "type": "array",
                        "items": {"$ref": "#/components/schemas/io.k8s.api.core.v1.Container"},
                    },
                },
            },
            "io.k8s.api.apps.v1.DeploymentStrategy": {
                "type": "object",
                "properties": {"type": {"type": "string", "description": "Type of deployment."}},
            },
            "io.k8s.api.core.v1.Container": {
                "type": "object",
                "required": ["name"],
                "properties": {"name": {"type": "string", "description": "Container name."}},
            },
        }
    }
}


def test_explain_finds_a_kind_by_its_gvk_marker() -> None:
    out = k8s_api.explain(DEPLOYMENT_DOC, "Deployment")
    assert out["path"] == "Deployment"
    assert "spec" in out["fields"]


def test_explain_walks_a_dotted_path_through_allof_refs() -> None:
    """Kubernetes wraps object refs in allOf, so a naive $ref lookup misses them."""
    out = k8s_api.explain(DEPLOYMENT_DOC, "Deployment", "spec")
    assert out["required"] == ["selector", "template"]
    assert out["fields"]["replicas"]["type"] == "integer"
    assert out["fields"]["replicas"]["required"] is False
    assert out["fields"]["selector"]["required"] is True

    nested = k8s_api.explain(DEPLOYMENT_DOC, "Deployment", "spec.strategy")
    assert "type" in nested["fields"]


def test_explain_steps_into_array_items() -> None:
    """spec.containers is a list; the useful schema is the element's."""
    out = k8s_api.explain(DEPLOYMENT_DOC, "Deployment", "spec.containers")
    assert "name" in out["fields"]
    assert out["fields"]["name"]["required"] is True


def test_explain_reports_array_types_readably() -> None:
    out = k8s_api.explain(DEPLOYMENT_DOC, "Deployment", "spec")
    assert out["fields"]["containers"]["type"] == "[]Container"


def test_explain_on_a_typo_lists_the_real_fields() -> None:
    """The error is the correction, so the model fixes it in one step."""
    with pytest.raises(KeyError) as exc:
        k8s_api.explain(DEPLOYMENT_DOC, "Deployment", "spec.replicaz")
    assert "replicas" in str(exc.value)


def test_explain_on_an_unknown_kind() -> None:
    with pytest.raises(KeyError, match="Nonesuch"):
        k8s_api.explain(DEPLOYMENT_DOC, "Nonesuch")


async def test_explain_tool_renders_and_tables(tmp_path) -> None:  # type: ignore[no-untyped-def]
    class SchemaClient(FakeClient):
        async def resolve_kind(self, kind, api_version=""):  # type: ignore[no-untyped-def]
            return api_version or "apps/v1"

        async def schema_document(self, group_version):  # type: ignore[no-untyped-def]
            return DEPLOYMENT_DOC

    out = await tool("k8s_explain").run(
        {"kind": "Deployment", "field": "spec"}, context_for(SchemaClient(), tmp_path)
    )
    assert not out.is_error
    assert "REQUIRED: selector, template" in out.content
    assert isinstance(out.visual, Table)


async def test_api_version_comes_from_discovery_not_a_guess(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The old hardcoded table knew a dozen kinds and was wrong for every CRD."""
    seen: list[str] = []

    class Recording(FakeClient):
        async def resolve_kind(self, kind, api_version=""):  # type: ignore[no-untyped-def]
            seen.append(kind)
            return "cert-manager.io/v1"

        async def list_kind(self, api_version, kind, namespace=None, **kw):  # type: ignore[no-untyped-def]
            self.calls.append((api_version, kind))
            return []

    client = Recording()
    await tool("k8s_list").run({"kind": "Certificate"}, context_for(client, tmp_path))
    assert seen == ["Certificate"]
    assert client.calls == [("cert-manager.io/v1", "Certificate")]


# ----------------------------------------------------------------- mutations


class MutableClient(FakeClient):
    """Records every call so tests can assert what did and did not happen."""

    def __init__(self, *a: Any, live: dict[str, Any] | None = None, **kw: Any) -> None:
        super().__init__(*a, **kw)
        self.applied: list[tuple[dict[str, Any], bool]] = []
        self.deleted: list[tuple[str, bool]] = []
        self.patched: list[tuple[dict[str, Any], bool]] = []
        self.replaced: list[tuple[str, bool]] = []
        self.collections: list[tuple[str | None, bool]] = []
        self.subresources: list[tuple[str, str, str, str, bool]] = []
        self.live = live

    async def get_one(self, api_version, kind, name, namespace=None, *, subresource=""):  # type: ignore[no-untyped-def]
        if self.live is None:
            raise KeyError("not found")
        return self.live

    async def apply(self, api_version, kind, body, namespace=None, *, dry_run=False):  # type: ignore[no-untyped-def]
        self.applied.append((body, dry_run))
        return body

    async def delete(self, api_version, kind, name, namespace=None, *, dry_run=False):  # type: ignore[no-untyped-def]
        self.deleted.append((name, dry_run))
        return {}

    async def patch(
        self, api_version, kind, name, patch, namespace=None, *, dry_run=False, patch_type="merge"
    ):  # type: ignore[no-untyped-def]
        self.patched.append((patch, dry_run))
        return {}

    @property
    def real_applies(self) -> list[dict[str, Any]]:
        return [body for body, dry in self.applied if not dry]

    async def create(self, api_version, kind, body, namespace=None, *, dry_run=False):  # type: ignore[no-untyped-def]
        self.created.append((kind, body, dry_run))
        if kind.endswith("Review"):
            return self.reviews.get(kind, {"status": {"allowed": False}})
        return {"metadata": {"name": (body.get("metadata") or {}).get("name", "new")}}

    async def replace(self, api_version, kind, name, body, namespace=None, *, dry_run=False):  # type: ignore[no-untyped-def]
        self.replaced.append((name, dry_run))
        return body

    async def delete_collection(
        self,
        api_version,
        kind,
        namespace=None,
        *,
        label_selector=None,
        field_selector=None,
        dry_run=False,
    ):  # type: ignore[no-untyped-def]
        self.collections.append((label_selector, dry_run))
        return {}

    async def subresource(
        self,
        method,
        api_version,
        kind,
        name,
        subresource,
        *,
        namespace=None,
        body=None,
        dry_run=False,
    ):  # type: ignore[no-untyped-def]
        self.subresources.append((method, kind, name, subresource, dry_run))
        return {"metadata": {"name": name}}

    @property
    def real_creates(self) -> list[str]:
        return [kind for kind, _body, dry in self.created if not dry]

    @property
    def real_replaces(self) -> list[str]:
        return [name for name, dry in self.replaced if not dry]

    @property
    def real_collections(self) -> list[str]:
        return [sel for sel, dry in self.collections if not dry]

    @property
    def real_deletes(self) -> list[str]:
        return [name for name, dry in self.deleted if not dry]

    @property
    def real_patches(self) -> list[dict[str, Any]]:
        return [p for p, dry in self.patched if not dry]


DEPLOY_LIVE: dict[str, Any] = {
    "kind": "Deployment",
    "metadata": {"name": "web", "namespace": "shop"},
    "spec": {"replicas": 2},
}

VERSIONED: dict[str, Any] = {
    "apiVersion": "apps/v1",
    "kind": "Deployment",
    "metadata": {"name": "web", "namespace": "shop", "resourceVersion": "42"},
    "spec": {"replicas": 3},
}

MANIFEST: dict[str, Any] = {
    "apiVersion": "apps/v1",
    "kind": "Deployment",
    "metadata": {"name": "web", "namespace": "shop"},
    "spec": {"replicas": 3},
}


def approving(decision: Decision = Decision.ALLOW):  # type: ignore[no-untyped-def]
    from altus.tools.approval import RecordingPolicy

    return RecordingPolicy(decision=decision)


def mutation_context(client, tmp_path, policy, patterns=("*prod*",)):  # type: ignore[no-untyped-def]
    return ToolContext(
        workspace=Workspace(root=tmp_path),
        approvals=policy,
        cloud=CloudContext(k8s=FakeProvider(client), protection=ProtectionRules(patterns=patterns)),
    )


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("k8s_apply", {"manifest": MANIFEST}),
        ("k8s_delete", {"kind": "Deployment", "name": "web", "namespace": "shop"}),
        ("k8s_scale", {"kind": "Deployment", "name": "web", "replicas": 5, "namespace": "shop"}),
        ("k8s_rollout", {"kind": "Deployment", "name": "web", "namespace": "shop"}),
        (
            "k8s_patch",
            {"kind": "Deployment", "name": "web", "namespace": "shop", "patch": {"spec": {}}},
        ),
        ("k8s_create", {"manifest": MANIFEST, "namespace": "shop"}),
        ("k8s_replace", {"manifest": VERSIONED, "namespace": "shop"}),
        (
            "k8s_delete",
            {"kind": "Deployment", "label_selector": "app=web", "namespace": "shop"},
        ),
    ],
)
async def test_every_mutation_asks_before_acting(tmp_path, name, args) -> None:  # type: ignore[no-untyped-def]
    policy = approving()
    client = MutableClient(live=DEPLOY_LIVE)
    out = await tool(name).run(args, mutation_context(client, tmp_path, policy))
    assert not out.is_error, out.content
    assert len(policy.seen) == 1
    request = policy.seen[0]
    assert request.target, "the blast radius must always be named"
    assert "cluster" in request.target and "namespace" in request.target
    assert request.dry_run, "the server's verdict must be shown"


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("k8s_apply", {"manifest": MANIFEST}),
        ("k8s_delete", {"kind": "Deployment", "name": "web", "namespace": "shop"}),
        ("k8s_scale", {"kind": "Deployment", "name": "web", "replicas": 5, "namespace": "shop"}),
        ("k8s_rollout", {"kind": "Deployment", "name": "web", "namespace": "shop"}),
        (
            "k8s_patch",
            {"kind": "Deployment", "name": "web", "namespace": "shop", "patch": {"spec": {}}},
        ),
        ("k8s_create", {"manifest": MANIFEST, "namespace": "shop"}),
        ("k8s_replace", {"manifest": VERSIONED, "namespace": "shop"}),
        (
            "k8s_delete",
            {"kind": "Deployment", "label_selector": "app=web", "namespace": "shop"},
        ),
    ],
)
async def test_rejection_changes_nothing(tmp_path, name, args) -> None:  # type: ignore[no-untyped-def]
    """The load-bearing assertion: only the dry run may have run."""
    client = MutableClient(live=DEPLOY_LIVE)
    out = await tool(name).run(args, mutation_context(client, tmp_path, approving(Decision.DENY)))
    assert out.is_error and out.denied
    assert client.real_applies == []
    assert client.real_deletes == []
    assert client.real_patches == []
    assert client.real_creates == []
    assert client.real_replaces == []
    assert client.real_collections == []
    assert client.subresources == [], "not even a subresource write"


async def test_dry_run_precedes_the_real_apply(tmp_path) -> None:  # type: ignore[no-untyped-def]
    client = MutableClient(live=DEPLOY_LIVE)
    await tool("k8s_apply").run(
        {"manifest": MANIFEST}, mutation_context(client, tmp_path, approving())
    )
    assert [dry for _, dry in client.applied] == [True, False], "dry run first, then the real one"


async def test_an_invalid_manifest_never_reaches_approval(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The API server rejects it during dry run, so the user is not asked."""

    class Rejecting(MutableClient):
        async def apply(self, *a: Any, **kw: Any) -> dict[str, Any]:
            raise ValueError("spec.replicaz: unknown field")

    policy = approving()
    out = await tool("k8s_apply").run(
        {"manifest": MANIFEST}, mutation_context(Rejecting(), tmp_path, policy)
    )
    assert out.is_error and out.summary == "invalid"
    assert "unknown field" in out.content
    assert policy.seen == [], "a manifest the server refuses must not be put to the user"


async def test_protected_context_is_flagged_on_the_request(tmp_path) -> None:  # type: ignore[no-untyped-def]
    class ProdClient(MutableClient):
        def __init__(self, **kw: Any) -> None:
            super().__init__(**kw)
            self.context = "AKS_EU_PROD"

    policy = approving()
    await tool("k8s_scale").run(
        {"kind": "Deployment", "name": "web", "replicas": 5, "namespace": "shop"},
        mutation_context(ProdClient(live=DEPLOY_LIVE), tmp_path, policy),
    )
    assert policy.seen[0].protected is True
    assert "AKS_EU_PROD" in policy.seen[0].target


async def test_deny_mode_refuses_outright(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from altus.cloud.base import ProtectionMode

    class ProdClient(MutableClient):
        def __init__(self, **kw: Any) -> None:
            super().__init__(**kw)
            self.context = "AKS_EU_PROD"

    policy = approving()
    ctx = ToolContext(
        workspace=Workspace(root=tmp_path),
        approvals=policy,
        cloud=CloudContext(
            k8s=FakeProvider(ProdClient(live=DEPLOY_LIVE)),
            protection=ProtectionRules(patterns=("*prod*",), mode=ProtectionMode.DENY),
        ),
    )
    out = await tool("k8s_delete").run({"kind": "Deployment", "name": "web"}, ctx)
    assert out.is_error and out.summary == "protected"
    assert policy.seen == [], "deny mode does not even ask"


async def test_scale_to_the_same_count_is_a_no_op(tmp_path) -> None:  # type: ignore[no-untyped-def]
    policy = approving()
    client = MutableClient(live=DEPLOY_LIVE)
    out = await tool("k8s_scale").run(
        {"kind": "Deployment", "name": "web", "replicas": 2},
        mutation_context(client, tmp_path, policy),
    )
    assert out.summary == "no change"
    assert policy.seen == [], "no change means no prompt"
    assert client.patched == []


async def test_scale_rejects_nonsense(tmp_path) -> None:  # type: ignore[no-untyped-def]
    ctx = mutation_context(MutableClient(live=DEPLOY_LIVE), tmp_path, approving())
    for replicas, expected in [(-1, "negative"), ("many", "whole number")]:
        out = await tool("k8s_scale").run(
            {"kind": "Deployment", "name": "web", "replicas": replicas}, ctx
        )
        assert out.is_error and expected in out.content


async def test_delete_says_whether_a_controller_will_recreate_it(tmp_path) -> None:  # type: ignore[no-untyped-def]
    owned = {
        **DEPLOY_LIVE,
        "metadata": {
            **DEPLOY_LIVE["metadata"],
            "ownerReferences": [{"kind": "ReplicaSet", "name": "r"}],
        },
    }
    policy = approving()
    await tool("k8s_delete").run(
        {"kind": "Pod", "name": "web-1"},
        mutation_context(MutableClient(live=owned), tmp_path, policy),
    )
    assert "recreated" in policy.seen[0].recoverability

    policy2 = approving()
    await tool("k8s_delete").run(
        {"kind": "Deployment", "name": "web"},
        mutation_context(MutableClient(live=DEPLOY_LIVE), tmp_path, policy2),
    )
    assert "permanent" in policy2.seen[0].recoverability


async def test_apply_diff_ignores_server_churn(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """managedFields and resourceVersion change on every apply; burying the
    real change in that noise defeats the point of showing a diff."""
    noisy = {
        **DEPLOY_LIVE,
        "metadata": {
            **DEPLOY_LIVE["metadata"],
            "managedFields": [{"manager": "kubectl", "time": "2024-01-01T00:00:00Z"}],
            "resourceVersion": "12345",
        },
    }
    policy = approving()
    await tool("k8s_apply").run(
        {"manifest": MANIFEST}, mutation_context(MutableClient(live=noisy), tmp_path, policy)
    )
    diff = policy.seen[0].diff
    assert "managedFields" not in diff
    assert "resourceVersion" not in diff
    assert "replicas" in diff


async def test_mutating_tools_are_marked_as_such() -> None:
    mutating = {t.name for t in k8s_tools() if not t.read_only}
    assert mutating == {
        "k8s_apply",
        "k8s_patch",
        "k8s_create",
        "k8s_replace",
        "k8s_delete",
        "k8s_scale",
        "k8s_rollout",
        "k8s_use_context",
        "k8s_node",
        "k8s_drain",
        "k8s_exec",
        "k8s_cp",
        "k8s_port_forward",
    }


# ------------------------------------------------------ the client itself

# Everything above drives a fake ``K8sClient``. These drive the real one
# against a fake *dynamic* client, because the operations added for the full
# API surface carry real logic --- patch semantics, the collection-delete
# guard, non-JSON responses --- and a fake K8sClient would assert nothing.


class FakeResponse:
    def __init__(self, data: bytes) -> None:
        self.data = data


class FakeApiClient:
    def __init__(self, responses: dict[str, bytes] | None = None) -> None:
        self.responses = responses or {}
        self.calls: list[tuple[str, str, Any]] = []

    def call_api(self, path: str, method: str, **kw: Any) -> FakeResponse:
        self.calls.append((method, path, kw.get("body")))
        return FakeResponse(self.responses.get(path, b'{"ok": true}'))


class FakeResource:
    def __init__(self, recorder: list[tuple[str, dict[str, Any]]], *, namespaced: bool = True):
        self.recorder = recorder
        self.namespaced = namespaced
        self.name = "deployments"

    def path(self, name: str | None = None, namespace: str | None = None) -> str:
        base = "/apis/apps/v1"
        if namespace:
            base += f"/namespaces/{namespace}"
        base += f"/{self.name}"
        return f"{base}/{name}" if name else base

    def _record(self, verb: str, **kw: Any) -> dict[str, Any]:
        self.recorder.append((verb, kw))
        return {"metadata": {"name": kw.get("name", "web")}}

    def patch(self, **kw: Any) -> dict[str, Any]:
        return self._record("patch", **kw)

    def create(self, **kw: Any) -> dict[str, Any]:
        return self._record("create", **kw)

    def replace(self, **kw: Any) -> dict[str, Any]:
        return self._record("replace", **kw)

    def delete(self, **kw: Any) -> dict[str, Any]:
        return self._record("delete", **kw)


class FakeResources:
    def __init__(self, resource: FakeResource) -> None:
        self.resource = resource

    def get(self, **kw: Any) -> FakeResource:
        return self.resource


class FakeDynamic:
    def __init__(self, *, responses: dict[str, bytes] | None = None, namespaced: bool = True):
        self.recorder: list[tuple[str, dict[str, Any]]] = []
        self.resource = FakeResource(self.recorder, namespaced=namespaced)
        self.resources = FakeResources(self.resource)
        self.client = FakeApiClient(responses)


def real_client(**kw: Any) -> tuple[k8s_api.K8sClient, FakeDynamic]:
    dynamic = FakeDynamic(**kw)
    return k8s_api.K8sClient(dynamic=dynamic, context="AKS_QAM"), dynamic


@pytest.mark.parametrize(
    ("patch_type", "content_type"),
    [
        ("merge", "application/merge-patch+json"),
        ("strategic", "application/strategic-merge-patch+json"),
        ("json", "application/json-patch+json"),
    ],
)
async def test_patch_sends_the_content_type_the_caller_asked_for(
    patch_type: str, content_type: str
) -> None:
    """Not interchangeable: a merge patch replaces spec.containers wholesale
    where a strategic one edits a single container in place."""
    client, dynamic = real_client()
    await client.patch("apps/v1", "Deployment", "web", {"spec": {}}, "shop", patch_type=patch_type)
    verb, kwargs = dynamic.recorder[0]
    assert verb == "patch"
    assert kwargs["content_type"] == content_type


async def test_patch_rejects_an_unknown_patch_type_before_calling_the_server() -> None:
    client, dynamic = real_client()
    with pytest.raises(ValueError, match="unknown patch type"):
        await client.patch("apps/v1", "Deployment", "web", {}, "shop", patch_type="telepathy")
    assert dynamic.recorder == [], "nothing must reach the API server"


async def test_collection_delete_refuses_without_a_selector() -> None:
    """An unfiltered deletecollection removes every object of the kind in
    scope. That must not be expressible by leaving an argument out."""
    client, dynamic = real_client()
    with pytest.raises(ValueError, match="requires a label or field selector"):
        await client.delete_collection("apps/v1", "Deployment", "shop")
    assert dynamic.recorder == []


async def test_collection_delete_passes_the_selector_through() -> None:
    client, dynamic = real_client()
    await client.delete_collection("apps/v1", "Deployment", "shop", label_selector="app=web")
    verb, kwargs = dynamic.recorder[0]
    assert verb == "delete"
    assert kwargs["label_selector"] == "app=web"
    assert "name" not in kwargs, "a collection delete names no single object"


async def test_dry_run_is_a_server_side_query_parameter() -> None:
    client, dynamic = real_client()
    await client.create(
        "apps/v1", "Deployment", {"metadata": {"name": "web"}}, "shop", dry_run=True
    )
    _, kwargs = dynamic.recorder[0]
    assert kwargs["query_params"] == [("dryRun", "All")]


async def test_raw_returns_text_when_the_response_is_not_json() -> None:
    """/healthz answers `ok` and /metrics answers Prometheus text. Neither is
    JSON, and neither should raise."""
    client, _ = real_client(responses={"/healthz": b"ok"})
    assert await client.raw("GET", "/healthz") == {"raw": "ok"}


async def test_raw_parses_json_when_it_is_json() -> None:
    client, _ = real_client(responses={"/version": b'{"gitVersion": "v1.29.4"}'})
    assert (await client.raw("GET", "/version"))["gitVersion"] == "v1.29.4"


async def test_subresource_builds_the_path_under_the_object() -> None:
    client, dynamic = real_client()
    await client.subresource(
        "POST", "v1", "Pod", "web-1", "eviction", namespace="shop", body={"kind": "Eviction"}
    )
    method, path, body = dynamic.client.calls[0]
    assert method == "POST"
    assert path == "/apis/apps/v1/namespaces/shop/deployments/web-1/eviction"
    assert body == {"kind": "Eviction"}


async def test_subresource_omits_the_namespace_for_a_cluster_scoped_kind() -> None:
    client, dynamic = real_client(namespaced=False)
    await client.subresource("GET", "v1", "Node", "node-7", "status", namespace="shop")
    _, path, _ = dynamic.client.calls[0]
    assert "/namespaces/" not in path


async def test_get_one_with_a_subresource_goes_through_the_subresource_path() -> None:
    client, dynamic = real_client()
    await client.get_one("apps/v1", "Deployment", "web", "shop", subresource="scale")
    assert dynamic.client.calls[0][1].endswith("/web/scale")


# --------------------------------------------------- the rest of the surface


SECRET_OBJ = {
    "kind": "Secret",
    "metadata": {
        "name": "db-creds",
        "namespace": "shop",
        "managedFields": [{"manager": "kubectl"}],
        "resourceVersion": "9",
        "annotations": {"kubectl.kubernetes.io/last-applied-configuration": "{...}", "keep": "me"},
    },
    "data": {"password": "aHVudGVyMg=="},
}


async def test_get_returns_the_whole_object_as_yaml(tmp_path) -> None:  # type: ignore[no-untyped-def]
    client = FakeClient({"Deployment": CLUSTER["Deployment"]})
    out = await tool("k8s_get").run(
        {"kind": "Deployment", "name": "web", "namespace": "shop"}, context_for(client, tmp_path)
    )
    assert not out.is_error
    assert "replicas: 2" in out.content, "the spec the model is about to change"


async def test_get_redacts_a_secret(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """k8s_get returns everything, which is exactly why redaction matters more
    here than in k8s_list."""
    client = FakeClient({"Secret": [SECRET_OBJ]})
    out = await tool("k8s_get").run(
        {"kind": "Secret", "name": "db-creds", "namespace": "shop"}, context_for(client, tmp_path)
    )
    assert "aHVudGVyMg" not in out.content
    assert MARKER in out.content


async def test_get_strips_server_bookkeeping_by_default(tmp_path) -> None:  # type: ignore[no-untyped-def]
    client = FakeClient({"Secret": [SECRET_OBJ]})
    ctx = context_for(client, tmp_path)
    out = await tool("k8s_get").run({"kind": "Secret", "name": "db-creds"}, ctx)
    assert "managedFields" not in out.content
    assert "resourceVersion" not in out.content
    assert "last-applied-configuration" not in out.content
    assert "keep" in out.content, "only the noise goes"

    verbose = await tool("k8s_get").run({"kind": "Secret", "name": "db-creds", "quiet": False}, ctx)
    assert "managedFields" in verbose.content


async def test_get_refuses_a_privileged_subresource_and_names_the_right_tool(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """`get pods/exec` is code execution wearing a read verb. It must not be
    reachable through the tool that asks for nothing."""
    out = await tool("k8s_get").run(
        {"kind": "Pod", "name": "web-1", "subresource": "exec"},
        context_for(FakeClient(), tmp_path),
    )
    assert out.is_error
    assert "k8s_exec" in out.content


async def test_raw_reaches_a_non_resource_endpoint(tmp_path) -> None:  # type: ignore[no-untyped-def]
    client = FakeClient()
    client.raw_responses["/healthz"] = {"raw": "ok"}
    out = await tool("k8s_raw").run({"path": "/healthz"}, context_for(client, tmp_path))
    assert not out.is_error
    assert "ok" in out.content
    assert client.raw_calls == [("GET", "/healthz")]


async def test_raw_refuses_a_privileged_path(tmp_path) -> None:  # type: ignore[no-untyped-def]
    client = FakeClient()
    out = await tool("k8s_raw").run(
        {"path": "/api/v1/namespaces/shop/pods/web-1/exec"}, context_for(client, tmp_path)
    )
    assert out.is_error
    assert "k8s_exec" in out.content
    assert client.raw_calls == [], "nothing must reach the API server"


async def test_raw_requires_an_absolute_path(tmp_path) -> None:  # type: ignore[no-untyped-def]
    out = await tool("k8s_raw").run({"path": "healthz"}, context_for(FakeClient(), tmp_path))
    assert out.is_error


async def test_can_i_asks_rbac_before_attempting(tmp_path) -> None:  # type: ignore[no-untyped-def]
    client = FakeClient()
    client.reviews["SelfSubjectAccessReview"] = {"status": {"allowed": True}}
    out = await tool("k8s_can_i").run(
        {"verb": "delete", "resource": "pods", "namespace": "shop"},
        context_for(client, tmp_path),
    )
    assert not out.is_error
    assert out.content.startswith("yes")
    kind, body, _ = client.created[0]
    assert kind == "SelfSubjectAccessReview"
    assert body["spec"]["resourceAttributes"]["verb"] == "delete"


async def test_can_i_reports_a_denial_with_the_reason(tmp_path) -> None:  # type: ignore[no-untyped-def]
    client = FakeClient()
    client.reviews["SelfSubjectAccessReview"] = {
        "status": {"allowed": False, "reason": "no RoleBinding grants this"}
    }
    out = await tool("k8s_can_i").run(
        {"verb": "delete", "resource": "nodes"}, context_for(client, tmp_path)
    )
    assert out.content.startswith("no")
    assert "no RoleBinding grants this" in out.content


async def test_can_i_without_a_verb_lists_every_rule(tmp_path) -> None:  # type: ignore[no-untyped-def]
    client = FakeClient()
    client.reviews["SelfSubjectRulesReview"] = {
        "status": {"resourceRules": [{"verbs": ["get", "list"], "resources": ["pods"]}]}
    }
    out = await tool("k8s_can_i").run({"namespace": "shop"}, context_for(client, tmp_path))
    assert "get,list" in out.content
    assert client.created[0][0] == "SelfSubjectRulesReview"


async def test_wait_needs_to_be_told_what_to_wait_for(tmp_path) -> None:  # type: ignore[no-untyped-def]
    out = await tool("k8s_wait").run(
        {"kind": "Pod", "name": "web"}, context_for(FakeClient(), tmp_path)
    )
    assert out.is_error
    assert "condition" in out.content


async def test_wait_reports_the_transitions_it_saw_on_timeout(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A timeout that only says `not ready` is not a diagnosis."""
    client = FakeClient()
    client.watch_result = (
        False,
        [
            {"type": "MODIFIED", "name": "web-1", "status": "Pending"},
            {"type": "MODIFIED", "name": "web-1", "status": "CrashLoopBackOff"},
        ],
        "the condition was not met within 120s",
    )
    out = await tool("k8s_wait").run(
        {"kind": "Pod", "name": "web-1", "condition": "Ready"}, context_for(client, tmp_path)
    )
    assert out.is_error
    assert "CrashLoopBackOff" in out.content
    assert out.summary == "timed out"


async def test_wait_succeeds_when_the_condition_is_met(tmp_path) -> None:  # type: ignore[no-untyped-def]
    client = FakeClient()
    client.watch_result = (
        True,
        [{"type": "MODIFIED", "name": "web-1", "status": "Running"}],
        "met",
    )
    out = await tool("k8s_wait").run(
        {"kind": "Pod", "name": "web-1", "condition": "Ready"}, context_for(client, tmp_path)
    )
    assert not out.is_error
    assert out.summary == "condition met"


async def test_wait_for_deletion_reads_absence_from_the_transitions() -> None:
    """No state of an object means `gone`, so deletion is read from the event
    stream rather than from a predicate."""
    from altus.tools.k8s.reads import _predicate

    ready = _predicate(deleted=True, condition="", want="", field="", value="")
    assert ready({"status": {"phase": "Running"}}) is False


def test_condition_and_field_predicates() -> None:
    from altus.tools.k8s.reads import _condition_met, _field_equals

    obj = {"status": {"phase": "Running", "conditions": [{"type": "Ready", "status": "True"}]}}
    assert _condition_met(obj, "Ready", "True")
    assert _condition_met(obj, "ready", "true"), "type and status compare case-insensitively"
    assert not _condition_met(obj, "Available", "True"), "a missing condition is not met"
    assert _field_equals(obj, "status.phase", "Running")
    assert not _field_equals(obj, "status.nope.deeper", "x"), "a missing path is not a crash"


async def test_every_new_read_tool_is_read_only() -> None:
    names = {"k8s_get", "k8s_raw", "k8s_can_i", "k8s_wait"}
    for entry in k8s_tools():
        if entry.name in names:
            assert entry.read_only, f"{entry.name} must not need approval"
            names.discard(entry.name)
    assert not names, f"not registered: {names}"


# ------------------------------------------------- the rest of the mutations


async def test_patch_defaults_to_strategic_merge(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """kubectl edit uses strategic, and it is the one that edits a single
    container rather than replacing the whole list."""
    client = MutableClient(live=DEPLOY_LIVE)
    captured: list[str] = []
    original = client.patch

    async def spy(*a: Any, **kw: Any) -> Any:
        captured.append(kw.get("patch_type", "merge"))
        return await original(*a, **kw)

    client.patch = spy  # type: ignore[method-assign]
    await tool("k8s_patch").run(
        {"kind": "Deployment", "name": "web", "patch": {"spec": {}}},
        mutation_context(client, tmp_path, approving()),
    )
    assert captured and all(c == "strategic" for c in captured)


async def test_patch_rejects_an_unknown_patch_type_without_asking(tmp_path) -> None:  # type: ignore[no-untyped-def]
    client = MutableClient(live=DEPLOY_LIVE)
    policy = approving()
    out = await tool("k8s_patch").run(
        {"kind": "Deployment", "name": "web", "patch": {}, "patch_type": "telepathy"},
        mutation_context(client, tmp_path, policy),
    )
    assert out.is_error
    assert policy.seen == [], "a bad argument must not reach a prompt"


async def test_json_patch_must_be_an_array(tmp_path) -> None:  # type: ignore[no-untyped-def]
    out = await tool("k8s_patch").run(
        {"kind": "Deployment", "name": "web", "patch": {"a": 1}, "patch_type": "json"},
        mutation_context(MutableClient(live=DEPLOY_LIVE), tmp_path, approving()),
    )
    assert out.is_error and "array" in out.content


async def test_replace_refuses_without_a_resource_version(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Without it the API server happily overwrites a concurrent edit. The
    point of replace over apply is that it fails loudly instead."""
    policy = approving()
    out = await tool("k8s_replace").run(
        {"manifest": MANIFEST}, mutation_context(MutableClient(live=DEPLOY_LIVE), tmp_path, policy)
    )
    assert out.is_error
    assert "resourceVersion" in out.content
    assert policy.seen == []


async def test_collection_delete_shows_the_objects_not_the_selector(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Approving `app=web` is not consent to delete whatever wears that label
    today, so the prompt lists what actually matched."""
    client = MutableClient({"Pod": [pod("web-1"), pod("web-2")]}, live=DEPLOY_LIVE)
    policy = approving()
    out = await tool("k8s_delete").run(
        {"kind": "Pod", "label_selector": "app=web", "namespace": "shop"},
        mutation_context(client, tmp_path, policy),
    )
    assert not out.is_error, out.content
    diff = policy.seen[0].diff
    assert "Pod/web-1" in diff and "Pod/web-2" in diff
    assert client.real_collections == ["app=web"]


async def test_collection_delete_is_classified_privileged(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """One call, an unbounded number of objects, and no per-object prompt."""
    client = MutableClient({"Pod": [pod("web-1")]}, live=DEPLOY_LIVE)
    policy = approving()
    await tool("k8s_delete").run(
        {"kind": "Pod", "label_selector": "app=web", "namespace": "shop"},
        mutation_context(client, tmp_path, policy),
    )
    request = policy.seen[0]
    assert request.sensitivity is Sensitivity.PRIVILEGED
    assert request.needs_challenge
    assert not request.may_grant_always


async def test_collection_delete_with_no_matches_does_nothing(tmp_path) -> None:  # type: ignore[no-untyped-def]
    client = MutableClient({"Pod": []}, live=DEPLOY_LIVE)
    policy = approving()
    out = await tool("k8s_delete").run(
        {"kind": "Pod", "label_selector": "app=nothing"},
        mutation_context(client, tmp_path, policy),
    )
    assert not out.is_error
    assert policy.seen == [], "nothing to delete, so nothing to ask about"
    assert client.collections == []


async def test_delete_refuses_a_kind_with_neither_name_nor_selector(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """There is deliberately no way to say `every Deployment`."""
    policy = approving()
    out = await tool("k8s_delete").run(
        {"kind": "Deployment"}, mutation_context(MutableClient(live=DEPLOY_LIVE), tmp_path, policy)
    )
    assert out.is_error
    assert policy.seen == []


async def test_creating_a_token_says_it_outlives_the_session(tmp_path) -> None:  # type: ignore[no-untyped-def]
    client = MutableClient(live=DEPLOY_LIVE)
    policy = approving()
    await tool("k8s_create").run(
        {
            "manifest": {"apiVersion": "v1", "kind": "ServiceAccount"},
            "on": "builder",
            "subresource": "token",
            "namespace": "shop",
        },
        mutation_context(client, tmp_path, policy),
    )
    request = policy.seen[0]
    assert request.sensitivity is Sensitivity.PRIVILEGED
    assert "outlives this session" in request.recoverability
    assert client.subresources == [("POST", "ServiceAccount", "builder", "token", False)]


async def test_rollout_undo_restores_the_previous_replicaset_template(tmp_path) -> None:  # type: ignore[no-untyped-def]
    def replicaset(name: str, revision: str, image: str) -> dict[str, Any]:
        return {
            "metadata": {
                "name": name,
                "namespace": "shop",
                "annotations": {"deployment.kubernetes.io/revision": revision},
                "ownerReferences": [{"kind": "Deployment", "name": "web"}],
            },
            "spec": {
                "template": {
                    "metadata": {"labels": {"app": "web", "pod-template-hash": "abc"}},
                    "spec": {"containers": [{"name": "c", "image": image}]},
                }
            },
        }

    client = MutableClient(
        {"ReplicaSet": [replicaset("web-1", "1", "app:v1"), replicaset("web-2", "2", "app:v2")]},
        live=DEPLOY_LIVE,
    )
    out = await tool("k8s_rollout").run(
        {"kind": "Deployment", "name": "web", "namespace": "shop", "action": "undo"},
        mutation_context(client, tmp_path, approving()),
    )
    assert not out.is_error, out.content
    template = client.real_patches[0]["spec"]["template"]
    assert template["spec"]["containers"][0]["image"] == "app:v1", "the older revision"
    assert "pod-template-hash" not in template["metadata"]["labels"], (
        "the hash belongs to the old ReplicaSet; carrying it over makes the new one unselectable"
    )


async def test_rollout_undo_says_so_when_there_is_no_history(tmp_path) -> None:  # type: ignore[no-untyped-def]
    client = MutableClient({"ReplicaSet": []}, live=DEPLOY_LIVE)
    policy = approving()
    out = await tool("k8s_rollout").run(
        {"kind": "Deployment", "name": "web", "action": "undo"},
        mutation_context(client, tmp_path, policy),
    )
    assert out.is_error and "nothing to roll back" in out.content
    assert policy.seen == []


async def test_rollout_pause_is_a_no_op_when_already_paused(tmp_path) -> None:  # type: ignore[no-untyped-def]
    client = MutableClient(live={**DEPLOY_LIVE, "spec": {"replicas": 2, "paused": True}})
    policy = approving()
    out = await tool("k8s_rollout").run(
        {"kind": "Deployment", "name": "web", "action": "pause"},
        mutation_context(client, tmp_path, policy),
    )
    assert not out.is_error and out.summary == "no change"
    assert policy.seen == [], "nothing changes, so nothing to approve"


async def test_rollout_status_needs_no_approval(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Checking on a deploy must never prompt --- which is why it is a
    separate tool from the one that drives the rollout."""
    client = FakeClient(
        {
            "Deployment": [
                {
                    "metadata": {"name": "web", "namespace": "shop", "generation": 2},
                    "spec": {"replicas": 3},
                    "status": {"updatedReplicas": 3, "readyReplicas": 3, "observedGeneration": 2},
                }
            ]
        }
    )
    out = await tool("k8s_rollout_status").run(
        {"kind": "Deployment", "name": "web", "namespace": "shop"},
        context_for(client, tmp_path),
    )
    assert not out.is_error
    assert out.summary == "complete"
    assert tool("k8s_rollout_status").read_only


# ------------------------------------------------------------ switching context


def kubeconfig_at(path, contexts=("AKS_QAM", "AKS_EU_PROD")):  # type: ignore[no-untyped-def]
    import yaml

    path.write_text(
        yaml.safe_dump(
            {
                "apiVersion": "v1",
                "kind": "Config",
                "current-context": contexts[0],
                "contexts": [
                    {"name": n, "context": {"cluster": n, "user": "u", "namespace": "default"}}
                    for n in contexts
                ],
                "clusters": [{"name": n, "cluster": {"server": "https://x"}} for n in contexts],
                "users": [{"name": "u", "user": {}}],
            }
        )
    )
    return path


async def test_switching_context_asks_first(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """It changes where every later command lands, so it is never silent."""
    monkeypatch.setenv("KUBECONFIG", str(kubeconfig_at(tmp_path / "kc")))
    switched: list[tuple[str, str]] = []
    policy = approving()
    ctx = ToolContext(
        workspace=Workspace(root=tmp_path),
        approvals=policy,
        cloud=CloudContext(
            protection=ProtectionRules(patterns=("*prod*",)),
            kube_context="AKS_QAM",
            on_context_change=lambda n, ns: switched.append((n, ns)),
        ),
    )
    out = await tool("k8s_use_context").run({"context": "AKS_EU_PROD"}, ctx)
    assert not out.is_error, out.content
    assert len(policy.seen) == 1
    assert switched == [("AKS_EU_PROD", "default")]


async def test_switching_to_a_protected_context_demands_the_typed_challenge(
    tmp_path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("KUBECONFIG", str(kubeconfig_at(tmp_path / "kc")))
    policy = approving()
    ctx = ToolContext(
        workspace=Workspace(root=tmp_path),
        approvals=policy,
        cloud=CloudContext(
            protection=ProtectionRules(patterns=("*prod*",)), kube_context="AKS_QAM"
        ),
    )
    await tool("k8s_use_context").run({"context": "AKS_EU_PROD"}, ctx)
    assert policy.seen[0].protected
    assert policy.seen[0].needs_challenge


async def test_switching_context_refuses_an_unknown_name(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("KUBECONFIG", str(kubeconfig_at(tmp_path / "kc")))
    policy = approving()
    ctx = ToolContext(workspace=Workspace(root=tmp_path), approvals=policy, cloud=CloudContext())
    out = await tool("k8s_use_context").run({"context": "nope"}, ctx)
    assert out.is_error
    assert "AKS_QAM" in out.content, "name what is available"
    assert policy.seen == []


async def test_switching_context_drops_the_cached_client() -> None:
    """The cached client holds a connection built for the old context; reusing
    it would send the next call to the cluster you just left."""
    from altus.cloud.k8s import K8sProvider

    provider = K8sProvider(context="AKS_QAM")
    provider._client = object()  # type: ignore[assignment]
    cloud = CloudContext(k8s=provider, kube_context="AKS_QAM")
    cloud.switch_context("AKS_EU_PROD", "default")
    assert provider.context == "AKS_EU_PROD"
    assert provider._client is None


# ---------------------------------------------------------------- node lifecycle


NODE = {"kind": "Node", "metadata": {"name": "node-7"}, "spec": {}}


def node_pod(name, *, owner="ReplicaSet", ns="shop", mirror=False, empty_dir=False):  # type: ignore[no-untyped-def]
    meta: dict[str, Any] = {"name": name, "namespace": ns}
    if owner:
        meta["ownerReferences"] = [{"kind": owner, "name": "x"}]
    if mirror:
        meta["annotations"] = {"kubernetes.io/config.mirror": "abc"}
    volumes = [{"name": "d", "emptyDir": {}}] if empty_dir else []
    return {"kind": "Pod", "metadata": meta, "spec": {"nodeName": "node-7", "volumes": volumes}}


class NodeClient(MutableClient):
    """A cluster with one node and whatever pods the test puts on it."""

    def __init__(self, pods: list[dict[str, Any]] | None = None, **kw: Any) -> None:
        super().__init__({"Pod": pods or [], "PodDisruptionBudget": []}, live=NODE, **kw)
        self.evicted: list[str] = []

    async def subresource(
        self,
        method,
        api_version,
        kind,
        name,
        subresource,
        *,
        namespace=None,
        body=None,
        dry_run=False,
    ):  # type: ignore[no-untyped-def]
        self.subresources.append((method, kind, name, subresource, dry_run))
        if subresource == "eviction":
            self.evicted.append(f"{namespace}/{name}")
        return {}


def node_context(client, tmp_path, policy):  # type: ignore[no-untyped-def]
    return mutation_context(client, tmp_path, policy)


async def test_cordon_is_privileged_and_needs_a_typed_confirmation(tmp_path) -> None:  # type: ignore[no-untyped-def]
    client = NodeClient()
    policy = approving()
    out = await tool("k8s_node").run(
        {"node": "node-7", "action": "cordon"}, node_context(client, tmp_path, policy)
    )
    assert not out.is_error, out.content
    request = policy.seen[0]
    assert request.sensitivity is Sensitivity.PRIVILEGED, "writing a Node is privileged"
    assert request.needs_challenge and not request.may_grant_always
    assert client.real_patches == [{"spec": {"unschedulable": True}}]


async def test_cordon_twice_is_a_no_op(tmp_path) -> None:  # type: ignore[no-untyped-def]
    client = NodeClient()
    client.live = {**NODE, "spec": {"unschedulable": True}}
    policy = approving()
    out = await tool("k8s_node").run(
        {"node": "node-7", "action": "cordon"}, node_context(client, tmp_path, policy)
    )
    assert out.summary == "no change"
    assert policy.seen == []


async def test_a_no_execute_taint_says_it_evicts(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """NoSchedule and NoExecute differ by whether running pods are thrown off.
    That distinction has to reach the prompt."""
    client = NodeClient()
    policy = approving()
    await tool("k8s_node").run(
        {"node": "node-7", "action": "taint", "key": "maintenance", "effect": "NoExecute"},
        node_context(client, tmp_path, policy),
    )
    assert "evicts running pods" in policy.seen[0].diff


async def test_taint_rejects_an_unknown_effect(tmp_path) -> None:  # type: ignore[no-untyped-def]
    policy = approving()
    out = await tool("k8s_node").run(
        {"node": "node-7", "action": "taint", "key": "k", "effect": "NoBananas"},
        node_context(NodeClient(), tmp_path, policy),
    )
    assert out.is_error
    assert policy.seen == []


async def test_untaint_removes_only_the_matching_taint(tmp_path) -> None:  # type: ignore[no-untyped-def]
    client = NodeClient()
    client.live = {
        **NODE,
        "spec": {
            "taints": [
                {"key": "a", "effect": "NoSchedule"},
                {"key": "b", "effect": "NoSchedule"},
            ]
        },
    }
    await tool("k8s_node").run(
        {"node": "node-7", "action": "untaint", "key": "a"},
        node_context(client, tmp_path, approving()),
    )
    remaining = client.real_patches[0]["spec"]["taints"]
    assert [t["key"] for t in remaining] == ["b"]


async def test_drain_skips_daemonset_and_mirror_pods(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Evicting a DaemonSet pod loops — it is recreated on the same node. A
    mirror pod cannot be evicted at all."""
    client = NodeClient(
        [
            node_pod("web-1"),
            node_pod("agent-1", owner="DaemonSet"),
            node_pod("static-1", owner=None, mirror=True),
        ]
    )
    out = await tool("k8s_drain").run(
        {"node": "node-7"}, node_context(client, tmp_path, approving())
    )
    assert not out.is_error, out.content
    assert client.evicted == ["shop/web-1"]
    assert "2 skipped" in out.content


async def test_drain_refuses_a_standalone_pod_unless_forced(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Nothing recreates it, so evicting it simply destroys it."""
    client = NodeClient([node_pod("orphan", owner=None)])
    policy = approving()
    out = await tool("k8s_drain").run({"node": "node-7"}, node_context(client, tmp_path, policy))
    assert out.is_error
    assert "no controller" in out.content
    assert policy.seen == [], "refused before anyone is asked"
    assert client.evicted == []

    forced = await tool("k8s_drain").run(
        {"node": "node-7", "force": True}, node_context(client, tmp_path, approving())
    )
    assert not forced.is_error, forced.content
    assert client.evicted == ["shop/orphan"]


async def test_drain_refuses_emptydir_data_unless_told(tmp_path) -> None:  # type: ignore[no-untyped-def]
    client = NodeClient([node_pod("cache-1", empty_dir=True)])
    out = await tool("k8s_drain").run(
        {"node": "node-7"}, node_context(client, tmp_path, approving())
    )
    assert out.is_error and "emptyDir" in out.content

    allowed = await tool("k8s_drain").run(
        {"node": "node-7", "delete_emptydir_data": True},
        node_context(client, tmp_path, approving()),
    )
    assert not allowed.is_error
    assert client.evicted == ["shop/cache-1"]


async def test_drain_evicts_rather_than_deletes(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The API server enforces PodDisruptionBudgets on an eviction and ignores
    them for a delete. Using the wrong one silently defeats every budget."""
    client = NodeClient([node_pod("web-1")])
    await tool("k8s_drain").run({"node": "node-7"}, node_context(client, tmp_path, approving()))
    assert client.real_deletes == [], "nothing may be deleted outright"
    assert [s[3] for s in client.subresources] == ["eviction"]


async def test_drain_names_the_pod_count_and_the_budgets(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """ "drain node-7" is not a decision anyone can make."""
    client = NodeClient([node_pod("web-1"), node_pod("web-2")])
    client.objects["PodDisruptionBudget"] = [{"metadata": {"name": "web-pdb"}}]
    policy = approving()
    await tool("k8s_drain").run({"node": "node-7"}, node_context(client, tmp_path, policy))
    request = policy.seen[0]
    assert "evict 2 pods" in request.diff
    assert "shop/web-1" in request.diff
    assert "web-pdb" in request.dry_run


async def test_drain_cordons_before_evicting(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Evicting without cordoning first lets the scheduler put the pods back."""
    client = NodeClient([node_pod("web-1")])
    await tool("k8s_drain").run({"node": "node-7"}, node_context(client, tmp_path, approving()))
    assert client.real_patches[0] == {"spec": {"unschedulable": True}}


async def test_drain_reports_a_budget_refusal_instead_of_retrying(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A 429 is the PodDisruptionBudget doing its job."""

    class Blocked(NodeClient):
        async def subresource(self, *a: Any, **kw: Any) -> Any:
            raise RuntimeError("429 Cannot evict pod as it would violate the budget")

    client = Blocked([node_pod("web-1")])
    out = await tool("k8s_drain").run(
        {"node": "node-7"}, node_context(client, tmp_path, approving())
    )
    assert out.is_error
    assert "PodDisruptionBudget" in out.content
    assert "1 refused" in out.content


async def test_drain_with_nothing_to_evict_does_nothing(tmp_path) -> None:  # type: ignore[no-untyped-def]
    client = NodeClient([])
    policy = approving()
    out = await tool("k8s_drain").run({"node": "node-7"}, node_context(client, tmp_path, policy))
    assert not out.is_error
    assert policy.seen == []
    assert client.real_patches == [], "not even the cordon"


# ------------------------------------------------------------------ the streams


class ExecClient(MutableClient):
    def __init__(self, out: str = "hello\n", err: str = "", **kw: Any) -> None:
        super().__init__(live=DEPLOY_LIVE, **kw)
        self.execs: list[tuple[str, list[str], str | None]] = []
        self._out, self._err = out, err

    async def exec_pod(
        self, name, namespace, command, *, container=None, stdin=None, timeout_seconds=60.0
    ):  # type: ignore[no-untyped-def]
        self.execs.append((name, list(command), stdin))
        return self._out, self._err


async def test_exec_is_privileged_and_never_grants_standing_approval(tmp_path) -> None:  # type: ignore[no-untyped-def]
    client = ExecClient()
    policy = approving()
    out = await tool("k8s_exec").run(
        {"pod": "web-1", "command": ["ls", "/data"], "namespace": "shop"},
        mutation_context(client, tmp_path, policy),
    )
    assert not out.is_error, out.content
    request = policy.seen[0]
    assert request.sensitivity is Sensitivity.PRIVILEGED
    assert request.needs_challenge and not request.may_grant_always


async def test_exec_refuses_a_string_command(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Splitting a string is where `sh -c` creeps back in."""
    client = ExecClient()
    policy = approving()
    out = await tool("k8s_exec").run(
        {"pod": "web-1", "command": "ls /data | grep x"},
        mutation_context(client, tmp_path, policy),
    )
    assert out.is_error
    assert client.execs == []
    assert policy.seen == [], "refused before anyone is asked"


async def test_exec_passes_argv_through_untouched(tmp_path) -> None:  # type: ignore[no-untyped-def]
    client = ExecClient()
    hostile = ["sh", "-c", "echo $(id); rm -rf /"]
    await tool("k8s_exec").run(
        {"pod": "web-1", "command": hostile}, mutation_context(client, tmp_path, approving())
    )
    assert client.execs[0][1] == hostile, "Altus neither splits nor rewrites the command"


async def test_exec_redacts_what_the_command_printed(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """`env` inside a container prints every secret the pod was given."""
    client = ExecClient(out="DB_PASSWORD=hunter2\nPATH=/usr/bin\n")
    out = await tool("k8s_exec").run(
        {"pod": "web-1", "command": ["env"]}, mutation_context(client, tmp_path, approving())
    )
    assert "hunter2" not in out.content
    assert MARKER in out.content


async def test_exec_shows_the_command_and_the_reason_in_the_prompt(tmp_path) -> None:  # type: ignore[no-untyped-def]
    policy = approving()
    await tool("k8s_exec").run(
        {"pod": "web-1", "command": ["cat", "/etc/nginx/nginx.conf"], "reason": "config on disk"},
        mutation_context(ExecClient(), tmp_path, policy),
    )
    diff = policy.seen[0].diff
    assert "cat /etc/nginx/nginx.conf" in diff
    assert "config on disk" in diff


async def test_exec_denied_runs_nothing(tmp_path) -> None:  # type: ignore[no-untyped-def]
    client = ExecClient()
    out = await tool("k8s_exec").run(
        {"pod": "web-1", "command": ["rm", "-rf", "/"]},
        mutation_context(client, tmp_path, approving(Decision.DENY)),
    )
    assert out.is_error and out.denied
    assert client.execs == []


async def test_cp_out_of_a_pod_writes_through_the_workspace_guard(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The same rule that stops write_file touching ~/.ssh."""
    import base64

    client = ExecClient(out=base64.b64encode(b"payload").decode())
    out = await tool("k8s_cp").run(
        {
            "pod": "web-1",
            "direction": "from_pod",
            "remote_path": "/etc/app.conf",
            "local_path": "app.conf",
        },
        mutation_context(client, tmp_path, approving()),
    )
    assert not out.is_error, out.content
    assert (tmp_path / "app.conf").read_bytes() == b"payload"


@pytest.mark.parametrize("target", ["../escape.txt", "/etc/passwd", ".env"])
async def test_cp_cannot_write_outside_the_workspace_or_over_a_secret(tmp_path, target) -> None:  # type: ignore[no-untyped-def]
    import base64

    client = ExecClient(out=base64.b64encode(b"x").decode())
    policy = approving()
    out = await tool("k8s_cp").run(
        {
            "pod": "web-1",
            "direction": "from_pod",
            "remote_path": "/x",
            "local_path": target,
        },
        mutation_context(client, tmp_path, policy),
    )
    assert out.is_error, f"{target} should have been refused"
    assert policy.seen == [], "refused before anyone is asked"
    assert client.execs == []


async def test_cp_quotes_the_remote_path_rather_than_interpolating(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """sh -c is unavoidable here, so a quote inside the path must not end the
    quoting and start a second command."""
    import base64

    client = ExecClient(out=base64.b64encode(b"x").decode())
    await tool("k8s_cp").run(
        {
            "pod": "web-1",
            "direction": "from_pod",
            "remote_path": "/tmp/a'; id; echo '",
            "local_path": "out.txt",
        },
        mutation_context(client, tmp_path, approving()),
    )
    script = client.execs[0][1][2]
    assert script.startswith("base64 < '")
    assert script.endswith("'")
    assert "'\\''" in script, "the embedded quote is escaped, not closed"


async def test_port_forward_binds_loopback_only_and_is_session_scoped() -> None:
    """A tunnel that outlives its session is a hole nobody remembers opening."""
    from altus.tools.k8s.streams import PortForwards

    closed: list[str] = []

    class Handle:
        def __init__(self, tag: str) -> None:
            self.tag = tag

        def close(self) -> None:
            closed.append(self.tag)

    forwards = PortForwards()
    forwards.add("shop/web:80", Handle("a"), "localhost:80 -> web:80")
    forwards.add("shop/api:81", Handle("b"), "localhost:81 -> api:81")
    assert len(forwards) == 2
    assert forwards.close("shop/web:80")
    assert closed == ["a"]
    assert not forwards.close("nope")
    assert forwards.close_all() == 1
    assert closed == ["a", "b"]


async def test_port_forward_rejects_an_impossible_port(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from altus.tools.k8s.streams import PortForwards

    policy = approving()
    ctx = mutation_context(ExecClient(), tmp_path, policy)
    ctx.cloud.port_forwards = PortForwards()
    out = await tool("k8s_port_forward").run(
        {"action": "start", "pod": "web-1", "remote_port": 99999}, ctx
    )
    assert out.is_error
    assert policy.seen == []


@pytest.mark.parametrize("name", ["k8s_exec", "k8s_cp", "k8s_port_forward"])
def test_a_tool_declares_the_subresource_its_classification_depends_on(name: str) -> None:
    """A tool whose danger lives in its subresource must declare it.

    Leaving `subresource` unset downgrades the tool to whatever its verb alone
    implies --- and for all three of these that verb is `get`, so each would
    classify as an ordinary read with a standing grant on offer.
    """
    from altus.cloud.kube import classify

    entry = tool(name)
    assert entry.subresource, f"{name} must declare its subresource"
    assert classify(entry.verb, "Pod", entry.subresource) is Sensitivity.PRIVILEGED
    assert classify(entry.verb, "Pod", "") is Sensitivity.READ, (
        "which is exactly what it would have been classified as without it"
    )


# ------------------------------------------------------------------ regressions


@pytest.mark.parametrize("form", ["exec", "EXEC", "Exec", "/exec", "/PORTFORWARD"])
async def test_the_privileged_guard_is_case_insensitive_like_classify(tmp_path, form) -> None:  # type: ignore[no-untyped-def]
    """classify() casefolds, so a guard that matched the raw string let `EXEC`
    walk straight past the tool that prompts for nothing."""
    client = FakeClient({"Pod": [pod("web-1")]})
    out = await tool("k8s_get").run(
        {"kind": "Pod", "name": "web-1", "subresource": form}, context_for(client, tmp_path)
    )
    assert out.summary == "wrong tool", f"{form} was not refused"

    raw = await tool("k8s_raw").run(
        {"path": f"/api/v1/namespaces/s/pods/p/{form.lstrip('/')}"},
        context_for(client, tmp_path),
    )
    assert raw.summary == "wrong tool", f"{form} was not refused by k8s_raw"


def test_a_cluster_scoped_operation_has_no_namespace_in_its_target() -> None:
    """Folding it into "default" tested the protection rules against a
    namespace the operation was never touching."""
    from altus.cloud.kube import KubeContext

    context = KubeContext(name="AKS_QAM", namespace="shop")
    assert context.target().scope == "shop", "None still means this context's namespace"
    assert context.target("other").scope == "other"
    assert context.target("").scope == "", "empty means cluster-scoped"


def test_cp_describes_the_tool_it_actually_needs() -> None:
    """It shells base64, not tar. A description that names the wrong binary
    sends the model debugging the wrong thing when a container lacks it."""
    description = tool("k8s_cp").description
    assert "base64" in description
    assert "not tar" in description
