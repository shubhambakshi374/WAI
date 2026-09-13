"""Kubernetes tools and the topology model. No cluster is contacted.

The dynamic client is injected, so everything here drives recorded API
payloads. The graph tests are the load-bearing ones: a wrong edge set means a
misleading picture of someone's production namespace.
"""

from __future__ import annotations

from typing import Any

import pytest

from wai.cloud import k8s as k8s_api
from wai.cloud.base import ProtectionRules
from wai.cloud.k8s import MetricsUnavailable, build_graph, parse_cpu, parse_memory
from wai.core.visuals import Bars, ResourceGraph, Table, VisualGroup
from wai.tools.base import CloudContext, ToolContext
from wai.tools.k8s import k8s_tools
from wai.workspace import Workspace

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

    async def api_groups(self) -> set[str]:
        return self._groups | ({k8s_api.METRICS_API} if self.has_metrics else set())

    async def api_available(self, group_version: str) -> bool:
        return group_version in await self.api_groups()

    async def list_kind(self, api_version, kind, namespace=None, **kw):  # type: ignore[no-untyped-def]
        self.calls.append((api_version, kind))
        return list(self.objects.get(kind, []))

    async def list_many(self, kinds, namespace=None, **kw):  # type: ignore[no-untyped-def]
        return {kind: list(self.objects.get(kind, [])) for _, kind in kinds}

    async def metrics(self, kind, namespace=None):  # type: ignore[no-untyped-def]
        if not self.has_metrics:
            raise MetricsUnavailable
        return self._metrics


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
    """to_text() is what `wai chat --once` and the Phase 3 engine will use."""
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
    from wai.core.types import ToolResultBlock

    out = await tool("k8s_topology").run({}, context_for(FakeClient(), tmp_path))
    block = ToolResultBlock(tool_use_id="x", content=out.content)
    assert "visual" not in ToolResultBlock.model_fields
    assert len(block.content) < 4000, "the summary must stay compact"


def test_scale_sizes_the_bar_without_inventing_a_percentage() -> None:
    """`limit` means a real ceiling and shows a percentage. `scale` is only
    bar length --- a percentage there would be read as a fill level."""
    from wai.core.visuals import Bar

    limited = Bar(label="cpu", value=500, limit=1000, unit="m").render()
    assert "500m / 1,000m" in limited and "50%" in limited

    scaled = Bar(label="vol", value=250, scale=300, unit="Gi").render()
    assert "250Gi" in scaled
    assert "%" not in scaled, "relative size must not be presented as a percentage"
    assert "/" not in scaled.split("Gi")[0]


def test_bar_labels_truncate_in_the_middle() -> None:
    """Elasticsearch PVCs share a long prefix; cutting the tail hides which is which."""
    from wai.core.visuals import Bar

    a = Bar(label="sandbox/elasticsearch-data-esmain-es-esmain-0", value=1).render(label_width=30)
    b = Bar(label="sandbox/elasticsearch-data-esmain-es-esmain-1", value=1).render(label_width=30)
    assert a != b
    assert a.strip().startswith("sandbox/")
    assert "-0" in a and "-1" in b


async def test_storage_does_not_claim_a_fill_level(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A bound PVC has requested == capacity, so charting one against the
    other shows every volume at 100% and reads as 'full'."""
    from wai.core.visuals import Bars

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
