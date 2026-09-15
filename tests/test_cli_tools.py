"""The CLI fallback, and the capability switches that decide what registers.

The load-bearing assertion here is that a command is an argument vector, never
a shell string. A model-generated argument containing shell syntax must arrive
at the binary as one inert literal.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from wai.cloud.base import Sensitivity
from wai.config.models import CloudSettings, K8sSettings
from wai.tools import default_registry
from wai.tools.approval import Decision, RecordingPolicy
from wai.tools.base import CloudContext, ToolContext
from wai.tools.cli import HelmTool, KubectlTool, cli_tools
from wai.workspace import Workspace


def context(tmp_path: Path, policy: Any = None, **cloud: Any) -> ToolContext:
    return ToolContext(
        workspace=Workspace(root=tmp_path),
        approvals=policy or RecordingPolicy(),
        cloud=CloudContext(kube_context="AKS_QAM", **cloud),
    )


def echoed_argv(content: str) -> list[str]:
    """What actually reached execve.

    Scanned for rather than taken by line number: an argument containing a
    newline splits the echoed header, which is itself evidence the argument
    survived whole.
    """
    import json

    for line in content.splitlines():
        if line.startswith("["):
            return list(json.loads(line))
    raise AssertionError(f"no argv in output:\n{content}")


def fake_binary(tmp_path: Path, name: str) -> Path:
    """A stand-in that prints its own argv, so a test can see exactly what
    reached execve rather than trusting that quoting worked."""
    script = tmp_path / name
    script.write_text(f"#!{sys.executable}\nimport sys, json\nprint(json.dumps(sys.argv[1:]))\n")
    script.chmod(0o755)
    return script


@pytest.fixture
def on_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    binaries = tmp_path / "bin"
    binaries.mkdir()
    for name in ("kubectl", "helm", "kustomize"):
        fake_binary(binaries, name)
    monkeypatch.setenv("PATH", str(binaries))
    return binaries


# ------------------------------------------------------------------ injection


@pytest.mark.parametrize(
    "hostile",
    [
        "; rm -rf /",
        "$(whoami)",
        "`id`",
        "web && curl evil.test",
        "a\nb",
        "| tee /tmp/x",
        "'; DROP TABLE --",
    ],
)
async def test_shell_metacharacters_arrive_as_one_inert_argument(
    tmp_path: Path, on_path: Path, hostile: str
) -> None:
    """No sh -c anywhere, so none of this is ever parsed as syntax."""
    out = await KubectlTool().run(
        {"args": ["get", "pods", hostile]}, context(tmp_path, RecordingPolicy(Decision.ALLOW))
    )
    printed = echoed_argv(out.content)
    assert hostile in printed, "the argument must survive verbatim"
    assert printed.count(hostile) == 1, "and must not have been split into more arguments"


async def test_args_must_be_a_list_not_a_shell_line(tmp_path: Path, on_path: Path) -> None:
    out = await KubectlTool().run({"args": "get pods -o yaml"}, context(tmp_path))
    assert out.is_error
    assert "array of strings" in out.content


# ------------------------------------------------------------------ the gate


async def test_a_read_subcommand_does_not_prompt(tmp_path: Path, on_path: Path) -> None:
    policy = RecordingPolicy()
    out = await KubectlTool().run({"args": ["get", "pods"]}, context(tmp_path, policy))
    assert not out.is_error, out.content
    assert policy.seen == []


async def test_an_unknown_subcommand_is_gated_rather_than_waved_through(
    tmp_path: Path, on_path: Path
) -> None:
    """Fail closed: a verb we have never heard of is a change until proven
    otherwise."""
    policy = RecordingPolicy(Decision.ALLOW)
    await KubectlTool().run({"args": ["frobnicate", "web"]}, context(tmp_path, policy))
    assert len(policy.seen) == 1
    assert policy.seen[0].sensitivity is Sensitivity.MUTATE


@pytest.mark.parametrize(
    "argv",
    [
        ["exec", "web", "--", "sh"],
        ["port-forward", "svc/web", "8080"],
        ["drain", "node-7"],
        ["cp", "web:/etc/passwd", "."],
        ["get", "--raw", "/api/v1/secrets"],
    ],
)
async def test_privileged_kubectl_subcommands_demand_a_typed_confirmation(
    tmp_path: Path, on_path: Path, argv: list[str]
) -> None:
    policy = RecordingPolicy(Decision.ALLOW)
    await KubectlTool().run({"args": argv}, context(tmp_path, policy))
    request = policy.seen[0]
    assert request.sensitivity is Sensitivity.PRIVILEGED
    assert request.needs_challenge
    assert not request.may_grant_always, "no standing grant on kubectl exec"


async def test_rejection_runs_nothing(tmp_path: Path, on_path: Path) -> None:
    out = await KubectlTool().run(
        {"args": ["delete", "deployment", "web"]},
        context(tmp_path, RecordingPolicy(Decision.DENY)),
    )
    assert out.is_error and out.denied


async def test_the_prompt_carries_the_reason_or_says_there_was_none(
    tmp_path: Path, on_path: Path
) -> None:
    """The native tools are better in four separate ways, so choosing the CLI
    over them is a decision the user should see justified."""
    policy = RecordingPolicy(Decision.ALLOW)
    await KubectlTool().run({"args": ["apply", "-f", "x.yaml"]}, context(tmp_path, policy))
    assert "no reason given" in policy.seen[0].diff

    policy = RecordingPolicy(Decision.ALLOW)
    await KubectlTool().run(
        {"args": ["apply", "-k", "overlays/prod"], "reason": "kustomize overlay"},
        context(tmp_path, policy),
    )
    assert "kustomize overlay" in policy.seen[0].diff


# ------------------------------------------------------------- reserved flags


@pytest.mark.parametrize(
    "flag",
    ["--context", "--kubeconfig", "--context=other", "--as", "--token"],
)
async def test_the_model_cannot_retarget_the_command(
    tmp_path: Path, on_path: Path, flag: str
) -> None:
    """WAI sets these from session state. A model-supplied one would send the
    command somewhere the approval prompt did not name."""
    policy = RecordingPolicy(Decision.ALLOW)
    out = await KubectlTool().run({"args": ["get", "pods", flag]}, context(tmp_path, policy))
    assert out.is_error
    assert "set by WAI" in out.content
    assert policy.seen == [], "refused before anyone is asked"


async def test_the_session_context_is_appended(tmp_path: Path, on_path: Path) -> None:
    out = await KubectlTool().run({"args": ["get", "pods"]}, context(tmp_path))
    printed = echoed_argv(out.content)
    assert printed[-2:] == ["--context", "AKS_QAM"]


async def test_a_missing_binary_says_so(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    out = await HelmTool().run({"args": ["list"]}, context(tmp_path))
    assert out.is_error
    assert "not installed" in out.content


# --------------------------------------------------------- capability switches


def registry_names(**cloud: Any) -> set[str]:
    settings = CloudSettings(**cloud)
    return set(default_registry(kubernetes=True, cloud=settings).names)


def test_everything_is_on_by_default() -> None:
    names = registry_names()
    for expected in ("k8s_exec", "k8s_cp", "k8s_port_forward", "k8s_drain", "k8s_node"):
        assert expected in names


@pytest.mark.parametrize(
    ("switch", "removed"),
    [
        ("allow_exec", {"k8s_exec", "k8s_cp"}),
        ("allow_port_forward", {"k8s_port_forward"}),
        ("allow_node_lifecycle", {"k8s_node", "k8s_drain"}),
    ],
)
def test_a_switch_removes_exactly_its_tools(switch: str, removed: set[str]) -> None:
    """Not registered, rather than registered and refused: a tool the model
    cannot see costs no context and cannot be argued into being used."""
    full = registry_names()
    reduced = registry_names(k8s=K8sSettings(**{switch: False}))
    assert full - reduced == removed
    assert reduced - full == set()


def test_cli_tools_need_both_switches() -> None:
    assert "k8s_kubectl" not in registry_names(cli_fallback=False)
    assert "k8s_kubectl" not in registry_names(k8s=K8sSettings(allow_cli=False))
    assert "k8s_kubectl" in registry_names(cli_fallback=True)


def test_the_allowlist_decides_which_binaries_register() -> None:
    names = registry_names(cli_fallback=True, cli_allowlist=["kubectl"])
    assert "k8s_kubectl" in names
    assert "helm" not in names and "kustomize" not in names


def test_no_cli_tool_is_read_only() -> None:
    """Whether a given invocation needs approval is decided per call, but the
    tool itself must never be marked safe."""
    assert all(not t.read_only for t in cli_tools())


def test_disabled_classes_are_reportable() -> None:
    from wai.tools.k8s import disabled_classes

    assert disabled_classes(K8sSettings()) == []
    off = disabled_classes(K8sSettings(allow_exec=False, allow_cli=False))
    assert len(off) == 2
    assert any("exec" in text for text in off)


# --------------------------------------------------------- RBAC writes switch


async def test_rbac_writes_off_refuses_at_the_gate(tmp_path: Path) -> None:
    """This one cannot be enforced by withholding a tool: the same k8s_apply
    writes a ConfigMap and a ClusterRoleBinding."""
    from wai.tools.k8s import k8s_tools

    apply_tool = next(t for t in k8s_tools() if t.name == "k8s_apply")
    policy = RecordingPolicy(Decision.ALLOW)
    ctx = ToolContext(
        workspace=Workspace(root=tmp_path),
        approvals=policy,
        cloud=CloudContext(allow_rbac_writes=False),
    )
    out = await apply_tool.confirm(
        ctx,
        client=None,
        context_name="AKS_QAM",
        kind="ClusterRoleBinding",
        name="admin",
        namespace="",
        diff="",
        dry_run="",
    )
    assert out is not None and out.is_error
    assert "allow_rbac_writes" in out.content
    assert policy.seen == [], "refused before anyone is asked"


async def test_rbac_writes_off_leaves_ordinary_objects_alone(tmp_path: Path) -> None:
    from wai.tools.k8s import k8s_tools

    apply_tool = next(t for t in k8s_tools() if t.name == "k8s_apply")
    ctx = ToolContext(
        workspace=Workspace(root=tmp_path),
        approvals=RecordingPolicy(Decision.ALLOW),
        cloud=CloudContext(allow_rbac_writes=False),
    )
    out = await apply_tool.confirm(
        ctx,
        client=None,
        context_name="AKS_QAM",
        kind="ConfigMap",
        name="app-config",
        namespace="shop",
        diff="",
        dry_run="",
    )
    assert out is None, "a ConfigMap is not an RBAC write"


async def test_minting_a_token_counts_as_an_rbac_write(tmp_path: Path) -> None:
    from wai.tools.k8s import k8s_tools

    create = next(t for t in k8s_tools() if t.name == "k8s_create")
    ctx = ToolContext(
        workspace=Workspace(root=tmp_path),
        approvals=RecordingPolicy(Decision.ALLOW),
        cloud=CloudContext(allow_rbac_writes=False),
    )
    out = await create.confirm(
        ctx,
        client=None,
        context_name="AKS_QAM",
        kind="ServiceAccount",
        name="builder",
        namespace="shop",
        diff="",
        dry_run="",
        subresource="token",
    )
    assert out is not None and out.is_error
