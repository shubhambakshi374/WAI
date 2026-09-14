"""Write, edit and delete: the approval gate and the mutations behind it.

The load-bearing assertion in most of these is that *nothing changed on disk*
when approval was refused. A tool that writes first and asks later is worse
than no gate at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from wai.cloud.base import Sensitivity
from wai.tools import ToolContext, default_registry
from wai.tools.approval import (
    AllowAll,
    ApprovalRequest,
    Decision,
    DenyAll,
    RecordingPolicy,
    SessionApprovals,
)
from wai.workspace import Workspace


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / ".git").mkdir()
    (root / "src" / "main.py").write_text("def hello():\n    return 'world'\n")
    (root / "README.md").write_text("# Repo\n")
    return root


@pytest.fixture
def registry():  # type: ignore[no-untyped-def]
    return default_registry(kubernetes=False)


def context(tree: Path, policy: object = None) -> ToolContext:
    return ToolContext(
        workspace=Workspace(root=tree),
        approvals=policy or AllowAll(),  # type: ignore[arg-type]
    )


async def run(registry, name, args, ctx):  # type: ignore[no-untyped-def]
    return await registry.execute(name, args, ctx)


# ------------------------------------------------------------------ the gate


async def test_default_policy_refuses_every_write(registry, tree) -> None:  # type: ignore[no-untyped-def]
    """A caller that forgets to wire a policy must not be able to write."""
    ctx = ToolContext(workspace=Workspace(root=tree))
    assert isinstance(ctx.approvals, DenyAll)

    out = await run(registry, "write_file", {"path": "new.txt", "content": "x"}, ctx)
    assert out.is_error and out.denied
    assert not (tree / "new.txt").exists()


@pytest.mark.parametrize(
    ("tool", "args"),
    [
        ("write_file", {"path": "src/main.py", "content": "wiped"}),
        ("edit_file", {"path": "src/main.py", "old_string": "world", "new_string": "moon"}),
        ("delete_path", {"path": "src/main.py"}),
    ],
)
async def test_rejection_leaves_the_file_untouched(registry, tree, tool, args) -> None:  # type: ignore[no-untyped-def]
    before = (tree / "src" / "main.py").read_text()
    ctx = context(tree, RecordingPolicy(decision=Decision.DENY))

    out = await run(registry, tool, args, ctx)
    assert out.is_error and out.denied
    assert out.summary == "rejected"
    assert (tree / "src" / "main.py").read_text() == before


@pytest.mark.parametrize(
    ("tool", "args", "action"),
    [
        ("write_file", {"path": "new.txt", "content": "hello\n"}, "create"),
        ("write_file", {"path": "README.md", "content": "# Changed\n"}, "overwrite"),
        ("edit_file", {"path": "README.md", "old_string": "Repo", "new_string": "Project"}, "edit"),
        ("delete_path", {"path": "README.md"}, "delete"),
    ],
)
async def test_every_mutation_asks_first(registry, tree, tool, args, action) -> None:  # type: ignore[no-untyped-def]
    policy = RecordingPolicy()
    await run(registry, tool, args, context(tree, policy))
    assert len(policy.seen) == 1
    assert policy.seen[0].tool == tool
    assert policy.seen[0].action == action


async def test_approval_carries_a_real_diff(registry, tree) -> None:  # type: ignore[no-untyped-def]
    """Approving a change you cannot see is not consent."""
    policy = RecordingPolicy()
    await run(
        registry,
        "edit_file",
        {"path": "src/main.py", "old_string": "'world'", "new_string": "'moon'"},
        context(tree, policy),
    )
    diff = policy.seen[0].diff
    assert "-    return 'world'" in diff
    assert "+    return 'moon'" in diff


async def test_read_only_tools_never_ask(registry, tree) -> None:  # type: ignore[no-untyped-def]
    policy = RecordingPolicy()
    ctx = context(tree, policy)
    for name, args in [
        ("read_file", {"path": "README.md"}),
        ("list_dir", {}),
        ("glob", {"pattern": "**/*.py"}),
        ("grep", {"pattern": "hello"}),
    ]:
        await run(registry, name, args, ctx)
    assert policy.seen == []


# ------------------------------------------------------- session-scoped always


async def test_always_allow_is_scoped_to_one_tool(registry, tree) -> None:  # type: ignore[no-untyped-def]
    inner = RecordingPolicy(decision=Decision.ALLOW_ALWAYS)
    approvals = SessionApprovals(inner)
    ctx = context(tree, approvals)

    await run(registry, "write_file", {"path": "a.txt", "content": "1"}, ctx)
    await run(registry, "write_file", {"path": "b.txt", "content": "2"}, ctx)
    assert len(inner.seen) == 1, "the second write should not re-prompt"
    assert approvals.always_allowed == frozenset({"write_file"})

    # A different tool still has to ask.
    await run(registry, "delete_path", {"path": "a.txt"}, ctx)
    assert len(inner.seen) == 2
    assert inner.seen[1].tool == "delete_path"


async def test_always_allow_can_be_revoked(registry, tree) -> None:  # type: ignore[no-untyped-def]
    inner = RecordingPolicy(decision=Decision.ALLOW_ALWAYS)
    approvals = SessionApprovals(inner)
    ctx = context(tree, approvals)

    await run(registry, "write_file", {"path": "a.txt", "content": "1"}, ctx)
    approvals.revoke_all()
    await run(registry, "write_file", {"path": "b.txt", "content": "2"}, ctx)
    assert len(inner.seen) == 2


async def test_privileged_requests_never_earn_a_standing_grant() -> None:
    """A standing `allow always` on exec is indistinguishable from no gate.

    Enforced in ``SessionApprovals`` rather than the modal, so a caller that
    never draws a modal --- the CLI, the Phase 3 engine --- cannot route around
    it by answering ALLOW_ALWAYS itself.
    """
    inner = RecordingPolicy(decision=Decision.ALLOW_ALWAYS)
    approvals = SessionApprovals(inner)
    req = ApprovalRequest(
        tool="k8s_exec", action="exec", path="pod/web", sensitivity=Sensitivity.PRIVILEGED
    )

    assert await approvals.request(req) is Decision.ALLOW, "this one call is still allowed"
    assert approvals.always_allowed == frozenset(), "but nothing was recorded"
    assert await approvals.request(req) is Decision.ALLOW
    assert len(inner.seen) == 2, "every privileged call must ask again"


async def test_a_standing_grant_does_not_carry_into_a_privileged_call() -> None:
    """Sensitivity is per call, but the grant is keyed by tool name.

    ``k8s_patch`` allowed-always for a label edit must not silently cover the
    same tool creating an eviction.
    """
    inner = RecordingPolicy(decision=Decision.ALLOW_ALWAYS)
    approvals = SessionApprovals(inner)

    ordinary = ApprovalRequest(tool="k8s_patch", action="patch", path="deployment/web")
    await approvals.request(ordinary)
    assert approvals.always_allowed == frozenset({"k8s_patch"})

    await approvals.request(ordinary)
    assert len(inner.seen) == 1, "the ordinary repeat is covered by the grant"

    escalated = ApprovalRequest(
        tool="k8s_patch",
        action="patch",
        path="clusterrolebinding/admin",
        sensitivity=Sensitivity.PRIVILEGED,
    )
    await approvals.request(escalated)
    assert len(inner.seen) == 2, "the privileged call must ask despite the grant"


async def test_protected_targets_also_refuse_a_standing_grant() -> None:
    inner = RecordingPolicy(decision=Decision.ALLOW_ALWAYS)
    approvals = SessionApprovals(inner)
    req = ApprovalRequest(tool="k8s_apply", action="apply", path="deployment/web", protected=True)

    await approvals.request(req)
    assert approvals.always_allowed == frozenset()
    await approvals.request(req)
    assert len(inner.seen) == 2


async def test_session_approvals_serialize_prompts() -> None:
    """Concurrent requests must queue, or two modals race to the screen."""
    import asyncio

    concurrent = 0
    peak = 0

    class Slow:
        async def request(self, req: ApprovalRequest) -> Decision:
            nonlocal concurrent, peak
            concurrent += 1
            peak = max(peak, concurrent)
            await asyncio.sleep(0.01)
            concurrent -= 1
            return Decision.ALLOW

    approvals = SessionApprovals(Slow())
    await asyncio.gather(
        *(
            approvals.request(ApprovalRequest(tool=f"t{i}", action="edit", path="p"))
            for i in range(5)
        )
    )
    assert peak == 1


# ------------------------------------------------------------------ write_file


async def test_write_creates_a_new_file(registry, tree) -> None:  # type: ignore[no-untyped-def]
    out = await run(
        registry, "write_file", {"path": "docs/new.md", "content": "# Hi\n"}, context(tree)
    )
    assert not out.is_error
    assert (tree / "docs" / "new.md").read_text() == "# Hi\n", "parents are created"


async def test_write_overwrites_and_preserves_nothing_else(registry, tree) -> None:  # type: ignore[no-untyped-def]
    out = await run(
        registry, "write_file", {"path": "README.md", "content": "# New\n"}, context(tree)
    )
    assert not out.is_error
    assert (tree / "README.md").read_text() == "# New\n"


async def test_write_identical_content_is_a_no_op(registry, tree) -> None:  # type: ignore[no-untyped-def]
    policy = RecordingPolicy()
    out = await run(
        registry, "write_file", {"path": "README.md", "content": "# Repo\n"}, context(tree, policy)
    )
    assert not out.is_error and out.summary == "no change"
    assert policy.seen == [], "an identical write should not bother the user"


async def test_write_refuses_to_clobber_a_binary(registry, tree) -> None:  # type: ignore[no-untyped-def]
    (tree / "blob.bin").write_bytes(b"\x00\x01\x02")
    out = await run(registry, "write_file", {"path": "blob.bin", "content": "x"}, context(tree))
    assert out.is_error and out.summary == "binary"
    assert (tree / "blob.bin").read_bytes() == b"\x00\x01\x02"


# ------------------------------------------------------------------- edit_file


async def test_edit_replaces_a_unique_string(registry, tree) -> None:  # type: ignore[no-untyped-def]
    out = await run(
        registry,
        "edit_file",
        {"path": "src/main.py", "old_string": "'world'", "new_string": "'moon'"},
        context(tree),
    )
    assert not out.is_error
    assert (tree / "src" / "main.py").read_text() == "def hello():\n    return 'moon'\n"


async def test_edit_refuses_an_ambiguous_match(registry, tree) -> None:  # type: ignore[no-untyped-def]
    (tree / "dup.py").write_text("x = 1\nx = 1\n")
    policy = RecordingPolicy()
    out = await run(
        registry,
        "edit_file",
        {"path": "dup.py", "old_string": "x = 1", "new_string": "x = 2"},
        context(tree, policy),
    )
    assert out.is_error and "appears 2 times" in out.content
    assert policy.seen == [], "an unresolvable edit must not reach the user"
    assert (tree / "dup.py").read_text() == "x = 1\nx = 1\n"


async def test_edit_replace_all(registry, tree) -> None:  # type: ignore[no-untyped-def]
    (tree / "dup.py").write_text("x = 1\nx = 1\n")
    out = await run(
        registry,
        "edit_file",
        {"path": "dup.py", "old_string": "x = 1", "new_string": "x = 2", "replace_all": True},
        context(tree),
    )
    assert not out.is_error and "2 replacements" in out.content
    assert (tree / "dup.py").read_text() == "x = 2\nx = 2\n"


async def test_edit_reports_a_missing_match_clearly(registry, tree) -> None:  # type: ignore[no-untyped-def]
    out = await run(
        registry,
        "edit_file",
        {"path": "src/main.py", "old_string": "nonexistent", "new_string": "x"},
        context(tree),
    )
    assert out.is_error and "not found" in out.content
    assert "whitespace" in out.content, "tell the model why exact matching failed"


async def test_edit_rejects_empty_and_identical_strings(registry, tree) -> None:  # type: ignore[no-untyped-def]
    ctx = context(tree)
    empty = await run(
        registry, "edit_file", {"path": "README.md", "old_string": "", "new_string": "x"}, ctx
    )
    assert empty.is_error and "must not be empty" in empty.content

    same = await run(
        registry, "edit_file", {"path": "README.md", "old_string": "a", "new_string": "a"}, ctx
    )
    assert same.is_error and "identical" in same.content


async def test_edit_on_a_missing_file(registry, tree) -> None:  # type: ignore[no-untyped-def]
    out = await run(
        registry,
        "edit_file",
        {"path": "nope.py", "old_string": "a", "new_string": "b"},
        context(tree),
    )
    assert out.is_error and "no such file" in out.content


# ----------------------------------------------------------------- delete_path


async def test_delete_removes_a_file(registry, tree) -> None:  # type: ignore[no-untyped-def]
    out = await run(registry, "delete_path", {"path": "README.md"}, context(tree))
    assert not out.is_error
    assert not (tree / "README.md").exists()


async def test_delete_refuses_a_non_empty_directory_without_recursive(registry, tree) -> None:  # type: ignore[no-untyped-def]
    policy = RecordingPolicy()
    out = await run(registry, "delete_path", {"path": "src"}, context(tree, policy))
    assert out.is_error and "recursive=true" in out.content
    assert policy.seen == []
    assert (tree / "src" / "main.py").exists()


async def test_delete_recursive_removes_the_tree(registry, tree) -> None:  # type: ignore[no-untyped-def]
    out = await run(registry, "delete_path", {"path": "src", "recursive": True}, context(tree))
    assert not out.is_error
    assert not (tree / "src").exists()


async def test_delete_refuses_the_workspace_root(registry, tree) -> None:  # type: ignore[no-untyped-def]
    out = await run(registry, "delete_path", {"path": "."}, context(tree))
    assert out.is_error and "workspace root" in out.content
    assert tree.exists()


async def test_delete_missing_path(registry, tree) -> None:  # type: ignore[no-untyped-def]
    out = await run(registry, "delete_path", {"path": "nope.txt"}, context(tree))
    assert out.is_error and "no such path" in out.content


# ------------------------------------------------------- boundaries still hold


@pytest.mark.parametrize(
    ("tool", "args"),
    [
        ("write_file", {"path": "/etc/evil", "content": "x"}),
        ("write_file", {"path": "../escape.txt", "content": "x"}),
        ("edit_file", {"path": "/etc/hosts", "old_string": "a", "new_string": "b"}),
        ("delete_path", {"path": "/etc/hosts"}),
    ],
)
async def test_writes_cannot_escape_the_workspace(registry, tree, tool, args) -> None:  # type: ignore[no-untyped-def]
    policy = RecordingPolicy()
    out = await run(registry, tool, args, context(tree, policy))
    assert out.is_error and out.summary == "denied"
    assert policy.seen == [], "containment is checked before the user is asked"


@pytest.mark.parametrize("name", [".env", "id_rsa", "server.pem", ".npmrc"])
async def test_writes_cannot_create_secret_shaped_files(registry, tree, name) -> None:  # type: ignore[no-untyped-def]
    out = await run(registry, "write_file", {"path": name, "content": "x"}, context(tree))
    assert out.is_error and out.summary == "denied"
    assert not (tree / name).exists()


@pytest.mark.parametrize(
    ("tool", "args"),
    [
        ("write_file", {"path": ".git/config", "content": "x"}),
        ("delete_path", {"path": ".git", "recursive": True}),
        ("edit_file", {"path": ".git/config", "old_string": "a", "new_string": "b"}),
    ],
)
async def test_git_directory_is_never_writable(registry, tree, tool, args) -> None:  # type: ignore[no-untyped-def]
    out = await run(registry, tool, args, context(tree))
    assert out.is_error and out.summary == "denied"
    assert (tree / ".git").exists()


async def test_atomic_write_leaves_no_temp_files(registry, tree) -> None:  # type: ignore[no-untyped-def]
    await run(registry, "write_file", {"path": "a.txt", "content": "x"}, context(tree))
    leftovers = [p.name for p in tree.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == []


# ------------------------------------------------------- git recoverability


async def _init_repo(path: Path) -> None:
    import asyncio

    for args in (
        ("init", "-q"),
        ("config", "user.email", "t@example.com"),
        ("config", "user.name", "t"),
    ):
        proc = await asyncio.create_subprocess_exec(
            "git", *args, cwd=str(path), stdout=asyncio.subprocess.DEVNULL
        )
        await proc.communicate()


async def _commit_all(path: Path) -> None:
    import asyncio

    for args in (("add", "-A"), ("commit", "-qm", "x")):
        proc = await asyncio.create_subprocess_exec(
            "git", *args, cwd=str(path), stdout=asyncio.subprocess.DEVNULL
        )
        await proc.communicate()


async def test_recoverability_reports_committed_files(tmp_path: Path) -> None:
    from wai.tools._git import MODIFIED, RECOVERABLE, UNTRACKED, recoverability

    repo = tmp_path / "r"
    repo.mkdir()
    await _init_repo(repo)
    (repo / "kept.txt").write_text("v1\n")
    await _commit_all(repo)

    assert await recoverability(repo / "kept.txt") == RECOVERABLE

    (repo / "kept.txt").write_text("v2\n")
    assert await recoverability(repo / "kept.txt") == MODIFIED

    (repo / "fresh.txt").write_text("new\n")
    assert await recoverability(repo / "fresh.txt") == UNTRACKED


async def test_recoverability_outside_a_repo(tmp_path: Path) -> None:
    from wai.tools._git import NO_REPO, recoverability

    loose = tmp_path / "loose"
    loose.mkdir()
    (loose / "f.txt").write_text("x\n")
    assert await recoverability(loose / "f.txt") == NO_REPO


async def test_delete_approval_states_recoverability(registry, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    """The one fact that makes a delete decision informed."""
    from wai.tools._git import UNTRACKED

    repo = tmp_path / "r"
    repo.mkdir()
    await _init_repo(repo)
    (repo / "doomed.txt").write_text("bye\n")

    policy = RecordingPolicy(decision=Decision.DENY)
    ctx = ToolContext(workspace=Workspace(root=repo), approvals=policy)
    await run(registry, "delete_path", {"path": "doomed.txt"}, ctx)

    assert policy.seen[0].recoverability == UNTRACKED
    assert policy.seen[0].destructive is True
