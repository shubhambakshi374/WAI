"""Containment and denylist tests.

This is the security boundary for every filesystem tool, so these are the
tests that matter most in this increment. Anything that gets past
``Workspace.resolve`` is readable by a model that then ships it to an external
API.
"""

from __future__ import annotations

import os
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from altus.core.errors import PathNotAllowed
from altus.workspace import Workspace, default_workspace


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "src" / "main.py").write_text("print('hi')\n")
    (tmp_path / "outside").mkdir()
    (tmp_path / "outside" / "secret.txt").write_text("nope\n")
    return Workspace(root=root)


# ------------------------------------------------------------------ containment


def test_relative_path_inside_root_is_allowed(ws: Workspace) -> None:
    assert ws.resolve("src/main.py") == ws.root / "src" / "main.py"


def test_absolute_path_inside_root_is_allowed(ws: Workspace) -> None:
    assert ws.resolve(str(ws.root / "src")) == ws.root / "src"


def test_root_itself_is_allowed(ws: Workspace) -> None:
    assert ws.resolve(".") == ws.root


def test_dotdot_traversal_is_rejected(ws: Workspace) -> None:
    with pytest.raises(PathNotAllowed) as exc:
        ws.resolve("../outside/secret.txt")
    assert exc.value.reason == "outside_workspace"


def test_deep_dotdot_traversal_is_rejected(ws: Workspace) -> None:
    with pytest.raises(PathNotAllowed):
        ws.resolve("src/../../../../../../etc/passwd")


def test_absolute_path_outside_root_is_rejected(ws: Workspace) -> None:
    with pytest.raises(PathNotAllowed):
        ws.resolve("/etc/passwd")


def test_symlink_escaping_the_root_is_rejected(ws: Workspace, tmp_path: Path) -> None:
    """The reason resolve() must follow symlinks BEFORE checking containment.

    The link itself lives inside the root, so a naive prefix check on the
    unresolved path would let this through.
    """
    link = ws.root / "escape"
    link.symlink_to(tmp_path / "outside")
    assert link.is_symlink()

    with pytest.raises(PathNotAllowed) as exc:
        ws.resolve("escape/secret.txt")
    assert exc.value.reason == "outside_workspace"


def test_symlink_staying_inside_the_root_is_allowed(ws: Workspace) -> None:
    link = ws.root / "alias"
    link.symlink_to(ws.root / "src")
    assert ws.resolve("alias/main.py") == ws.root / "src" / "main.py"


def test_sibling_directory_sharing_a_name_prefix_is_rejected(tmp_path: Path) -> None:
    """ "/work" must not match "/workspace-other" --- no string-prefix checks."""
    (tmp_path / "work").mkdir()
    (tmp_path / "work-other").mkdir()
    (tmp_path / "work-other" / "f.txt").write_text("x")

    ws = Workspace(root=tmp_path / "work")
    with pytest.raises(PathNotAllowed):
        ws.resolve(str(tmp_path / "work-other" / "f.txt"))


def test_nonexistent_path_inside_root_still_resolves(ws: Workspace) -> None:
    """Tools need to report 'no such file', not 'outside workspace'."""
    assert ws.resolve("src/does_not_exist.py").name == "does_not_exist.py"


def test_nonexistent_path_outside_root_is_still_rejected(ws: Workspace) -> None:
    with pytest.raises(PathNotAllowed):
        ws.resolve("../outside/does_not_exist.txt")


def test_is_allowed_mirrors_resolve(ws: Workspace) -> None:
    assert ws.is_allowed("src/main.py") is True
    assert ws.is_allowed("/etc/passwd") is False


# ----------------------------------------------------------------- extra roots


def test_extra_root_is_allowed(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    extra = tmp_path / "etc-nginx"
    root.mkdir()
    extra.mkdir()
    (extra / "nginx.conf").write_text("server {}\n")

    ws = Workspace(root=root, extra_roots=(extra,))
    assert ws.resolve(str(extra / "nginx.conf")) == extra / "nginx.conf"


def test_path_outside_both_root_and_extras_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    extra = tmp_path / "extra"
    other = tmp_path / "other"
    for d in (root, extra, other):
        d.mkdir()

    ws = Workspace(root=root, extra_roots=(extra,))
    with pytest.raises(PathNotAllowed):
        ws.resolve(str(other / "f.txt"))


def test_roots_are_canonicalized_at_construction(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    ws = Workspace(root=tmp_path / "repo" / "src" / "..")
    assert ws.root == root.resolve()


def test_tilde_is_expanded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    ws = Workspace(root="~")
    assert ws.root == tmp_path.resolve()


# -------------------------------------------------------------------- denylist


@pytest.mark.parametrize(
    "name",
    [
        ".env",
        ".env.local",
        ".env.production",
        "server.pem",
        "private.key",
        "cert.p12",
        "store.jks",
        "id_rsa",
        "id_rsa.pub",
        "id_ed25519",
        ".netrc",
        ".npmrc",
        ".pypirc",
        ".pgpass",
        "credentials.json",
    ],
)
def test_secret_files_are_denied_inside_the_root(ws: Workspace, name: str) -> None:
    (ws.root / name).write_text("SECRET=1\n")
    with pytest.raises(PathNotAllowed) as exc:
        ws.resolve(name)
    assert exc.value.reason == "denylisted"


def test_secret_denial_names_the_rule_and_the_reason(ws: Workspace) -> None:
    """The model must see a clear refusal, not a silent skip, or it retries."""
    (ws.root / ".env").write_text("KEY=1\n")
    with pytest.raises(PathNotAllowed) as exc:
        ws.resolve(".env")
    message = str(exc.value)
    assert ".env" in message
    assert "external model providers" in message


def test_ssh_directory_is_denied_even_as_an_explicit_extra_root(tmp_path: Path) -> None:
    """Opting in deliberately does not unlock private keys."""
    root = tmp_path / "repo"
    ssh = tmp_path / ".ssh"
    root.mkdir()
    ssh.mkdir()
    (ssh / "known_hosts").write_text("host\n")

    ws = Workspace(root=root, extra_roots=(ssh,))
    with pytest.raises(PathNotAllowed) as exc:
        ws.resolve(str(ssh / "known_hosts"))
    assert exc.value.reason == "denylisted"


def test_aws_credentials_are_denied_by_relative_shape(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / ".aws").mkdir(parents=True)
    (root / ".aws" / "credentials").write_text("[default]\n")
    (root / ".aws" / "config").write_text("[default]\n")

    ws = Workspace(root=root)
    with pytest.raises(PathNotAllowed):
        ws.resolve(".aws/credentials")
    # config holds no keys, so it stays readable.
    assert ws.resolve(".aws/config").name == "config"


def test_ordinary_files_are_not_denied(ws: Workspace) -> None:
    for name in ("main.py", "README.md", "environment.yml", "keyboard.ts"):
        (ws.root / name).write_text("x\n")
        assert ws.is_allowed(name), f"{name} should not match the denylist"


def test_denylist_can_be_disabled_explicitly(ws: Workspace) -> None:
    (ws.root / ".env").write_text("KEY=1\n")
    open_ws = Workspace(root=ws.root, deny_secrets=False)
    assert open_ws.resolve(".env").name == ".env"


def test_containment_still_applies_when_denylist_is_off(ws: Workspace) -> None:
    open_ws = Workspace(root=ws.root, deny_secrets=False)
    with pytest.raises(PathNotAllowed):
        open_ws.resolve("/etc/passwd")


# --------------------------------------------------------------------- display


def test_relative_shortens_paths_under_the_root(ws: Workspace) -> None:
    assert ws.relative(ws.root / "src" / "main.py") == "src/main.py"
    assert ws.relative(ws.root) == "."


def test_relative_leaves_outside_paths_absolute(ws: Workspace) -> None:
    assert ws.relative("/etc/hosts") == "/etc/hosts"


# ---------------------------------------------------------------------- shared


def test_workspaces_carry_an_id_for_flows_to_reference(ws: Workspace) -> None:
    other = Workspace(root=ws.root)
    assert ws.id.startswith("ws_")
    assert ws.id != other.id


def test_default_workspace_uses_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert default_workspace().root == tmp_path.resolve()


def test_workspace_is_hashable_and_shareable(ws: Workspace) -> None:
    """Flows pass one instance between steps, so it must be a value object."""
    assert hash(ws) == hash(ws)
    assert {ws, ws} == {ws}
    with pytest.raises(FrozenInstanceError):
        ws.root = Path(os.sep)  # type: ignore[misc]
