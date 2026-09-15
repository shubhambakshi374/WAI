"""The Phase 2 guard.

``wai.core``, ``wai.providers``, ``wai.config`` and ``wai.storage`` must stay
importable without Textual, because the Phase 2 workflow engine drives them
headlessly. If this test fails, do not delete it — move the offending code
into ``wai.tui`` instead.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import wai

HEADLESS_PACKAGES = ("core", "providers", "config", "storage", "tools", "cloud", "render")
SRC = Path(wai.__file__).parent


def _modules(package: str) -> list[Path]:
    return sorted((SRC / package).rglob("*.py"))


def _imported_roots(tree: ast.AST) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.parametrize("package", HEADLESS_PACKAGES)
def test_headless_packages_do_not_import_textual(package: str) -> None:
    offenders = [
        path.relative_to(SRC).as_posix()
        for path in _modules(package)
        if "textual" in _imported_roots(ast.parse(path.read_text(encoding="utf-8")))
    ]
    assert not offenders, f"textual imported by headless module(s): {offenders}"


@pytest.mark.parametrize("package", HEADLESS_PACKAGES)
def test_headless_packages_do_not_import_tui(package: str) -> None:
    for path in _modules(package):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("wai.tui"):
                pytest.fail(f"{path.relative_to(SRC)} imports {node.module}")


HEADLESS_MODULES = ("runner.py", "workspace.py", "agent.py")


@pytest.mark.parametrize("module", HEADLESS_MODULES)
def test_headless_modules_do_not_import_textual(module: str) -> None:
    path = SRC / module
    if not path.exists():
        pytest.skip(f"{module} not built yet")
    assert "textual" not in _imported_roots(ast.parse(path.read_text(encoding="utf-8")))
