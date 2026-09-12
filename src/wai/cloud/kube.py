"""Kubeconfig discovery and context selection.

**Nothing here writes to the user's kubeconfig.** Listing contexts reads the
file; selecting one records the choice in WAI's own state. Changing the global
context as a side effect of a chat message would silently retarget every other
terminal the user has open, which is not a surprise worth risking.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wai.cloud.base import CloudTarget, Sensitivity

DEFAULT_KUBECONFIG = Path.home() / ".kube" / "config"

#: Verbs the Kubernetes API itself defines. Classification is exact here ---
#: unlike AWS there is no guessing from a name.
READ_VERBS = frozenset({"get", "list", "watch"})
MUTATE_VERBS = frozenset(
    {"create", "update", "patch", "delete", "deletecollection", "apply", "scale", "rollout"}
)

#: Kinds whose payload is credential material whatever the verb.
SECRET_KINDS = frozenset({"Secret"})


def classify(verb: str, kind: str = "") -> Sensitivity:
    verb = verb.casefold()
    if verb in MUTATE_VERBS:
        return Sensitivity.MUTATE
    if verb in READ_VERBS:
        return Sensitivity.SENSITIVE_READ if kind in SECRET_KINDS else Sensitivity.READ
    return Sensitivity.MUTATE  # unknown verbs fail closed


@dataclass(frozen=True)
class KubeContext:
    name: str
    cluster: str = ""
    user: str = ""
    namespace: str = "default"
    source: str = ""
    """Which kubeconfig file it came from."""

    def target(self, namespace: str | None = None) -> CloudTarget:
        return CloudTarget(
            cloud="k8s",
            context=self.name,
            location=self.cluster,
            scope=namespace or self.namespace,
        )


def kubeconfig_paths(extra: tuple[str, ...] = ()) -> list[Path]:
    """Every kubeconfig to consider: KUBECONFIG, the default, plus registered ones."""
    paths: list[Path] = []
    env = os.environ.get("KUBECONFIG")
    if env:
        paths.extend(Path(p).expanduser() for p in env.split(os.pathsep) if p)
    elif DEFAULT_KUBECONFIG.exists():
        paths.append(DEFAULT_KUBECONFIG)
    paths.extend(Path(p).expanduser() for p in extra)
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        resolved = path.expanduser()
        if resolved not in seen and resolved.exists():
            seen.add(resolved)
            unique.append(resolved)
    return unique


def list_contexts(extra: tuple[str, ...] = ()) -> tuple[list[KubeContext], str | None]:
    """Every context across every kubeconfig, plus the file's own active one.

    Read-only: opens the files and closes them.
    """
    from kubernetes import config as kube_config

    found: list[KubeContext] = []
    active: str | None = None
    for path in kubeconfig_paths(extra):
        try:
            contexts, current = kube_config.list_kube_config_contexts(config_file=str(path))
        except Exception:  # unreadable or malformed; skip rather than fail the command
            continue
        if active is None and current:
            active = current.get("name")
        for entry in contexts or []:
            detail = entry.get("context", {})
            found.append(
                KubeContext(
                    name=entry.get("name", ""),
                    cluster=detail.get("cluster", ""),
                    user=detail.get("user", ""),
                    namespace=detail.get("namespace") or "default",
                    source=str(path),
                )
            )
    return found, active


def resolve_context(name: str | None, extra: tuple[str, ...] = ()) -> KubeContext | None:
    """The context WAI should use: the selected one, else the kubeconfig's own."""
    contexts, active = list_contexts(extra)
    if not contexts:
        return None
    wanted = name or active
    return next((c for c in contexts if c.name == wanted), None) or (
        contexts[0] if name is None else None
    )


def build_client(context: str | None, extra: tuple[str, ...] = ()) -> Any:
    """A dynamic client bound to one context, without touching the kubeconfig.

    ``new_client_from_config`` takes the context by name and returns a client
    configured for it; the file on disk is never rewritten.
    """
    from kubernetes import client as kube_client
    from kubernetes import config as kube_config
    from kubernetes.dynamic import DynamicClient

    paths = kubeconfig_paths(extra)
    config_file = str(paths[0]) if paths else None
    api = kube_config.new_client_from_config(config_file=config_file, context=context)
    return DynamicClient(kube_client.ApiClient(configuration=api.configuration))
