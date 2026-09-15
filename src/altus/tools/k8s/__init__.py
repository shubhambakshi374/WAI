"""Kubernetes tools.

Split by what the operation needs from the user rather than by API group,
because that is the distinction that matters when deciding what to register:
reads run freely, mutations are gated, and the privileged set can be switched
off entirely.
"""

from __future__ import annotations

from typing import Any

from altus.cloud.base import Sensitivity
from altus.cloud.kube import classify
from altus.tools.base import BaseTool
from altus.tools.k8s.base import MAX_LOG_LINES, K8sMutatingTool, K8sTool
from altus.tools.k8s.context import K8sUseContextTool
from altus.tools.k8s.insight import (
    K8sExplainTool,
    K8sStorageTool,
    K8sTopologyTool,
    K8sTopTool,
    K8sUsageTool,
)
from altus.tools.k8s.mutations import (
    K8sApplyTool,
    K8sCreateTool,
    K8sDeleteTool,
    K8sPatchTool,
    K8sReplaceTool,
    K8sRolloutTool,
    K8sScaleTool,
)
from altus.tools.k8s.nodes import K8sDrainTool, K8sNodeTool
from altus.tools.k8s.reads import (
    K8sApiResourcesTool,
    K8sCanITool,
    K8sContextsTool,
    K8sEventsTool,
    K8sGetTool,
    K8sListTool,
    K8sLogsTool,
    K8sRawTool,
    K8sRolloutStatusTool,
    K8sWaitTool,
)
from altus.tools.k8s.streams import (
    K8sCpTool,
    K8sExecTool,
    K8sPortForwardTool,
    PortForwards,
)


def k8s_tools(settings: Any = None) -> list[BaseTool]:
    """The tool set, minus whatever `[cloud.k8s]` switches off.

    A disabled class is *not registered*, rather than registered and refused:
    a tool the model cannot see costs no context and cannot be argued into
    being used. ``settings`` is a ``K8sSettings``; None means everything on.
    """
    allow = _Allow(settings)
    tools: list[BaseTool] = [
        K8sContextsTool(),
        K8sApiResourcesTool(),
        K8sListTool(),
        K8sGetTool(),
        K8sEventsTool(),
        K8sLogsTool(),
        K8sWaitTool(),
        K8sCanITool(),
        K8sRawTool(),
        K8sTopTool(),
        K8sUsageTool(),
        K8sStorageTool(),
        K8sTopologyTool(),
        K8sExplainTool(),
        K8sRolloutStatusTool(),
        K8sApplyTool(),
        K8sPatchTool(),
        K8sCreateTool(),
        K8sReplaceTool(),
        K8sDeleteTool(),
        K8sScaleTool(),
        K8sRolloutTool(),
        K8sUseContextTool(),
    ]
    if allow("allow_node_lifecycle"):
        tools += [K8sNodeTool(), K8sDrainTool()]
    if allow("allow_exec"):
        tools += [K8sExecTool(), K8sCpTool()]
    if allow("allow_port_forward"):
        tools.append(K8sPortForwardTool())
    return tools


class _Allow:
    """Reads a switch, defaulting to on when there is no settings object.

    Defaulting *on* is deliberate: a caller that has not been taught about
    these switches --- a test, the Phase 3 engine --- should get the full tool
    set, and turning something off should require saying so.
    """

    def __init__(self, settings: Any) -> None:
        self._settings = settings

    def __call__(self, name: str) -> bool:
        if self._settings is None:
            return True
        return bool(getattr(self._settings, name, True))


def disabled_classes(settings: Any) -> list[str]:
    """Which capability classes are off, for `/tools` to show. A capability
    that is missing should be visibly missing, not mysteriously absent."""
    if settings is None:
        return []
    labels = {
        "allow_node_lifecycle": "node lifecycle (cordon, drain, taint)",
        "allow_exec": "exec, attach and cp",
        "allow_port_forward": "port forwarding",
        "allow_rbac_writes": "RBAC writes (Roles, Bindings, tokens)",
        "allow_cli": "kubectl, helm and kustomize",
    }
    return [text for key, text in labels.items() if not getattr(settings, key, True)]


def sensitivity_for(kind: str) -> Sensitivity:
    return classify("get", kind)


__all__ = [
    "MAX_LOG_LINES",
    "K8sMutatingTool",
    "K8sTool",
    "PortForwards",
    "disabled_classes",
    "k8s_tools",
    "sensitivity_for",
]
