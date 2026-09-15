"""Kubernetes tools.

Split by what the operation needs from the user rather than by API group,
because that is the distinction that matters when deciding what to register:
reads run freely, mutations are gated, and the privileged set can be switched
off entirely.
"""

from __future__ import annotations

from wai.cloud.base import Sensitivity
from wai.cloud.kube import classify
from wai.tools.base import BaseTool
from wai.tools.k8s.base import MAX_LOG_LINES, K8sMutatingTool, K8sTool
from wai.tools.k8s.insight import (
    K8sExplainTool,
    K8sStorageTool,
    K8sTopologyTool,
    K8sTopTool,
    K8sUsageTool,
)
from wai.tools.k8s.mutations import (
    K8sApplyTool,
    K8sDeleteTool,
    K8sRolloutTool,
    K8sScaleTool,
)
from wai.tools.k8s.reads import (
    K8sApiResourcesTool,
    K8sContextsTool,
    K8sEventsTool,
    K8sListTool,
    K8sLogsTool,
)


def k8s_tools() -> list[BaseTool]:
    return [
        K8sContextsTool(),
        K8sApiResourcesTool(),
        K8sListTool(),
        K8sEventsTool(),
        K8sLogsTool(),
        K8sTopTool(),
        K8sUsageTool(),
        K8sStorageTool(),
        K8sTopologyTool(),
        K8sExplainTool(),
        K8sApplyTool(),
        K8sDeleteTool(),
        K8sScaleTool(),
        K8sRolloutTool(),
    ]


def sensitivity_for(kind: str) -> Sensitivity:
    return classify("get", kind)


__all__ = [
    "MAX_LOG_LINES",
    "K8sMutatingTool",
    "K8sTool",
    "k8s_tools",
    "sensitivity_for",
]
