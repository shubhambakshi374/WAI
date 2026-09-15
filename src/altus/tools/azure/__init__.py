"""Azure tools.

Split by what the operation needs from the user, the way ``tools/k8s`` and
``tools/aws`` are: reads run freely, changes are gated, and whole classes can
be switched off in ``[cloud.azure]`` so the model is never told they exist.
"""

from __future__ import annotations

from typing import Any

from altus.tools.azure.base import AzureMutatingTool, AzureTool
from altus.tools.azure.insight import (
    AzureCostTool,
    AzureInventoryTool,
    AzureQuotasTool,
    AzureTopologyTool,
)
from altus.tools.azure.mutations import AzureActionTool, AzureDeleteTool, AzureWriteTool
from altus.tools.azure.reads import (
    AzureCanITool,
    AzureExplainTool,
    AzureGetTool,
    AzureProvidersTool,
    AzureQueryTool,
    AzureSubscriptionsTool,
    AzureWhoamiTool,
)
from altus.tools.base import BaseTool


def azure_tools(settings: Any = None) -> list[BaseTool]:
    """The tool set, minus whatever `[cloud.azure]` switches off."""
    tools: list[BaseTool] = [
        AzureWhoamiTool(),
        AzureSubscriptionsTool(),
        AzureProvidersTool(),
        AzureExplainTool(),
        AzureGetTool(),
        AzureQueryTool(),
        AzureCanITool(),
        AzureInventoryTool(),
        AzureTopologyTool(),
        AzureCostTool(),
        AzureQuotasTool(),
    ]
    if settings is None or getattr(settings, "allow_writes", True):
        tools += [AzureWriteTool(), AzureActionTool()]
        if getattr(settings, "allow_delete", True):
            # Switched off means never registered, so the model is not told
            # about a tool that would prompt and then refuse.
            tools.append(AzureDeleteTool())
    return tools


__all__ = ["AzureMutatingTool", "AzureTool", "azure_tools"]
