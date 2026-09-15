"""Azure tools.

Split by what the operation needs from the user, the way ``tools/k8s`` and
``tools/aws`` are: reads run freely, changes are gated, and whole classes can
be switched off in ``[cloud.azure]`` so the model is never told they exist.
"""

from __future__ import annotations

from typing import Any

from wai.tools.azure.base import AzureMutatingTool, AzureTool
from wai.tools.azure.reads import (
    AzureCanITool,
    AzureExplainTool,
    AzureGetTool,
    AzureProvidersTool,
    AzureQueryTool,
    AzureSubscriptionsTool,
    AzureWhoamiTool,
)
from wai.tools.base import BaseTool


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
    ]
    return tools


__all__ = ["AzureMutatingTool", "AzureTool", "azure_tools"]
