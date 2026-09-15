"""AWS tools.

Split by what the operation needs from the user, the way ``tools/k8s`` is:
reads run freely, writes are gated, and whole classes can be switched off in
``[cloud.aws]`` so the model is never told they exist.
"""

from __future__ import annotations

from typing import Any

from wai.tools.aws.base import AwsMutatingTool, AwsTool
from wai.tools.aws.insight import (
    AwsCostTool,
    AwsInventoryTool,
    AwsQuotasTool,
    AwsTopologyTool,
)
from wai.tools.aws.reads import (
    AwsCallTool,
    AwsCanITool,
    AwsExplainTool,
    AwsRegionsTool,
    AwsServicesTool,
    AwsWhoamiTool,
)
from wai.tools.base import BaseTool


def aws_tools(settings: Any = None) -> list[BaseTool]:
    """The tool set, minus whatever `[cloud.aws]` switches off."""
    tools: list[BaseTool] = [
        AwsWhoamiTool(),
        AwsServicesTool(),
        AwsExplainTool(),
        AwsCallTool(),
        AwsCanITool(),
        AwsRegionsTool(),
        AwsInventoryTool(),
        AwsTopologyTool(),
        AwsQuotasTool(),
    ]
    if settings is None or getattr(settings, "allow_cost_explorer", True):
        # Not registered when switched off, so the model is never told about a
        # tool that would charge the user and then refuse.
        tools.append(AwsCostTool())
    return tools


__all__ = ["AwsMutatingTool", "AwsTool", "aws_tools"]
