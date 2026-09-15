"""AWS tools.

Split by what the operation needs from the user, the way ``tools/k8s`` is:
reads run freely, writes are gated, and whole classes can be switched off in
``[cloud.aws]`` so the model is never told they exist.
"""

from __future__ import annotations

from typing import Any

from wai.tools.aws.base import AwsMutatingTool, AwsTool
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
    return [
        AwsWhoamiTool(),
        AwsServicesTool(),
        AwsExplainTool(),
        AwsCallTool(),
        AwsCanITool(),
        AwsRegionsTool(),
    ]


__all__ = ["AwsMutatingTool", "AwsTool", "aws_tools"]
