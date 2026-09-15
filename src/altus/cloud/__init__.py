"""Cloud integrations. No ``textual`` imports.

Each cloud is an optional extra; ``INTEGRATIONS`` reports what is installed so
a missing SDK is a visible, fixable state rather than a tool that silently
isn't there.
"""

from altus.cloud.base import (
    INTEGRATIONS,
    CloudTarget,
    Integration,
    ProtectionMode,
    ProtectionRules,
    Sensitivity,
    available_integrations,
    integration,
    missing_integrations,
)
from altus.cloud.redact import redact, redact_text

__all__ = [
    "INTEGRATIONS",
    "CloudTarget",
    "Integration",
    "ProtectionMode",
    "ProtectionRules",
    "Sensitivity",
    "available_integrations",
    "integration",
    "missing_integrations",
    "redact",
    "redact_text",
]
