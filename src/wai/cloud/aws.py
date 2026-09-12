"""AWS: operation classification and introspection over botocore.

431 services and 19,189 operations, so per-operation tools are impossible and
a generic invoke over the service models is the only workable shape. Those
same models are what make it safe: they carry parameter shapes, required
members and documentation, so the model can be told exactly what an operation
takes instead of guessing.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from wai.cloud.base import CloudTarget, Sensitivity

#: Verb prefixes that only read. Anything not listed falls through to MUTATE,
#: so a new or unrecognised verb fails closed rather than open.
READ_PREFIXES: tuple[str, ...] = (
    "Describe",
    "List",
    "Get",
    "Search",
    "Scan",
    "Query",
    "Head",
    "BatchGet",
    "BatchDescribe",
    "Lookup",
    "Select",
    "Preview",
    "Estimate",
    "Simulate",
    "Check",
    "Verify",
    "Retrieve",
    "Detect",
    "View",
    "Count",
    "Discover",
    "Filter",
    "Read",
    "Test",
    "Validate",
)

#: Read-shaped operations that hand back live credentials. Measured: 98 of
#: these exist across the service models. Classifying them as plain reads
#: would let the model mint and exfiltrate credentials with nobody asked.
SENSITIVE_PATTERN = re.compile(
    r"(Credential|Token|Password|Secret|Session|Signin|SigninToken|Authoriz"
    r"|PrivateKey|Keypair|KeyPair|Federation|AssumeRole|Presigned)",
    re.IGNORECASE,
)

#: Operations that are sensitive whatever their name suggests.
SENSITIVE_OPERATIONS: frozenset[str] = frozenset(
    {
        "sts:GetSessionToken",
        "sts:GetFederationToken",
        "sts:AssumeRole",
        "sts:AssumeRoleWithWebIdentity",
        "sts:AssumeRoleWithSAML",
        "secretsmanager:GetSecretValue",
        "ssm:GetParameter",
        "ssm:GetParameters",
        "ssm:GetParametersByPath",
        "ecr:GetAuthorizationToken",
        "ecr-public:GetAuthorizationToken",
        "eks:DescribeCluster",  # returns the CA cert + endpoint used to build kubeconfig
        "iam:CreateAccessKey",
        "cognito-identity:GetCredentialsForIdentity",
        "connect:GetFederationToken",
        "gamelift:GetInstanceAccess",
        "lightsail:GetInstanceAccessDetails",
    }
)


def classify(service: str, operation: str) -> Sensitivity:
    """Sensitivity of one operation. Unknown verbs fail closed to MUTATE."""
    qualified = f"{service}:{operation}"
    if qualified in SENSITIVE_OPERATIONS:
        return Sensitivity.SENSITIVE_READ
    if not operation.startswith(READ_PREFIXES):
        return Sensitivity.MUTATE
    if SENSITIVE_PATTERN.search(operation):
        return Sensitivity.SENSITIVE_READ
    return Sensitivity.READ


@lru_cache(maxsize=1)
def _session() -> Any:
    import botocore.session

    return botocore.session.get_session()


def available_services() -> list[str]:
    return sorted(_session().get_available_services())


@lru_cache(maxsize=64)
def _service_model(service: str) -> Any:
    return _session().get_service_model(service)


def operations(service: str) -> list[str]:
    return sorted(_service_model(service).operation_names)


def describe_operation(service: str, operation: str) -> dict[str, Any]:
    """The parameter contract, straight from the service model.

    This is what makes the SDK path easier than guessing CLI flags, which is
    how the agent is steered towards it.
    """
    model = _service_model(service).operation_model(operation)
    shape = model.input_shape
    params: dict[str, Any] = {}
    if shape is not None:
        for name, member in shape.members.items():
            params[name] = {
                "type": member.type_name,
                "required": name in shape.required_members,
                "documentation": _strip_html(getattr(member, "documentation", ""))[:300],
            }
    return {
        "service": service,
        "operation": operation,
        "sensitivity": classify(service, operation).value,
        "documentation": _strip_html(model.documentation or "")[:600],
        "parameters": params,
    }


def python_method(operation: str) -> str:
    """botocore's API name for an operation: DescribeInstances -> describe_instances."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", operation).lower()


def _strip_html(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", text or "")).strip()


def target_for(account: str, region: str, service: str = "") -> CloudTarget:
    return CloudTarget(
        cloud="aws", context=account or "unknown-account", location=region, scope=service
    )
