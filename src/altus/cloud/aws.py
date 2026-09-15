"""AWS: operation classification and introspection over botocore.

431 services and 19,189 operations, so per-operation tools are impossible and
a generic invoke over the service models is the only workable shape. Those
same models are what make it safe: they carry parameter shapes, required
members and documentation, so the model can be told exactly what an operation
takes instead of guessing.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from altus.cloud.base import CloudTarget, Sensitivity

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


#: Services whose *writes* decide who may do what, or hold key material.
#: Reading them is ordinary --- listing roles is how you understand an account
#: --- but changing one is the AWS equivalent of editing a ClusterRoleBinding.
PRIVILEGED_SERVICES: frozenset[str] = frozenset(
    {
        "iam",
        "sts",
        "organizations",
        "sso",
        "sso-admin",
        "identitystore",
        "kms",
        "cloudhsm",
        "cloudhsmv2",
        "acm-pca",
        "account",
        "ram",
    }
)

#: Destroying something stateful. Losing a security group is an inconvenience;
#: losing a database is not, and MUTATE cannot tell the two apart.
IRREVERSIBLE_PATTERN = re.compile(
    r"^(Terminate|Delete|Destroy|Purge|Remove|Deregister|Disassociate|Revoke|Cancel)"
)

#: Kinds where deletion is unrecoverable rather than merely inconvenient ---
#: the object holds data, or something else depends on it existing.
STATEFUL_NOUNS = re.compile(
    r"(Bucket|DBInstance|DBCluster|DBSnapshot|Cluster|Table|Volume|Snapshot|FileSystem"
    r"|Instance|Domain|Repository|Vault|Backup|Archive|Stack|Distribution|Queue|Stream"
    r"|Secret|Parameter|LogGroup|Image|Workspace|Broker|Replication)",
    re.IGNORECASE,
)

#: Operations that open something to the network, or widen who can reach it.
#: A single call here can expose a private subnet to the internet.
EXPOSURE_OPERATIONS: frozenset[str] = frozenset(
    {
        "ec2:AuthorizeSecurityGroupIngress",
        "ec2:ModifyInstanceAttribute",
        "ec2:CreateInternetGateway",
        "ec2:AttachInternetGateway",
        "ec2:ModifyVpcEndpointServicePermissions",
        "ec2:ModifySnapshotAttribute",
        "ec2:ModifyImageAttribute",
        "s3:PutBucketPolicy",
        "s3:PutBucketAcl",
        "s3:PutObjectAcl",
        "s3:DeletePublicAccessBlock",
        "s3:PutPublicAccessBlock",
        "rds:ModifyDBInstance",
        "lambda:AddPermission",
        "sns:AddPermission",
        "sqs:AddPermission",
        "ecr:SetRepositoryPolicy",
        "secretsmanager:PutResourcePolicy",
        "kms:PutKeyPolicy",
        "efs:PutFileSystemPolicy",
        "opensearch:UpdateDomainConfig",
        "es:UpdateElasticsearchDomainConfig",
    }
)


#: Verbs that change something. Destructive ones are here too, deliberately:
#: without them every Delete* fell through to the unrecognised branch and came
#: out PRIVILEGED --- 2,281 of them, including DeleteTag and DeleteAlarm. That
#: made STATEFUL_NOUNS decide nothing, and would have demanded a typed
#: confirmation for deleting a CloudWatch alarm. A challenge that fires on
#: everything trains people to type through it, which is how the gate stops
#: working. Destruction of something *stateful* is still privileged; this is
#: only the floor for the rest.
WRITE_PREFIXES: tuple[str, ...] = (
    "Delete",
    "Remove",
    "Revoke",
    "Cancel",
    "Deregister",
    "Disassociate",
    "Detach",
    "Terminate",
    "Purge",
    "Destroy",
    "Create",
    "Put",
    "Update",
    "Modify",
    "Set",
    "Add",
    "Attach",
    "Associate",
    "Register",
    "Enable",
    "Disable",
    "Start",
    "Stop",
    "Reboot",
    "Restore",
    "Run",
    "Invoke",
    "Send",
    "Publish",
    "Tag",
    "Untag",
    "Copy",
    "Import",
    "Export",
    "Upload",
    "Apply",
    "Accept",
    "Reject",
    "Activate",
    "Deactivate",
    "Allocate",
    "Release",
    "Assign",
    "Unassign",
    "Replace",
    "Reset",
    "Resume",
    "Suspend",
    "Rotate",
    "Switch",
    "Transfer",
    "Move",
    "Promote",
    "Failover",
    "Rollback",
    "Retry",
    "Complete",
    "Confirm",
    "Batch",
    "Bulk",
    "Request",
    "Provision",
    "Deploy",
    "Execute",
    "Initiate",
    "Submit",
    "Sign",
    "Encrypt",
    "Decrypt",
    "Generate",
)


def classify(service: str, operation: str) -> Sensitivity:
    """Sensitivity of one operation.

    Precedence runs highest-first, because an operation can be several things
    at once and the most dangerous reading is the one that should win:
    ``iam:DeleteRole`` is both an identity write and a destruction.

    Unknown verbs fail closed to PRIVILEGED --- one step stricter than the
    MUTATE this used to return. AWS ships new operations constantly, and a verb
    nobody anticipated is exactly where guessing low is unrecoverable.
    """
    qualified = f"{service}:{operation}"

    if operation.startswith(READ_PREFIXES):
        # Reading the authorization graph is how you understand an account, so
        # it stays a read --- only the credential-shaped ones escalate.
        if qualified in SENSITIVE_OPERATIONS or SENSITIVE_PATTERN.search(operation):
            return Sensitivity.SENSITIVE_READ
        return Sensitivity.READ

    # Past here nothing is a read, so the SENSITIVE_OPERATIONS shortcut must
    # not apply: iam:CreateAccessKey and sts:AssumeRole are in that set and
    # *also* create lasting access. Returning SENSITIVE_READ for them would
    # have let a long-lived credential be minted on a single keypress, because
    # only PRIVILEGED demands the name typed out.

    if service in PRIVILEGED_SERVICES:
        return Sensitivity.PRIVILEGED
    if qualified in EXPOSURE_OPERATIONS:
        return Sensitivity.PRIVILEGED
    if IRREVERSIBLE_PATTERN.match(operation) and STATEFUL_NOUNS.search(operation):
        return Sensitivity.PRIVILEGED
    if operation.startswith(WRITE_PREFIXES):
        return Sensitivity.MUTATE
    # A verb we have never seen, on a service we do not know. Fail closed to
    # the strictest level rather than the second-strictest.
    return Sensitivity.PRIVILEGED


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


def input_shape(service: str, operation: str) -> set[str] | None:
    """Member names an operation accepts, or None when it takes nothing.

    Used to ask whether DryRun is available rather than assuming it from the
    service: 758 of EC2's 802 operations take it and 44 do not.
    """
    model = _service_model(service).operation_model(operation)
    shape = model.input_shape
    return None if shape is None else set(shape.members)


def python_method(operation: str) -> str:
    """botocore's API name for an operation: DescribeInstances -> describe_instances."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", operation).lower()


def _strip_html(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", text or "")).strip()


def target_for(account: str, region: str, service: str = "") -> CloudTarget:
    return CloudTarget(
        cloud="aws", context=account or "unknown-account", location=region, scope=service
    )


# --------------------------------------------------------------- the session

#: Results returned from one call. AWS paginators will happily walk a hundred
#: thousand objects; the model pays for every one of them.
MAX_RESULTS = 500

#: Keys whose values are pagination bookkeeping, not answers.
PAGINATION_KEYS = frozenset(
    {"NextToken", "NextMarker", "Marker", "ContinuationToken", "IsTruncated", "ResponseMetadata"}
)


@dataclass
class AwsProvider:
    """Sessions and clients, built on first use and cached.

    Mirrors ``K8sProvider``: constructing a client is a real cost --- botocore
    loads and parses a JSON service model --- so a session that never mentions
    S3 should never build an S3 client. Cached per (service, region), because
    the region is part of the endpoint.
    """

    profile: str | None = None
    region: str | None = None
    _session: Any = field(default=None, repr=False)
    _clients: dict[tuple[str, str], Any] = field(default_factory=dict, repr=False)
    _identity: dict[str, str] | None = field(default=None, repr=False)

    def reset(self) -> None:
        """Drop everything cached. Called when the profile or region changes:
        a cached client is bound to the credentials it was built with, and
        reusing one would send the next call to the account just left."""
        self._session = None
        self._clients.clear()
        self._identity = None

    def session(self) -> Any:
        if self._session is None:
            import boto3

            self._session = boto3.Session(
                profile_name=self.profile or None, region_name=self.region or None
            )
        return self._session

    def default_region(self) -> str:
        return str(self.session().region_name or "us-east-1")

    def client(self, service: str, region: str = "") -> Any:
        where = region or self.default_region()
        key = (service, where)
        if key not in self._clients:
            self._clients[key] = self.session().client(service, region_name=where)
        return self._clients[key]

    async def whoami(self) -> dict[str, str]:
        """Account, ARN and user id, cached for the session.

        The ARN is not decoration: ``iam:SimulatePrincipalPolicy`` needs it as
        ``PolicySourceArn`` to answer "may *I* do this", so every preflight
        depends on this call having happened.
        """
        if self._identity is None:

            def _call() -> dict[str, str]:
                raw = self.client("sts").get_caller_identity()
                return {
                    "account": str(raw.get("Account", "")),
                    "arn": str(raw.get("Arn", "")),
                    "user_id": str(raw.get("UserId", "")),
                }

            self._identity = await asyncio.to_thread(_call)
        return self._identity

    async def regions(self, service: str = "ec2") -> list[str]:
        def _call() -> list[str]:
            return sorted(self.session().get_available_regions(service))

        return await asyncio.to_thread(_call)

    async def call(
        self,
        service: str,
        operation: str,
        params: dict[str, Any] | None = None,
        *,
        region: str = "",
        limit: int = MAX_RESULTS,
    ) -> dict[str, Any]:
        """One operation, paginated where the service offers a paginator.

        Returns the response dict with pagination bookkeeping stripped, plus a
        ``_truncated`` marker when the cap was hit --- silently returning the
        first page would have the model reason about a partial answer as
        though it were the whole one.
        """
        method = python_method(operation)
        arguments = dict(params or {})

        def _call() -> dict[str, Any]:
            client = self.client(service, region)
            if not client.can_paginate(method):
                raw = getattr(client, method)(**arguments)
                return {k: v for k, v in raw.items() if k not in PAGINATION_KEYS}

            merged: dict[str, Any] = {}
            counted = 0
            truncated = False
            for page in client.get_paginator(method).paginate(**arguments):
                for key, value in page.items():
                    if key in PAGINATION_KEYS:
                        continue
                    if isinstance(value, list):
                        room = limit - counted
                        existing = merged.setdefault(key, [])
                        existing.extend(value[:room])
                        counted += min(len(value), max(0, room))
                    else:
                        merged.setdefault(key, value)
                if counted >= limit:
                    truncated = True
                    break
            if truncated:
                merged["_truncated"] = f"stopped at {limit} results"
            return merged

        return await asyncio.to_thread(_call)
