"""AWS tools. No account is contacted.

botocore's service models ship with the library, so classification and
introspection are tested for real --- ``aws_explain`` really does read EC2's
model. Anything that would reach the network goes through a stub in the shape
of ``tests/test_k8s.py``'s FakeClient.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from wai.cloud.base import ProtectionRules
from wai.cloud.redact import MARKER
from wai.config.models import AwsSettings, CloudSettings
from wai.tools import default_registry
from wai.tools.approval import RecordingPolicy
from wai.tools.aws import aws_tools
from wai.tools.base import CloudContext, ToolContext
from wai.workspace import Workspace

IDENTITY = {
    "account": "123456789012",
    "arn": "arn:aws:iam::123456789012:user/dev",
    "user_id": "AIDAEXAMPLE",
}


class FakeProvider:
    """An AwsProvider's shape, recording what was asked for."""

    def __init__(self, responses: dict[str, Any] | None = None, *, fail: str = "") -> None:
        self.responses = responses or {}
        self.fail = fail
        self.calls: list[tuple[str, str, dict[str, Any], str]] = []
        self.identity = dict(IDENTITY)

    def default_region(self) -> str:
        return "eu-west-1"

    async def whoami(self) -> dict[str, str]:
        return self.identity

    async def regions(self, service: str = "ec2") -> list[str]:
        return ["eu-west-1", "us-east-1"]

    async def call(
        self, service: str, operation: str, params: dict[str, Any] | None = None, **kw: Any
    ) -> dict[str, Any]:
        key = f"{service}:{operation}"
        self.calls.append((service, operation, dict(params or {}), str(kw.get("region", ""))))
        if self.fail and self.fail == key:
            raise RuntimeError("boom")
        return dict(self.responses.get(key, {}))

    @property
    def real_calls(self) -> list[str]:
        """Calls that were not a dry run --- what actually happened."""
        return [
            f"{service}:{operation}"
            for service, operation, params, _region in self.calls
            if not params.get("DryRun")
        ]


def tool(name: str) -> Any:
    from wai.tools.aws import aws_tools as build

    return next(t for t in build() if t.name == name)


def context(
    provider: FakeProvider | None = None,
    *,
    policy: Any = None,
    patterns: tuple[str, ...] = ("*prod*",),
    accounts: tuple[str, ...] = (),
    settings: AwsSettings | None = None,
    tmp_path: Path | None = None,
) -> ToolContext:
    return ToolContext(
        workspace=Workspace(root=tmp_path or Path(".")),
        approvals=policy or RecordingPolicy(),
        cloud=CloudContext(
            aws=provider,
            protection=ProtectionRules(patterns=patterns, accounts=accounts),
            aws_settings=settings or AwsSettings(),
        ),
    )


# ------------------------------------------------------------------ identity


async def test_whoami_names_the_account_and_region() -> None:
    out = await tool("aws_whoami").run({}, context(FakeProvider()))
    assert not out.is_error
    assert "123456789012" in out.content
    assert "eu-west-1" in out.content


async def test_whoami_flags_a_protected_account() -> None:
    """ProtectedSettings.accounts has been plumbed into ProtectionRules since
    Phase 2 and never had a caller. AWS is what makes it live."""
    ctx = context(FakeProvider(), accounts=("123456789012",))
    out = await tool("aws_whoami").run({}, ctx)
    assert "PROTECTED" in out.content


async def test_every_tool_says_so_when_there_is_no_session() -> None:
    for name in ("aws_whoami", "aws_call", "aws_can_i", "aws_regions"):
        out = await tool(name).run(
            {"service": "ec2", "operation": "DescribeInstances", "actions": ["s3:GetObject"]},
            context(None),
        )
        assert out.is_error and "no AWS session" in out.content, name


# ------------------------------------------------------------ introspection


def test_explain_reads_the_real_service_model() -> None:
    """Not a fixture: botocore ships the models, so this is the contract the
    SDK will actually enforce."""
    from wai.cloud import aws as aws_api

    described = aws_api.describe_operation("ec2", "DescribeInstances")
    assert described["sensitivity"] == "read"
    assert "InstanceIds" in described["parameters"]
    assert described["documentation"]


async def test_explain_lists_operations_with_their_sensitivity() -> None:
    out = await tool("aws_explain").run({"service": "s3", "filter": "Bucket"}, context())
    assert not out.is_error
    assert "DeleteBucket" in out.content
    assert out.visual is not None, "a table the human can read"


async def test_explain_marks_required_parameters() -> None:
    out = await tool("aws_explain").run({"service": "s3", "operation": "GetObject"}, context())
    assert "* Bucket" in out.content or "*   Bucket" in out.content.replace("  ", " ")
    assert "required" in out.content


async def test_explain_on_nonsense_says_so() -> None:
    out = await tool("aws_explain").run(
        {"service": "ec2", "operation": "FrobnicateWidget"}, context()
    )
    assert out.is_error


# ----------------------------------------------------------------- aws_call


async def test_call_returns_the_response() -> None:
    provider = FakeProvider({"ec2:DescribeInstances": {"Reservations": [{"InstanceId": "i-1"}]}})
    out = await tool("aws_call").run(
        {"service": "ec2", "operation": "DescribeInstances"}, context(provider)
    )
    assert not out.is_error
    assert "i-1" in out.content


async def test_call_refuses_anything_that_changes_something() -> None:
    """The read tool must not be a way around the gate."""
    provider = FakeProvider()
    for service, operation in [
        ("ec2", "TerminateInstances"),
        ("s3", "PutObject"),
        ("iam", "AttachRolePolicy"),
    ]:
        out = await tool("aws_call").run(
            {"service": service, "operation": operation}, context(provider)
        )
        assert out.is_error, f"{service}:{operation} was allowed through the read tool"
        assert "aws_write" in out.content
    assert provider.calls == [], "nothing may reach AWS"


async def test_call_redacts_secret_material() -> None:
    """Tool output goes verbatim to whichever LLM is configured."""
    provider = FakeProvider(
        {"secretsmanager:GetSecretValue": {"SecretString": "hunter2", "Name": "db"}}
    )
    out = await tool("aws_call").run(
        {"service": "secretsmanager", "operation": "GetSecretValue"}, context(provider)
    )
    assert "hunter2" not in out.content
    assert MARKER in out.content


async def test_call_reports_a_failure_rather_than_raising() -> None:
    provider = FakeProvider(fail="ec2:DescribeInstances")
    out = await tool("aws_call").run(
        {"service": "ec2", "operation": "DescribeInstances"}, context(provider)
    )
    assert out.is_error and "boom" in out.content


async def test_call_rejects_params_that_are_not_an_object() -> None:
    out = await tool("aws_call").run(
        {"service": "ec2", "operation": "DescribeInstances", "params": "InstanceIds=i-1"},
        context(FakeProvider()),
    )
    assert out.is_error


async def test_call_uses_the_region_it_was_given() -> None:
    provider = FakeProvider({"ec2:DescribeInstances": {}})
    await tool("aws_call").run(
        {"service": "ec2", "operation": "DescribeInstances", "region": "us-east-1"},
        context(provider),
    )
    assert provider.calls[0][3] == "us-east-1"


# ---------------------------------------------------------------- aws_can_i


async def test_can_i_reports_each_decision() -> None:
    provider = FakeProvider(
        {
            "iam:SimulatePrincipalPolicy": {
                "EvaluationResults": [
                    {"EvalActionName": "s3:GetObject", "EvalDecision": "allowed"},
                    {"EvalActionName": "s3:DeleteBucket", "EvalDecision": "implicitDeny"},
                ]
            }
        }
    )
    out = await tool("aws_can_i").run(
        {"actions": ["s3:GetObject", "s3:DeleteBucket"]}, context(provider)
    )
    assert not out.is_error
    assert out.summary == "1/2 allowed"


async def test_can_i_says_so_when_it_cannot_check() -> None:
    """Simulating needs iam:SimulatePrincipalPolicy, and plenty of roles that
    can do the thing cannot ask whether they can."""
    provider = FakeProvider(fail="iam:SimulatePrincipalPolicy")
    out = await tool("aws_can_i").run({"actions": ["s3:GetObject"]}, context(provider))
    assert out.is_error
    assert "iam:SimulatePrincipalPolicy" in out.content


async def test_can_i_needs_actions() -> None:
    out = await tool("aws_can_i").run({"actions": []}, context(FakeProvider()))
    assert out.is_error


# --------------------------------------------------------------- the tool set


def test_every_read_tool_is_read_only() -> None:
    for entry in aws_tools():
        assert entry.read_only, f"{entry.name} is registered as a read but is not one"


def test_aws_tools_register_because_boto3_is_a_core_dependency() -> None:
    """Unlike k8s, AWS needs no extra --- bedrock already depends on boto3."""
    names = set(default_registry(kubernetes=False, cloud=CloudSettings()).names)
    assert {"aws_whoami", "aws_call", "aws_explain", "aws_can_i"} <= names
