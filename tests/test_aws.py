"""AWS tools. No account is contacted.

botocore's service models ship with the library, so classification and
introspection are tested for real --- ``aws_explain`` really does read EC2's
model. Anything that would reach the network goes through a stub in the shape
of ``tests/test_k8s.py``'s FakeClient.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from wai.cloud.base import ProtectionRules, Sensitivity
from wai.cloud.redact import MARKER
from wai.config.models import AwsSettings, CloudSettings
from wai.tools import default_registry
from wai.tools.approval import Decision, RecordingPolicy
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


def test_only_the_write_tool_prompts() -> None:
    """Everything else must be a read. A tool that quietly changes something
    without being marked would never reach the gate at all."""
    for entry in aws_tools():
        if entry.name == "aws_write":
            continue
        assert entry.read_only, f"{entry.name} is registered as a read but is not one"


def test_aws_tools_register_because_boto3_is_a_core_dependency() -> None:
    """Unlike k8s, AWS needs no extra --- bedrock already depends on boto3."""
    names = set(default_registry(kubernetes=False, cloud=CloudSettings()).names)
    assert {"aws_whoami", "aws_call", "aws_explain", "aws_can_i"} <= names


# ------------------------------------------------------------ curated reads


VPC_REGION = {
    "ec2:DescribeVpcs": {
        "Vpcs": [
            {
                "VpcId": "vpc-1",
                "CidrBlock": "10.0.0.0/16",
                "State": "available",
                "Tags": [{"Key": "Name", "Value": "prod-vpc"}],
            }
        ]
    },
    "ec2:DescribeSubnets": {
        "Subnets": [
            {
                "SubnetId": "subnet-a",
                "VpcId": "vpc-1",
                "AvailabilityZone": "eu-west-1a",
                "State": "available",
            },
            {
                "SubnetId": "subnet-b",
                "VpcId": "vpc-1",
                "AvailabilityZone": "eu-west-1b",
                "State": "available",
            },
        ]
    },
    "ec2:DescribeSecurityGroups": {
        "SecurityGroups": [
            {"GroupId": "sg-1", "GroupName": "web", "VpcId": "vpc-1"},
            {"GroupId": "sg-unused", "GroupName": "orphan", "VpcId": "vpc-1"},
        ]
    },
    "ec2:DescribeInstances": {
        "Reservations": [
            {
                "Instances": [
                    {
                        "InstanceId": "i-1",
                        "VpcId": "vpc-1",
                        "SubnetId": "subnet-a",
                        "InstanceType": "t3.small",
                        "State": {"Name": "running"},
                        "SecurityGroups": [{"GroupId": "sg-1"}],
                        "Tags": [{"Key": "Name", "Value": "web-1"}],
                    },
                    {
                        "InstanceId": "i-2",
                        "VpcId": "vpc-1",
                        "SubnetId": "subnet-b",
                        "InstanceType": "t3.small",
                        "State": {"Name": "stopped"},
                        "SecurityGroups": [{"GroupId": "sg-1"}],
                    },
                ]
            }
        ]
    },
}


async def test_inventory_gathers_several_services() -> None:
    provider = FakeProvider(
        {
            **VPC_REGION,
            "rds:DescribeDBInstances": {
                "DBInstances": [
                    {
                        "DBInstanceIdentifier": "orders",
                        "DBInstanceStatus": "available",
                        "Engine": "postgres",
                        "DBInstanceClass": "db.t3.medium",
                    }
                ]
            },
            "lambda:ListFunctions": {
                "Functions": [{"FunctionName": "resize", "Runtime": "python3.12"}]
            },
        }
    )
    out = await tool("aws_inventory").run({}, context(provider))
    assert not out.is_error
    assert "web-1" in out.content and "orders" in out.content and "resize" in out.content
    assert out.visual is not None


async def test_one_service_failing_does_not_take_the_inventory_down() -> None:
    """IAM routinely permits some of these and not others."""
    provider = FakeProvider({**VPC_REGION}, fail="rds:DescribeDBInstances")
    out = await tool("aws_inventory").run({}, context(provider))
    assert not out.is_error
    assert "web-1" in out.content, "EC2 still listed"
    assert "rds" in (out.visual.caption if out.visual else ""), "and the failure is named"


async def test_topology_builds_the_vpc_tree() -> None:
    provider = FakeProvider(VPC_REGION)
    out = await tool("aws_topology").run({}, context(provider))
    assert not out.is_error
    graph = out.visual
    assert graph is not None
    kinds = {node.kind for node in graph.nodes}
    assert {"Vpc", "Subnet", "Instance", "SecurityGroup"} <= kinds
    owns = [e for e in graph.edges if e.relation == "owns"]
    assert len(owns) == 4, "vpc->2 subnets, subnet->instance twice"


async def test_topology_nodes_know_which_tool_opens_them() -> None:
    """Inferring the reader from the kind would mean the front end guessing
    which cloud a graph came from, and guessing wrong would send an instance
    id to a Kubernetes tool."""
    provider = FakeProvider(VPC_REGION)
    out = await tool("aws_topology").run({}, context(provider))
    assert out.visual is not None
    assert {node.reader for node in out.visual.nodes} == {"aws_call"}


async def test_topology_identities_split_the_way_drill_down_expects() -> None:
    from wai.cloud.k8s import split_node_id

    provider = FakeProvider(VPC_REGION)
    out = await tool("aws_topology").run({}, context(provider))
    assert out.visual is not None
    for node in out.visual.nodes:
        assert split_node_id(node.id) is not None, node.id


async def test_topology_omits_security_groups_nothing_uses() -> None:
    """An unattached group is noise on a map of what talks to what."""
    provider = FakeProvider(VPC_REGION)
    out = await tool("aws_topology").run({}, context(provider))
    assert out.visual is not None
    names = {node.name for node in out.visual.nodes}
    assert "web" in names and "orphan" not in names


async def test_topology_security_groups_cut_across_the_tree() -> None:
    """They are what make this a graph rather than a tree: membership ignores
    the VPC hierarchy entirely."""
    provider = FakeProvider(VPC_REGION)
    out = await tool("aws_topology").run({}, context(provider))
    assert out.visual is not None
    secures = [e for e in out.visual.edges if e.relation == "secures"]
    assert len(secures) == 2


async def test_topology_draws_what_it_can_when_a_read_is_denied() -> None:
    provider = FakeProvider(VPC_REGION, fail="ec2:DescribeSecurityGroups")
    out = await tool("aws_topology").run({}, context(provider))
    assert not out.is_error
    assert out.visual is not None
    assert not [e for e in out.visual.edges if e.relation == "secures"]


async def test_topology_on_an_empty_region_says_so() -> None:
    out = await tool("aws_topology").run({}, context(FakeProvider()))
    assert not out.is_error
    assert "no VPC resources" in out.content


# ----------------------------------------------------------------- aws_cost


COST = {
    "ce:GetCostAndUsage": {
        "ResultsByTime": [
            {
                "TimePeriod": {"Start": "2026-09-01"},
                "Groups": [
                    {"Keys": ["Amazon EC2"], "Metrics": {"UnblendedCost": {"Amount": "12.50"}}},
                    {"Keys": ["Amazon S3"], "Metrics": {"UnblendedCost": {"Amount": "1.20"}}},
                ],
            },
            {
                "TimePeriod": {"Start": "2026-09-02"},
                "Groups": [
                    {"Keys": ["Amazon EC2"], "Metrics": {"UnblendedCost": {"Amount": "13.10"}}},
                    {"Keys": ["Amazon S3"], "Metrics": {"UnblendedCost": {"Amount": "1.30"}}},
                ],
            },
        ]
    }
}


async def test_cost_returns_a_chart_with_a_real_time_axis() -> None:
    provider = FakeProvider(COST)
    out = await tool("aws_cost").run({"days": 2}, context(provider))
    assert not out.is_error
    chart = out.visual
    assert chart is not None
    assert {line.label for line in chart.series} == {"Amazon EC2", "Amazon S3"}
    assert all(line.timed for line in chart.series), "timestamps, not just a shape"
    assert "28.10" in out.summary


async def test_cost_says_in_its_description_that_it_charges() -> None:
    """It bills per request. The model has to be able to warn the user before
    spending their money."""
    assert "COSTS MONEY" in tool("aws_cost").description


async def test_cost_can_be_switched_off_entirely() -> None:
    """Not registered when off --- a tool that would charge and then refuse is
    worse than one that is simply absent."""
    off = AwsSettings(allow_cost_explorer=False)
    assert not any(t.name == "aws_cost" for t in aws_tools(off))
    assert any(t.name == "aws_cost" for t in aws_tools(AwsSettings()))


async def test_cost_with_no_data_says_so() -> None:
    out = await tool("aws_cost").run({}, context(FakeProvider()))
    assert not out.is_error and "no cost data" in out.content


# --------------------------------------------------------------- aws_quotas


async def test_quotas_come_back_as_bars() -> None:
    provider = FakeProvider(
        {
            "service-quotas:ListServiceQuotas": {
                "Quotas": [
                    {
                        "QuotaName": "Running On-Demand Standard instances",
                        "Value": 640,
                        "Unit": "None",
                    },
                    {"QuotaName": "VPCs per Region", "Value": 5, "Unit": "None"},
                ]
            }
        }
    )
    out = await tool("aws_quotas").run({"service": "ec2"}, context(provider))
    assert not out.is_error
    assert out.visual is not None and len(out.visual.bars) == 2


# ------------------------------------------------------------------- writes

# The Kubernetes gate promises "the server says this will work". AWS offers
# that for 4.3% of operations, so these assert the prompt tells the truth about
# which check actually ran.


ALLOWED = {
    "iam:SimulatePrincipalPolicy": {
        "EvaluationResults": [{"EvalActionName": "x", "EvalDecision": "allowed"}]
    }
}
DENIED = {
    "iam:SimulatePrincipalPolicy": {
        "EvaluationResults": [{"EvalActionName": "x", "EvalDecision": "implicitDeny"}]
    }
}


def approving(decision: Decision = Decision.ALLOW) -> RecordingPolicy:
    return RecordingPolicy(decision=decision)


async def test_a_non_ec2_write_never_claims_a_dry_run_happened() -> None:
    """The load-bearing honesty. AWS cannot preview an S3 delete, and a prompt
    that implies otherwise buys false confidence at the moment of consent."""
    provider = FakeProvider({**ALLOWED, "s3:DeleteObject": {}})
    policy = approving()
    out = await tool("aws_write").run(
        {"service": "s3", "operation": "DeleteObject", "params": {"Bucket": "b", "Key": "k"}},
        context(provider, policy=policy),
    )
    assert not out.is_error, out.content
    dry_run = policy.seen[0].dry_run
    assert "not dry-run" in dry_run
    assert "permission check: allowed" in dry_run
    assert "succeeded" not in dry_run


async def test_an_ec2_write_really_is_dry_run_first() -> None:
    class DryRunProvider(FakeProvider):
        async def call(self, service, operation, params=None, **kw):  # type: ignore[no-untyped-def]
            self.calls.append((service, operation, dict(params or {}), str(kw.get("region", ""))))
            if (params or {}).get("DryRun"):
                raise RuntimeError("An error occurred (DryRunOperation) when calling StopInstances")
            return {}

    provider = DryRunProvider()
    policy = approving()
    out = await tool("aws_write").run(
        {"service": "ec2", "operation": "StopInstances", "params": {"InstanceIds": ["i-1"]}},
        context(provider, policy=policy),
    )
    assert not out.is_error, out.content
    assert "dry run succeeded" in policy.seen[0].dry_run
    assert "not dry-run" not in policy.seen[0].dry_run


async def test_a_denied_permission_check_refuses_before_prompting() -> None:
    """Asking about a decision that does not exist spends attention for
    nothing, and teaches people the prompt is noise."""
    provider = FakeProvider(DENIED)
    policy = approving()
    out = await tool("aws_write").run(
        {"service": "s3", "operation": "DeleteObject", "params": {}},
        context(provider, policy=policy),
    )
    assert out.is_error
    assert policy.seen == []
    assert "s3:DeleteObject" not in provider.real_calls


async def test_a_permission_check_that_cannot_run_is_reported_not_treated_as_denial() -> None:
    """Simulating needs iam:SimulatePrincipalPolicy, which plenty of roles that
    can do the thing do not have."""
    provider = FakeProvider({"s3:DeleteObject": {}}, fail="iam:SimulatePrincipalPolicy")
    policy = approving()
    out = await tool("aws_write").run(
        {"service": "s3", "operation": "DeleteObject", "params": {}},
        context(provider, policy=policy),
    )
    assert not out.is_error, out.content
    assert "could not run" in policy.seen[0].dry_run


@pytest.mark.parametrize(
    ("service", "operation"),
    [
        ("s3", "DeleteObject"),
        ("ec2", "CreateTags"),
        ("iam", "AttachRolePolicy"),
        ("rds", "DeleteDBInstance"),
    ],
)
async def test_rejection_changes_nothing(service: str, operation: str) -> None:
    """The load-bearing assertion: only the preflight may have run."""
    provider = FakeProvider({**ALLOWED, f"{service}:{operation}": {}})
    out = await tool("aws_write").run(
        {"service": service, "operation": operation, "params": {}},
        context(provider, policy=approving(Decision.DENY)),
    )
    assert out.is_error and out.denied
    assert f"{service}:{operation}" not in provider.real_calls


async def test_the_blast_radius_is_always_named() -> None:
    provider = FakeProvider({**ALLOWED, "s3:DeleteObject": {}})
    policy = approving()
    await tool("aws_write").run(
        {"service": "s3", "operation": "DeleteObject", "params": {}},
        context(provider, policy=policy),
    )
    target = policy.seen[0].target
    assert "123456789012" in target and "eu-west-1" in target


async def test_a_privileged_write_demands_the_typed_challenge() -> None:
    provider = FakeProvider({**ALLOWED, "iam:AttachRolePolicy": {}})
    policy = approving()
    await tool("aws_write").run(
        {"service": "iam", "operation": "AttachRolePolicy", "params": {}},
        context(provider, policy=policy),
    )
    request = policy.seen[0]
    assert request.sensitivity is Sensitivity.PRIVILEGED
    assert request.needs_challenge
    assert not request.may_grant_always, "no standing grant on rewriting IAM"


async def test_an_ordinary_change_does_not_demand_one() -> None:
    """A challenge that fires on everything trains people to type through it."""
    provider = FakeProvider({**ALLOWED, "ec2:CreateTags": {}})
    policy = approving()
    await tool("aws_write").run(
        {"service": "ec2", "operation": "CreateTags", "params": {}},
        context(provider, policy=policy),
    )
    assert policy.seen[0].sensitivity is Sensitivity.MUTATE
    assert not policy.seen[0].needs_challenge


async def test_a_protected_account_is_flagged_in_the_prompt() -> None:
    provider = FakeProvider({**ALLOWED, "s3:DeleteObject": {}})
    policy = approving()
    await tool("aws_write").run(
        {"service": "s3", "operation": "DeleteObject", "params": {}},
        context(provider, policy=policy, accounts=("123456789012",)),
    )
    assert policy.seen[0].protected


async def test_the_write_tool_refuses_reads() -> None:
    out = await tool("aws_write").run(
        {"service": "ec2", "operation": "DescribeInstances"}, context(FakeProvider())
    )
    assert out.is_error and "aws_call" in out.content


async def test_the_prompt_carries_the_reason_or_says_there_was_none() -> None:
    provider = FakeProvider({**ALLOWED, "s3:DeleteObject": {}})
    policy = approving()
    await tool("aws_write").run(
        {"service": "s3", "operation": "DeleteObject", "params": {}},
        context(provider, policy=policy),
    )
    assert "no reason given" in policy.seen[0].diff

    policy = approving()
    await tool("aws_write").run(
        {"service": "s3", "operation": "DeleteObject", "params": {}, "reason": "stale export"},
        context(provider, policy=policy),
    )
    assert "stale export" in policy.seen[0].diff


# ------------------------------------------------------------ the switches


@pytest.mark.parametrize(
    ("settings", "service", "operation"),
    [
        (AwsSettings(allow_writes=False), "ec2", "CreateTags"),
        (AwsSettings(allow_iam_writes=False), "iam", "AttachRolePolicy"),
        (AwsSettings(allow_delete=False), "s3", "DeleteObject"),
    ],
)
async def test_a_switch_refuses_at_the_gate(
    settings: AwsSettings, service: str, operation: str
) -> None:
    """These cannot work by withholding a tool: the same aws_write tags a
    volume and rewrites a trust policy."""
    provider = FakeProvider({**ALLOWED, f"{service}:{operation}": {}})
    policy = approving()
    out = await tool("aws_write").run(
        {"service": service, "operation": operation, "params": {}},
        context(provider, policy=policy, settings=settings),
    )
    assert out.is_error and "[cloud.aws]" in out.content
    assert policy.seen == []
    assert f"{service}:{operation}" not in provider.real_calls


def test_allow_writes_off_removes_the_tool_as_well() -> None:
    names = {t.name for t in aws_tools(AwsSettings(allow_writes=False))}
    assert "aws_write" not in names
    assert "aws_call" in names, "reads are unaffected"


def test_only_the_write_tool_is_mutating() -> None:
    mutating = {t.name for t in aws_tools() if not t.read_only}
    assert mutating == {"aws_write"}


# ------------------------------------------------------------ the CLI door


def cli_tool() -> Any:
    from wai.tools.cli import AwsCliTool

    return AwsCliTool()


@pytest.mark.parametrize(
    ("argv", "service", "operation"),
    [
        (["s3", "ls"], "s3", "ListBuckets"),
        (["s3", "rb", "--force"], "s3", "DeleteBucket"),
        (["ec2", "describe-instances"], "ec2", "DescribeInstances"),
        (["ec2", "terminate-instances"], "ec2", "TerminateInstances"),
        (["iam", "list-roles"], "iam", "ListRoles"),
        (["iam", "attach-role-policy"], "iam", "AttachRolePolicy"),
        (["rds", "delete-db-instance"], "rds", "DeleteDBInstance"),
    ],
)
def test_the_cli_and_the_sdk_agree(argv: list[str], service: str, operation: str) -> None:
    """Two answers for terminate-instances depending on which door it came
    through is exactly the gap a gate is supposed not to have. The CLI asks the
    same classifier rather than keeping a second table to drift from."""
    from wai.cloud.aws import classify

    assert cli_tool().classify(argv) is classify(service, operation)


def test_a_bare_aws_invocation_is_treated_as_a_change() -> None:
    assert cli_tool().classify(["s3"]) is Sensitivity.MUTATE


def test_the_model_cannot_retarget_the_aws_cli() -> None:
    """--profile and --region would send the command at another account, and
    the approval prompt would then name the wrong blast radius."""
    from wai.tools.cli import RESERVED_FLAGS

    for flag in ("--profile", "--region", "--endpoint-url"):
        assert flag in RESERVED_FLAGS


# ------------------------------------------------------------- the dashboard


def test_every_aws_dashboard_panel_is_a_read() -> None:
    """The dashboard runs these on open without asking, which is only
    acceptable because none of them can change anything."""
    from wai.tui.screens.dashboard import AWS_PANELS

    by_name = {t.name: t for t in aws_tools()}
    for _title, name, _args in AWS_PANELS:
        assert by_name[name].read_only, f"{name} is not a read"


def test_cost_explorer_is_not_on_a_refreshing_screen() -> None:
    """It bills per request, and `r` is one keypress."""
    from wai.tui.screens.dashboard import AWS_PANELS

    assert "aws_cost" not in {name for _t, name, _a in AWS_PANELS}
