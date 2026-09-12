"""Cloud foundations: classification, redaction, protection, kubeconfig.

The redaction and classification tests are the load-bearing ones. Tool results
are transmitted to whichever LLM provider is active, so a gap here is a
credential leaving the machine.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from wai.cloud import aws as aws_cloud
from wai.cloud import kube as kube_cloud
from wai.cloud.base import (
    INTEGRATIONS,
    CloudTarget,
    ProtectionMode,
    ProtectionRules,
    Sensitivity,
    integration,
)
from wai.cloud.redact import MARKER, redact, redact_text

# ------------------------------------------------------------- classification


@pytest.mark.parametrize(
    ("service", "operation", "expected"),
    [
        ("ec2", "DescribeInstances", Sensitivity.READ),
        ("s3", "ListBuckets", Sensitivity.READ),
        ("s3", "GetObject", Sensitivity.READ),
        ("cloudwatch", "GetMetricData", Sensitivity.READ),
        ("ec2", "TerminateInstances", Sensitivity.MUTATE),
        ("rds", "DeleteDBInstance", Sensitivity.MUTATE),
        ("lambda", "Invoke", Sensitivity.MUTATE),
        ("s3", "PutObject", Sensitivity.MUTATE),
        ("sts", "GetSessionToken", Sensitivity.SENSITIVE_READ),
        ("sts", "AssumeRole", Sensitivity.SENSITIVE_READ),
        ("secretsmanager", "GetSecretValue", Sensitivity.SENSITIVE_READ),
        ("ecr", "GetAuthorizationToken", Sensitivity.SENSITIVE_READ),
        ("ssm", "GetParameter", Sensitivity.SENSITIVE_READ),
        ("ec2", "GetPasswordData", Sensitivity.SENSITIVE_READ),
        ("iam", "CreateAccessKey", Sensitivity.SENSITIVE_READ),
        ("eks", "DescribeCluster", Sensitivity.SENSITIVE_READ),
    ],
)
def test_aws_classification(service: str, operation: str, expected: Sensitivity) -> None:
    assert aws_cloud.classify(service, operation) is expected


def test_unknown_verbs_fail_closed() -> None:
    for operation in ("FrobnicateWidget", "YeetInstance", "Whatever"):
        assert aws_cloud.classify("madeup", operation) is Sensitivity.MUTATE


def test_no_credential_shaped_operation_is_classified_read() -> None:
    """The whole corpus, not a sample: a leak here mints credentials silently."""
    leaked: list[str] = []
    for service in aws_cloud.available_services():
        try:
            operations = aws_cloud.operations(service)
        except Exception:
            continue
        for operation in operations:
            if aws_cloud.classify(service, operation) is not Sensitivity.READ:
                continue
            if any(
                word in operation
                for word in ("Credential", "Token", "Password", "Secret", "Session", "PrivateKey")
            ):
                leaked.append(f"{service}:{operation}")
    assert leaked == [], f"credential-shaped operations classified as plain reads: {leaked[:10]}"


def test_only_approval_free_level_is_read() -> None:
    assert Sensitivity.READ.needs_approval is False
    assert Sensitivity.SENSITIVE_READ.needs_approval is True
    assert Sensitivity.MUTATE.needs_approval is True


def test_describe_operation_reports_required_params_and_docs() -> None:
    described = aws_cloud.describe_operation("ec2", "TerminateInstances")
    assert described["sensitivity"] == "mutate"
    assert [k for k, v in described["parameters"].items() if v["required"]] == ["InstanceIds"]
    assert described["documentation"]
    assert "<" not in described["documentation"], "HTML must be stripped"


def test_python_method_name_conversion() -> None:
    assert aws_cloud.python_method("DescribeInstances") == "describe_instances"
    assert aws_cloud.python_method("GetObject") == "get_object"


@pytest.mark.parametrize(
    ("verb", "kind", "expected"),
    [
        ("get", "Pod", Sensitivity.READ),
        ("list", "Deployment", Sensitivity.READ),
        ("watch", "Pod", Sensitivity.READ),
        ("get", "Secret", Sensitivity.SENSITIVE_READ),
        ("list", "Secret", Sensitivity.SENSITIVE_READ),
        ("delete", "Deployment", Sensitivity.MUTATE),
        ("apply", "Deployment", Sensitivity.MUTATE),
        ("frobnicate", "Pod", Sensitivity.MUTATE),
    ],
)
def test_k8s_classification(verb: str, kind: str, expected: Sensitivity) -> None:
    assert kube_cloud.classify(verb, kind) is expected


# ------------------------------------------------------------------ redaction


def test_kubernetes_secret_is_scrubbed_but_keeps_its_shape() -> None:
    secret = {
        "kind": "Secret",
        "metadata": {"name": "db-creds", "namespace": "default"},
        "data": {"password": "aHVudGVyMg==", "username": "YWRtaW4="},
    }
    out = redact(secret)
    assert out["data"] == {"password": MARKER, "username": MARKER}
    assert out["metadata"]["name"] == "db-creds", "names are not secret"
    assert "aHVudGVyMg==" not in str(out)


def test_configmap_is_left_alone() -> None:
    cm = {"kind": "ConfigMap", "data": {"LOG_LEVEL": "debug", "REPLICAS": "3"}}
    assert redact(cm) == cm


def test_sts_response_is_scrubbed() -> None:
    response = {
        "Credentials": {
            "AccessKeyId": "ASIAEXAMPLE",
            "SecretAccessKey": "wJalrXUtnFEMI",
            "SessionToken": "FQoGZXIvYXdzEBYa",
        }
    }
    out = redact(response)
    assert out["Credentials"]["SecretAccessKey"] == MARKER
    assert out["Credentials"]["SessionToken"] == MARKER
    assert "wJalrXUtnFEMI" not in str(out)


def test_nested_and_listed_secrets_are_reached() -> None:
    payload = {"items": [{"spec": {"env": [{"name": "API_TOKEN", "value": "abc"}]}}]}
    out = redact(payload)
    assert "abc" not in str(out) or out["items"][0]["spec"]["env"][0]["value"] == MARKER


def test_key_name_lists_are_not_mistaken_for_secrets() -> None:
    """`keys` holding names, and `PublicKey`, are not secret material."""
    payload = {"keys": ["password", "username"], "PublicKey": "ssh-rsa AAAA", "KeyId": "k-1"}
    out = redact(payload)
    assert out["keys"] == ["password", "username"]
    assert out["KeyId"] == "k-1"


def test_redaction_can_be_disabled() -> None:
    secret = {"kind": "Secret", "data": {"password": "x"}}
    assert redact(secret, enabled=False) == secret


def test_text_redaction_catches_loose_credentials() -> None:
    text = (
        "export AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMIabcdefgh\n"
        "token ASIAIOSFODNN7EXAMPLE\n"
        "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9\n"
    )
    out = redact_text(text)
    assert "wJalrXUtnFEMIabcdefgh" not in out
    assert "ASIAIOSFODNN7EXAMPLE" not in out
    assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in out


def test_text_redaction_strips_private_keys() -> None:
    pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIEow\n-----END RSA PRIVATE KEY-----"
    assert "MIIEow" not in redact_text(pem)


# ----------------------------------------------------------------- protection


def test_protection_matches_case_insensitively() -> None:
    """A real cluster is as likely to be AKS_EU_PROD as prod-eu."""
    rules = ProtectionRules(patterns=("*prod*",))
    assert rules.matches(CloudTarget("k8s", "AKS_EU_PROD"))
    assert rules.matches(CloudTarget("k8s", "prod-eu-west"))
    assert rules.matches(CloudTarget("k8s", "Production-Cluster"))
    assert not rules.matches(CloudTarget("k8s", "AKS_QAM"))
    assert not rules.matches(CloudTarget("k8s", "staging"))


def test_protection_matches_any_part_of_the_target() -> None:
    rules = ProtectionRules(patterns=("*prod*",))
    assert rules.matches(CloudTarget("k8s", "cluster-a", "eu", "prod-namespace"))
    assert rules.matches(CloudTarget("aws", "111122223333", "prod-region"))


def test_protection_matches_account_ids_exactly() -> None:
    rules = ProtectionRules(accounts=("123456789012",))
    assert rules.matches(CloudTarget("aws", "123456789012", "eu-west-1"))
    assert not rules.matches(CloudTarget("aws", "999988887777", "eu-west-1"))


def test_protection_default_is_confirm_not_deny() -> None:
    assert ProtectionRules().mode is ProtectionMode.CONFIRM


def test_target_renders_the_blast_radius() -> None:
    target = CloudTarget("k8s", "prod-eu", "cluster-1", "payments")
    assert target.render() == "k8s: prod-eu · cluster-1 · payments"


# ---------------------------------------------------------------- kubeconfig


@pytest.fixture
def kubeconfig(tmp_path: Path) -> Path:
    path = tmp_path / "config"
    path.write_text(
        """
apiVersion: v1
kind: Config
current-context: staging
clusters:
- name: prod-cluster
  cluster: {server: https://prod.example.com}
- name: staging-cluster
  cluster: {server: https://staging.example.com}
contexts:
- name: AKS_EU_PROD
  context: {cluster: prod-cluster, user: u1, namespace: payments}
- name: staging
  context: {cluster: staging-cluster, user: u1}
users:
- name: u1
  user: {token: secret-token-value}
""".strip()
    )
    return path


def test_contexts_are_listed_without_touching_the_file(
    kubeconfig: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Selecting a context must never rewrite the user's kubeconfig."""
    monkeypatch.setenv("KUBECONFIG", str(kubeconfig))
    before = kubeconfig.read_bytes()

    contexts, active = kube_cloud.list_contexts()

    assert {c.name for c in contexts} == {"AKS_EU_PROD", "staging"}
    assert active == "staging"
    prod = next(c for c in contexts if c.name == "AKS_EU_PROD")
    assert prod.cluster == "prod-cluster"
    assert prod.namespace == "payments"
    assert kubeconfig.read_bytes() == before, "kubeconfig was modified"


def test_listed_context_targets_are_protection_matched(
    kubeconfig: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KUBECONFIG", str(kubeconfig))
    contexts, _ = kube_cloud.list_contexts()
    rules = ProtectionRules(patterns=("*prod*",))
    protected = {c.name for c in contexts if rules.matches(c.target())}
    assert protected == {"AKS_EU_PROD"}


def test_missing_kubeconfig_is_not_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KUBECONFIG", str(tmp_path / "nope"))
    contexts, active = kube_cloud.list_contexts()
    assert contexts == [] and active is None


def test_malformed_kubeconfig_is_skipped_not_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bad = tmp_path / "bad"
    bad.write_text("this is not: [valid yaml")
    monkeypatch.setenv("KUBECONFIG", str(bad))
    contexts, _ = kube_cloud.list_contexts()
    assert contexts == []


def test_context_target_includes_namespace(
    kubeconfig: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KUBECONFIG", str(kubeconfig))
    contexts, _ = kube_cloud.list_contexts()
    prod = next(c for c in contexts if c.name == "AKS_EU_PROD")
    assert prod.target().render() == "k8s: AKS_EU_PROD · prod-cluster · payments"
    assert prod.target("other").scope == "other"


# --------------------------------------------------------------- integrations


def test_integrations_report_availability_and_install_hints() -> None:
    assert {i.name for i in INTEGRATIONS} == {"k8s", "aws", "azure", "gcp"}
    entry = integration("k8s")
    assert entry is not None
    assert "wai[k8s]" in entry.install_hint


def test_missing_integration_is_reported_not_crashed(monkeypatch: pytest.MonkeyPatch) -> None:
    from wai.cloud.base import Integration

    absent = Integration("nope", "nope", ("definitely_not_a_module",), "Nothing")
    assert absent.available is False
    assert "wai[nope]" in absent.install_hint


def test_name_value_pairs_redact_the_value_not_the_label() -> None:
    """`{"name": "API_TOKEN", "value": ...}` hides the indicator in a sibling.

    The label must survive, or the model cannot tell the pairs apart and will
    report "there are two secrets" instead of naming which is which.
    """
    payload = {
        "env": [
            {"name": "API_TOKEN", "value": "abc123"},
            {"name": "LOG_LEVEL", "value": "debug"},
        ],
        "Tags": [
            {"Key": "db_password", "Value": "hunter2"},
            {"Key": "env", "Value": "prod"},
        ],
    }
    out = redact(payload)
    assert out["env"][0] == {"name": "API_TOKEN", "value": MARKER}
    assert out["env"][1] == {"name": "LOG_LEVEL", "value": "debug"}
    assert out["Tags"][0] == {"Key": "db_password", "Value": MARKER}
    assert out["Tags"][1] == {"Key": "env", "Value": "prod"}


def test_a_bare_key_field_is_still_treated_as_secret() -> None:
    """Outside a name/value pair, `key` may well hold key material."""
    assert redact({"key": "-----BEGIN RSA PRIVATE KEY-----"})["key"] == MARKER
