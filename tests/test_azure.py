"""Azure. No subscription is contacted.

The classification tests are the interesting half. botocore ships all 19,189
AWS operations on disk, so ``test_cloud.py`` can sweep them offline; Azure's
equivalent catalog is a network call, so there is nothing to sweep in CI.

The replacement is not weaker. Azure's operation grammar is regular and
closed --- ``{namespace}/{type}/{read|write|delete|action}`` --- so the sweep
below generates the cross product and asserts invariants over all of it. A
curated table pins the operations that actually matter.
"""

from __future__ import annotations

import itertools

import pytest

from wai.cloud import azure as az
from wai.cloud.base import Sensitivity

RESOURCE_SCOPE = (
    "/subscriptions/0000/resourceGroups/rg/providers/Microsoft.Compute/virtualMachines/vm"
)
GROUP_SCOPE = "/subscriptions/0000/resourceGroups/rg"
SUBSCRIPTION_SCOPE = "/subscriptions/0000"
MANAGEMENT_SCOPE = "/providers/Microsoft.Management/managementGroups/mg"


# ------------------------------------------------------------------- grammar


@pytest.mark.parametrize(
    ("operation", "namespace", "type_path", "verb", "action"),
    [
        (
            "Microsoft.Compute/virtualMachines/read",
            "Microsoft.Compute",
            "virtualMachines",
            "read",
            "",
        ),
        (
            "Microsoft.Compute/virtualMachines/start/action",
            "Microsoft.Compute",
            "virtualMachines",
            "action",
            "start",
        ),
        (
            "Microsoft.Network/virtualNetworks/subnets/write",
            "Microsoft.Network",
            "virtualNetworks/subnets",
            "write",
            "",
        ),
    ],
)
def test_parse_operation(
    operation: str, namespace: str, type_path: str, verb: str, action: str
) -> None:
    parsed = az.parse_operation(operation)
    assert parsed is not None
    assert (parsed.namespace, parsed.type_path, parsed.verb, parsed.action) == (
        namespace,
        type_path,
        verb,
        action,
    )


@pytest.mark.parametrize(
    "operation", ["", "/", "Microsoft.Compute", "Microsoft.Compute/vm/frobnicate"]
)
def test_unparseable_operations(operation: str) -> None:
    assert az.parse_operation(operation) is None
    assert az.classify(operation) is Sensitivity.PRIVILEGED


@pytest.mark.parametrize(
    ("scope", "level"),
    [
        (RESOURCE_SCOPE, "resource"),
        (GROUP_SCOPE, "resource-group"),
        (SUBSCRIPTION_SCOPE, "subscription"),
        (MANAGEMENT_SCOPE, "management-group"),
        ("", "unknown"),
    ],
)
def test_scope_level(scope: str, level: str) -> None:
    assert az.scope_level(scope) == level


# -------------------------------------------------------------- the classifier


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        # Ordinary reads stay reads --- listing role assignments is how you
        # understand a subscription.
        ("Microsoft.Compute/virtualMachines/read", Sensitivity.READ),
        ("Microsoft.Authorization/roleAssignments/read", Sensitivity.READ),
        ("Microsoft.Storage/storageAccounts/read", Sensitivity.READ),
        # Credential-shaped actions. Azure's sts:GetSessionToken.
        ("Microsoft.Storage/storageAccounts/listKeys/action", Sensitivity.SENSITIVE_READ),
        (
            "Microsoft.DocumentDB/databaseAccounts/listConnectionStrings/action",
            Sensitivity.SENSITIVE_READ,
        ),
        (
            "Microsoft.ContainerService/managedClusters/listClusterAdminCredential/action",
            Sensitivity.SENSITIVE_READ,
        ),
        ("Microsoft.Web/sites/publishxml/action", Sensitivity.SENSITIVE_READ),
        ("Microsoft.KeyVault/vaults/secrets/read", Sensitivity.SENSITIVE_READ),
        # Authorization writes.
        ("Microsoft.Authorization/roleAssignments/write", Sensitivity.PRIVILEGED),
        ("Microsoft.Authorization/roleDefinitions/delete", Sensitivity.PRIVILEGED),
        ("Microsoft.Authorization/locks/delete", Sensitivity.PRIVILEGED),
        ("Microsoft.ManagedIdentity/userAssignedIdentities/write", Sensitivity.PRIVILEGED),
        ("Microsoft.KeyVault/vaults/write", Sensitivity.PRIVILEGED),
        # Key material.
        ("Microsoft.Storage/storageAccounts/regenerateKey/action", Sensitivity.PRIVILEGED),
        # Network exposure.
        (
            "Microsoft.Network/networkSecurityGroups/securityRules/write",
            Sensitivity.PRIVILEGED,
        ),
        ("Microsoft.Network/publicIPAddresses/write", Sensitivity.PRIVILEGED),
        ("Microsoft.Sql/servers/firewallRules/write", Sensitivity.PRIVILEGED),
        # Irreversible destruction of something stateful.
        ("Microsoft.Sql/servers/databases/delete", Sensitivity.PRIVILEGED),
        ("Microsoft.Compute/virtualMachines/delete", Sensitivity.PRIVILEGED),
        # Ordinary changes.
        ("Microsoft.Compute/virtualMachines/write", Sensitivity.MUTATE),
        ("Microsoft.Compute/virtualMachines/start/action", Sensitivity.MUTATE),
        ("Microsoft.Insights/diagnosticSettings/delete", Sensitivity.MUTATE),
        ("Microsoft.Resources/tags/write", Sensitivity.MUTATE),
    ],
)
def test_classification(operation: str, expected: Sensitivity) -> None:
    assert az.classify(operation, RESOURCE_SCOPE) is expected


def test_wildcards_are_privileged() -> None:
    """A wildcard in a role definition covers everything under it."""
    for operation in ("*", "Microsoft.Compute/*", "*/read", "Microsoft.Compute/virtualMachines/*"):
        assert az.classify(operation, RESOURCE_SCOPE) is Sensitivity.PRIVILEGED


def test_scope_escalates_the_same_operation() -> None:
    """The axis AWS did not have: blast radius is a function of scope too.

    The same delete removes one diagnostic setting at a resource and every
    diagnostic setting in a subscription one level up.
    """
    operation = "Microsoft.Insights/diagnosticSettings/delete"
    assert az.classify(operation, RESOURCE_SCOPE) is Sensitivity.MUTATE
    assert az.classify(operation, GROUP_SCOPE) is Sensitivity.MUTATE
    assert az.classify(operation, SUBSCRIPTION_SCOPE) is Sensitivity.PRIVILEGED
    assert az.classify(operation, MANAGEMENT_SCOPE) is Sensitivity.PRIVILEGED


def test_data_actions_classify_one_level_stricter() -> None:
    """A data action reads the contents, not the container."""
    operation = "Microsoft.Storage/storageAccounts/blobServices/containers/blobs/read"
    assert az.classify(operation, RESOURCE_SCOPE) is Sensitivity.READ
    assert az.classify(operation, RESOURCE_SCOPE, data_action=True) is Sensitivity.SENSITIVE_READ


# ------------------------------------------------------------------ the sweep

NAMESPACES = (
    "Microsoft.Compute",
    "Microsoft.Storage",
    "Microsoft.Network",
    "Microsoft.Sql",
    "Microsoft.Web",
    "Microsoft.KeyVault",
    "Microsoft.Authorization",
    "Microsoft.ManagedIdentity",
    "Microsoft.ContainerService",
    "Microsoft.DocumentDB",
    "Microsoft.Insights",
    "Microsoft.Resources",
    "Microsoft.EventHub",
    "Microsoft.RecoveryServices",
    "Microsoft.AAD",
    "Microsoft.Management",
)

TYPES = (
    "virtualMachines",
    "storageAccounts",
    "networkSecurityGroups/securityRules",
    "servers/databases",
    "sites",
    "vaults/secrets",
    "roleAssignments",
    "locks",
    "managedClusters",
    "diagnosticSettings",
    "tags",
    "namespaces",
    "userAssignedIdentities",
    "publicIPAddresses",
)

ACTIONS = ("start", "restart", "listKeys", "listConnectionStrings", "regenerateKey", "failover")

SCOPES = (RESOURCE_SCOPE, GROUP_SCOPE, SUBSCRIPTION_SCOPE, MANAGEMENT_SCOPE, "")


def _corpus() -> list[tuple[str, str]]:
    """Every operation the grammar can express over the known vocabulary."""
    out: list[tuple[str, str]] = []
    for namespace, type_path, scope in itertools.product(NAMESPACES, TYPES, SCOPES):
        for verb in ("read", "write", "delete"):
            out.append((f"{namespace}/{type_path}/{verb}", scope))
        for action in ACTIONS:
            out.append((f"{namespace}/{type_path}/{action}/action", scope))
    return out


def test_the_corpus_is_worth_sweeping() -> None:
    assert len(_corpus()) > 5_000


def test_no_change_is_ever_classified_read() -> None:
    """The invariant the AWS corpus sweep exists to protect, restated.

    A write, a delete, or a non-credential action must never come back READ ---
    that would be a change made with nobody asked.
    """
    for operation, scope in _corpus():
        parsed = az.parse_operation(operation)
        assert parsed is not None
        if parsed.verb == "read":
            continue
        found = az.classify(operation, scope)
        assert found is not Sensitivity.READ, f"{operation} at {scope or '(no scope)'}"


def test_every_authorization_write_is_privileged() -> None:
    """Never merely MUTATE: these decide who may do what, and one of them
    removes the lock that protects everything else."""
    for operation, scope in _corpus():
        parsed = az.parse_operation(operation)
        assert parsed is not None
        if parsed.namespace.casefold() not in az.PRIVILEGED_NAMESPACES:
            continue
        if parsed.verb == "read" or az._is_credential_read(parsed.action):
            continue
        assert az.classify(operation, scope) is Sensitivity.PRIVILEGED, operation


def test_every_credential_action_is_at_least_sensitive() -> None:
    """Shaped like a read, and capable of minting access that outlives the
    conversation. Classifying one as a plain read is the whole failure mode."""
    for operation, scope in _corpus():
        parsed = az.parse_operation(operation)
        assert parsed is not None
        if not az._is_credential_read(parsed.action):
            continue
        assert az.classify(operation, scope) is not Sensitivity.READ, operation


def test_the_common_case_is_not_escalated() -> None:
    """A challenge that fires on everything trains people to type through it.

    The AWS work found exactly this: every ``Delete*`` came out privileged,
    2,281 of them including DeleteTag and DeleteAlarm, so the typed
    confirmation stopped meaning anything. The check that matters is not a
    ratio --- it is that an ordinary change to an ordinary resource, at the
    scope real calls are made at, stays MUTATE.
    """
    ordinary = [
        "Microsoft.Compute/virtualMachines/write",
        "Microsoft.Web/sites/write",
        "Microsoft.Insights/diagnosticSettings/write",
        "Microsoft.Resources/tags/write",
        "Microsoft.Compute/virtualMachines/start/action",
        "Microsoft.Compute/virtualMachines/restart/action",
        "Microsoft.Web/sites/restart/action",
        "Microsoft.Insights/diagnosticSettings/delete",
    ]
    for operation in ordinary:
        assert az.classify(operation, RESOURCE_SCOPE) is Sensitivity.MUTATE, operation


def test_the_corpus_reaches_every_level() -> None:
    """A sanity check on the sweep itself, not on the real world.

    The corpus pairs every namespace with every type, so it contains plenty of
    combinations that do not exist --- ``Microsoft.Compute/roleAssignments``
    and the like --- and its privileged share is an artifact of that, not a
    rate anyone would see. What it is good for is proving no level is
    unreachable and no branch is dead.
    """
    counts: dict[Sensitivity, int] = dict.fromkeys(Sensitivity, 0)
    for operation, scope in _corpus():
        counts[az.classify(operation, scope)] += 1
    assert all(counts[level] > 0 for level in Sensitivity), counts


# ------------------------------------------------------------- resource ids


@pytest.mark.parametrize(
    ("resource_id", "expected"),
    [
        (
            RESOURCE_SCOPE,
            {
                "subscription": "0000",
                "group": "rg",
                "namespace": "Microsoft.Compute",
                "type": "virtualMachines",
                "name": "vm",
            },
        ),
        (
            "/subscriptions/1/resourceGroups/g/providers/Microsoft.Network/virtualNetworks/v/subnets/s",
            {
                "subscription": "1",
                "group": "g",
                "namespace": "Microsoft.Network",
                "type": "virtualNetworks/subnets",
                "name": "s",
            },
        ),
        (
            SUBSCRIPTION_SCOPE,
            {"subscription": "0000", "group": "", "namespace": "", "type": "", "name": ""},
        ),
    ],
)
def test_parse_resource_id(resource_id: str, expected: dict[str, str]) -> None:
    assert az.parse_resource_id(resource_id) == expected


def test_parse_resource_id_survives_nonsense() -> None:
    """A bad id is something to report to the model, not an exception that
    aborts a turn it could have recovered from."""
    assert az.parse_resource_id("not-an-id")["subscription"] == ""


@pytest.mark.parametrize(
    ("resource_id", "verb", "action", "expected"),
    [
        (RESOURCE_SCOPE, "delete", "", "Microsoft.Compute/virtualMachines/delete"),
        (RESOURCE_SCOPE, "action", "start", "Microsoft.Compute/virtualMachines/start/action"),
        (SUBSCRIPTION_SCOPE, "read", "", ""),
    ],
)
def test_operation_for(resource_id: str, verb: str, action: str, expected: str) -> None:
    """ARM takes a URL and RBAC talks about operations; something has to
    translate, and every preflight depends on it."""
    assert az.operation_for(resource_id, verb, action) == expected


def test_target_for_feeds_the_protection_rules() -> None:
    """Protecting a subscription by id starts working with no new config ---
    ProtectedSettings.accounts already holds opaque ids."""
    from wai.cloud.base import ProtectionRules

    rules = ProtectionRules.build([], ["0000"], "confirm")
    assert rules.matches(az.target_for("0000", "westeurope", "rg"))
    assert not rules.matches(az.target_for("1111", "westeurope", "rg"))
