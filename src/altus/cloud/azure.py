"""Azure: operation classification and the ARM session.

Azure needs a generic invoke for the opposite reason AWS did. AWS has 431
services and 19,189 distinct operations, so per-operation tools were
impossible. Azure Resource Manager *is already* a generic API --- every
management operation is an HTTP verb on a resource path --- so wrapping it in
two hundred ``azure-mgmt-*`` packages would fragment one uniform surface into
many partial ones. This talks to ARM directly: ``azure.identity`` for the
token, ``httpx`` for the call.

The one thing a model cannot guess is ``api-version``. It is mandatory, it
differs per resource type, and omitting it is an error rather than a default.
``GET /subscriptions/{id}/providers/{ns}`` answers it, and that call --- cached
--- is Azure's equivalent of botocore's shipped service models.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any

from altus.cloud.base import CloudTarget, Sensitivity

ARM = "https://management.azure.com"
ARM_SCOPE = "https://management.azure.com/.default"

#: Pinned where the endpoint is a fixed, well-known one rather than a resource
#: type whose version has to be resolved.
PROVIDERS_API = "2021-04-01"
PROVIDER_OPERATIONS_API = "2022-04-01"
SUBSCRIPTIONS_API = "2022-12-01"
RESOURCE_GRAPH_API = "2022-10-01"
LOCKS_API = "2016-09-01"
CHECK_ACCESS_API = "2018-09-01-preview"
DEPLOYMENTS_API = "2021-04-01"
COST_API = "2023-03-01"

#: Results returned from one call. ARM pages with ``nextLink`` and will walk a
#: whole subscription given the chance; the model pays for every row.
MAX_RESULTS = 500

# ------------------------------------------------------------- the grammar

#: The entire vocabulary. Azure states the verb instead of leaving it to be
#: inferred from an English word, which is why this classifier can parse where
#: the AWS one has to guess.
VERBS: frozenset[str] = frozenset({"read", "write", "action", "delete"})

#: Namespaces where a *write* decides who may do what, or touches key
#: material. Reading them is ordinary --- listing role assignments is how you
#: understand a subscription --- but changing one is the Azure equivalent of
#: editing a ClusterRoleBinding.
#:
#: ``Microsoft.Authorization`` covers resource locks as well as role
#: assignments, and that is deliberate: removing the lock is how you get around
#: the lock, so it cannot be a lesser operation than what it protects.
PRIVILEGED_NAMESPACES: frozenset[str] = frozenset(
    {
        "microsoft.authorization",
        "microsoft.managedidentity",
        "microsoft.aad",
        "microsoft.azureactivedirectory",
        "microsoft.keyvault",
        "microsoft.management",
        "microsoft.subscription",
        "microsoft.billing",
        "microsoft.blueprint",
        "microsoft.customerlockbox",
        "microsoft.graph",
    }
)

#: Action names that hand back live credentials. These are Azure's
#: ``sts:GetSessionToken``: shaped like a read, classified like one, and
#: capable of minting access that outlives the conversation.
CREDENTIAL_ACTION = re.compile(
    r"^(list|get)\w*?"
    r"(keys?|secrets?|credentials?|password|token|sas|connectionstrings?)$",
    re.IGNORECASE,
)

#: The ones the pattern cannot reach, because their names say nothing about
#: what they return. ``publishxml`` hands back deployment credentials for an
#: App Service in plain text.
CREDENTIAL_ACTIONS: frozenset[str] = frozenset(
    {
        "publishxml",
        "publishingcredentials",
        "listsecrets",
        "listkeyvalue",
        "listconnectioninfo",
        "listsyncfunctiontriggers",
        "listquerykeys",
        "listadminkeys",
        "listclusteradmincredential",
        "listclusterusercredential",
        "listclustermonitoringusercredential",
        "getsharedaccesssignature",
    }
)

#: Types whose *metadata* is already secret-adjacent, so even a plain read is
#: worth classifying up. The values themselves live on the data plane, which
#: this module does not reach at all.
SECRET_TYPES = re.compile(r"(secret|credential|password|certificate|/keys)", re.IGNORECASE)

#: Actions that mint or rotate key material. Distinct from the credential
#: reads above: these change the credential rather than reveal it, so they
#: break whatever is using the old one.
KEY_MATERIAL = re.compile(r"^(regenerate|rotate|renew|recover|purge)", re.IGNORECASE)

#: Type-path fragments where a write or delete changes who can reach the
#: network. A single call here can put a private workload on the internet.
#:
#: Deliberately tight. Storage's public-blob-access is a property of an
#: ordinary ``storageAccounts/write``, and classifying every storage account
#: write as privileged would fire the typed challenge on creating one --- a
#: challenge that fires on everything trains people to type through it.
EXPOSURE_TYPES: tuple[str, ...] = (
    "networksecuritygroups",
    "securityrules",
    "publicipaddresses",
    "firewallrules",
    "azurefirewalls",
    "routetables",
    "privateendpointconnections",
    "virtualnetworkgateways",
)

#: Types where deletion is unrecoverable rather than merely inconvenient ---
#: the object holds data, or everything inside it goes with it.
STATEFUL_TYPES = re.compile(
    r"(virtualmachines|disks|snapshots|databases|servers|storageaccounts|blobservices"
    r"|containers|fileservices|shares|vaults|managedclusters|registries|namespaces"
    r"|redis|clusters|instances|accounts|workspaces|resourcegroups|subscriptions"
    r"|sqlpools|elasticpools|databaseaccounts|caches|queues|topics|eventhubs"
    r"|factories|sites|components|backupvaults|images|galleries)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Operation:
    """An RBAC operation string, taken apart.

    ``Microsoft.Compute/virtualMachines/start/action`` is namespace
    ``Microsoft.Compute``, type ``virtualMachines``, verb ``action`` and action
    name ``start``. The action name is the part that carries the meaning, and
    the reason ``action`` cannot be classified as one thing: it covers both
    ``start`` and ``listKeys``.
    """

    namespace: str
    type_path: str
    verb: str
    action: str = ""

    @property
    def wildcard(self) -> bool:
        return "*" in self.namespace or "*" in self.type_path or "*" in self.verb


def parse_operation(operation: str) -> Operation | None:
    """Split an RBAC string, or None when it is not one."""
    parts = [p for p in str(operation).strip().strip("/").split("/") if p]
    if not parts:
        return None
    if parts == ["*"]:
        return Operation(namespace="*", type_path="*", verb="*")
    verb = parts[-1].casefold()
    if verb == "action":
        if len(parts) < 3:
            return None
        return Operation(parts[0], "/".join(parts[1:-2]), "action", parts[-2])
    if verb not in VERBS and verb != "*":
        return None
    return Operation(parts[0], "/".join(parts[1:-1]), verb)


def scope_level(scope: str) -> str:
    """How wide a scope path is.

    Blast radius in Azure is a function of scope as well as operation: the same
    ``delete`` removes one virtual machine at a resource scope and an entire
    subscription one level up. AWS had no equivalent axis.
    """
    lowered = str(scope or "").casefold().strip("/")
    if not lowered:
        return "unknown"
    if "managementgroups/" in lowered:
        return "management-group"
    if "/providers/" in lowered:
        return "resource"
    if "resourcegroups/" in lowered:
        return "resource-group"
    if lowered.startswith("subscriptions/"):
        return "subscription"
    return "tenant"


#: Scopes at which a change is not about one object.
WIDE_SCOPES: frozenset[str] = frozenset({"tenant", "management-group", "subscription"})


def classify(operation: str, scope: str = "", *, data_action: bool = False) -> Sensitivity:
    """Sensitivity of one operation at one scope.

    Precedence runs highest-first, because an operation is often several things
    at once and the most dangerous reading is the one that should win:
    ``Microsoft.Authorization/roleAssignments/delete`` is both an authorization
    write and a destruction.

    An operation string that does not parse fails closed to PRIVILEGED. Azure
    ships new resource providers constantly, and a shape nobody anticipated is
    exactly where guessing low is unrecoverable.
    """
    parsed = parse_operation(operation)
    if parsed is None:
        return Sensitivity.PRIVILEGED
    if parsed.wildcard:
        # A wildcard in a role definition covers everything under it, so it is
        # as dangerous as the most dangerous thing it matches.
        return Sensitivity.PRIVILEGED

    namespace = parsed.namespace.casefold()
    types = parsed.type_path.casefold()
    verb = parsed.verb
    action = parsed.action
    writes = verb in {"write", "delete"} or (verb == "action" and not _is_credential_read(action))

    if verb == "action" and KEY_MATERIAL.match(action):
        return Sensitivity.PRIVILEGED
    if writes and namespace in PRIVILEGED_NAMESPACES:
        return Sensitivity.PRIVILEGED
    if writes and scope_level(scope) in WIDE_SCOPES:
        return Sensitivity.PRIVILEGED
    if writes and any(fragment in types for fragment in EXPOSURE_TYPES):
        return Sensitivity.PRIVILEGED
    if verb == "delete" and STATEFUL_TYPES.search(types):
        return Sensitivity.PRIVILEGED

    if verb == "action":
        found = Sensitivity.SENSITIVE_READ if _is_credential_read(action) else Sensitivity.MUTATE
        return _bump(found) if data_action else found
    if verb == "read":
        found = (
            Sensitivity.SENSITIVE_READ
            if SECRET_TYPES.search(parsed.type_path)
            else Sensitivity.READ
        )
        return _bump(found) if data_action else found

    # write and plain delete
    return _bump(Sensitivity.MUTATE) if data_action else Sensitivity.MUTATE


def _is_credential_read(action: str) -> bool:
    if not action:
        return False
    return action.casefold() in CREDENTIAL_ACTIONS or bool(CREDENTIAL_ACTION.match(action))


def _bump(found: Sensitivity) -> Sensitivity:
    """One level stricter. A data action touches the contents rather than the
    container, which is a different question from the one ARM's verb answers."""
    order = [
        Sensitivity.READ,
        Sensitivity.SENSITIVE_READ,
        Sensitivity.MUTATE,
        Sensitivity.PRIVILEGED,
    ]
    return order[min(order.index(found) + 1, len(order) - 1)]


# ---------------------------------------------------------------- resource ids

RESOURCE_ID = re.compile(
    r"^/subscriptions/(?P<subscription>[^/]+)"
    r"(?:/resourceGroups/(?P<group>[^/]+))?"
    r"(?:/providers/(?P<namespace>[^/]+)/(?P<rest>.+))?$",
    re.IGNORECASE,
)


def parse_resource_id(resource_id: str) -> dict[str, str]:
    """Subscription, group, namespace, type and name from an ARM id.

    Returns empty strings rather than raising: an id that does not parse is a
    thing to report to the model, not an exception to abort a turn with.
    """
    match = RESOURCE_ID.match(str(resource_id or "").rstrip("/"))
    if match is None:
        return {"subscription": "", "group": "", "namespace": "", "type": "", "name": ""}
    rest = (match.group("rest") or "").split("/")
    # A type path alternates type/name, so the types are the even positions.
    types = rest[0::2]
    names = rest[1::2]
    return {
        "subscription": match.group("subscription") or "",
        "group": match.group("group") or "",
        "namespace": match.group("namespace") or "",
        "type": "/".join(types),
        "name": names[-1] if names else "",
    }


def operation_for(resource_id: str, verb: str, action: str = "") -> str:
    """The RBAC operation string a call against this id would need.

    This is what makes ``azure_can_i`` and every preflight possible: ARM takes
    a URL, RBAC talks about operations, and something has to translate.
    """
    parts = parse_resource_id(resource_id)
    if not parts["namespace"] or not parts["type"]:
        return ""
    base = f"{parts['namespace']}/{parts['type']}"
    return f"{base}/{action}/action" if verb == "action" and action else f"{base}/{verb}"


def target_for(subscription: str, region: str = "", group: str = "") -> CloudTarget:
    return CloudTarget(
        cloud="azure",
        context=subscription or "unknown-subscription",
        location=region,
        scope=group,
    )


# ------------------------------------------------------------------ the session


class ArmError(RuntimeError):
    """An error ARM reported, with its own message rather than a status code.

    ARM replies to a failure with ``{"error": {"code": ..., "message": ...}}``
    and the message is almost always the actionable part --- "the api-version
    is not supported for this type", not "400".
    """

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(f"{code or status}: {message}" if message else f"HTTP {status}")
        self.status = status
        self.code = code


@dataclass
class AzureProvider:
    """A credential, a client and the api-version cache, built on first use.

    Mirrors ``AwsProvider`` and ``K8sProvider``: acquiring a token is a network
    call and building a credential chain walks the filesystem, so a session
    that never mentions Azure should do neither. ``reset()`` drops everything,
    and is what makes switching subscription safe --- a cached token is bound
    to the tenant it was issued for.
    """

    subscription: str = ""
    tenant: str | None = None
    max_results: int = MAX_RESULTS
    _credential: Any = field(default=None, repr=False)
    _client: Any = field(default=None, repr=False)
    _token: tuple[str, float] | None = field(default=None, repr=False)
    _api_versions: dict[str, dict[str, str]] = field(default_factory=dict, repr=False)
    _identity: dict[str, str] | None = field(default=None, repr=False)

    def reset(self) -> None:
        """Drop everything cached. Called when the subscription or tenant
        changes: reusing a token issued for the tenant just left would send the
        next call somewhere the user did not approve."""
        self._credential = None
        self._token = None
        self._api_versions.clear()
        self._identity = None

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def credential(self) -> Any:
        if self._credential is None:
            from azure.identity import DefaultAzureCredential

            self._credential = DefaultAzureCredential(exclude_interactive_browser_credential=True)
        return self._credential

    def _http(self) -> Any:
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client

    async def token(self) -> str:
        """A bearer token for ARM, refreshed a minute before it expires."""
        import time

        if self._token is not None and self._token[1] - 60 > time.time():
            return self._token[0]

        def _acquire() -> tuple[str, float]:
            got = self.credential().get_token(ARM_SCOPE)
            return got.token, float(got.expires_on)

        self._token = await asyncio.to_thread(_acquire)
        return self._token[0]

    # ------------------------------------------------------------- one request

    async def call(
        self,
        method: str,
        path: str,
        *,
        api_version: str = "",
        body: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
        limit: int = 0,
    ) -> dict[str, Any]:
        """One ARM request, following ``nextLink`` for collections.

        Returns the response body with a ``_truncated`` marker when the cap was
        hit. Silently returning the first page would have the model reason
        about a partial answer as though it were the whole one --- the same
        argument the AWS paginator makes.
        """
        cap = limit or self.max_results
        query = dict(params or {})
        if api_version:
            query["api-version"] = api_version
        url = path if path.startswith("http") else f"{ARM}{path}"

        first = await self._request(method, url, body, query)
        values = first.get("value")
        if not isinstance(values, list) or "nextLink" not in first:
            return first

        merged: list[Any] = list(values[:cap])
        link = first.get("nextLink")
        truncated = len(values) > cap
        while link and len(merged) < cap and not truncated:
            page = await self._request("GET", str(link), None, {})
            rows = page.get("value") or []
            merged.extend(rows[: cap - len(merged)])
            truncated = len(merged) >= cap and bool(page.get("nextLink"))
            link = page.get("nextLink")

        out = {k: v for k, v in first.items() if k not in {"value", "nextLink"}}
        out["value"] = merged
        if truncated:
            out["_truncated"] = f"stopped at {cap} results"
        return out

    async def _request(
        self, method: str, url: str, body: dict[str, Any] | None, params: dict[str, str]
    ) -> dict[str, Any]:
        _status, _headers, payload = await self._send(method, url, body, params)
        return payload

    async def _send(
        self, method: str, url: str, body: dict[str, Any] | None, params: dict[str, str]
    ) -> tuple[int, dict[str, str], dict[str, Any]]:
        """The raw exchange. Headers are returned because long-running
        operations carry their poll URL in one, and nowhere else."""
        token = await self.token()
        response = await self._http().request(
            method.upper(),
            url,
            params=params or None,
            json=body,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        )
        headers = {key.casefold(): value for key, value in response.headers.items()}
        if response.status_code == 204 or not (response.content or b"").strip():
            return response.status_code, headers, {"status": response.status_code}
        try:
            payload = response.json()
        except ValueError:
            payload = {"body": response.text}
        if response.status_code >= 400:
            error = payload.get("error") if isinstance(payload, dict) else None
            error = error if isinstance(error, dict) else {}
            raise ArmError(
                response.status_code,
                str(error.get("code", "")),
                str(error.get("message", "")) or str(payload)[:400],
            )
        if not isinstance(payload, dict):
            payload = {"value": payload}
        return response.status_code, headers, payload

    #: Statuses a long-running operation reports while it is still going.
    PENDING: frozenset[str] = frozenset({"inprogress", "running", "accepted", "notstarted"})

    async def call_lro(
        self,
        method: str,
        path: str,
        *,
        api_version: str = "",
        body: dict[str, Any] | None = None,
        wait_seconds: float = 120.0,
        interval: float = 2.0,
    ) -> dict[str, Any]:
        """A request that ARM may answer asynchronously, followed to its end.

        What-If is the one that forces this: ARM accepts it with a 202 and a
        poll URL, so a caller that only reads the immediate response gets an
        empty acknowledgement where the diff should be. Returning that as the
        preflight would put "no changes" in an approval prompt for a change
        that has not been evaluated yet --- the exact false confidence the
        whole gate exists to avoid.
        """
        import time

        query = {"api-version": api_version} if api_version else {}
        url = path if path.startswith("http") else f"{ARM}{path}"
        status, headers, payload = await self._send(method, url, body, query)
        poll = headers.get("azure-asyncoperation") or headers.get("location") or ""
        if status < 202 or not poll:
            return payload

        deadline = time.monotonic() + wait_seconds
        while time.monotonic() < deadline:
            await asyncio.sleep(interval)
            _status, _headers, payload = await self._send("GET", poll, None, {})
            state = str(payload.get("status", "")).casefold()
            if state and state not in self.PENDING:
                return payload
            if not state:
                return payload
        raise ArmError(
            408,
            "Timeout",
            f"the operation was still running after {wait_seconds:g}s; Altus stopped waiting "
            "rather than report a result it does not have",
        )

    async def what_if(
        self,
        group_scope: str,
        resource: dict[str, Any],
        *,
        mode: str = "Incremental",
        wait_seconds: float = 120.0,
    ) -> dict[str, Any]:
        """A server-side, property-level diff of one resource.

        The closest any cloud gets to ``kubectl diff``, and the reason the
        Azure gate can promise more than the AWS one. A generic ARM PUT body
        *is* a resource definition, so it can be wrapped in a one-resource
        template and genuinely evaluated against what is deployed.

        Incremental mode on purpose: Complete mode would report deleting
        everything in the scope that the template does not mention, which for a
        one-resource template is the entire resource group.
        """
        name = f"altus-whatif-{int(asyncio.get_running_loop().time() * 1000) % 1_000_000}"
        return await self.call_lro(
            "POST",
            f"/{group_scope.strip('/')}/providers/Microsoft.Resources/deployments/{name}/whatIf",
            api_version=DEPLOYMENTS_API,
            body={
                "properties": {
                    "mode": mode,
                    "template": {
                        "$schema": (
                            "https://schema.management.azure.com/schemas/2019-04-01/"
                            "deploymentTemplate.json#"
                        ),
                        "contentVersion": "1.0.0.0",
                        "resources": [resource],
                    },
                    "parameters": {},
                }
            },
            wait_seconds=wait_seconds,
        )

    # --------------------------------------------------------- introspection

    async def providers(self, namespace: str = "") -> dict[str, Any]:
        """Resource providers and the api-versions each type supports."""
        subscription = self.require_subscription()
        path = f"/subscriptions/{subscription}/providers"
        if namespace:
            path += f"/{namespace}"
        return await self.call("GET", path, api_version=PROVIDERS_API)

    async def provider_operations(self, namespace: str) -> dict[str, Any]:
        """The RBAC operations a provider defines, with their descriptions.

        Azure's answer to botocore's service models, and the reason
        ``azure_explain`` can hand over an exact contract instead of leaving
        the model to guess. Unlike botocore's, it is a network call --- which
        is why the classification tests sweep a generated grammar rather than
        a shipped corpus.
        """
        return await self.call(
            "GET",
            f"/providers/Microsoft.Authorization/providerOperations/{namespace}",
            api_version=PROVIDER_OPERATIONS_API,
            params={"$expand": "resourceTypes"},
        )

    async def api_version_for(self, namespace: str, resource_type: str) -> str:
        """The version to send for one resource type.

        The single call everything else depends on. Prefers a stable version:
        a preview one can be withdrawn, and a call pinned to a withdrawn
        version fails in a way that looks like the resource is gone.
        """
        key = namespace.casefold()
        if key not in self._api_versions:
            payload = await self.providers(namespace)
            found: dict[str, str] = {}
            for entry in payload.get("resourceTypes") or []:
                versions = [str(v) for v in entry.get("apiVersions") or []]
                stable = [v for v in versions if "preview" not in v]
                if versions:
                    found[str(entry.get("resourceType", "")).casefold()] = (
                        stable[0] if stable else versions[0]
                    )
            self._api_versions[key] = found

        wanted = resource_type.casefold()
        table = self._api_versions[key]
        if wanted in table:
            return table[wanted]
        raise ArmError(
            404,
            "UnknownResourceType",
            f"{namespace}/{resource_type} is not a type this subscription's "
            f"providers offer. Call azure_providers to see what {namespace} has.",
        )

    def require_subscription(self) -> str:
        if not self.subscription:
            raise ArmError(
                400,
                "NoSubscription",
                "no subscription is selected. Call azure_subscriptions, then /azure sub <id>.",
            )
        return self.subscription

    # -------------------------------------------------------------- identity

    async def whoami(self) -> dict[str, str]:
        """Who this token says we are, read from the token itself.

        Decoded locally rather than asked of Microsoft Graph: the claims are
        already in hand, Graph is a second audience needing a second token, and
        an identity panel should not cost a round trip. The token is not
        *verified* here --- it is our own, and nothing is being authorised on
        the strength of it.
        """
        if self._identity is None:
            claims = _token_claims(await self.token())
            self._identity = {
                "tenant": str(claims.get("tid", "")),
                "principal": str(
                    claims.get("upn") or claims.get("unique_name") or claims.get("appid") or ""
                ),
                "object_id": str(claims.get("oid", "")),
                "subscription": self.subscription,
            }
        return self._identity

    async def subscriptions(self) -> list[dict[str, str]]:
        payload = await self.call("GET", "/subscriptions", api_version=SUBSCRIPTIONS_API)
        return [
            {
                "id": str(item.get("subscriptionId", "")),
                "name": str(item.get("displayName", "")),
                "state": str(item.get("state", "")),
                "tenant": str(item.get("tenantId", "")),
            }
            for item in payload.get("value") or []
        ]

    # -------------------------------------------------------- resource graph

    async def graph(self, query: str, subscriptions: list[str] | None = None) -> dict[str, Any]:
        """Resource Graph, KQL over every resource at once.

        AWS needed a loop of per-service describes to answer "what have we
        got". This is one call, and it is free.
        """
        targets = subscriptions or [self.require_subscription()]
        return await self.call(
            "POST",
            "/providers/Microsoft.ResourceGraph/resources",
            api_version=RESOURCE_GRAPH_API,
            body={
                "subscriptions": targets,
                "query": query,
                "options": {"resultFormat": "objectArray", "$top": self.max_results},
            },
        )

    # ------------------------------------------------------------- preflight

    async def locks(self, scope: str) -> list[dict[str, Any]]:
        """Locks that apply at or above a scope.

        Azure's own answer to "would this delete actually work", and a
        mechanism AWS has no equivalent of at all. ``atScope()`` walks up the
        hierarchy, so a lock on the resource group is found from the resource.
        """
        payload = await self.call(
            "GET",
            f"/{scope.strip('/')}/providers/Microsoft.Authorization/locks",
            api_version=LOCKS_API,
            params={"$filter": "atScope()"},
        )
        return [
            {
                "name": str(item.get("name", "")),
                "level": str((item.get("properties") or {}).get("level", "")),
                "notes": str((item.get("properties") or {}).get("notes", "")),
                "scope": str(item.get("id", "")),
            }
            for item in payload.get("value") or []
        ]

    async def check_access(self, scope: str, actions: list[str]) -> list[dict[str, str]]:
        """RBAC's "may I" --- the twin of k8s_can_i and SimulatePrincipalPolicy.

        Unlike IAM's simulate, asking about yourself needs no extra permission,
        so this answers far more often than the AWS preflight could.
        """
        identity = await self.whoami()
        out: list[dict[str, str]] = []
        for action in actions:
            payload = await self.call(
                "POST",
                f"/{scope.strip('/')}/providers/Microsoft.Authorization/checkAccess",
                api_version=CHECK_ACCESS_API,
                body={
                    "Subject": {"ObjectId": identity["object_id"]},
                    "Actions": [{"Id": action, "IsDataAction": False}],
                },
            )
            decisions = payload.get("AccessDecisions") or payload.get("accessDecisions") or []
            verdict = ""
            if decisions:
                first = decisions[0]
                verdict = str(first.get("AccessDecision") or first.get("accessDecision") or "")
            out.append({"action": action, "decision": verdict or "unknown"})
        return out


def _token_claims(token: str) -> dict[str, Any]:
    """The payload of a JWT, decoded but not verified.

    Not verified on purpose: this is our own token and nothing is authorised on
    the strength of what it says. A malformed one yields no claims rather than
    an exception, because a missing display name is not worth failing a call
    over.
    """
    import base64
    import json

    parts = token.split(".")
    if len(parts) < 2:
        return {}
    padded = parts[1] + "=" * (-len(parts[1]) % 4)
    try:
        return dict(json.loads(base64.urlsafe_b64decode(padded)))
    except Exception:
        return {}
