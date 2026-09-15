"""Strip secret material out of cloud responses before the model sees them.

This is not defence in depth --- it is the primary control. Tool results are
sent verbatim to whichever of the eight LLM providers is active, so an
unredacted ``k8s_get(kind=Secret)`` ships base64 credentials to a third party
and into their logs. There is no undo for that.

Redaction is visible: the value becomes a marker, so the model knows a field
existed and can say "there is a password here" rather than hallucinating one.
"""

from __future__ import annotations

import re
from typing import Any

MARKER = "«redacted by wai»"

SECRET_KEY_PATTERN = re.compile(
    r"(secret|password|passwd|token|credential|private[-_]?key|client[-_]?secret"
    r"|access[-_]?key|api[-_]?key|session[-_]?token|bearer|authorization|certificate"
    r"|\bkey\b|kubeconfig|connection[-_]?string|sas|signature)",
    re.IGNORECASE,
)

#: Whole subtrees that are secret regardless of key names inside them.
SECRET_CONTAINERS = frozenset({"data", "stringData", "binaryData"})

#: Kinds whose entire payload is credential material.
SECRET_KINDS = frozenset({"Secret"})

#: `{"name": "API_TOKEN", "value": "..."}` --- the secret indicator sits in a
#: sibling value, not the key, so key-matching alone walks straight past it.
#: This shape is everywhere: container env vars, AWS tags, ARM parameters.
NAME_FIELDS = ("name", "Name", "key", "Key")
VALUE_FIELDS = ("value", "Value", "stringValue", "secretValue")

_SAFE_KEYS = frozenset(
    {
        "keys",  # a list of key *names* is not secret
        "publickey",
        "public_key",
        "keyid",
        "key_id",
        "keyarn",
        "keyname",
        "key_name",
        "keyring",
        "secretname",
        "secret_name",
        "secretarn",
        "tokencount",
        "keypairname",
    }
)


def _is_secret_key(key: str) -> bool:
    if key.casefold() in _SAFE_KEYS:
        return False
    return bool(SECRET_KEY_PATTERN.search(key))


def _pair_label(value: dict[Any, Any]) -> str | None:
    """The label of a name/value pair, or None when this is not one."""
    if not any(field in value for field in VALUE_FIELDS):
        return None
    for field in NAME_FIELDS:
        label = value.get(field)
        if isinstance(label, str):
            return label
    return None


def redact(value: Any, *, enabled: bool = True, _in_secret: bool = False) -> Any:
    """Recursively replace secret-looking values. Structure is preserved."""
    if not enabled:
        return value

    if isinstance(value, dict):
        kind = value.get("kind")
        inside = _in_secret or (isinstance(kind, str) and kind in SECRET_KINDS)
        out: dict[Any, Any] = {}
        label = _pair_label(value)
        is_pair = label is not None
        labelled = is_pair and _is_secret_key(label or "")
        for key, item in value.items():
            name = str(key)
            if inside and name in SECRET_CONTAINERS and isinstance(item, dict):
                # A Kubernetes Secret's data block: keep the key names, drop
                # every value, so the model can still reason about shape.
                out[key] = dict.fromkeys(item, MARKER)
            elif is_pair and name in NAME_FIELDS:
                # The label is metadata --- "which tag is this" --- and blanking
                # it would leave the model unable to tell the pairs apart. Only
                # the paired value is secret. A bare `key` field outside a
                # name/value pair still falls through to the normal rule below.
                out[key] = item
            elif (labelled and name in VALUE_FIELDS and not isinstance(item, dict | list)) or (
                _is_secret_key(name) and not isinstance(item, dict | list)
            ):
                out[key] = MARKER if item not in (None, "") else item
            else:
                out[key] = redact(item, enabled=True, _in_secret=inside)
        return out

    if isinstance(value, list):
        return [redact(item, enabled=True, _in_secret=_in_secret) for item in value]

    return value


def redact_text(text: str, *, enabled: bool = True) -> str:
    """Last-resort scrub for text that never went through a structured form."""
    if not enabled:
        return text
    patterns = (
        r"(?i)\b(aws_secret_access_key|aws_session_token)\s*[=:]\s*\S+",
        r"\bASIA[0-9A-Z]{16}\b",
        r"\bAKIA[0-9A-Z]{16}\b",
        r"(?i)\bbearer\s+[A-Za-z0-9._\-]{20,}",
        r"\beyJ[A-Za-z0-9._\-]{20,}",  # JWT
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]+?-----END [A-Z ]*PRIVATE KEY-----",
    )
    for pattern in patterns:
        text = re.sub(pattern, MARKER, text)
    return _ASSIGNMENT.sub(_scrub_assignment, text)


#: `NAME=value` at the start of a line. This is the shape of `env`, of a
#: .env file, and of a properties file --- and `k8s_exec -- env` is one of the
#: first things anyone reaches for when debugging a pod, which would otherwise
#: ship every secret the pod was given straight to the model provider.
_ASSIGNMENT = re.compile(r"(?m)^([ \t]*[A-Za-z_][A-Za-z0-9_.\-]*)([ \t]*[=:][ \t]*)(\S.*)$")


def _scrub_assignment(match: re.Match[str]) -> str:
    """Judged by the same vocabulary as the structured path, so PATH and
    HOSTNAME survive while DB_PASSWORD and AWS_SESSION_TOKEN do not."""
    key, separator, _value = match.groups()
    if not _is_secret_key(key.strip()):
        return match.group(0)
    return f"{key}{separator}{MARKER}"
