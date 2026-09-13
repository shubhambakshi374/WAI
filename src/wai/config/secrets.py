"""Credential resolution: environment first, OS keyring second.

Environment wins so CI and headless runs work without a keyring backend. No
secret is ever written to the config file.

Bedrock is deliberately exempt from this path: it uses the standard boto3
credential chain (env, shared profiles, instance/IRSA roles), because forcing
an API key there would break the normal way a DevOps tool gets deployed.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from wai.core.errors import CredentialsError

log = logging.getLogger(__name__)

KEYRING_SERVICE = "wai"

NATIVE_ENV_VARS: dict[str, tuple[str, ...]] = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "openrouter": ("OPENROUTER_API_KEY",),
    "deepseek": ("DEEPSEEK_API_KEY",),
    "mistral": ("MISTRAL_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "azure_foundry": ("AZURE_OPENAI_API_KEY", "AZURE_AI_API_KEY"),
    "bedrock": (),
    "local": (),
}

USES_CREDENTIAL_CHAIN = frozenset({"bedrock"})
"""Providers that authenticate through their cloud SDK, not an API key."""

NEEDS_NO_API_KEY = frozenset({"bedrock", "local"})
"""Asking these for a key is the wrong question: Bedrock uses the AWS chain,
and a self-hosted endpoint usually has no auth at all."""


@dataclass(frozen=True)
class CredentialStatus:
    provider: str
    available: bool
    source: str
    detail: str = ""


def _wai_env_var(provider: str) -> str:
    return f"WAI_{provider.upper()}_API_KEY"


def get_api_key(provider: str) -> str | None:
    """Resolve a key: ``WAI_<PROVIDER>_API_KEY`` -> native env var -> keyring."""
    if provider in USES_CREDENTIAL_CHAIN:
        # Bedrock authenticates through the AWS chain; a key here would be
        # ignored by the SDK anyway, so returning one would only mislead.
        return None
    if provider in NEEDS_NO_API_KEY:
        # A self-hosted endpoint usually has no auth, but may have one --- so
        # honour an explicitly exported key and never the keyring.
        return _env_key(provider)
    for var in (_wai_env_var(provider), *NATIVE_ENV_VARS.get(provider, ())):
        value = os.environ.get(var)
        if value:
            return value
    return _keyring_get(provider)


def _env_key(provider: str) -> str | None:
    """A key only if one was explicitly exported; never from the keyring."""
    return os.environ.get(_wai_env_var(provider))


def require_api_key(provider: str) -> str:
    key = get_api_key(provider)
    if not key:
        raise CredentialsError(
            f"no API key for {provider!r}. Set {_wai_env_var(provider)} "
            f"or run: wai config set-key {provider}",
            provider=provider,
        )
    return key


def set_api_key(provider: str, key: str) -> None:
    import keyring

    keyring.set_password(KEYRING_SERVICE, provider, key)


def delete_api_key(provider: str) -> bool:
    import keyring
    from keyring.errors import PasswordDeleteError

    try:
        keyring.delete_password(KEYRING_SERVICE, provider)
    except PasswordDeleteError:
        return False
    return True


def _keyring_get(provider: str) -> str | None:
    try:
        import keyring

        return keyring.get_password(KEYRING_SERVICE, provider)
    except Exception as exc:  # no backend, locked keychain, DBus missing
        log.debug("keyring unavailable for %s: %s", provider, exc)
        return None


def credential_status(provider: str) -> CredentialStatus:
    """Report whether a provider can authenticate, without revealing the key."""
    if provider == "local":
        # There is no key to look for. Whether it is *usable* depends on a
        # profile naming an endpoint, which this function cannot see --- so
        # report the honest thing and let callers ask providers.local.
        return CredentialStatus(provider, False, "", "needs an endpoint — /setup local")
    if provider in USES_CREDENTIAL_CHAIN:
        return _aws_status(provider)
    for var in (_wai_env_var(provider), *NATIVE_ENV_VARS.get(provider, ())):
        if os.environ.get(var):
            return CredentialStatus(provider, True, "env", var)
    if _keyring_get(provider):
        return CredentialStatus(provider, True, "keyring", KEYRING_SERVICE)
    return CredentialStatus(provider, False, "none", "run: wai config set-key " + provider)


def _aws_status(provider: str) -> CredentialStatus:
    try:
        import botocore.session

        session = botocore.session.get_session()
        credentials = session.get_credentials()
    except Exception as exc:
        return CredentialStatus(provider, False, "none", f"boto3 unavailable: {exc}")
    if credentials is None:
        return CredentialStatus(provider, False, "none", "no AWS credentials found")
    return CredentialStatus(provider, True, "aws-chain", credentials.method)


PROVIDER_NAMES_HINT = ", ".join(sorted(NATIVE_ENV_VARS))
"""Human-readable provider list, for error messages, without importing the registry."""
