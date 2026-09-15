"""Error hierarchy. Every adapter maps its SDK's exceptions onto these."""

from __future__ import annotations


class AltusError(Exception):
    """Base for every error Altus raises. ``retryable`` drives ``core.retry``."""

    retryable: bool = False

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        retry_after: float | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.retry_after = retry_after
        self.__cause__ = cause

    def __str__(self) -> str:
        return f"[{self.provider}] {self.message}" if self.provider else self.message


class ConfigError(AltusError):
    """Malformed or missing configuration."""


class CredentialsError(AltusError):
    """No usable credential could be resolved for a provider."""


class WorkspaceError(AltusError):
    """The workspace refused an operation."""


class PathNotAllowed(WorkspaceError):
    """A path resolved outside every allowed root, or onto a denied file."""

    def __init__(self, message: str, *, path: str, reason: str) -> None:
        super().__init__(message)
        self.path = path
        self.reason = reason


class ToolError(AltusError):
    """A tool failed. Surfaced to the model as an error tool_result."""


class ProviderError(AltusError):
    """A provider rejected or failed the request."""


class AuthenticationError(ProviderError):
    """Credentials were present but rejected."""


class InvalidRequestError(ProviderError):
    """The request was malformed or unsupported. Never retried."""


class ModelNotFoundError(InvalidRequestError):
    """The model id is unknown to this provider, or not enabled on the account."""


class ContextLengthError(InvalidRequestError):
    """The conversation exceeds the model's context window."""


class ContentFilterError(ProviderError):
    """The provider blocked the request or response on safety grounds."""


class RateLimitError(ProviderError):
    retryable = True


class TransientProviderError(ProviderError):
    """5xx, connection reset, timeout — worth another attempt."""

    retryable = True


class StreamInterrupted(AltusError):
    """The stream ended before the provider signalled completion."""
