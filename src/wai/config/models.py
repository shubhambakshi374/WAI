"""Configuration schema. Secrets are never represented here."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ProviderSettings(BaseModel):
    """Non-secret, per-provider connection settings."""

    model_config = ConfigDict(extra="forbid")

    base_url: str | None = None
    api_version: str | None = None
    """Azure AI Foundry only."""
    region: str | None = None
    """AWS Bedrock only."""
    aws_profile: str | None = None
    """AWS Bedrock only; names a profile in the standard boto3 credential chain."""
    timeout: float = 600.0
    max_retries: int = 4
    extra_headers: dict[str, str] = Field(default_factory=dict)


class Profile(BaseModel):
    """A named provider + model + sampling combination."""

    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str
    max_tokens: int = 4096
    temperature: float | None = None
    system: str | None = None


class UISettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    theme: str = "textual-dark"
    stream_flush_ms: int = 50
    """How often streamed text is flushed into the transcript widget."""
    show_reasoning: bool = True


DEFAULT_PROFILE_NAME = "default"


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default_profile: str = DEFAULT_PROFILE_NAME
    profiles: dict[str, Profile] = Field(
        default_factory=lambda: {
            DEFAULT_PROFILE_NAME: Profile(
                provider="anthropic", model="claude-sonnet-5", max_tokens=8192
            )
        }
    )
    providers: dict[str, ProviderSettings] = Field(default_factory=dict)
    ui: UISettings = Field(default_factory=UISettings)

    def provider_settings(self, name: str) -> ProviderSettings:
        return self.providers.get(name, ProviderSettings())
