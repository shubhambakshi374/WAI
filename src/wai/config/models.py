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


class WorkspaceSettings(BaseModel):
    """The filesystem context tools operate in. See wai/workspace.py."""

    model_config = ConfigDict(extra="forbid")

    extra_roots: list[str] = Field(default_factory=list)
    """Opt-in paths outside the working directory, e.g. /etc/nginx."""
    deny_secrets: bool = True
    """Block credential-shaped files even inside an allowed root."""


class ToolSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    max_iterations: int = 25
    max_file_bytes: int = 262_144
    max_output_bytes: int = 102_400
    """Total tool-result bytes per loop iteration."""


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
    workspace: WorkspaceSettings = Field(default_factory=WorkspaceSettings)
    tools: ToolSettings = Field(default_factory=ToolSettings)
    ui: UISettings = Field(default_factory=UISettings)

    def provider_settings(self, name: str) -> ProviderSettings:
        return self.providers.get(name, ProviderSettings())
