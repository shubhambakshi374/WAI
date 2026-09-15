"""Configuration schema. Secrets are never represented here."""

from __future__ import annotations

from typing import Literal

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
    base_url: str = ""
    """For self-hosted endpoints. Required by the `local` provider."""
    api_key_env: str = ""
    """Environment variable holding the key, for endpoints that need one."""
    supports_tools: bool | None = None
    """Override capability detection. None means ask the server, or assume yes."""


class ProtectedSettings(BaseModel):
    """Targets too important to change on a single keypress."""

    model_config = ConfigDict(extra="forbid")

    patterns: list[str] = Field(default_factory=lambda: ["*prod*", "*production*"])
    """Matched case-insensitively against context, cluster, region and namespace."""
    accounts: list[str] = Field(default_factory=list)
    mode: Literal["confirm", "deny"] = "confirm"


class K8sSettings(BaseModel):
    """Which classes of Kubernetes capability this machine offers at all.

    A `false` here means the tool is never registered, so the model is not told
    it exists. That is deliberately stronger than refusing at call time: a tool
    the model cannot see costs no context and cannot be argued into being used.
    """

    model_config = ConfigDict(extra="forbid")

    allow_exec: bool = True
    """exec, attach and cp. Running a command inside a container is the single
    largest escalation here: whatever the container can reach, so can a chat
    message."""
    allow_port_forward: bool = True
    """Opens a tunnel from this machine into the cluster network."""
    allow_node_lifecycle: bool = True
    """cordon, uncordon, taint, drain."""
    allow_rbac_writes: bool = True
    """Creating or changing Roles, Bindings, ServiceAccounts and CSRs."""
    allow_cli: bool = True
    """kubectl, helm and kustomize, when the native tools cannot express it."""
    exec_timeout: int = 60
    """Seconds before an exec is cut off and what it printed so far returned."""


class AwsSettings(BaseModel):
    """Which classes of AWS capability this machine offers.

    Unlike the Kubernetes switches, most of these cannot work by withholding a
    tool: the same ``aws_write`` tags a volume and rewrites a trust policy. They
    are checked at the approval gate instead.
    """

    model_config = ConfigDict(extra="forbid")

    allow_writes: bool = True
    """Any mutating call at all."""
    allow_iam_writes: bool = True
    """Writes to IAM, STS, Organizations, KMS and the other identity services."""
    allow_delete: bool = True
    """Terminate, delete, destroy --- the irreversible verbs."""
    allow_cost_explorer: bool = True
    """Cost Explorer bills per request, so it can be switched off entirely."""
    max_results: int = 500
    """Results returned from one call. Paginators will happily walk a hundred
    thousand objects, and the model pays for every one of them."""


class CloudSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    secret_redaction: bool = True
    """Scrub secret material from tool output before it reaches the model.

    Tool results are transmitted to the active LLM provider, so turning this
    off means Kubernetes Secrets and AWS session tokens leave your machine.
    """
    kubeconfigs: list[str] = Field(default_factory=list)
    """Extra kubeconfig files, added with /kube add."""
    kube_context: str | None = None
    """The context WAI uses."""
    kube_context_scope: Literal["wai", "global"] = "wai"
    """`wai` keeps the selection to this tool. `global` also writes
    current-context to your kubeconfig, like `kubectl config use-context` ---
    which retargets every other terminal you have open, so it is opt-in."""
    default_region: str | None = None
    dry_run_first: bool = True
    cli_fallback: bool = True
    cli_allowlist: list[str] = Field(
        default_factory=lambda: ["kubectl", "aws", "az", "gcloud", "helm", "terraform"]
    )
    protected: ProtectedSettings = Field(default_factory=ProtectedSettings)
    k8s: K8sSettings = Field(default_factory=K8sSettings)
    aws: AwsSettings = Field(default_factory=AwsSettings)


class UISettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    theme: str = "textual-dark"
    stream_flush_ms: int = 50
    """How often streamed text is flushed into the transcript widget."""
    show_reasoning: bool = True
    graphics: Literal["auto", "image", "cells", "off"] = "auto"
    """How visuals are drawn.

    `auto` follows the terminal: images where it speaks Kitty's protocol or
    Sixel, box-drawing characters everywhere else. The explicit values are a
    ceiling rather than a floor --- asking for images on a terminal that cannot
    show them still yields cells, because the alternative is a broken screen.
    """
    graphics_font: str = ""
    """Absolute path to a TTF for drawn labels. Empty means discover one."""


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
    cloud: CloudSettings = Field(default_factory=CloudSettings)
    ui: UISettings = Field(default_factory=UISettings)

    def provider_settings(self, name: str) -> ProviderSettings:
        return self.providers.get(name, ProviderSettings())
