"""XDG-aware config loading and saving."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

import tomli_w
from platformdirs import user_config_dir, user_data_dir
from pydantic import ValidationError

from altus.config.models import Config, Profile
from altus.core.errors import ConfigError

APP_NAME = "altus"

#: What the application was called before. Kept so a rename does not silently
#: strand an existing install's config, sessions and cached tokens in a
#: directory nothing looks in any more.
LEGACY_APP_NAME = "wai"


def config_dir() -> Path:
    """Honours ``ALTUS_CONFIG_DIR`` so tests and CI never touch the real config."""
    override = _override("CONFIG_DIR")
    if override:
        return Path(override)
    return _preferred(user_config_dir(APP_NAME), user_config_dir(LEGACY_APP_NAME))


def config_path() -> Path:
    return config_dir() / "config.toml"


def data_dir() -> Path:
    override = _override("DATA_DIR")
    if override:
        return Path(override)
    return _preferred(user_data_dir(APP_NAME), user_data_dir(LEGACY_APP_NAME))


def _override(name: str) -> str:
    """``ALTUS_<NAME>``, or the pre-rename ``WAI_<NAME>``.

    The old spelling stays honoured because it lives in people's shell
    profiles and CI files, and a rename is not a reason to break those.
    """
    return os.environ.get(f"{APP_NAME.upper()}_{name}") or os.environ.get(
        f"{LEGACY_APP_NAME.upper()}_{name}", ""
    )


def _preferred(current: str, legacy: str) -> Path:
    """The new location, unless only the old one exists.

    Deliberately not a migration: moving someone's directory as a side effect
    of importing a module is a surprise, and a half-finished move is worse
    than either location. An existing install keeps working exactly where it
    is; a fresh one starts in the new place.
    """
    new = Path(current)
    old = Path(legacy)
    if not new.exists() and old.exists():
        return old
    return new


def sessions_dir() -> Path:
    return data_dir() / "sessions"


def load_config() -> Config:
    """Load config, falling back to defaults when the file does not exist."""
    path = config_path()
    if not path.exists():
        return Config()
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"could not read {path}: {exc}", cause=exc) from exc
    try:
        return Config.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"invalid config at {path}:\n{exc}", cause=exc) from exc


def save_config(config: Config) -> Path:
    """Write config atomically. Never contains secrets."""
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = config.model_dump(mode="json", exclude_defaults=True, exclude_none=True)
    tmp = path.with_suffix(".toml.tmp")
    tmp.write_text(tomli_w.dumps(payload), encoding="utf-8")
    tmp.replace(path)
    return path


STARTER_CONFIG = """\
# Altus configuration. This file never contains secrets --- API keys live in the
# OS keyring (altus config set-key <provider>) or the environment.
# Check what resolves with: altus config doctor

default_profile = "default"

[profiles.default]
provider = "anthropic"
model = "claude-sonnet-5"
max_tokens = 8192

# A second profile, selected with: altus chat --profile ops
# [profiles.ops]
# provider = "bedrock"
# model = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
# system = "You are a careful SRE. Explain before you act."

# Bedrock uses the standard AWS credential chain, not an API key.
# [providers.bedrock]
# region = "eu-west-1"
# aws_profile = "prod"

# Azure AI Foundry needs your resource endpoint; "model" is a deployment name.
# [providers.azure_foundry]
# base_url = "https://my-resource.openai.azure.com"
# api_version = "2024-10-21"

# [ui]
# theme = "textual-dark"
# stream_flush_ms = 50
# show_reasoning = true
"""


def write_starter_config() -> Path:
    """Write an annotated starter file.

    Deliberately not ``save_config(Config())``: that serializes only
    non-default values, which for a fresh config is nothing at all, leaving
    the user an empty file to edit.
    """
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(STARTER_CONFIG, encoding="utf-8")
    return path


def resolve_profile(config: Config, name: str | None = None) -> tuple[str, Profile]:
    """Return the named profile, or the configured default."""
    wanted = name or config.default_profile
    profile = config.profiles.get(wanted)
    if profile is None:
        known = ", ".join(sorted(config.profiles)) or "none"
        raise ConfigError(f"no such profile {wanted!r} (known profiles: {known})")
    return wanted, profile
