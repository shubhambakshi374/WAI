from __future__ import annotations

import pytest

from altus.config.loader import config_path, load_config, resolve_profile, save_config
from altus.config.models import Config, Profile, ProviderSettings
from altus.config.secrets import (
    USES_CREDENTIAL_CHAIN,
    credential_status,
    get_api_key,
    require_api_key,
)
from altus.core.errors import ConfigError, CredentialsError


def test_defaults_when_no_file() -> None:
    config = load_config()
    assert not config_path().exists()
    assert config.default_profile == "default"
    assert config.profiles["default"].provider == "anthropic"


def test_save_and_reload_round_trip() -> None:
    config = Config()
    config.profiles["ops"] = Profile(provider="bedrock", model="m", max_tokens=1024)
    config.providers["bedrock"] = ProviderSettings(region="eu-west-1")
    config.default_profile = "ops"
    save_config(config)

    reloaded = load_config()
    assert reloaded.default_profile == "ops"
    assert reloaded.profiles["ops"].max_tokens == 1024
    assert reloaded.providers["bedrock"].region == "eu-west-1"


def test_saved_config_never_contains_a_key() -> None:
    save_config(Config())
    assert "api_key" not in config_path().read_text().lower()


def test_invalid_config_raises_config_error() -> None:
    config_path().parent.mkdir(parents=True, exist_ok=True)
    config_path().write_text('default_profile = "x"\nnonsense_field = 1\n')
    with pytest.raises(ConfigError):
        load_config()


def test_malformed_toml_raises_config_error() -> None:
    config_path().parent.mkdir(parents=True, exist_ok=True)
    config_path().write_text("this is [not toml")
    with pytest.raises(ConfigError):
        load_config()


def test_resolve_unknown_profile_lists_known_ones() -> None:
    with pytest.raises(ConfigError, match="default"):
        resolve_profile(Config(), "missing")


def test_env_beats_keyring(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("altus.config.secrets._keyring_get", lambda _p: "from-keyring")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
    assert get_api_key("anthropic") == "from-env"
    assert credential_status("anthropic").source == "env"


def test_wai_prefixed_env_beats_native(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "native")
    monkeypatch.setenv("ALTUS_ANTHROPIC_API_KEY", "altus")
    assert get_api_key("anthropic") == "altus"


def test_keyring_used_when_env_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("altus.config.secrets._keyring_get", lambda _p: "stored")
    assert get_api_key("openai") == "stored"
    assert credential_status("openai").source == "keyring"


def test_keyring_failure_is_not_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing or locked keyring backend must degrade, not crash."""

    def boom(_service: str, _user: str) -> str:
        raise RuntimeError("no backend")

    monkeypatch.undo()
    monkeypatch.setattr("keyring.get_password", boom)
    assert get_api_key("mistral") is None


def test_require_api_key_error_names_the_fix() -> None:
    with pytest.raises(CredentialsError, match="altus config set-key"):
        require_api_key("openai")


def test_bedrock_never_uses_the_keyring(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bedrock must go through the boto3 credential chain, not an API key."""
    monkeypatch.setattr("altus.config.secrets._keyring_get", lambda _p: "should-be-ignored")
    monkeypatch.setenv("ALTUS_BEDROCK_API_KEY", "should-be-ignored")
    assert "bedrock" in USES_CREDENTIAL_CHAIN
    assert get_api_key("bedrock") is None
    assert credential_status("bedrock").source in {"aws-chain", "none"}


def test_starter_config_is_not_empty_and_reloads() -> None:
    """`config init` must give the user something to edit.

    save_config(Config()) would write an empty file, since it serializes only
    non-default values.
    """
    from altus.config.loader import write_starter_config

    path = write_starter_config()
    text = path.read_text()
    assert "[profiles.default]" in text
    assert "#" in text, "starter config should be annotated"
    assert load_config().profiles["default"].model == "claude-sonnet-5"


def test_starter_config_documents_bedrock_credential_chain() -> None:
    from altus.config.loader import STARTER_CONFIG

    assert "AWS credential chain" in STARTER_CONFIG
    assert "api_key" not in STARTER_CONFIG.lower()


def test_the_test_harness_cannot_reach_the_real_keyring() -> None:
    """A regression guard with teeth: an earlier ad-hoc run overwrote a real
    stored API key because only keyring *reads* were isolated, not writes."""
    import keyring

    from altus.config.secrets import KEYRING_SERVICE, delete_api_key, set_api_key

    set_api_key("openai", "sk-not-real")
    assert keyring.get_password(KEYRING_SERVICE, "openai") == "sk-not-real"
    assert delete_api_key("openai") is True

    # The stand-in is a plain dict, so nothing here touched the OS keychain.
    assert keyring.get_password.__qualname__ != "get_password"
