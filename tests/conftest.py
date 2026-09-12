from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Never read or write the developer's real config, data or keyring."""
    monkeypatch.setenv("WAI_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("WAI_DATA_DIR", str(tmp_path / "data"))
    for key in list(os.environ):
        if key.endswith("_API_KEY"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("wai.config.secrets._keyring_get", lambda _provider: None)
