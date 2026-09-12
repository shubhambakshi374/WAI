"""Provider adapters. No ``textual`` imports.

Adapters are imported lazily by the registry so that starting the TUI does not
pay the import cost of every SDK.
"""

from wai.providers.base import BaseProvider, Provider, ProviderCapabilities
from wai.providers.registry import (
    PROVIDER_NAMES,
    catalog_for,
    create_provider,
    known_models,
)

__all__ = [
    "PROVIDER_NAMES",
    "BaseProvider",
    "Provider",
    "ProviderCapabilities",
    "catalog_for",
    "create_provider",
    "known_models",
]
