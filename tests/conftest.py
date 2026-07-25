"""Shared fixtures for Ollama Cloud web search plugin tests.

The plugin is a user-installed plugin (not bundled in the Hermes source
tree), so we load it via importlib.util.spec_from_file_location -- the
same mechanism the PluginManager uses for user plugins.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

PLUGIN_DIR = Path(__file__).resolve().parent.parent
INIT_PATH = PLUGIN_DIR / "__init__.py"

# Hermes source tree -- needed for the WebSearchProvider ABC.
_HERMES_SRC = os.environ.get("HERMES_SRC", "/home/hermes/.hermes/hermes-agent")


def _load_plugin_module():
    """Load the plugin __init__.py as a module and return it.

    Uses the same spec_from_file_location pattern as PluginManager.
    Caches in sys.modules so repeated imports return the same object.
    """
    if "hermes_plugins.web_ollama_cloud" in sys.modules:
        return sys.modules["hermes_plugins.web_ollama_cloud"]

    if _HERMES_SRC not in sys.path:
        sys.path.insert(0, _HERMES_SRC)

    if "hermes_plugins" not in sys.modules:
        import types
        ns_pkg = types.ModuleType("hermes_plugins")
        ns_pkg.__path__ = []  # type: ignore[attr-defined]
        ns_pkg.__package__ = "hermes_plugins"
        sys.modules["hermes_plugins"] = ns_pkg

    spec = importlib.util.spec_from_file_location(
        "hermes_plugins.web_ollama_cloud",
        str(INIT_PATH),
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create module spec for {INIT_PATH}")

    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "hermes_plugins.web_ollama_cloud"
    mod.__path__ = [str(PLUGIN_DIR)]  # type: ignore[attr-defined]
    sys.modules["hermes_plugins.web_ollama_cloud"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def plugin_module():
    """Load and cache the plugin module for the test session."""
    return _load_plugin_module()


@pytest.fixture(scope="session")
def provider_cls(plugin_module):
    """The OllamaCloudWebSearchProvider class."""
    return plugin_module.OllamaCloudWebSearchProvider


@pytest.fixture
def provider(provider_cls):
    """A fresh provider instance per test."""
    return provider_cls()


@pytest.fixture
def require_api_key():
    """Skip the test if OLLAMA_API_KEY is not set.

    These tests make real API calls against Ollama's web search/fetch
    endpoints. No mocking -- we exercise the actual interface.
    """
    from agent.web_search_provider import get_provider_env

    key = get_provider_env("OLLAMA_API_KEY")
    if not key:
        pytest.skip("OLLAMA_API_KEY not set -- live API tests skipped")
    return key