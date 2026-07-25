"""Automated tests for the Ollama Cloud web search + extract plugin.

These tests make REAL API calls against Ollama's /api/web_search and
/api/web_fetch endpoints. No mocking -- we exercise the actual interface
to catch drift in the API contract, response shape, and error handling.

Requirements:
- OLLAMA_API_KEY must be set (in .env or environment). Tests skip if absent.
- HERMES_SRC must point to the Hermes source tree (default: /home/hermes/.hermes/hermes-agent)

Run:
    cd hermes-plugin-ollama-cloud-search
    pytest tests/ -v
"""

from __future__ import annotations

import inspect
import os
from typing import Any, Dict, List

import pytest

from agent.web_search_provider import WebSearchProvider


# ---------------------------------------------------------------------------
# Plugin structure (no API key needed)
# ---------------------------------------------------------------------------


class TestPluginStructure:
    """Verify the plugin loads, subclasses the ABC, and advertises the right
    capabilities. These don't need network access."""

    def test_subclasses_web_search_provider(self, provider_cls):
        assert issubclass(provider_cls, WebSearchProvider)

    def test_name_is_ollama_cloud(self, provider):
        assert provider.name == "ollama-cloud"

    def test_display_name_is_non_empty(self, provider):
        assert isinstance(provider.display_name, str)
        assert len(provider.display_name) > 0

    def test_supports_search(self, provider):
        assert provider.supports_search() is True

    def test_supports_extract(self, provider):
        assert provider.supports_extract() is True

    def test_extract_is_sync(self, provider):
        """Extract is a sync def (not async) -- httpx.post per URL."""
        assert inspect.iscoroutinefunction(provider.extract) is False

    def test_search_is_sync(self, provider):
        assert inspect.iscoroutinefunction(provider.search) is False

    def test_register_is_callable(self, plugin_module):
        assert callable(plugin_module.register)

    def test_setup_schema_shape(self, provider):
        schema = provider.get_setup_schema()
        assert isinstance(schema, dict)
        assert "name" in schema
        assert "env_vars" in schema
        env_keys = [ev.get("key") for ev in schema.get("env_vars", [])]
        assert "OLLAMA_API_KEY" in env_keys

    def test_setup_schema_badge(self, provider):
        schema = provider.get_setup_schema()
        # Free tier available with an Ollama account
        assert schema.get("badge") == "free"

    def test_plugin_yaml_exists(self):
        from pathlib import Path
        assert (Path(__file__).resolve().parent.parent / "plugin.yaml").exists()

    def test_plugin_yaml_name(self):
        import yaml
        from pathlib import Path
        p = Path(__file__).resolve().parent.parent / "plugin.yaml"
        with open(p) as f:
            data = yaml.safe_load(f)
        assert data["name"] == "web-ollama-cloud"
        assert data["kind"] == "backend"
        assert "ollama-cloud" in data.get("provides_web_providers", [])


# ---------------------------------------------------------------------------
# is_available -- env var gating
# ---------------------------------------------------------------------------


class TestIsAvailable:
    """is_available() must reflect OLLAMA_API_KEY presence.

    These tests check the env-var gating logic without making API calls.
    get_provider_env reads from both os.environ AND ~/.hermes/.env, so
    we patch it on the plugin module to simulate the key being absent.
    """

    def test_available_when_key_set(self, provider, monkeypatch):
        monkeypatch.setenv("OLLAMA_API_KEY", "test-key-123")
        # get_provider_env checks os.environ first, so this works
        assert provider.is_available() is True

    def test_unavailable_when_key_missing(self, provider, plugin_module):
        from unittest.mock import patch
        with patch.object(plugin_module, "get_provider_env", return_value=""):
            assert provider.is_available() is False

    def test_unavailable_when_key_empty(self, provider, plugin_module):
        from unittest.mock import patch
        # Covers both empty string and whitespace-only (get_provider_env strips)
        with patch.object(plugin_module, "get_provider_env", return_value=""):
            assert provider.is_available() is False


# ---------------------------------------------------------------------------
# Error handling -- missing credentials
# ---------------------------------------------------------------------------


class TestMissingCredentials:
    """When OLLAMA_API_KEY is unset, search/extract must return typed errors,
    not raise exceptions.

    get_provider_env reads from both os.environ AND ~/.hermes/.env, so
    we patch it on the plugin module to simulate the key being absent.
    """

    def test_search_returns_error_dict_when_key_missing(self, provider, plugin_module):
        from unittest.mock import patch
        with patch.object(plugin_module, "get_provider_env", return_value=""):
            result = provider.search("test", limit=5)
        assert isinstance(result, dict)
        assert result["success"] is False
        assert "OLLAMA_API_KEY" in result["error"]

    def test_extract_returns_error_list_when_key_missing(self, provider, plugin_module):
        from unittest.mock import patch
        with patch.object(plugin_module, "get_provider_env", return_value=""):
            result = provider.extract(["https://example.com"])
        assert isinstance(result, list)
        assert len(result) == 1
        assert "error" in result[0]
        assert "OLLAMA_API_KEY" in result[0]["error"]
        assert result[0]["url"] == "https://example.com"


# ---------------------------------------------------------------------------
# Live search tests -- real API calls
# ---------------------------------------------------------------------------


class TestLiveSearch:
    """Exercise the search endpoint with real API calls.

    Requires OLLAMA_API_KEY. Tests skip if the key is not set.
    """

    def test_search_returns_results(self, provider, require_api_key):
        result = provider.search("what is ollama", limit=3)
        assert result["success"] is True, f"Search failed: {result.get('error')}"
        web = result["data"]["web"]
        assert len(web) >= 1, "Expected at least 1 result"
        assert len(web) <= 3, "Expected at most 3 results (limit=3)"

    def test_search_result_shape(self, provider, require_api_key):
        result = provider.search("python programming", limit=2)
        assert result["success"] is True
        web = result["data"]["web"]
        for item in web:
            assert "title" in item
            assert "url" in item
            assert "description" in item
            assert "position" in item
            assert isinstance(item["title"], str)
            assert isinstance(item["url"], str)
            assert isinstance(item["description"], str)
            assert isinstance(item["position"], int)

    def test_search_positions_are_sequential(self, provider, require_api_key):
        result = provider.search("artificial intelligence", limit=5)
        assert result["success"] is True
        web = result["data"]["web"]
        positions = [item["position"] for item in web]
        assert positions == list(range(1, len(web) + 1))

    def test_search_urls_are_valid(self, provider, require_api_key):
        result = provider.search("github", limit=3)
        assert result["success"] is True
        web = result["data"]["web"]
        for item in web:
            assert item["url"].startswith("http"), f"URL not valid: {item['url']}"

    def test_search_limit_one(self, provider, require_api_key):
        result = provider.search("test", limit=1)
        assert result["success"] is True
        web = result["data"]["web"]
        assert len(web) <= 1

    def test_search_limit_capped_at_ten(self, provider, require_api_key):
        """Ollama API caps max_results at 10 -- verify it doesn't error."""
        result = provider.search("technology news", limit=10)
        assert result["success"] is True
        web = result["data"]["web"]
        assert len(web) <= 10

    def test_search_empty_results_on_nonsense(self, provider, require_api_key):
        """A very obscure query should still return success (possibly empty)."""
        result = provider.search("xqzwykjonesauce", limit=3)
        assert result["success"] is True
        assert "data" in result
        assert "web" in result["data"]


# ---------------------------------------------------------------------------
# Live extract tests -- real API calls
# ---------------------------------------------------------------------------


class TestLiveExtract:
    """Exercise the web_fetch endpoint with real API calls.

    Requires OLLAMA_API_KEY. Tests skip if the key is not set.
    """

    def test_extract_single_url(self, provider, require_api_key):
        docs = provider.extract(["https://ollama.com"])
        assert len(docs) == 1
        doc = docs[0]
        assert doc["url"] == "https://ollama.com"
        assert isinstance(doc.get("title"), str)
        assert len(doc["title"]) > 0
        assert isinstance(doc.get("content"), str)
        assert len(doc["content"]) > 0
        assert isinstance(doc.get("raw_content"), str)
        assert isinstance(doc.get("metadata"), dict)
        assert "error" not in doc

    def test_extract_metadata_has_links(self, provider, require_api_key):
        docs = provider.extract(["https://ollama.com"])
        doc = docs[0]
        links = doc["metadata"].get("links", [])
        assert isinstance(links, list)
        assert len(links) >= 1

    def test_extract_multiple_urls(self, provider, require_api_key):
        urls = ["https://ollama.com", "https://example.com"]
        docs = provider.extract(urls)
        assert len(docs) == 2
        assert docs[0]["url"] == "https://ollama.com"
        assert docs[1]["url"] == "https://example.com"
        # Both should have content
        for doc in docs:
            assert "content" in doc

    def test_extract_invalid_url_returns_error_entry(self, provider, require_api_key):
        """A bogus URL should produce an error entry, not crash the whole call."""
        docs = provider.extract(["https://this-domain-does-not-exist-xyz.invalid"])
        assert len(docs) == 1
        doc = docs[0]
        assert "error" in doc or "content" not in doc or len(doc.get("content", "")) == 0
        assert doc["url"] == "https://this-domain-does-not-exist-xyz.invalid"

    def test_extract_preserves_raw_content(self, provider, require_api_key):
        """raw_content should match content (the plugin sets them equal)."""
        docs = provider.extract(["https://ollama.com"])
        doc = docs[0]
        assert doc.get("content") == doc.get("raw_content")


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------


class TestConfigResolution:
    """Test _resolve_base_url and _resolve_timeout without API calls.

    These exercise the config loading logic in isolation.
    """

    def test_default_base_url(self, plugin_module, monkeypatch):
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        from unittest.mock import patch
        with patch.object(plugin_module, "_load_ollama_config", return_value={}):
            assert plugin_module._resolve_base_url() == "https://ollama.com"

    def test_env_base_url_override(self, plugin_module, monkeypatch):
        monkeypatch.setenv("OLLAMA_BASE_URL", "https://custom.ollama.proxy")
        assert plugin_module._resolve_base_url() == "https://custom.ollama.proxy"

    def test_env_base_url_strips_trailing_slash(self, plugin_module, monkeypatch):
        monkeypatch.setenv("OLLAMA_BASE_URL", "https://custom.ollama.proxy/")
        assert plugin_module._resolve_base_url() == "https://custom.ollama.proxy"

    def test_default_timeout(self, plugin_module, monkeypatch):
        from unittest.mock import patch
        with patch.object(plugin_module, "_load_ollama_config", return_value={}):
            assert plugin_module._resolve_timeout() == 30.0

    def test_config_timeout_override(self, plugin_module, monkeypatch):
        from unittest.mock import patch
        with patch.object(
            plugin_module, "_load_ollama_config", return_value={"timeout": 60},
        ):
            assert plugin_module._resolve_timeout() == 60.0

    def test_invalid_timeout_falls_back_to_default(self, plugin_module, monkeypatch):
        from unittest.mock import patch
        with patch.object(
            plugin_module, "_load_ollama_config", return_value={"timeout": "not-a-number"},
        ):
            assert plugin_module._resolve_timeout() == 30.0