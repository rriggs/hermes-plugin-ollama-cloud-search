"""Ollama Cloud web search + content extraction -- user-installed plugin.

Subclasses :class:`agent.web_search_provider.WebSearchProvider`. Two
capabilities advertised:

- ``supports_search()``  -> True (Ollama ``POST /api/web_search``)
- ``supports_extract()`` -> True (Ollama ``POST /api/web_fetch``)

Both are sync -- the underlying calls are ``httpx.post(...)``.

Config keys this provider responds to::

    web:
      search_backend: "ollama-cloud"   # explicit per-capability
      extract_backend: "ollama-cloud"  # explicit per-capability
      backend: "ollama-cloud"          # shared fallback for both

Optional knobs (under ``web.ollama-cloud`` in ``config.yaml``)::

    web:
      ollama-cloud:
        timeout: 30                    # seconds (default 30)

Env vars::

    OLLAMA_API_KEY=...              # https://ollama.com/settings/keys (required)
    OLLAMA_BASE_URL=...             # optional override of https://ollama.com

The API key env var is the same one the ollama-cloud model provider uses,
so profiles already configured for Ollama Cloud chat completions get web
search for free -- no extra credential setup.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from agent.web_search_provider import WebSearchProvider, get_provider_env

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://ollama.com"
_DEFAULT_TIMEOUT = 30
_SEARCH_ENDPOINT = "/api/web_search"
_FETCH_ENDPOINT = "/api/web_fetch"
_MAX_RESULTS_CAP = 10  # Ollama API hard cap


def _native_base_url(raw: str) -> str:
    """Normalize a configured Ollama base URL to the NATIVE API root.

    ``OLLAMA_BASE_URL`` is shared with the ollama-cloud MODEL provider, whose
    OpenAI-compatible base is ``https://ollama.com/v1``. The native endpoints
    this plugin calls (``/api/web_search``, ``/api/web_fetch``) live at the
    ORIGIN, not under ``/v1``, so appending them to the shared value produced
    ``/v1/api/web_search`` -> HTTP 404 on any host configured for chat
    completions. Strip the OpenAI-compat suffix (and a trailing ``/api``, the
    same class of mistake) so one variable can serve both consumers.
    """
    base = (raw or "").strip().rstrip("/")
    for suffix in ("/v1", "/api"):
        if base.endswith(suffix):
            base = base[: -len(suffix)].rstrip("/")
            break
    return base or _DEFAULT_BASE_URL


def _load_ollama_config() -> Dict[str, Any]:
    """Read ``web.ollama-cloud`` from config.yaml (returns {} on miss)."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        web_section = cfg.get("web") if isinstance(cfg, dict) else None
        oc_section = web_section.get("ollama-cloud") if isinstance(web_section, dict) else None
        return oc_section if isinstance(oc_section, dict) else {}
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not load web.ollama-cloud config: %s", exc)
        return {}


def _resolve_base_url() -> str:
    """Resolve the Ollama NATIVE API base URL.

    Checks ``OLLAMA_BASE_URL`` env var first, then config, then falls back
    to the default. The shared variable may carry the model provider's
    OpenAI-compatible ``/v1`` suffix -- :func:`_native_base_url` strips it so
    the native ``/api/*`` paths still resolve.
    """
    env_url = get_provider_env("OLLAMA_BASE_URL")
    if env_url:
        return _native_base_url(env_url)
    cfg = _load_ollama_config()
    cfg_url = cfg.get("base_url")
    if isinstance(cfg_url, str) and cfg_url.strip():
        return _native_base_url(cfg_url)
    return _DEFAULT_BASE_URL


def _resolve_timeout() -> float:
    """Resolve the timeout from config or default."""
    cfg = _load_ollama_config()
    try:
        return float(cfg.get("timeout", _DEFAULT_TIMEOUT))
    except (TypeError, ValueError):
        return _DEFAULT_TIMEOUT


class OllamaCloudWebSearchProvider(WebSearchProvider):
    """Ollama Cloud search + extract provider.

    Wraps Ollama's native REST APIs:

    - Search: ``POST /api/web_search`` with ``{query, max_results}``
      Returns ``{results: [{title, url, content}, ...]}``
    - Extract: ``POST /api/web_fetch`` with ``{url}`` per URL
      Returns ``{title, content, links}``

    Both use Bearer auth with ``OLLAMA_API_KEY`` -- the same env var the
    ollama-cloud model provider uses, so no extra credential setup is
    needed for profiles already running Ollama Cloud chat completions.
    """

    @property
    def name(self) -> str:
        return "ollama-cloud"

    @property
    def display_name(self) -> str:
        return "Ollama Cloud Web Search"

    def is_available(self) -> bool:
        """Return True when ``OLLAMA_API_KEY`` is set to a non-empty value."""
        return bool(get_provider_env("OLLAMA_API_KEY"))

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return True

    # -- Search -----------------------------------------------------------

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute a web search via Ollama's /api/web_search endpoint.

        Returns ``{"success": True, "data": {"web": [{title, url, description, position}, ...]}}``
        on success, ``{"success": False, "error": str}`` on failure.
        """
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return {"success": False, "error": "Interrupted"}
        except Exception:  # noqa: BLE001 -- interrupt module is best-effort
            pass

        api_key = get_provider_env("OLLAMA_API_KEY")
        if not api_key:
            return {
                "success": False,
                "error": (
                    "OLLAMA_API_KEY is not set. Create an API key at "
                    "https://ollama.com/settings/keys"
                ),
            }

        # Ollama API caps max_results at 10.
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = 5
        max_results = max(1, min(limit, _MAX_RESULTS_CAP))

        base_url = _resolve_base_url()
        timeout = _resolve_timeout()

        try:
            import httpx
        except ImportError:
            return {
                "success": False,
                "error": "httpx is not installed (required for Ollama Cloud web search)",
            }

        logger.info(
            "Ollama Cloud web search: '%s' (max_results=%d, timeout=%.0fs)",
            query, max_results, timeout,
        )

        try:
            resp = httpx.post(
                f"{base_url}{_SEARCH_ENDPOINT}",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={"query": query, "max_results": max_results},
                timeout=timeout,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            body = ""
            try:
                body = exc.response.text[:300]
            except Exception:
                pass
            logger.warning("Ollama web search HTTP %d: %s", exc.response.status_code, body)
            return {
                "success": False,
                "error": f"Ollama web search returned HTTP {exc.response.status_code}: {body}".rstrip(),
            }
        except httpx.RequestError as exc:
            logger.warning("Ollama web search request error: %s", exc)
            return {"success": False, "error": f"Could not reach Ollama: {exc}"}

        try:
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Ollama web search response parse error: %s", exc)
            return {"success": False, "error": "Could not parse Ollama search response as JSON"}

        raw_results = data.get("results", []) or []
        web_results = [
            {
                "title": str(r.get("title", "")),
                "url": str(r.get("url", "")),
                "description": str(r.get("content", "")),
                "position": i + 1,
            }
            for i, r in enumerate(raw_results[:limit])
            if isinstance(r, dict)
        ]

        logger.info(
            "Ollama web search '%s': %d results (limit %d)",
            query, len(web_results), limit,
        )

        return {"success": True, "data": {"web": web_results}}

    # -- Extract ----------------------------------------------------------

    def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        """Extract content from one or more URLs via Ollama's /api/web_fetch.

        Fetches each URL individually (Ollama's API accepts one URL per call).
        Returns the standard list-of-documents shape; per-URL failures become
        entries with an ``error`` field.
        """
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return [{"url": u, "error": "Interrupted", "title": ""} for u in urls]
        except Exception:  # noqa: BLE001
            pass

        api_key = get_provider_env("OLLAMA_API_KEY")
        if not api_key:
            return [
                {
                    "url": u,
                    "title": "",
                    "content": "",
                    "error": (
                        "OLLAMA_API_KEY is not set. Create an API key at "
                        "https://ollama.com/settings/keys"
                    ),
                }
                for u in urls
            ]

        base_url = _resolve_base_url()
        timeout = _resolve_timeout()

        try:
            import httpx
        except ImportError:
            return [
                {"url": u, "title": "", "content": "", "error": "httpx is not installed"}
                for u in urls
            ]

        logger.info("Ollama Cloud web fetch: %d URL(s)", len(urls))

        documents: List[Dict[str, Any]] = []
        for url in urls:
            try:
                resp = httpx.post(
                    f"{base_url}{_FETCH_ENDPOINT}",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={"url": url},
                    timeout=timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                content = str(data.get("content", ""))
                title = str(data.get("title", ""))
                links = data.get("links", [])
                documents.append({
                    "url": url,
                    "title": title,
                    "content": content,
                    "raw_content": content,
                    "metadata": {
                        "sourceURL": url,
                        "title": title,
                        "links": links if isinstance(links, list) else [],
                    },
                })
            except httpx.HTTPStatusError as exc:
                body = ""
                try:
                    body = exc.response.text[:200]
                except Exception:
                    pass
                logger.warning("Ollama web fetch HTTP %d for %s: %s", exc.response.status_code, url, body)
                documents.append({
                    "url": url,
                    "title": "",
                    "content": "",
                    "raw_content": "",
                    "error": f"HTTP {exc.response.status_code}: {body}".rstrip(),
                    "metadata": {"sourceURL": url},
                })
            except httpx.RequestError as exc:
                logger.warning("Ollama web fetch request error for %s: %s", url, exc)
                documents.append({
                    "url": url,
                    "title": "",
                    "content": "",
                    "raw_content": "",
                    "error": f"Could not reach Ollama: {exc}",
                    "metadata": {"sourceURL": url},
                })
            except Exception as exc:  # noqa: BLE001
                logger.warning("Ollama web fetch error for %s: %s", url, exc)
                documents.append({
                    "url": url,
                    "title": "",
                    "content": "",
                    "raw_content": "",
                    "error": f"Ollama fetch failed: {exc}",
                    "metadata": {"sourceURL": url},
                })

        return documents

    # -- Setup picker -----------------------------------------------------

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Ollama Cloud Web Search",
            "badge": "free",
            "tag": (
                "Web search + page fetch via Ollama's native API. "
                "Free tier available with an Ollama account. "
                "Reuses OLLAMA_API_KEY."
            ),
            "env_vars": [
                {
                    "key": "OLLAMA_API_KEY",
                    "prompt": "Ollama API key",
                    "url": "https://ollama.com/settings/keys",
                },
            ],
        }


def register(ctx) -> None:
    """Register the Ollama Cloud web search provider with the plugin context."""
    ctx.register_web_search_provider(OllamaCloudWebSearchProvider())