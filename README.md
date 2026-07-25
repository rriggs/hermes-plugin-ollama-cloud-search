# hermes-plugin-ollama-cloud-search

Ollama Cloud web search + content extraction provider plugin for [Hermes Agent](https://hermes-agent.nousresearch.com).

Wraps Ollama's native REST APIs -- `POST /api/web_search` for search and `POST /api/web_fetch` for page extraction -- behind the Hermes `WebSearchProvider` interface. Both capabilities are served from one provider, so a single config entry covers `web_search` and `web_extract`.

## Setup

### 1. Get an Ollama API key

Create one at [ollama.com/settings/keys](https://ollama.com/settings/keys). A free Ollama account includes a generous web search tier.

### 2. Set the env var

```
OLLAMA_API_KEY=ollama-xxxxxxxxxxxx
```

This is the **same env var** the ollama-cloud model provider uses. If you're already running Ollama Cloud for chat completions, you're done -- no extra credentials needed.

### 3. Install the plugin

```
hermes plugins install rriggs/hermes-plugin-ollama-cloud-search
```

### 4. Enable and configure

Add `web-ollama-cloud` to `plugins.enabled` in your `config.yaml`, then set the web backend:

```yaml
plugins:
  enabled:
    - web-ollama-cloud

web:
  backend: ollama-cloud
  # Or set per-capability:
  # search_backend: ollama-cloud
  # extract_backend: ollama-cloud
  ollama-cloud:
    timeout: 30  # optional, default 30 seconds
```

Restart your session (`/reset`) to pick up the new plugin.

## Config reference

| Key | Default | Description |
|-----|---------|-------------|
| `web.backend` | -- | Shared fallback for search + extract |
| `web.search_backend` | -- | Explicit search override |
| `web.extract_backend` | -- | Explicit extract override |
| `web.ollama-cloud.timeout` | `30` | HTTP timeout in seconds |
| `OLLAMA_API_KEY` | -- | Required. API key for Ollama |
| `OLLAMA_BASE_URL` | `https://ollama.com` | Optional base URL override |

## API details

### Search (`POST /api/web_search`)

Request:
```json
{"query": "what is ollama?", "max_results": 5}
```

Response:
```json
{
  "results": [
    {"title": "Ollama", "url": "https://ollama.com/", "content": "Cloud models are now available..."}
  ]
}
```

The plugin maps `content` to `description` in the Hermes search result shape. `max_results` is capped at 10 by the API.

### Extract (`POST /api/web_fetch`)

Request (one URL per call):
```json
{"url": "https://ollama.com"}
```

Response:
```json
{
  "title": "Ollama",
  "content": "Cloud models are now available...",
  "links": ["https://ollama.com/", "https://ollama.com/models"]
}
```

The plugin fetches each URL sequentially and returns the standard Hermes document list shape with `url`, `title`, `content`, `raw_content`, and `metadata`.

## Free tier

Ollama provides a generous free tier of web searches for individuals. Higher rate limits are available via [Ollama Cloud](https://ollama.com/cloud).

## License

MIT