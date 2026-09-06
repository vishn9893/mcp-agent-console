# A2A + local llama.cpp + MCP (without LangGraph)

This is a small learning example based on the A2A project's
[a2a-mcp-without-framework sample](https://github.com/a2aproject/a2a-samples/tree/main/samples/python/agents/a2a-mcp-without-framework).
It keeps the model and tool orchestration explicit and adds a browser-managed gateway:

`Browser UI -> FastAPI gateway -> MCP servers`

`A2A client -> A2A server -> gateway -> llama.cpp (/v1/chat/completions)`

The default MCP server is `mcp-server-fetch`, launched through `uvx`. No LangGraph,
LangChain, or hosted model is required.

## Run it

Install with uv:

```bash
uv sync
cp .env.example .env
```

Start llama.cpp with an instruction-tuned model and tool calling enabled. The model
must be served on port 8082 and the server should be started with `--jinja` when the
model's chat template supports function calling:

```bash
llama-server -m /path/to/model.gguf --host 127.0.0.1 --port 8082 --jinja
```

In another terminal, start the A2A server:

```bash
uv run a2a-local-server --host 127.0.0.1 --port 9999
```

Open [http://127.0.0.1:9999/](http://127.0.0.1:9999/) in a browser. The UI shows
configured MCP servers, connection status, tool descriptions and JSON schemas,
enabled/disabled checkboxes, test buttons, and llama.cpp status. Configuration is
stored in `.data/mcp_config.json` by default.

Then send a question:

```bash
uv run a2a-local-client --url http://127.0.0.1:9999/ \
  --question "Fetch https://a2aproject.github.io/A2A/specification/ and summarize the protocol."
```

If your model does not support tool calls reliably, first test the A2A path with a
plain question. The tool loop is bounded by `LLAMA_MAX_TOOL_ROUNDS`.

## Gateway API

The FastAPI endpoints are available under `/api`:

- `GET /api/config` lists servers, connection state, tools, descriptions, schemas,
  and enabled flags.
- `PUT /api/config` saves server commands and enabled tool names.
- `POST /api/servers/{id}/connect` and `/disconnect` manage a connection.
- `POST /api/tools/test` calls one enabled tool directly.
- `GET /api/model/status` checks the llama.cpp `/models` endpoint.

The gateway validates every model tool call against the saved allowlist. Duplicate
tool names across enabled servers are rejected as ambiguous until they are
namespaced, which is a safe starting point for adding more MCP servers.

## Safety notes

Treat MCP tool descriptions and tool results as untrusted input. Keep an allowlist
of servers in a real gateway, validate tool arguments, enforce timeouts, and avoid
passing secrets to child processes. The reference A2A sample carries the same
warning about treating remote agent data as untrusted.

# mcp-agent-console
