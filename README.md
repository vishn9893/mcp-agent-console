# A2A + local llama.cpp + MCP (without LangGraph)

This is a small learning example based on the A2A project's
[a2a-mcp-without-framework sample](https://github.com/a2aproject/a2a-samples/tree/main/samples/python/agents/a2a-mcp-without-framework).
It keeps the model and tool orchestration explicit and adds a browser-managed gateway:

`Browser UI -> FastAPI gateway -> MCP servers`

`A2A client -> A2A server -> gateway -> llama.cpp (/v1/chat/completions)`

The default MCP server is `mcp-server-fetch`, launched through `uvx`. No LangGraph,
LangChain, or hosted model is required.

## Architecture

The runtime has two entry points that share the same MCP gateway and llama.cpp
orchestration layer:

```text
                         +----------------------+
                         | Browser console       |
                         | config + live events  |
                         +----------+-----------+
                                    |
                         WebSocket /ws/agent/{id}
                                    |
+-------------------+     +--------v---------+     +------------------+
| A2A client        +---->| FastAPI gateway  +---->| MCP servers      |
| JSON-RPC          |     | config + sessions|     | stdio processes  |
+-------------------+     +--------+---------+     +------------------+
                                    |
                         +----------v-----------+
                         | AgentOrchestrator    |
                         | llama.cpp tool loop  |
                         | namespacing + HITL   |
                         +----------+-----------+
                                    |
                         OpenAI-compatible API
                                    |
                         +----------v-----------+
                         | Local llama.cpp      |
                         +----------------------+
```

`ToolNamespacer` exposes tools to the model as `{server_id}__{tool_name}` and
resolves each selection back to the correct MCP server. The orchestrator sends
structured `info`, `turn_start`, `tool_call`, `tool_result`, `error`, `final`, and
`awaiting_approval` events to the browser while retaining the normal A2A response
path.

Tools listed in `RESTRICTED_TOOLS` pause the session before execution. The browser
receives an `awaiting_approval` event containing the call ID, tool, and arguments;
it then sends an `approval_response`. Approval resumes the MCP call, while denial
feeds a structured rejection result back to llama.cpp so the model can recover.
Disconnects cancel active tasks and pending approval futures.

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
- `WS /ws/agent` accepts `start_task` and `approval_response` messages.
- `WS /ws/agent/{session_id}` provides the session-aware HITL stream used by the UI.

The gateway validates every model tool call against the saved allowlist. Duplicate
tool names across enabled servers are namespaced before they are sent to the model,
then resolved back to the originating server before execution.

## Safety notes

Treat MCP tool descriptions and tool results as untrusted input. Keep an allowlist
of servers in a real gateway, validate tool arguments, enforce timeouts, and avoid
passing secrets to child processes. The reference A2A sample carries the same
warning about treating remote agent data as untrusted.

# mcp-agent-console
