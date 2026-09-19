"""Conversation/tool orchestration for the local llama.cpp agent."""

from __future__ import annotations

import json
from typing import Any, Protocol

from .namespacer import ToolNamespacer


class LlamaChatClient(Protocol):
    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]: ...


class MCPToolClient(Protocol):
    async def call(self, server_id: str, name: str, arguments: dict[str, Any]) -> str: ...


class EventSink(Protocol):
    async def send_json(self, data: dict[str, Any]) -> Any: ...


class AgentOrchestrator:
    """Run model turns and turn tool failures into retryable tool messages."""

    def __init__(
        self,
        namespacer: ToolNamespacer,
        mcp_client: MCPToolClient,
        llama_client: LlamaChatClient,
        max_tool_rounds: int = 4,
    ) -> None:
        self.namespacer = namespacer
        self.mcp_client = mcp_client
        self.llama = llama_client
        self.max_tool_rounds = max_tool_rounds

    async def execute_query(
        self,
        user_question: str,
        available_mcp_servers: dict[str, list[dict[str, Any]]],
        event_sink: EventSink | None = None,
    ) -> str:
        await self.send_log(event_sink, "info", "Initializing agent workspace history stream...")
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "You are an A2A agent with access to enabled MCP tools. Use a tool when useful. After tool results, answer plainly and cite source URLs when relevant."},
            {"role": "user", "content": user_question},
        ]
        self.namespacer.registry.clear()
        all_tools: list[dict[str, Any]] = []
        for server_id, tools in available_mcp_servers.items():
            all_tools.extend(self.namespacer.register_server_tools(server_id, tools))

        for round_index in range(self.max_tool_rounds):
            turn = round_index + 1
            await self.send_log(event_sink, "turn_start", f"Starting loop evaluation turn {turn}...", {"turn": turn})
            try:
                message = await self.llama.chat(messages, all_tools)
            except Exception as exc:
                error = f"Critical error communicating with local llama-server: {exc}"
                await self.send_log(event_sink, "error", error)
                return error
            messages.append(message)
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                final_answer = message.get("content") or "(The local model returned an empty response.)"
                await self.send_log(event_sink, "final", "Agent determined task context complete.", {"content": final_answer})
                return final_answer

            for call in tool_calls:
                function = call.get("function") or {}
                name = function.get("name")
                call_id = call.get("id", name)
                raw_arguments = function.get("arguments", {})
                await self.send_log(event_sink, "tool_call", f"Agent requested tool: {name}", {
                    "tool": name, "arguments": raw_arguments,
                })
                try:
                    arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
                    if not isinstance(arguments, dict):
                        raise ValueError("arguments must be a JSON object")
                    server_id, tool_name = self.namespacer.resolve(name)
                    result = await self.mcp_client.call(server_id, tool_name, arguments)
                    content = result if isinstance(result, str) else json.dumps(result, default=str)
                    await self.send_log(event_sink, "tool_result", f"Tool {name} executed successfully.", {"result": result})
                except json.JSONDecodeError as exc:
                    content = json.dumps({
                        "error": "Invalid JSON arguments.",
                        "details": str(exc),
                        "remedy": "Retry with strict JSON matching the tool parameter schema.",
                    })
                    await self.send_log(event_sink, "error", f"Self-correction: malformed JSON arguments for {name}", json.loads(content))
                except Exception as exc:
                    content = json.dumps({
                        "error": "Tool execution failed.",
                        "details": str(exc),
                        "remedy": "Review the error and retry with corrected tool and arguments.",
                    })
                    await self.send_log(event_sink, "error", f"Self-correction: tool execution failed for {name}", json.loads(content))
                messages.append({"role": "tool", "tool_call_id": call_id, "name": name, "content": content})

        error = "The model reached the maximum number of MCP tool calls without producing a final answer."
        await self.send_log(event_sink, "error", error)
        return error

    async def send_log(self, event_sink: EventSink | None, event_type: str, message: str, data: Any = None) -> None:
        """Send a best-effort event without interrupting the agent if the client disconnects."""
        if event_sink is None:
            return
        try:
            await event_sink.send_json({"type": event_type, "message": message, "data": data})
        except Exception:
            pass
