"""Conversation/tool orchestration for the local llama.cpp agent."""

from __future__ import annotations

import json
import asyncio
from typing import Any, Protocol

from .namespacer import ToolNamespacer


RESTRICTED_TOOLS = frozenset({
    "filesystem__write_file",
    "filesystem__delete_file",
    "shell__execute_command",
})


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
        restricted_tools: set[str] | frozenset[str] | None = None,
    ) -> None:
        self.namespacer = namespacer
        self.mcp_client = mcp_client
        self.llama = llama_client
        self.max_tool_rounds = max_tool_rounds
        self.restricted_tools = frozenset(restricted_tools if restricted_tools is not None else RESTRICTED_TOOLS)
        self.approval_futures: dict[str, asyncio.Future[dict[str, Any]]] = {}

    def resolve_approval(self, call_id: str, approved: bool, reason: str = "") -> bool:
        """Resolve a pending approval request; return false for stale request IDs."""
        future = self.approval_futures.get(call_id)
        if future is None or future.done():
            return False
        future.set_result({"approved": approved, "reason": reason})
        return True

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
                call_id = call.get("id") or f"{name}:{id(call)}"
                raw_arguments = function.get("arguments", {})
                await self.send_log(event_sink, "tool_call", f"Agent requested tool: {name}", {
                    "tool": name, "arguments": raw_arguments,
                })
                try:
                    arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
                    if not isinstance(arguments, dict):
                        raise ValueError("arguments must be a JSON object")
                    server_id, tool_name = self.namespacer.resolve(name)
                    if name in self.restricted_tools:
                        await self.send_log(event_sink, "awaiting_approval", f"Action hold: tool '{name}' requires approval.", {
                            "call_id": call_id, "tool": name, "arguments": arguments,
                        })
                        approval_loop = asyncio.get_running_loop()
                        approval_future = approval_loop.create_future()
                        self.approval_futures[call_id] = approval_future
                        try:
                            decision = await approval_future
                        finally:
                            self.approval_futures.pop(call_id, None)
                        if not decision.get("approved", False):
                            reason = decision.get("reason") or "No reason provided."
                            content = json.dumps({
                                "error": "Execution denied by user.",
                                "details": reason,
                            })
                            await self.send_log(event_sink, "info", f"User rejected execution of tool {name}.")
                            messages.append({"role": "tool", "tool_call_id": call_id, "name": name, "content": content})
                            continue
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
