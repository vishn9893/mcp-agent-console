from __future__ import annotations

import json
from typing import Any

from .config import Settings
from .llama_client import LlamaCppClient
from .mcp_gateway import MCPGateway


class LocalMCPAgent:
    def __init__(self, config: Settings, gateway: MCPGateway) -> None:
        self.config = config
        self.gateway = gateway
        self.llama = LlamaCppClient(config.llama_base_url, config.llama_api_key, config.llama_model)

    async def answer(self, question: str) -> str:
        await self.gateway.ensure_connected()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "You are an A2A agent with access to enabled MCP tools. Use a tool when useful. After tool results, answer plainly and cite source URLs when relevant."},
            {"role": "user", "content": question},
        ]
        for _ in range(self.config.max_tool_rounds):
            message = await self.llama.chat(messages, self.gateway.enabled_tools())
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                return message.get("content") or "(The local model returned an empty response.)"
            messages.append(message)
            for call in tool_calls:
                function = call.get("function", {})
                name = function.get("name")
                raw_arguments = function.get("arguments", {})
                try:
                    arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"Model produced invalid arguments for MCP tool {name!r}") from exc
                if not isinstance(arguments, dict):
                    raise RuntimeError(f"Arguments for MCP tool {name!r} must be a JSON object")
                server_id, tool_name = self.gateway.resolve_tool(name)
                result = await self.gateway.call(server_id, tool_name, arguments)
                messages.append({"role": "tool", "tool_call_id": call.get("id", name), "name": name, "content": result})
        return "The model reached the maximum number of MCP tool calls without producing a final answer."
