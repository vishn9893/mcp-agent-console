"""Namespace MCP tools so multiple servers can expose the same tool name."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


class ToolNamespacer:
    """Translate between llama.cpp tool names and MCP server/tool pairs."""

    SEPARATOR = "__"

    def __init__(self) -> None:
        self.registry: dict[str, tuple[str, str]] = {}

    def register_server_tools(
        self, server_id: str, tools: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        namespaced_tools: list[dict[str, Any]] = []
        for tool in tools:
            function = tool.get("function")
            if not isinstance(function, dict) or not isinstance(function.get("name"), str):
                raise ValueError("MCP tool schema must contain function.name")
            original_name = function["name"]
            namespaced_name = f"{server_id}{self.SEPARATOR}{original_name}"
            self.registry[namespaced_name] = (server_id, original_name)
            namespaced_tool = deepcopy(tool)
            namespaced_tool["function"]["name"] = namespaced_name
            namespaced_tools.append(namespaced_tool)
        return namespaced_tools

    def resolve(self, namespaced_name: str) -> tuple[str, str]:
        try:
            return self.registry[namespaced_name]
        except KeyError as exc:
            raise ValueError(f"Unrecognized namespaced tool: {namespaced_name}") from exc

