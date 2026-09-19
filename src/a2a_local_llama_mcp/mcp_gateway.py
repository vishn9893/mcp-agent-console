from __future__ import annotations

import asyncio
import json
from contextlib import AsyncExitStack
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def _schema(tool: Any) -> dict[str, Any]:
    input_schema = getattr(tool, "inputSchema", None) or getattr(tool, "input_schema", None)
    return {"type": "function", "function": {
        "name": tool.name, "description": tool.description or "",
        "parameters": input_schema or {"type": "object", "properties": {}},
    }}


@dataclass
class MCPServerSpec:
    command: str
    args: list[str] = field(default_factory=list)
    enabled_tools: list[str] = field(default_factory=list)


class MCPConfigStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self, default: dict[str, MCPServerSpec]) -> dict[str, MCPServerSpec]:
        if not self.path.exists():
            return default
        raw = json.loads(self.path.read_text())
        return {server_id: MCPServerSpec(
            command=value["command"], args=list(value.get("args", [])),
            enabled_tools=list(value.get("enabled_tools", [])),
        ) for server_id, value in raw.get("servers", {}).items()}

    def save(self, servers: dict[str, MCPServerSpec]) -> None:
        payload = {"servers": {server_id: asdict(spec) for server_id, spec in servers.items()}}
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.replace(self.path)


class MCPConnection:
    def __init__(self, spec: MCPServerSpec) -> None:
        self.spec = spec
        self.exit_stack = AsyncExitStack()
        self.session: ClientSession | None = None
        self.tools: list[dict[str, Any]] = []
        self.error: str | None = None

    async def connect(self) -> None:
        try:
            read, write = await self.exit_stack.enter_async_context(
                stdio_client(StdioServerParameters(command=self.spec.command, args=self.spec.args))
            )
            self.session = await self.exit_stack.enter_async_context(ClientSession(read, write))
            await self.session.initialize()
            result = await self.session.list_tools()
            self.tools = [_schema(tool) for tool in result.tools]
            self.error = None
        except Exception as exc:
            await self.exit_stack.aclose()
            self.exit_stack = AsyncExitStack()
            self.session = None
            self.tools = []
            self.error = str(exc)
            raise

    async def close(self) -> None:
        await self.exit_stack.aclose()
        self.session = None

    async def call(self, name: str, arguments: dict[str, Any]) -> str:
        if self.session is None:
            raise RuntimeError("MCP server is not connected")
        result = await self.session.call_tool(name, arguments=arguments)
        if getattr(result, "is_error", False):
            raise RuntimeError(f"MCP tool {name!r} returned an error")
        structured = getattr(result, "structured_content", None)
        if structured is not None:
            return json.dumps(structured, default=str)
        return "\n".join(getattr(item, "text", None) or str(item) for item in getattr(result, "content", []))


class MCPGateway:
    """Persistent multi-server MCP gateway with a saved tool allowlist."""

    def __init__(self, config_path: str, default_servers: dict[str, MCPServerSpec]) -> None:
        self.store = MCPConfigStore(Path(config_path))
        self.servers = self.store.load(default_servers)
        self.connections: dict[str, MCPConnection] = {}
        self.lock = asyncio.Lock()

    async def connect_server(self, server_id: str) -> None:
        async with self.lock:
            spec = self._spec(server_id)
            old = self.connections.pop(server_id, None)
            if old:
                await old.close()
            connection = MCPConnection(spec)
            self.connections[server_id] = connection
            await connection.connect()

    async def disconnect_server(self, server_id: str) -> None:
        async with self.lock:
            connection = self.connections.pop(server_id, None)
            if connection:
                await connection.close()

    async def ensure_connected(self) -> None:
        for server_id in self.servers:
            if server_id not in self.connections or self.connections[server_id].session is None:
                try:
                    await self.connect_server(server_id)
                except Exception:
                    continue

    async def save(self, servers: dict[str, MCPServerSpec]) -> None:
        async with self.lock:
            for connection in self.connections.values():
                await connection.close()
            self.connections.clear()
            self.servers = servers
            self.store.save(servers)

    def _spec(self, server_id: str) -> MCPServerSpec:
        try:
            return self.servers[server_id]
        except KeyError as exc:
            raise KeyError(f"Unknown MCP server: {server_id}") from exc

    def enabled_tools(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        seen: set[str] = set()
        for server_id, spec in self.servers.items():
            connection = self.connections.get(server_id)
            if not connection:
                continue
            for tool in connection.tools:
                name = tool["function"]["name"]
                if name in spec.enabled_tools and name not in seen:
                    result.append(tool)
                    seen.add(name)
        return result

    def enabled_tools_by_server(self) -> dict[str, list[dict[str, Any]]]:
        """Return enabled tool schemas grouped by their MCP server."""
        result: dict[str, list[dict[str, Any]]] = {}
        for server_id, spec in self.servers.items():
            connection = self.connections.get(server_id)
            if not connection:
                continue
            tools = [
                tool for tool in connection.tools
                if tool["function"]["name"] in spec.enabled_tools
            ]
            if tools:
                result[server_id] = tools
        return result

    def resolve_tool(self, name: str) -> tuple[str, str]:
        matches: list[tuple[str, str]] = []
        for server_id, spec in self.servers.items():
            connection = self.connections.get(server_id)
            available = {tool["function"]["name"] for tool in connection.tools} if connection else set()
            if name in spec.enabled_tools and name in available:
                matches.append((server_id, name))
        if len(matches) != 1:
            raise KeyError(f"Tool {name!r} is unavailable or ambiguous")
        return matches[0]

    async def call(self, server_id: str, name: str, arguments: dict[str, Any]) -> str:
        spec = self._spec(server_id)
        if name not in spec.enabled_tools:
            raise PermissionError(f"MCP tool {server_id}/{name} is disabled")
        connection = self.connections.get(server_id)
        if not connection or name not in {tool["function"]["name"] for tool in connection.tools}:
            raise KeyError(f"MCP tool {server_id}/{name} does not exist")
        return await connection.call(name, arguments)

    def snapshot(self) -> dict[str, Any]:
        servers: dict[str, Any] = {}
        for server_id, spec in self.servers.items():
            connection = self.connections.get(server_id)
            servers[server_id] = {
                "command": spec.command, "args": spec.args, "enabled_tools": spec.enabled_tools,
                "connected": bool(connection and connection.session),
                "error": connection.error if connection else None,
                "tools": [{
                    "name": tool["function"]["name"],
                    "description": tool["function"]["description"],
                    "schema": tool["function"]["parameters"],
                    "enabled": tool["function"]["name"] in spec.enabled_tools,
                } for tool in (connection.tools if connection else [])],
            }
        return {"servers": servers}
