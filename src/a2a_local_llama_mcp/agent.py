from __future__ import annotations

from .config import Settings
from .llama_client import LlamaCppClient
from .mcp_gateway import MCPGateway
from .namespacer import ToolNamespacer
from .orchestrator import AgentOrchestrator


class LocalMCPAgent:
    def __init__(self, config: Settings, gateway: MCPGateway) -> None:
        self.config = config
        self.gateway = gateway
        self.llama = LlamaCppClient(config.llama_base_url, config.llama_api_key, config.llama_model)
        self.orchestrator = AgentOrchestrator(
            ToolNamespacer(), gateway, self.llama, config.max_tool_rounds
        )

    async def answer(self, question: str) -> str:
        await self.gateway.ensure_connected()
        return await self.orchestrator.execute_query(question, self.gateway.enabled_tools_by_server())
