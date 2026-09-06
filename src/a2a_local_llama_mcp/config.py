from __future__ import annotations

import os
import shlex
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    a2a_host: str = os.getenv("A2A_HOST", "127.0.0.1")
    a2a_port: int = int(os.getenv("A2A_PORT", "9999"))
    llama_base_url: str = os.getenv("LLAMA_BASE_URL", "http://127.0.0.1:8082/v1").rstrip("/")
    llama_api_key: str = os.getenv("LLAMA_API_KEY", "sk-no-key-required")
    llama_model: str = os.getenv("LLAMA_MODEL", "local-model")
    mcp_command: str = os.getenv("MCP_COMMAND", "uvx")
    mcp_args: tuple[str, ...] = tuple(shlex.split(os.getenv("MCP_ARGS", "mcp-server-fetch")))
    mcp_timeout_seconds: float = float(os.getenv("MCP_TIMEOUT_SECONDS", "60"))
    max_tool_rounds: int = int(os.getenv("LLAMA_MAX_TOOL_ROUNDS", "4"))
    mcp_config_path: str = os.getenv("MCP_CONFIG_PATH", ".data/mcp_config.json")


settings = Settings()
