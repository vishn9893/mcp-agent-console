from pathlib import Path

from fastapi.testclient import TestClient

from a2a_local_llama_mcp.config import settings
from a2a_local_llama_mcp.gateway_api import create_gateway_api
from a2a_local_llama_mcp.mcp_gateway import MCPGateway, MCPServerSpec


def test_allowlist_is_persisted(tmp_path: Path) -> None:
    gateway = MCPGateway(str(tmp_path / "mcp.json"), {
        "fetch": MCPServerSpec("uvx", ["mcp-server-fetch"], ["fetch"]),
    })
    api = create_gateway_api(gateway, settings)
    client = TestClient(api)

    response = client.put("/api/config", json={"servers": {
        "fetch": {"command": "uvx", "args": ["mcp-server-fetch"], "enabled_tools": []},
    }})
    assert response.status_code == 200
    assert gateway.servers["fetch"].enabled_tools == []
    assert (tmp_path / "mcp.json").exists()


def test_ui_is_served(tmp_path: Path) -> None:
    api = create_gateway_api(MCPGateway(str(tmp_path / "mcp.json"), {}), settings)
    response = TestClient(api).get("/")
    assert response.status_code == 200
    assert "MCP Gateway" in response.text
