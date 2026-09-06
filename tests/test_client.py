from a2a_local_llama_mcp.client import send


def test_module_imports() -> None:
    assert callable(send)
