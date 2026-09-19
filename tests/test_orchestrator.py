import asyncio

from a2a_local_llama_mcp.namespacer import ToolNamespacer
from a2a_local_llama_mcp.orchestrator import AgentOrchestrator


def tool(name: str) -> dict:
    return {"type": "function", "function": {"name": name, "description": "", "parameters": {"type": "object"}}}


class FakeLlama:
    def __init__(self, messages):
        self.messages = iter(messages)
        self.requests = []

    async def chat(self, messages, tools):
        self.requests.append((messages, tools))
        return next(self.messages)


class FakeGateway:
    def __init__(self, result="ok"):
        self.calls = []
        self.result = result

    async def call(self, server_id, name, arguments):
        self.calls.append((server_id, name, arguments))
        return self.result


class FakeSink:
    def __init__(self):
        self.events = []

    async def send_json(self, data):
        self.events.append(data)


def test_namespacer_handles_duplicate_tool_names_without_mutating_input():
    original = tool("fetch")
    namespacer = ToolNamespacer()
    result = namespacer.register_server_tools("one", [original])
    result += namespacer.register_server_tools("two", [original])

    assert original["function"]["name"] == "fetch"
    assert [item["function"]["name"] for item in result] == ["one__fetch", "two__fetch"]
    assert namespacer.resolve("two__fetch") == ("two", "fetch")


def test_execution_routes_namespaced_call_and_returns_final_answer():
    llama = FakeLlama([
        {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "server__fetch", "arguments": '{"url":"https://example.com"}'}}]},
        {"role": "assistant", "content": "done"},
    ])
    gateway = FakeGateway("page")
    orchestrator = AgentOrchestrator(ToolNamespacer(), gateway, llama)

    assert asyncio.run(orchestrator.execute_query("fetch it", {"server": [tool("fetch")] })) == "done"
    assert gateway.calls == [("server", "fetch", {"url": "https://example.com"})]


def test_malformed_arguments_are_feedback_for_model_retry():
    llama = FakeLlama([
        {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "server__fetch", "arguments": "{"}}]},
        {"role": "assistant", "content": "recovered"},
    ])
    gateway = FakeGateway()
    orchestrator = AgentOrchestrator(ToolNamespacer(), gateway, llama)

    assert asyncio.run(orchestrator.execute_query("fetch it", {"server": [tool("fetch")] })) == "recovered"
    feedback = llama.requests[1][0][-2]
    assert feedback["role"] == "tool"
    assert "Invalid JSON arguments" in feedback["content"]
    assert gateway.calls == []


def test_tool_exception_is_feedback_and_does_not_escape():
    class BrokenGateway(FakeGateway):
        async def call(self, server_id, name, arguments):
            raise RuntimeError("transient failure")

    llama = FakeLlama([
        {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "server__fetch", "arguments": "{}"}}]},
        {"role": "assistant", "content": "recovered"},
    ])
    orchestrator = AgentOrchestrator(ToolNamespacer(), BrokenGateway(), llama)
    assert asyncio.run(orchestrator.execute_query("fetch it", {"server": [tool("fetch")] })) == "recovered"
    assert "Tool execution failed" in llama.requests[1][0][-2]["content"]


def test_event_sink_receives_turn_tool_and_final_events():
    llama = FakeLlama([
        {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "server__fetch", "arguments": "{}"}}]},
        {"role": "assistant", "content": "done"},
    ])
    sink = FakeSink()
    orchestrator = AgentOrchestrator(ToolNamespacer(), FakeGateway(), llama)
    assert asyncio.run(orchestrator.execute_query("fetch it", {"server": [tool("fetch")]}, sink)) == "done"
    assert [event["type"] for event in sink.events] == [
        "info", "turn_start", "tool_call", "tool_result", "turn_start", "final"
    ]
