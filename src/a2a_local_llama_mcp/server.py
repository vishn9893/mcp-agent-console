from __future__ import annotations

import uuid

import uvicorn
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill, Artifact, Part, TaskArtifactUpdateEvent, TaskState, TaskStatus, TaskStatusUpdateEvent
from a2a.helpers import new_task_from_user_message
from starlette.applications import Starlette
from starlette.routing import Mount
import typer

from .agent import LocalMCPAgent
from .config import settings
from .gateway_api import create_gateway_api
from .mcp_gateway import MCPGateway, MCPServerSpec


class LocalAgentExecutor(AgentExecutor):
    def __init__(self, gateway: MCPGateway) -> None:
        self.agent = LocalMCPAgent(settings, gateway)

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        if not context.message:
            raise ValueError("No A2A message provided")
        task = context.current_task or new_task_from_user_message(context.message)
        if context.current_task is None:
            await event_queue.enqueue_event(task)
        await event_queue.enqueue_event(TaskStatusUpdateEvent(
            status=TaskStatus(state=TaskState.TASK_STATE_WORKING),
            context_id=task.context_id, task_id=task.id,
        ))
        answer = await self.agent.answer(context.get_user_input())
        await event_queue.enqueue_event(TaskArtifactUpdateEvent(
            append=False, context_id=task.context_id, task_id=task.id, last_chunk=True,
            artifact=Artifact(artifact_id=str(uuid.uuid4()), name="answer", description="MCP-backed answer", parts=[Part(text=answer)]),
        ))
        await event_queue.enqueue_event(TaskStatusUpdateEvent(
            status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
            context_id=task.context_id, task_id=task.id,
        ))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("Cancellation is not implemented in this example")


def build_app(host: str = settings.a2a_host, port: int = settings.a2a_port) -> Starlette:
    gateway = MCPGateway(settings.mcp_config_path, {
        "fetch": MCPServerSpec(command=settings.mcp_command, args=list(settings.mcp_args), enabled_tools=["fetch"]),
    })
    card = AgentCard(
        name="Local llama.cpp MCP Agent",
        description="An A2A agent that calls a local llama.cpp model and MCP tools directly.",
        supported_interfaces=[AgentInterface(protocol_binding="JSONRPC", url=f"http://{host}:{port}/")],
        version="0.1.0", default_input_modes=["text"], default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[AgentSkill(id="mcp-answer", name="MCP-backed answers", description="Answer questions using configured MCP tools.", tags=["mcp", "llama.cpp"])],
    )
    handler = DefaultRequestHandler(agent_executor=LocalAgentExecutor(gateway), task_store=InMemoryTaskStore(), agent_card=card)
    api = create_gateway_api(gateway, settings)
    return Starlette(routes=[*create_agent_card_routes(card), *create_jsonrpc_routes(handler, rpc_url="/"), Mount("/", app=api)])


def main() -> None:
    def run(host: str = settings.a2a_host, port: int = settings.a2a_port) -> None:
        uvicorn.run(build_app(host, port), host=host, port=port)
    typer.run(run)
