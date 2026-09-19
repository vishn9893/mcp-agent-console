from __future__ import annotations

import asyncio
import uuid
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .config import Settings
from .llama_client import LlamaCppClient
from .mcp_gateway import MCPGateway, MCPServerSpec
from .namespacer import ToolNamespacer
from .orchestrator import AgentOrchestrator


class ServerInput(BaseModel):
    command: str
    args: list[str] = Field(default_factory=list)
    enabled_tools: list[str] = Field(default_factory=list)


class ConfigInput(BaseModel):
    servers: dict[str, ServerInput]


class ToolTestInput(BaseModel):
    server_id: str
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


INDEX_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MCP Gateway</title><style>
body{font:15px system-ui;margin:0;background:#f6f7f9;color:#20242b}main{max-width:1000px;margin:32px auto;padding:0 20px}
h1{margin-bottom:4px}.muted{color:#697386}.toolbar{display:flex;gap:10px;margin:20px 0}button{border:0;border-radius:7px;padding:9px 14px;background:#2563eb;color:white;cursor:pointer}button.secondary{background:#e5e7eb;color:#20242b}.card{background:white;border:1px solid #e2e5ea;border-radius:10px;margin:14px 0;padding:18px;box-shadow:0 1px 2px #0000000b}.head{display:flex;justify-content:space-between;gap:10px}.status{font-size:13px;padding:4px 8px;border-radius:99px;background:#fee2e2}.connected{background:#dcfce7}.tool{border-top:1px solid #edf0f3;padding:12px 0;display:grid;grid-template-columns:28px 150px 1fr auto;gap:10px;align-items:start}.tool:first-child{border-top:0}.desc{color:#697386}.schema{font:12px ui-monospace,monospace;white-space:pre-wrap;color:#596273}.notice{margin:10px 0;color:#b42318}.ok{color:#087443}
</style></head><body><main><h1>MCP Gateway</h1><div class="muted">Enable only the tools you want llama.cpp and A2A to use.</div>
<div class="toolbar"><button onclick="save()">Save configuration</button><button class="secondary" onclick="connectAll()">Test connections</button><span id="model" class="muted"></span></div><div id="notice"></div><div id="servers"></div>
<section class="card"><h2>Live agent console</h2><div class="muted">Watch model turns, MCP calls, retries, and final responses.</div><div class="toolbar"><input id="question" style="flex:1;padding:9px;border:1px solid #d1d5db;border-radius:7px" value="Fetch https://example.com and summarize it."><button onclick="runAgent()">Run agent</button></div><div id="approvalCard" style="display:none;background:#fffbeb;border:2px solid #eab308;padding:15px;border-radius:7px;margin:15px 0"><h3 style="color:#92400e;margin-top:0">Action requires approval</h3><div id="approvalDetails"></div><input id="rejectionReason" placeholder="Reason if denying" style="width:70%;padding:8px;margin:10px 0"><br><button onclick="respondToGate(true)" style="background:#059669">Approve</button> <button onclick="respondToGate(false)" style="background:#dc2626">Deny</button></div><div id="console" style="background:#111827;color:#e5e7eb;border-radius:8px;padding:12px;min-height:120px;max-height:360px;overflow:auto;font:13px ui-monospace,monospace"></div></section></main>
<script>
let state={servers:{}}; const el=id=>document.getElementById(id);
async function load(){state=await fetch('/api/config').then(r=>r.json()); render(); modelStatus();}
function render(){el('servers').innerHTML=Object.entries(state.servers).map(([id,s])=>`<section class="card"><div class="head"><div><h2>${esc(id)}</h2><div class="muted">${esc(s.command+' '+s.args.join(' '))}</div></div><span class="status ${s.connected?'connected':''}">${s.connected?'connected':'disconnected'}</span></div>${s.error?'<div class="notice">'+esc(s.error)+'</div>':''}${(s.tools||[]).map(t=>`<label class="tool"><input type="checkbox" ${t.enabled?'checked':''} onchange="toggle('${esc(id)}','${esc(t.name)}',this.checked)"><b>${esc(t.name)}</b><span class="desc">${esc(t.description||'')}</span><button class="secondary" onclick="testTool('${esc(id)}','${esc(t.name)}')">Test</button><div></div><div></div><pre class="schema">${esc(JSON.stringify(t.schema,null,2))}</pre><div></div></label>`).join('')}</section>`).join('')||'<div class="card">No MCP servers configured.</div>'}
function toggle(id,name,on){let s=state.servers[id];s.enabled_tools=on?[...new Set([...s.enabled_tools,name])]:s.enabled_tools.filter(x=>x!==name);}
async function save(){let servers={};for(let [id,s] of Object.entries(state.servers))servers[id]={command:s.command,args:s.args,enabled_tools:s.enabled_tools};let r=await fetch('/api/config',{method:'PUT',headers:{'content-type':'application/json'},body:JSON.stringify({servers})});state=await r.json();render();notice('Saved. Servers will reconnect on the next request.','ok');}
async function connectAll(){for(let id of Object.keys(state.servers))await fetch('/api/servers/'+encodeURIComponent(id)+'/connect',{method:'POST'});await load();notice('Connection checks complete.','ok');}
async function testTool(id,name){let r=await fetch('/api/tools/test',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({server_id:id,tool_name:name,arguments:{}})});let j=await r.json();notice(r.ok?name+': '+j.result:'Test failed: '+(j.detail||'unknown error'),r.ok?'ok':'');}
async function modelStatus(){let r=await fetch('/api/model/status');let j=await r.json();el('model').textContent='llama.cpp: '+(j.connected?'connected':'unavailable')+(j.model?' · '+j.model:'');}
function notice(t,c){el('notice').className=c||'notice';el('notice').textContent=t;setTimeout(()=>el('notice').textContent='',5000)} function esc(x){return String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))} load();setInterval(load,10000);
let socket; let currentPendingCallId=null; const sessionId='session_'+Math.random().toString(36).slice(2);
function logEvent(event){const line=document.createElement('div');line.style.cssText='margin:6px 0;padding-left:8px;border-left:3px solid '+(event.type==='error'?'#ef4444':event.type==='final'?'#a855f7':'#60a5fa');line.textContent='['+event.type.toUpperCase()+'] '+event.message;if(event.data){const pre=document.createElement('pre');pre.style.whiteSpace='pre-wrap';pre.textContent=JSON.stringify(event.data,null,2);line.appendChild(pre)}el('console').appendChild(line);el('console').scrollTop=el('console').scrollHeight}
function runAgent(){const question=el('question').value.trim();if(!question)return;el('console').innerHTML='';if(socket&&socket.readyState===WebSocket.OPEN)socket.send(JSON.stringify({type:'start_task',question}));}
function respondToGate(approved){if(!currentPendingCallId||!socket)return;socket.send(JSON.stringify({type:'approval_response',call_id:currentPendingCallId,approved,reason:el('rejectionReason').value}));el('approvalCard').style.display='none';el('rejectionReason').value='';currentPendingCallId=null;}
function setupSocket(){const scheme=location.protocol==='https:'?'wss':'ws';socket=new WebSocket(scheme+'://'+location.host+'/ws/agent/'+sessionId);socket.onmessage=e=>{const event=JSON.parse(e.data);logEvent(event);if(event.type==='awaiting_approval'){currentPendingCallId=event.data.call_id;el('approvalDetails').textContent='Tool '+event.data.tool+' requested with arguments: '+JSON.stringify(event.data.arguments);el('approvalCard').style.display='block';}};socket.onclose=()=>logEvent({type:'info',message:'Agent stream disconnected.'});} setupSocket();
</script></body></html>"""


def create_gateway_api(gateway: MCPGateway, settings: Settings) -> FastAPI:
    app = FastAPI(title="MCP Gateway", docs_url="/api/docs")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return INDEX_HTML

    async def run_agent_socket(websocket: WebSocket, session_id: str) -> None:
        await websocket.accept()
        orchestrator = AgentOrchestrator(
            ToolNamespacer(),
            gateway,
            LlamaCppClient(settings.llama_base_url, settings.llama_api_key, settings.llama_model),
            settings.max_tool_rounds,
        )
        active_task: asyncio.Task[str] | None = None
        try:
            while True:
                payload = await websocket.receive_json()
                if not isinstance(payload, dict):
                    await websocket.send_json({"type": "error", "message": "WebSocket payload must be a JSON object.", "data": None})
                    continue
                message_type = payload.get("type", "start_task")
                if message_type == "approval_response":
                    call_id = payload.get("call_id")
                    if not isinstance(call_id, str) or not orchestrator.resolve_approval(
                        call_id, bool(payload.get("approved", False)), str(payload.get("reason", ""))
                    ):
                        await websocket.send_json({"type": "error", "message": f"Stale or unknown approval request: {call_id}", "data": None})
                    continue
                if message_type != "start_task":
                    await websocket.send_json({"type": "error", "message": f"Unknown WebSocket message type: {message_type}", "data": None})
                    continue
                question = payload.get("question")
                if not isinstance(question, str) or not question.strip():
                    await websocket.send_json({"type": "error", "message": "Missing 'question' parameter.", "data": None})
                    continue
                if active_task and not active_task.done():
                    await websocket.send_json({"type": "error", "message": "An agent task is already running for this session.", "data": None})
                    continue
                await gateway.ensure_connected()
                active_task = asyncio.create_task(orchestrator.execute_query(
                    question.strip(), gateway.enabled_tools_by_server(), event_sink=websocket
                ), name=f"agent-session-{session_id}")
        except WebSocketDisconnect:
            pass
        finally:
            if active_task and not active_task.done():
                active_task.cancel()
                await asyncio.gather(active_task, return_exceptions=True)
            for future in orchestrator.approval_futures.values():
                if not future.done():
                    future.cancel()
            orchestrator.approval_futures.clear()

    @app.websocket("/ws/agent")
    async def agent_websocket(websocket: WebSocket) -> None:
        await run_agent_socket(websocket, str(uuid.uuid4()))

    @app.websocket("/ws/agent/{session_id}")
    async def session_agent_websocket(websocket: WebSocket, session_id: str) -> None:
        await run_agent_socket(websocket, session_id)

    @app.get("/api/config")
    async def get_config() -> dict[str, Any]:
        await gateway.ensure_connected()
        return gateway.snapshot()

    @app.put("/api/config")
    async def put_config(payload: ConfigInput) -> dict[str, Any]:
        servers = {server_id: MCPServerSpec(**value.model_dump()) for server_id, value in payload.servers.items()}
        await gateway.save(servers)
        return gateway.snapshot()

    @app.post("/api/servers/{server_id}/connect")
    async def connect(server_id: str) -> dict[str, Any]:
        try:
            await gateway.connect_server(server_id)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return gateway.snapshot()

    @app.post("/api/servers/{server_id}/disconnect")
    async def disconnect(server_id: str) -> dict[str, Any]:
        await gateway.disconnect_server(server_id)
        return gateway.snapshot()

    @app.post("/api/tools/test")
    async def test_tool(payload: ToolTestInput) -> dict[str, str]:
        try:
            await gateway.connect_server(payload.server_id)
            result = await gateway.call(payload.server_id, payload.tool_name, payload.arguments)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"result": result}

    @app.get("/api/model/status")
    async def model_status() -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(f"{settings.llama_base_url}/models", headers={"Authorization": f"Bearer {settings.llama_api_key}"})
                response.raise_for_status()
            return {"connected": True, "model": settings.llama_model}
        except Exception as exc:
            return {"connected": False, "model": settings.llama_model, "error": str(exc)}

    return app
