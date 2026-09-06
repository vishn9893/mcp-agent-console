from __future__ import annotations

import asyncio
import json
import uuid

import httpx
import typer


async def send(url: str, question: str) -> str:
    payload = {
        "jsonrpc": "2.0", "id": str(uuid.uuid4()), "method": "message/send",
        "params": {"message": {"kind": "message", "messageId": str(uuid.uuid4()), "role": "ROLE_USER", "parts": [{"kind": "text", "text": question}]}},
    }
    async with httpx.AsyncClient(timeout=180) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        body = response.json()
    if "error" in body:
        raise RuntimeError(json.dumps(body["error"], indent=2))
    result = body.get("result", {})
    texts: list[str] = []
    def walk(value: object) -> None:
        if isinstance(value, dict):
            if isinstance(value.get("text"), str): texts.append(value["text"])
            for child in value.values(): walk(child)
        elif isinstance(value, list):
            for child in value: walk(child)
    walk(result)
    return texts[-1] if texts else json.dumps(result, indent=2)


def main() -> None:
    def run(question: str, url: str = "http://127.0.0.1:9999/") -> None:
        print(asyncio.run(send(url, question)))
    typer.run(run)

