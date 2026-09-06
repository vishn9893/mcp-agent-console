from __future__ import annotations

from typing import Any

import httpx


class LlamaCppClient:
    """Minimal OpenAI-compatible client for llama-server; no LLM framework required."""

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 120) -> None:
        self.endpoint = f"{base_url.rstrip('/')}/chat/completions"
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.model = model
        self.timeout = timeout

    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(self.endpoint, headers=self.headers, json=payload)
            response.raise_for_status()
            body = response.json()
        try:
            return body["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected llama.cpp response: {body!r}") from exc
