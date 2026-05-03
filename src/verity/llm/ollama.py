"""Chat client for a local Ollama server (/api/chat, with tool calling)."""

from __future__ import annotations

import httpx

from verity.config import get_settings


class OllamaChatModel:
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        settings = get_settings()
        self._model = model or settings.llm_model
        self._base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self._timeout = timeout

    def chat(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self._model,
            "messages": messages,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
        resp = httpx.post(
            f"{self._base_url}/api/chat", json=payload, timeout=self._timeout
        )
        resp.raise_for_status()
        data = resp.json()
        message: dict[str, object] = data["message"]
        return message
