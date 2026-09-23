"""Minimal client for OpenAI-compatible /v1/chat/completions endpoints."""

from __future__ import annotations

import re

import httpx

from .config import Backend

_THINK = re.compile(r"<think>.*?</think>\s*", re.S)


class ChatBackend:
    def __init__(self, cfg: Backend, client: httpx.Client | None = None):
        self.cfg = cfg
        self.client = client or httpx.Client(timeout=cfg.timeout_s)

    def chat(self, messages: list[dict]) -> tuple[str, dict]:
        """Return (reply text without <think> blocks, usage dict)."""
        body = {
            "model": self.cfg.model,
            "messages": messages,
            "max_tokens": self.cfg.max_tokens,
            **self.cfg.extra_body,
        }
        resp = self.client.post(
            self.cfg.base_url.rstrip("/") + "/chat/completions",
            json=body,
            headers={"Authorization": f"Bearer {self.cfg.api_key}"},
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"].get("content") or ""
        return _THINK.sub("", text).strip(), data.get("usage") or {}
