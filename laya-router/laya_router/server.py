"""OpenAI-compatible HTTP server, so any chat UI or SDK can use the router as one model.

model="auto" (or anything unknown) routes with Laya; model="easy"/"medium"/"hard" forces a tier.
"""

from __future__ import annotations

import time
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import TIERS
from .pipeline import Pipeline


class ChatRequest(BaseModel):
    model: str = "auto"
    messages: list[dict]
    stream: bool = False


def create_app(pipeline: Pipeline) -> FastAPI:
    app = FastAPI(title="laya-router")

    @app.get("/v1/models")
    def models():
        ids = ["auto", *pipeline.available]
        return {"object": "list", "data": [{"id": i, "object": "model", "owned_by": "laya-router"} for i in ids]}

    @app.post("/v1/route")
    def route(req: ChatRequest):
        tier, d = pipeline.route(req.messages)
        return {"tier": tier, "decision": d.__dict__}

    @app.post("/v1/chat/completions")
    def chat(req: ChatRequest):
        if req.stream:
            raise HTTPException(400, "streaming is not supported by this prototype; send stream=false")
        force = req.model if req.model in TIERS else None
        r = pipeline.run(req.messages, force_tier=force)
        routing = r.to_dict()
        routing.pop("reply")
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": pipeline.backends[r.tier].cfg.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": r.reply}, "finish_reason": "stop"}],
            "usage": r.usage,
            "routing": routing,
        }

    return app
