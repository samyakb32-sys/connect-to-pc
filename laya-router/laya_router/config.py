"""Configuration: which classifier to use and which LLM backs each tier."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

TIERS = ("easy", "medium", "hard")


@dataclass
class Backend:
    """One OpenAI-compatible chat endpoint (Ollama, vLLM, llama.cpp server, LM Studio...)."""

    base_url: str
    model: str
    api_key: str = "none"
    max_tokens: int = 1024
    timeout_s: float = 600.0
    # Sent as-is in the request body, e.g. to switch a model's thinking mode on/off.
    extra_body: dict = field(default_factory=dict)


@dataclass
class Config:
    # "laya" (in-process), "laya-http" (a running `laya-serve`), or "heuristic" (no model).
    classifier: str = "laya"
    laya_url: str = "http://localhost:8000"
    laya_device: str | None = None
    # Below this Laya confidence, the request goes one tier up.
    min_confidence: float = 0.6
    # If a lower-tier model says it is unsure, retry on the next tier.
    escalate_on_unsure_reply: bool = True
    backends: dict[str, Backend] = field(default_factory=dict)


def load_config(path: str | os.PathLike | None = None) -> Config:
    path = path or os.environ.get("LAYA_ROUTER_CONFIG", "config.yaml")
    raw = yaml.safe_load(Path(path).read_text()) or {}
    backends = {name: Backend(**b) for name, b in (raw.pop("backends", None) or {}).items()}
    unknown = set(backends) - set(TIERS)
    if unknown:
        raise ValueError(f"unknown tier(s) in backends: {sorted(unknown)}; use {TIERS}")
    if not backends:
        raise ValueError("config needs at least one backend under 'backends'")
    return Config(**raw, backends=backends)
