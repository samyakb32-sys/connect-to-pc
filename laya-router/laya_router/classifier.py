"""Difficulty classifiers. Each one maps a chat request to easy / medium / hard.

`LayaClassifier` runs Laya in-process, `LayaHTTPClassifier` calls a running
`laya-serve`, and `HeuristicClassifier` is a keyword fallback that needs no model
(useful for trying the pipeline before downloading Laya, and for tests).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import httpx

from .config import TIERS, Config

# One `choice` question: Laya's README reports `choice` as its most reliable
# primitive, while `noul` can stick to its labels and `score` is the weakest.
QUESTIONS = {
    "difficulty": {
        "type": "choice",
        "instructions": "How much reasoning does an AI assistant need to answer this request well?",
        "criteria": {
            "easy": "greeting, small talk, a simple fact, a unit conversion, a short rewrite or fix of one sentence",
            "medium": "writing an email or message, summarising, explaining a concept, translating, a short piece of code",
            "hard": "multi-step planning, math or logic problems, debugging or designing software, comparing many options, deep analysis",
        },
    }
}


@dataclass
class Decision:
    tier: str
    confidence: float
    source: str
    latency_ms: float
    probabilities: dict[str, float] = field(default_factory=dict)


def request_text(messages: list[dict]) -> str:
    """The text of the latest user message (handles OpenAI content-part lists)."""
    for m in reversed(messages):
        if m.get("role") != "user":
            continue
        content = m.get("content") or ""
        if isinstance(content, list):
            content = " ".join(p.get("text", "") for p in content if p.get("type") == "text")
        return content.strip()
    return ""


def _from_laya_result(result: dict, source: str, t0: float) -> Decision:
    ans = result["answers"]["difficulty"]
    return Decision(
        tier=ans["choice"],
        confidence=float(ans["confidence"]),
        probabilities=ans.get("probabilities", {}),
        source=source,
        latency_ms=(time.perf_counter() - t0) * 1000,
    )


class LayaClassifier:
    def __init__(self, device: str | None = None):
        from laya import Router  # heavy import (torch); only when this classifier is used

        self.router = Router(preload=True, device=device)

    def classify(self, messages: list[dict]) -> Decision:
        t0 = time.perf_counter()
        result = self.router.predict({"request": request_text(messages)}, QUESTIONS)
        return _from_laya_result(result, "laya", t0)


class LayaHTTPClassifier:
    def __init__(self, url: str, client: httpx.Client | None = None):
        self.url = url.rstrip("/") + "/v1/systemone"
        self.client = client or httpx.Client(timeout=30)

    def classify(self, messages: list[dict]) -> Decision:
        t0 = time.perf_counter()
        resp = self.client.post(
            self.url, json={"state": {"request": request_text(messages)}, "questions": QUESTIONS}
        )
        resp.raise_for_status()
        return _from_laya_result(resp.json(), "laya-http", t0)


_HARD = re.compile(
    r"\b(plan|design|architect\w*|debug|prove|proof|derive|optimi[sz]e|algorithm|analy[sz]e|"
    r"compare|trade-?offs?|step[- ]by[- ]step|calculate|solve|strategy|itinerary|refactor)\b",
    re.I,
)
_MEDIUM = re.compile(
    r"\b(write|draft|email|letter|summari[sz]e|explain|translate|code|function|script|"
    r"rewrite|describe|list|outline|review)\b",
    re.I,
)


class HeuristicClassifier:
    """Keyword/length rules. Crude, but free and instant."""

    def classify(self, messages: list[dict]) -> Decision:
        t0 = time.perf_counter()
        text = request_text(messages)
        words = len(text.split())
        if _HARD.search(text) or words > 120:
            tier, conf = "hard", 0.7
        elif _MEDIUM.search(text) or words > 40:
            tier, conf = "medium", 0.7
        else:
            tier, conf = "easy", 0.8
        return Decision(tier, conf, "heuristic", (time.perf_counter() - t0) * 1000)


def make_classifier(cfg: Config):
    if cfg.classifier == "laya":
        return LayaClassifier(device=cfg.laya_device)
    if cfg.classifier == "laya-http":
        return LayaHTTPClassifier(cfg.laya_url)
    if cfg.classifier == "heuristic":
        return HeuristicClassifier()
    raise ValueError(f"unknown classifier {cfg.classifier!r}; use laya, laya-http or heuristic")


assert set(QUESTIONS["difficulty"]["criteria"]) == set(TIERS)
