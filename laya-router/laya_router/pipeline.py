"""Route a request with the classifier, answer it with the matching tier, escalate if needed."""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field

from .backends import ChatBackend
from .classifier import Decision, make_classifier
from .config import TIERS, Config

UNSURE = re.compile(
    r"\b(i'?m not sure|i am not sure|i don'?t know|i cannot (answer|help|solve)|"
    r"i can'?t (answer|help|solve)|beyond my (ability|capabilit\w+)|too complex for me)\b",
    re.I,
)


@dataclass
class Result:
    reply: str
    tier: str                      # tier that produced the reply
    routed_tier: str               # tier chosen before any escalation
    decision: Decision | None      # None when the tier was forced
    attempts: list[dict] = field(default_factory=list)
    route_ms: float = 0.0
    total_ms: float = 0.0
    usage: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class Pipeline:
    def __init__(self, cfg: Config, classifier=None, backends: dict | None = None):
        self.cfg = cfg
        self.classifier = classifier or make_classifier(cfg)
        self.backends = backends or {t: ChatBackend(b) for t, b in cfg.backends.items()}
        self.available = [t for t in TIERS if t in self.backends]

    def resolve(self, tier: str) -> str:
        """The cheapest configured tier at or above `tier` (else the strongest one)."""
        want = TIERS.index(tier)
        for t in self.available:
            if TIERS.index(t) >= want:
                return t
        return self.available[-1]

    def next_tier(self, tier: str) -> str | None:
        i = self.available.index(tier)
        return self.available[i + 1] if i + 1 < len(self.available) else None

    def route(self, messages: list[dict]) -> tuple[str, Decision]:
        d = self.classifier.classify(messages)
        tier = d.tier
        if d.confidence < self.cfg.min_confidence and tier != TIERS[-1]:
            tier = TIERS[TIERS.index(tier) + 1]  # unsure -> be safe, go one tier up
        return self.resolve(tier), d

    def run(self, messages: list[dict], force_tier: str | None = None) -> Result:
        t0 = time.perf_counter()
        if force_tier:
            tier, decision = self.resolve(force_tier), None
        else:
            tier, decision = self.route(messages)
        route_ms = (time.perf_counter() - t0) * 1000
        result = Result("", tier, tier, decision, route_ms=route_ms)

        while True:
            t1 = time.perf_counter()
            reply, usage = self.backends[tier].chat(messages)
            result.attempts.append({"tier": tier, "ms": (time.perf_counter() - t1) * 1000})
            result.reply, result.tier, result.usage = reply, tier, usage
            nxt = self.next_tier(tier)
            unsure = not reply or UNSURE.search(reply[:400])
            if not (self.cfg.escalate_on_unsure_reply and unsure and nxt):
                break
            tier = nxt

        result.total_ms = (time.perf_counter() - t0) * 1000
        return result
