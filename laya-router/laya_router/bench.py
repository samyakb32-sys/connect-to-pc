"""Measure the router: does it pick the right tier, and how much time does it save?

Prompts file: JSON Lines, one {"prompt": "...", "tier": "easy|medium|hard"} per line
("tier" is your expected label and is optional).
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path

from .config import TIERS
from .pipeline import Pipeline


def load_prompts(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _stats(ms: list[float]) -> str:
    return f"mean {statistics.mean(ms) / 1000:6.2f}s | p50 {statistics.median(ms) / 1000:6.2f}s"


def run_bench(pipeline: Pipeline, prompts: list[dict], baseline: str = "hard", route_only: bool = False) -> dict:
    rows = []
    for i, p in enumerate(prompts, 1):
        msgs = [{"role": "user", "content": p["prompt"]}]
        tier, d = pipeline.route(msgs)
        row = {"prompt": p["prompt"], "label": p.get("tier"), "predicted": d.tier, "conf": d.confidence,
               "tier": tier, "route_ms": d.latency_ms}
        if not route_only:
            routed = pipeline.run(msgs)
            base = pipeline.run(msgs, force_tier=baseline)
            row |= {"final_tier": routed.tier, "routed_ms": routed.total_ms, "baseline_ms": base.total_ms}
        rows.append(row)
        print(f"[{i}/{len(prompts)}] {row['label'] or '?':6} -> {tier:6} ({d.tier} @ {d.confidence:.2f})  {p['prompt'][:60]}")

    report: dict = {"n": len(rows), "tiers": dict(Counter(r["tier"] for r in rows))}
    labelled = [r for r in rows if r["label"]]
    if labelled:
        idx = TIERS.index
        report["classifier_accuracy"] = sum(r["predicted"] == r["label"] for r in labelled) / len(labelled)
        report["routed_accuracy"] = sum(r["tier"] == r["label"] for r in labelled) / len(labelled)
        # Under-routing is the costly mistake: a hard task answered by a weaker model.
        report["under_routed"] = sum(idx(r["tier"]) < idx(r["label"]) for r in labelled)
        report["over_routed"] = sum(idx(r["tier"]) > idx(r["label"]) for r in labelled)
    report["route_ms_mean"] = statistics.mean(r["route_ms"] for r in rows)
    if not route_only:
        routed = [r["routed_ms"] for r in rows]
        base = [r["baseline_ms"] for r in rows]
        report |= {"routed": _stats(routed), "baseline": _stats(base),
                   "speedup": statistics.mean(base) / statistics.mean(routed),
                   "escalations": sum(r["final_tier"] != r["tier"] for r in rows)}

    print("\n=== report ===")
    for k, v in report.items():
        print(f"{k:16}: {v:.3f}" if isinstance(v, float) else f"{k:16}: {v}")
    return {"report": report, "rows": rows}
