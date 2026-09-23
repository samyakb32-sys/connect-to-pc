"""Command line: python -m laya_router {ask,route,serve,bench} ..."""

from __future__ import annotations

import argparse
import json

from .bench import load_prompts, run_bench
from .config import TIERS, load_config
from .pipeline import Pipeline


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="laya_router")
    ap.add_argument("-c", "--config", help="config file (default: $LAYA_ROUTER_CONFIG or config.yaml)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ask = sub.add_parser("ask", help="route and answer one message")
    ask.add_argument("text")
    ask.add_argument("--tier", choices=TIERS, help="skip routing and force a tier")

    route = sub.add_parser("route", help="only show which tier a message would go to")
    route.add_argument("text")

    serve = sub.add_parser("serve", help="run the OpenAI-compatible server")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)

    bench = sub.add_parser("bench", help="compare routed vs always-big-model latency")
    bench.add_argument("prompts", help="JSON Lines file of {prompt, tier}")
    bench.add_argument("--baseline", choices=TIERS, default="hard")
    bench.add_argument("--route-only", action="store_true", help="only check routing accuracy, no LLM calls")
    bench.add_argument("--out", help="write per-prompt results as JSON")

    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.cmd == "route":
        # Routing needs no LLM backends, so this works before any are running.
        tier, d = Pipeline(cfg).route([{"role": "user", "content": args.text}])
        print(json.dumps({"tier": tier, "decision": d.__dict__}, indent=2))
    elif args.cmd == "ask":
        r = Pipeline(cfg).run([{"role": "user", "content": args.text}], force_tier=args.tier)
        print(r.reply)
        hops = " -> ".join(f"{a['tier']} ({a['ms'] / 1000:.1f}s)" for a in r.attempts)
        conf = f", {r.decision.source} confidence {r.decision.confidence:.2f}" if r.decision else ""
        print(f"\n[route {r.route_ms:.0f} ms{conf} | {hops} | total {r.total_ms / 1000:.1f}s]")
    elif args.cmd == "serve":
        import uvicorn

        from .server import create_app

        uvicorn.run(create_app(Pipeline(cfg)), host=args.host, port=args.port)
    elif args.cmd == "bench":
        out = run_bench(Pipeline(cfg), load_prompts(args.prompts), args.baseline, args.route_only)
        if args.out:
            with open(args.out, "w") as f:
                json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
