import json

import httpx
import pytest
from fastapi.testclient import TestClient

from laya_router.backends import ChatBackend
from laya_router.bench import run_bench
from laya_router.classifier import Decision, HeuristicClassifier, LayaHTTPClassifier, request_text
from laya_router.config import Backend, Config, load_config
from laya_router.pipeline import Pipeline
from laya_router.server import create_app


class FakeClassifier:
    def __init__(self, tier, confidence=0.9):
        self.decision = Decision(tier, confidence, "fake", 1.0)

    def classify(self, messages):
        return self.decision


class FakeBackend:
    def __init__(self, name, reply="ok"):
        self.cfg = Backend(base_url="http://x", model=name)
        self.reply = reply
        self.calls = 0

    def chat(self, messages):
        self.calls += 1
        return self.reply, {"total_tokens": 3}


def make(tier, confidence=0.9, replies=None, tiers=("easy", "medium", "hard"), **cfg):
    replies = replies or {}
    backends = {t: FakeBackend(t, replies.get(t, f"answer from {t}")) for t in tiers}
    return Pipeline(Config(**cfg), FakeClassifier(tier, confidence), backends), backends


MSG = [{"role": "user", "content": "hello"}]


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_routes_to_classified_tier(tier):
    p, b = make(tier)
    r = p.run(MSG)
    assert r.tier == r.routed_tier == tier
    assert r.reply == f"answer from {tier}"
    assert b[tier].calls == 1 and sum(x.calls for x in b.values()) == 1


def test_low_confidence_goes_one_tier_up():
    p, _ = make("easy", confidence=0.4, min_confidence=0.6)
    assert p.run(MSG).tier == "medium"


def test_unsure_reply_escalates_until_confident():
    p, b = make("easy", replies={"easy": "I'm not sure about that.", "medium": ""})
    r = p.run(MSG)
    assert [a["tier"] for a in r.attempts] == ["easy", "medium", "hard"]
    assert r.tier == "hard" and r.routed_tier == "easy"


def test_escalation_can_be_disabled():
    p, _ = make("easy", replies={"easy": "I don't know"}, escalate_on_unsure_reply=False)
    assert p.run(MSG).tier == "easy"


def test_missing_tier_falls_through_to_stronger_one():
    p, _ = make("medium", tiers=("easy", "hard"))
    assert p.run(MSG).tier == "hard"
    p, _ = make("hard", tiers=("easy",))
    assert p.run(MSG).tier == "easy"


def test_forced_tier_skips_classifier():
    p, _ = make("easy")
    r = p.run(MSG, force_tier="hard")
    assert r.tier == "hard" and r.decision is None


def test_request_text_uses_latest_user_message_and_content_parts():
    msgs = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "x"},
        {"role": "user", "content": [{"type": "text", "text": "new"}, {"type": "image_url"}]},
    ]
    assert request_text(msgs) == "new"


@pytest.mark.parametrize("text,tier", [
    ("hi there", "easy"),
    ("write an email to my landlord", "medium"),
    ("plan a 3 day trip to Goa", "hard"),
])
def test_heuristic_classifier(text, tier):
    assert HeuristicClassifier().classify([{"role": "user", "content": text}]).tier == tier


def test_laya_http_classifier_parses_systemone_response():
    seen = {}

    def handler(req):
        seen["url"], seen["body"] = str(req.url), json.loads(req.content)
        return httpx.Response(200, json={"answers": {"difficulty": {
            "type": "choice", "choice": "hard", "confidence": 0.77,
            "probabilities": {"easy": 0.1, "medium": 0.13, "hard": 0.77}}}})

    c = LayaHTTPClassifier("http://laya:8000/", httpx.Client(transport=httpx.MockTransport(handler)))
    d = c.classify([{"role": "user", "content": "prove it"}])
    assert (d.tier, d.confidence, d.source) == ("hard", 0.77, "laya-http")
    assert seen["url"] == "http://laya:8000/v1/systemone"
    assert seen["body"]["state"] == {"request": "prove it"}
    assert seen["body"]["questions"]["difficulty"]["type"] == "choice"


def test_chat_backend_sends_extra_body_and_strips_think():
    seen = {}

    def handler(req):
        seen["url"], seen["body"] = str(req.url), json.loads(req.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "<think>hmm</think>\n42"}}],
                                         "usage": {"total_tokens": 5}})

    cfg = Backend(base_url="http://llm/v1/", model="m", max_tokens=7,
                  extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    text, usage = ChatBackend(cfg, httpx.Client(transport=httpx.MockTransport(handler))).chat(MSG)
    assert text == "42" and usage == {"total_tokens": 5}
    assert seen["url"] == "http://llm/v1/chat/completions"
    assert seen["body"]["model"] == "m" and seen["body"]["max_tokens"] == 7
    assert seen["body"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_server_openai_shape_and_forced_tier():
    p, _ = make("easy")
    client = TestClient(create_app(p))
    r = client.post("/v1/chat/completions", json={"model": "auto", "messages": MSG}).json()
    assert r["choices"][0]["message"]["content"] == "answer from easy"
    assert r["model"] == "easy" and r["routing"]["tier"] == "easy"
    r = client.post("/v1/chat/completions", json={"model": "hard", "messages": MSG}).json()
    assert r["routing"]["tier"] == "hard"
    assert client.post("/v1/chat/completions", json={"messages": MSG, "stream": True}).status_code == 400
    assert [m["id"] for m in client.get("/v1/models").json()["data"]] == ["auto", "easy", "medium", "hard"]


def test_bench_reports_accuracy_and_under_routing(capsys):
    p, _ = make("easy")
    prompts = [{"prompt": "a", "tier": "easy"}, {"prompt": "b", "tier": "hard"}]
    rep = run_bench(p, prompts)["report"]
    assert rep["classifier_accuracy"] == 0.5 and rep["under_routed"] == 1 and rep["over_routed"] == 0
    assert "speedup" in rep


def test_load_config(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("classifier: heuristic\nbackends:\n  easy: {base_url: http://a/v1, model: s}\n")
    cfg = load_config(f)
    assert cfg.classifier == "heuristic" and cfg.backends["easy"].model == "s"
    f.write_text("backends:\n  tiny: {base_url: http://a/v1, model: s}\n")
    with pytest.raises(ValueError, match="unknown tier"):
        load_config(f)
