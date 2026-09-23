# laya-router

Use a big open model without waiting for it on every message.
[Laya](https://huggingface.co/convaiinnovations/laya) (a ~400M model that makes a decision in ~30 ms)
reads each request first and sends it to the cheapest model that can handle it:

```
request ──► Laya (~30 ms): easy / medium / hard?
              ├─ easy   ──► small LLM (3–8B)            fast
              ├─ medium ──► big LLM, thinking OFF       medium
              └─ hard   ──► big LLM, thinking ON        slow, only when needed
Laya unsure (confidence < min_confidence) ──► one tier up
small model replies "I'm not sure…"       ──► retried one tier up
```

Laya never generates text, so it can't be merged into the LLM. The speedup comes
from not calling the big model for easy requests.

## Setup

```bash
cd laya-router
pip install -r requirements.txt         # includes laya (torch + transformers)
cp config.example.yaml config.yaml      # set your models / servers
```

Serve your LLMs from any OpenAI-compatible server. With Ollama, for example:

```bash
ollama pull qwen3:4b        # easy tier
ollama pull qwen3:32b       # medium + hard tiers
```

The model names are examples. Use any small and big model and put them in `config.yaml`.
To try the pipeline before downloading Laya, set `classifier: heuristic` (keyword rules).

## Use

```bash
# Which tier would this go to? (no LLM needed)
python -m laya_router route "Plan a 3-day Goa trip under 15000 rupees"

# Route and answer
python -m laya_router ask "hi, how are you?"
python -m laya_router ask "Prove sqrt(2) is irrational" --tier hard   # force a tier

# OpenAI-compatible server: point Open WebUI, scripts, SDKs at http://127.0.0.1:8080/v1
python -m laya_router serve
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "auto", "messages": [{"role": "user", "content": "hi"}]}'
```

`model: "auto"` routes with Laya, and `"easy"` / `"medium"` / `"hard"` force a tier. Each
response has a `routing` field showing Laya's decision, any escalations and timings.
Streaming is not supported yet.

## Measure it on your own requests

`prompts/sample.jsonl` holds 20 labelled everyday requests. Put your own requests in the same format,
because the router only helps if it is right on the kind of requests you actually send.

```bash
# 1. Routing accuracy only (fast, no LLM calls)
python -m laya_router bench prompts/sample.jsonl --route-only

# 2. Full run: routed vs always using the big model with thinking
python -m laya_router bench prompts/sample.jsonl --baseline hard --out results.json
```

The report shows:
- `classifier_accuracy`: how often Laya picked the labelled tier
- `under_routed`: hard requests sent to a weaker model. This is the costly mistake, so keep it near 0
  by raising `min_confidence`.
- `routed` vs `baseline` latency and `speedup`
- `escalations`: requests that the small model gave up on and sent up a tier

As a pipeline check, with mock LLMs (small 0.2 s, big 1.5 s, big + thinking 4 s) and the heuristic
classifier, the sample set gave 90% routing accuracy and a 2.6× average speedup. Real numbers
depend on your models, your hardware and Laya's accuracy on your requests.

## Laya's limits (from its model card)

- **The base checkpoints are not reliable zero-shot deciders.** On its typed-decisions benchmark,
  base Laya is close to chance. The 0.766 score comes from a fine-tuned checkpoint. Run `bench --route-only`
  on your own labelled requests. If accuracy is low, fine-tune Laya on them with the
  [fine-tuning notebook](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb).
- **It ships over-confident.** Its README recommends fitting a temperature on your own data.
  Until you do, keep `min_confidence` conservative.
- This router asks one `choice` question. Laya's README reports `choice` as its most reliable question type,
  while `noul` (yes/no) can get stuck on its labels and `score` is the weakest. Wording the tier descriptions
  in `laya_router/classifier.py` (`QUESTIONS`) to match your requests is the cheapest improvement.
- For Hindi and other non-Latin scripts, Laya's `Router` switches to its multilingual checkpoint automatically.

## Layout

```
laya_router/
  classifier.py   Laya (in-process or laya-serve over HTTP) + keyword fallback
  pipeline.py     route -> call tier -> escalate on low confidence / unsure reply
  backends.py     OpenAI-compatible chat client (strips <think> blocks)
  server.py       OpenAI-compatible /v1/chat/completions server
  bench.py        routing accuracy + latency vs always-big baseline
config.example.yaml
prompts/sample.jsonl
tests/            pytest, no GPU or network needed
```

Run the tests with `pip install pytest && python -m pytest tests`.
