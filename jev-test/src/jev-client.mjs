// Jev client: calls TypeSafe's Jev via Vercel AI Gateway when AI_GATEWAY_API_KEY
// is set, otherwise falls back to a local heuristic mock so the orchestrator is
// runnable and demoable without live credentials.

const GATEWAY_URL = 'https://ai-gateway.vercel.sh/v1/chat/completions';
const MODEL = 'typesafe-ai/jev';

const COMPLEX_HINTS = [
  'write', 'draft', 'design', 'plan', 'explain', 'summarize', 'refactor',
  'generate', 'create', 'compose', 'analyze', 'debug', 'architecture',
];

function mockClassify(taskText) {
  const lower = taskText.toLowerCase();
  const hit = COMPLEX_HINTS.find((w) => lower.includes(w));
  const isComplex = Boolean(hit) || taskText.split(' ').length > 14;
  return {
    mode: 'mock',
    category: isComplex ? 'complex' : 'simple',
    confidence: isComplex ? 0.82 : 0.91,
    reason: isComplex
      ? `matched complexity signal: "${hit ?? 'long/ambiguous request'}"`
      : 'short, well-defined action with no reasoning/generation signal',
  };
}

export async function classifyTask(taskText, { state = {} } = {}) {
  const apiKey = process.env.AI_GATEWAY_API_KEY;
  const t0 = performance.now();

  if (!apiKey) {
    const result = mockClassify(taskText);
    return { ...result, latencyMs: Math.round(performance.now() - t0) };
  }

  const body = {
    model: MODEL,
    messages: [
      {
        role: 'user',
        content: JSON.stringify({
          state: { task: taskText, ...state },
          questions: {
            category: {
              type: 'choice',
              options: ['simple', 'complex'],
              prompt: 'Can this be handled by a direct action/lookup (simple), or does it need open-ended reasoning or content generation (complex)?',
            },
          },
        }),
      },
    ],
  };

  const res = await fetch(GATEWAY_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    throw new Error(`Jev call failed: ${res.status} ${await res.text()}`);
  }

  const data = await res.json();
  const latencyMs = Math.round(performance.now() - t0);
  // NOTE: exact response shape depends on Jev's actual schema (docs.typesafe.ai) —
  // adjust this parse once you've inspected a live response.
  return { mode: 'live', raw: data, latencyMs };
}
