// Complex-path handler: calls a full reasoning model (Claude) when
// ANTHROPIC_API_KEY is set, otherwise returns a mock plan so the orchestrator
// demo runs without live credentials.

export async function runComplexTask(taskText) {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  const t0 = performance.now();

  if (!apiKey) {
    return {
      mode: 'mock',
      result: `[mock plan] Would call Claude here to reason through: "${taskText}"`,
      latencyMs: Math.round(performance.now() - t0),
    };
  }

  const res = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': apiKey,
      'anthropic-version': '2023-06-01',
    },
    body: JSON.stringify({
      model: 'claude-sonnet-5',
      max_tokens: 512,
      messages: [{ role: 'user', content: taskText }],
    }),
  });

  if (!res.ok) {
    throw new Error(`Complex model call failed: ${res.status} ${await res.text()}`);
  }

  const data = await res.json();
  const latencyMs = Math.round(performance.now() - t0);
  return { mode: 'live', result: data.content?.[0]?.text ?? data, latencyMs };
}
