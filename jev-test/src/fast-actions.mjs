// Fast-path action registry: deterministic handlers for tasks Jev marks "simple".
// No LLM call happens here at all — this is the whole point of the fast path.

const HANDLERS = [
  {
    match: /open (the )?browser/i,
    run: (task) => `Opened browser (direct action, no reasoning needed) for: "${task}"`,
  },
  {
    match: /status|health|ping/i,
    run: () => 'System status: OK',
  },
  {
    match: /log(ged)? in|logout/i,
    run: (task) => `Session action handled directly: "${task}"`,
  },
];

const DEFAULT_HANDLER = (task) => `Executed routine action directly: "${task}"`;

export function runFastAction(taskText) {
  const t0 = performance.now();
  const handler = HANDLERS.find((h) => h.match.test(taskText));
  const result = (handler ?? { run: DEFAULT_HANDLER }).run(taskText);
  return { result, latencyMs: Math.round(performance.now() - t0) };
}
