import { classifyTask } from './jev-client.mjs';
import { runFastAction } from './fast-actions.mjs';
import { runComplexTask } from './complex-model.mjs';

export async function handleTask(taskText, { state } = {}) {
  const decision = await classifyTask(taskText, { state });
  const category = decision.category ?? decision.raw?.category ?? 'complex';

  if (category === 'simple') {
    const { result, latencyMs } = runFastAction(taskText);
    return {
      task: taskText,
      path: 'fast',
      decision,
      result,
      totalLatencyMs: decision.latencyMs + latencyMs,
    };
  }

  const { result, latencyMs, mode } = await runComplexTask(taskText);
  return {
    task: taskText,
    path: 'complex',
    decision,
    result,
    complexModelMode: mode,
    totalLatencyMs: decision.latencyMs + latencyMs,
  };
}
