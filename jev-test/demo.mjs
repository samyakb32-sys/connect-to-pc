import { handleTask } from './src/orchestrator.mjs';

const tasks = [
  'open the browser',
  'check system status',
  'write a summary of this week\'s sales report for the leadership team',
  'design a database schema for a multi-tenant SaaS app with role-based access',
  'log out the current user',
];

for (const task of tasks) {
  const outcome = await handleTask(task);
  console.log('\n---');
  console.log('Task     :', outcome.task);
  console.log('Path     :', outcome.path.toUpperCase(), `(Jev mode: ${outcome.decision.mode})`);
  console.log('Decision :', outcome.decision.category ?? outcome.decision.raw?.category, '| confidence:', outcome.decision.confidence ?? 'n/a');
  console.log('Result   :', outcome.result);
  console.log('Latency  :', `${outcome.totalLatencyMs}ms total`);
}
