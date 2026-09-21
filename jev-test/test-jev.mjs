// Quick test script for calling Jev (TypeSafe AI) via Vercel AI Gateway.
// Prereqs:
//   1. npm install ai
//   2. export AI_GATEWAY_API_KEY=your_key_from_vercel_dashboard
//   3. Confirm "typesafe/jev" is actually listed in your Vercel AI Gateway model catalog

import { generateObject } from 'ai';
import { z } from 'zod';

const state = {
  ticket_text: 'App crashes every time I try to upload a photo larger than 5MB.',
};

const result = await generateObject({
  model: 'typesafe/jev',
  schema: z.object({
    category: z.enum(['bug', 'billing', 'feature_request', 'other']),
    urgency: z.number().min(0).max(1),
  }),
  prompt: JSON.stringify({
    state,
    questions: {
      category: { type: 'choice', options: ['bug', 'billing', 'feature_request', 'other'] },
      urgency: { type: 'score', range: [0, 1] },
    },
  }),
});

console.log(JSON.stringify(result.object, null, 2));
