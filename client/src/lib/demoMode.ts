/**
 * Static-demo mode.
 *
 * When VITE_DEMO_MODE=true, the client runs entirely without a backend:
 *   - all API calls are stubbed out with hardcoded responses (mockApi.ts)
 *   - the WebSocket connection is replaced with a replay of the bundled
 *     Adhi 2026-02-20 recording (demoData.ts)
 *   - startSimulation() short-circuits to a fake demo patient
 *
 * Enable during local dev by adding `VITE_DEMO_MODE=true` to `.env.local`.
 * The production Vercel build sets it via .env.production.
 */
export const DEMO_MODE: boolean =
  (import.meta.env.VITE_DEMO_MODE as string | undefined) === "true";
