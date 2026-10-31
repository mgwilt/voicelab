import { expect, test } from "bun:test";
import type { Emit, VoiceSession, Outcome } from "../frontend/domain/session.js";
import { SessionRunner, type RunState } from "../frontend/domain/runner.js";

class DeferredSession implements VoiceSession {
  resolve: ((outcome: Outcome) => void) | null = null;
  stopped = false;
  async run(signal: AbortSignal): Promise<Outcome> {
    return new Promise((resolve, reject) => {
      this.resolve = resolve;
      signal.addEventListener("abort", () => reject(new DOMException("stopped", "AbortError")), { once: true });
    });
  }
  stop(): void { this.stopped = true; }
}

test("runner prevents overlap, supports stop and awaits cancellation", async () => {
  const states: RunState[] = [];
  const runner = new SessionRunner(state => states.push(state), () => {});
  const first = new DeferredSession();
  let started = 0;
  runner.start(() => { started++; return first; });
  runner.start(() => { started++; return new DeferredSession(); });
  expect(started).toBe(1);
  expect(runner.busy).toBe(true);
  runner.stop();
  expect(first.stopped).toBe(true);
  await runner.cancel();
  expect(runner.busy).toBe(false);
  expect(states.at(-1)?.phase).toBe("idle");
  expect(states.at(-1)?.error).toBe(false);
});

test("callbacks from a cancelled run cannot affect a later run", async () => {
  const messages: string[] = [];
  const runner = new SessionRunner(state => messages.push(state.message), () => {});
  const callbacks: { stale: Emit | null } = { stale: null };
  runner.start(emit => { callbacks.stale = emit; return new DeferredSession(); });
  await runner.cancel();
  runner.start(() => new DeferredSession());
  const previous = messages.length;
  // Closure assignment occurs synchronously inside start's factory.
  const emit = callbacks.stale;
  if (emit !== null) emit({ type: "status", phase: "running", message: "Stale" });
  expect(messages.length).toBe(previous);
  await runner.dispose();
  runner.start(() => { throw new Error("Disposed runner started"); });
});

test("failed sessions return to an actionable idle state", async () => {
  const states: RunState[] = [];
  const runner = new SessionRunner(state => states.push(state), () => {});
  runner.start(() => ({ run: async () => { throw new Error("Model unavailable"); }, stop: () => {} }));
  await Promise.resolve();
  expect(states.at(-1)).toEqual({ phase: "idle", message: "Model unavailable", error: true });
  expect(runner.busy).toBe(false);
});
