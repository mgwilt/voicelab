import { afterEach, beforeEach, expect, test } from "bun:test";
import { Playback } from "../frontend/audio/playback.js";

class SourceDouble {
  buffer: unknown = null;
  onended: (() => void) | null = null;
  stopped = false;
  disconnected = false;
  connect(_destination: unknown): void {}
  start(_time: number): void {}
  stop(): void { this.stopped = true; }
  disconnect(): void { this.disconnected = true; }
  finish(): void { this.onended?.(); }
}
class ContextDouble {
  static latest: ContextDouble | null = null;
  state = "running";
  currentTime = 0;
  destination = {};
  readonly sources: SourceDouble[] = [];
  constructor() { ContextDouble.latest = this; }
  async resume(): Promise<void> {}
  async close(): Promise<void> { this.state = "closed"; }
  createBuffer(_channels: number, count: number, rate: number): { duration: number; getChannelData(channel: number): Float32Array } {
    return { duration: count / rate, getChannelData: () => new Float32Array(count) };
  }
  createBufferSource(): SourceDouble {
    const source = new SourceDouble(); this.sources.push(source); return source;
  }
}
const original = Object.getOwnPropertyDescriptor(globalThis, "AudioContext");
beforeEach(() => {
  ContextDouble.latest = null;
  Object.defineProperty(globalThis, "AudioContext", { configurable: true, writable: true, value: ContextDouble });
});
afterEach(() => {
  if (original !== undefined) Object.defineProperty(globalThis, "AudioContext", original);
  else Reflect.deleteProperty(globalThis, "AudioContext");
});
function context(): ContextDouble {
  const value = ContextDouble.latest;
  if (value === null) throw new Error("Missing audio context");
  return value;
}

test("completion drains scheduled audio before closing the context", async () => {
  const playback = new Playback();
  const signal = new AbortController().signal;
  await playback.open(signal);
  playback.play(new ArrayBuffer(4800), 24000, 3);
  let drained = false;
  const pending = playback.drain(signal).then(() => { drained = true; });
  await Promise.resolve();
  expect(drained).toBe(false);
  context().sources[0]?.finish();
  await pending;
  expect(drained).toBe(true);
  await playback.close();
  expect(context().state).toBe("closed");
});

test("cancellation releases queued audio and rejects a pending drain", async () => {
  const playback = new Playback();
  const cancellation = new AbortController();
  await playback.open(cancellation.signal);
  playback.play(new ArrayBuffer(4800), 24000, 3);
  const pending = playback.drain(cancellation.signal);
  cancellation.abort();
  await expect(pending).rejects.toThrow("Playback cancelled");
  await playback.close();
  expect(context().sources[0]?.stopped).toBe(true);
  expect(context().state).toBe("closed");
});

test("PCM validation and playback bounds reject malformed or excessive audio", async () => {
  const playback = new Playback();
  await playback.open(new AbortController().signal);
  expect(() => playback.play(new ArrayBuffer(3), 24000, 3)).toThrow("Invalid PCM16");
  expect(() => playback.play(new ArrayBuffer(48000), 24000, .2)).toThrow("time limit");
  await playback.close();
});
