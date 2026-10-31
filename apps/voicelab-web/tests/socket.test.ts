import { afterEach, beforeEach, expect, test } from "bun:test";
import { streamSession, type SessionSender } from "../frontend/transport/socket.js";

class SocketDouble {
  static readonly OPEN = 1;
  static latest: SocketDouble | null = null;
  binaryType = "blob";
  bufferedAmount = 0;
  readyState = 0;
  closed = false;
  readonly sent: (string | ArrayBuffer)[] = [];
  onopen: (() => void) | null = null;
  onmessage: ((event: MessageEvent<unknown>) => void) | null = null;
  onerror: (() => void) | null = null;
  onclose: (() => void) | null = null;
  constructor(readonly url: URL) { SocketDouble.latest = this; }
  send(value: string | ArrayBuffer): void { this.sent.push(value); }
  close(): void { this.closed = true; }
  open(): void { this.readyState = 1; this.onopen?.(); }
  message(data: unknown): void { this.onmessage?.(new MessageEvent("message", { data })); }
}
const original = Object.getOwnPropertyDescriptor(globalThis, "WebSocket");
beforeEach(() => {
  SocketDouble.latest = null;
  Object.defineProperty(globalThis, "WebSocket", { configurable: true, writable: true, value: SocketDouble });
});
afterEach(() => {
  if (original !== undefined) Object.defineProperty(globalThis, "WebSocket", original);
});
function connection(): SocketDouble {
  const socket = SocketDouble.latest;
  if (socket === null) throw new Error("Connection was not created");
  return socket;
}

test("model-specific deadlines remain bounded and close expired sessions", async () => {
  const options = {
    path: "/api/qwen_omni/session", request: {}, signal: new AbortController().signal,
    decode: (raw: string) => raw, onAudio: () => {}, onMessage: () => undefined,
  };
  for (const timeoutMs of [0, Number.NaN, Number.POSITIVE_INFINITY, 1_800_001]) {
    expect(() => streamSession({ ...options, timeoutMs })).toThrow("Session timeout");
  }
  expect(SocketDouble.latest).toBeNull();
  const pending = streamSession({ ...options, timeoutMs: 1 });
  const socket = connection();
  await expect(pending).rejects.toThrow("Session timed out");
  expect(socket.closed).toBe(true);
  expect(socket.onmessage).toBeNull();
  expect(socket.onclose).toBeNull();
});

test("socket decodes messages and closes all handlers after completion", async () => {
  const pending = streamSession({
    path: "/api/sesame/session", request: { text: "hello" }, signal: new AbortController().signal,
    decode: (raw: string) => raw, onAudio: () => {},
    onMessage: message => message === "done" ? { kind: "complete", message: "Done" } : undefined,
  });
  const socket = connection();
  expect(socket.url.toString()).toBe("wss://lab.test/api/sesame/session");
  socket.open();
  expect(socket.sent).toEqual(['{"text":"hello"}']);
  socket.message("done");
  expect(await pending).toEqual({ kind: "complete", message: "Done" });
  expect(socket.closed).toBe(true);
  expect(socket.onmessage).toBeNull();
  expect(socket.onclose).toBeNull();
});

test("malformed data and abrupt disconnect are failures, never completion", async () => {
  const pending = streamSession({
    path: "/api/sesame/session", request: {}, signal: new AbortController().signal,
    decode: () => { throw new Error("Malformed message"); }, onAudio: () => {}, onMessage: () => undefined,
  });
  connection().message("bad");
  await expect(pending).rejects.toThrow("Malformed message");
  expect(connection().closed).toBe(true);

  const disconnected = streamSession({
    path: "/api/sesame/session", request: {}, signal: new AbortController().signal,
    decode: raw => raw, onAudio: () => {}, onMessage: () => undefined,
  });
  connection().onclose?.();
  await expect(disconnected).rejects.toThrow("before the model completed");
});

test("abort and network backpressure release a connection", async () => {
  const cancellation = new AbortController();
  const pending = streamSession({
    path: "/api/nemotron/session", request: {}, signal: cancellation.signal,
    decode: raw => raw, onAudio: () => {}, onMessage: () => undefined,
  });
  cancellation.abort();
  await expect(pending).rejects.toThrow("Session cancelled");
  expect(connection().closed).toBe(true);

  const state: { sender: SessionSender | null } = { sender: null };
  const blocked = streamSession({
    path: "/api/nemotron/session", request: {}, signal: new AbortController().signal,
    decode: raw => raw, onAudio: () => {},
    onMessage: (_message, sender) => { state.sender = sender; return undefined; },
  });
  const socket = connection();
  socket.open(); socket.message("ready"); socket.bufferedAmount = 65_000;
  state.sender?.audio(new ArrayBuffer(2560));
  await expect(blocked).rejects.toThrow("Network is falling behind");
  expect(socket.closed).toBe(true);
});
