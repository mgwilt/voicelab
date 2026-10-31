import type { Outcome } from "../domain/experiment.js";

export interface SessionSender {
  send(value: object): void;
  audio(value: ArrayBuffer): void;
}
interface StreamOptions<Event> {
  readonly path: string;
  readonly request: object;
  readonly signal: AbortSignal;
  readonly decode: (raw: string) => Event;
  readonly onMessage: (message: Event, sender: SessionSender) => Outcome | undefined;
  readonly onAudio: (pcm: ArrayBuffer) => void;
}
/** Own all handlers and timers for one connection; never reconnect inference implicitly. */
export function streamSession<Event>(options: StreamOptions<Event>): Promise<Outcome> {
  options.signal.throwIfAborted();
  return new Promise((resolve, reject) => {
    const url = new URL(options.path, location.href);
    url.protocol = location.protocol === "https:" ? "wss:" : "ws:";
    const socket = new WebSocket(url);
    socket.binaryType = "arraybuffer";
    let settled = false;
    const finish = (result: Outcome | Error): void => {
      if (settled) return;
      settled = true;
      clearTimeout(timeout);
      options.signal.removeEventListener("abort", abort);
      socket.onopen = null; socket.onmessage = null; socket.onerror = null; socket.onclose = null;
      socket.close();
      if (result instanceof Error) reject(result); else resolve(result);
    };
    const abort = (): void => finish(new DOMException("Session cancelled", "AbortError"));
    const timeout = setTimeout(() => finish(new Error("Session timed out")), 180_000);
    const send = (value: string | ArrayBuffer): void => {
      if (settled) return;
      if (socket.readyState !== WebSocket.OPEN || socket.bufferedAmount > 64_000) {
        finish(new Error("Network is falling behind. Session stopped."));
        return;
      }
      socket.send(value);
    };
    const sender: SessionSender = { send: value => send(JSON.stringify(value)), audio: send };
    options.signal.addEventListener("abort", abort, { once: true });
    socket.onopen = () => sender.send(options.request);
    socket.onerror = () => finish(new Error("Connection failed. Check the server and HTTPS certificate."));
    socket.onclose = () => finish(new Error("Connection ended before the model completed."));
    socket.onmessage = (event: MessageEvent<unknown>) => {
      if (settled) return;
      try {
        if (event.data instanceof ArrayBuffer) options.onAudio(event.data);
        else if (typeof event.data === "string") {
          const result = options.onMessage(options.decode(event.data), sender);
          if (result !== undefined) finish(result);
        } else throw new Error("Unexpected server data");
      } catch (error) {
        finish(error instanceof Error ? error : new Error(String(error)));
      }
    };
    if (options.signal.aborted) abort();
  });
}
