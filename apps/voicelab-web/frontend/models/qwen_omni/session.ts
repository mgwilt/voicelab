import { Playback } from "../../audio/playback.js";
import type { Emit, VoiceSession, Outcome, QwenOmniModel } from "../../domain/session.js";
import { unreachable } from "../../domain/session.js";
import { streamSession } from "../../transport/socket.js";
import { decode } from "./protocol.js";
import type { Request } from "./request.js";
export class QwenOmniSession implements VoiceSession {
  private readonly playback = new Playback();
  private readonly cancellation = new AbortController();
  constructor(private readonly model: QwenOmniModel, private readonly read: () => Promise<Request>, private readonly emit: Emit) {}
  async run(parent: AbortSignal): Promise<Outcome> {
    const signal = AbortSignal.any([parent, this.cancellation.signal]);
    try {
      await this.playback.open(signal);
      const request = await this.read();
      signal.throwIfAborted();
      this.emit({ type: "status", phase: "preparing", message: "Preparing local Qwen inference…" });
      const outcome = await streamSession({ path: this.model.endpoint, request: { type: "start", ...request }, signal, decode, timeoutMs: 900000,
        onAudio: pcm => this.playback.play(pcm, this.model.outputSampleRate, 125),
        onMessage: (message): Outcome | undefined => {
          switch (message.type) {
            case "loading": this.emit({ type: "status", phase: "preparing", message: "Loading Qwen. First use downloads the checkpoint…" }); return undefined;
            case "ready": this.emit({ type: "status", phase: "running", message: "Generating the response…" }); return undefined;
            case "assistant_text_delta": this.emit({ type: "transcript", speaker: "assistant", text: message.text }); return undefined;
            case "done": this.emit({ type: "measurement", value: message.value }); return { kind: message.complete ? "complete" : "incomplete", message: message.complete ? "Qwen response complete" : "Qwen reached a generation limit" };
            case "error": throw new Error(message.message);
            default: return unreachable(message);
          }
        },
      });
      this.emit({ type: "status", phase: "draining", message: "Finishing playback…" });
      await this.playback.drain(signal);
      return outcome;
    } finally { await this.playback.close(); }
  }
  stop(): void { this.cancellation.abort(); }
}
