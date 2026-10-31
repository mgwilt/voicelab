import { Playback } from "../../audio/playback.js";
import type { Emit, VoiceSession, Outcome } from "../../domain/session.js";
import { unreachable } from "../../domain/session.js";
import { streamSession } from "../../transport/socket.js";
import { decodeTool } from "./tool_protocol.js";

export interface ToolQuestion {
  readonly file: File; readonly prompt: string; readonly seed: number;
  readonly trailingSilence: number; readonly tools: unknown; readonly response: unknown;
}
export class ToolSession implements VoiceSession {
  private readonly playback = new Playback();
  private readonly cancellation = new AbortController();
  constructor(private readonly question: ToolQuestion, private readonly emit: Emit) {}
  async run(parent: AbortSignal): Promise<Outcome> {
    const signal = AbortSignal.any([parent, this.cancellation.signal]);
    try {
      await this.playback.open(signal);
      const bytes = new Uint8Array(await this.question.file.arrayBuffer());
      let binary = "";
      for (const byte of bytes) binary += String.fromCharCode(byte);
      signal.throwIfAborted();
      const request = { type: "start", audio: { wav_base64: btoa(binary) },
        prompt: this.question.prompt, seed: this.question.seed, trailing_silence: this.question.trailingSilence,
        tools: this.question.tools, api_response: this.question.response };
      this.emit({ type: "status", phase: "preparing", message: "Offline tool test: detecting a call, then injecting the supplied response…" });
      const result = await streamSession({ path: "/api/nemotron/tools", request, signal, decode: decodeTool,
        onAudio: pcm => this.playback.play(pcm, 22050, 150),
        onMessage: (message): Outcome | undefined => {
          switch (message.type) {
            case "loading": case "ready": return undefined;
            case "assistant_text_delta": case "function_delta":
              this.emit({ type: "transcript", speaker: message.type === "function_delta" ? "function" : "assistant", text: message.text }); return undefined;
            case "done":
              this.emit({ type: "measurement", value: message.value });
              return { kind: message.complete ? "complete" : "incomplete", message: message.complete ? "Offline tool response completed" : "No complete continuation after tool response" };
            case "error": throw new Error(message.message);
            default: return unreachable(message);
          }
        },
      });
      this.emit({ type: "status", phase: "draining", message: "Playing the expanded offline timeline…" });
      await this.playback.drain(signal);
      return result;
    } finally { await this.playback.close(); }
  }
  stop(): void { this.cancellation.abort(); }
}
