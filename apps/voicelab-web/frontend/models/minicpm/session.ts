import { Microphone } from "../../audio/microphone.js";
import { Playback } from "../../audio/playback.js";
import type { Emit, ExperimentSession, MiniCPMModel, Outcome } from "../../domain/experiment.js";
import { unreachable } from "../../domain/experiment.js";
import { streamSession, type SessionSender } from "../../transport/socket.js";
import type { ReferenceAudio } from "../synthesis/reference.js";
import { decode, type Mode } from "./protocol.js";
export interface StartRequest {
  readonly type: "start"; readonly mode: Mode; readonly prompt: string;
  readonly text: string; readonly audio: ReferenceAudio | null;
  readonly history: readonly { readonly role: "user" | "assistant"; readonly text: string }[];
  readonly reference_audio: ReferenceAudio | null; readonly seed: number;
  readonly temperature: number; readonly top_p: number; readonly top_k: number;
  readonly tts_temperature: number; readonly length_penalty: number;
  readonly listen_prob_scale: number; readonly max_speak_tokens: number; readonly max_tokens: number;
}
/** The native decoder consumes one second at a time; microphone worklet chunks vary. */
export class FrameBuffer {
  private pending = new Uint8Array(0);
  push(pcm: ArrayBuffer, send: (frame: ArrayBuffer) => void): void {
    if (pcm.byteLength % 2 !== 0) throw new Error("Invalid microphone PCM16 frame.");
    const combined = new Uint8Array(this.pending.length + pcm.byteLength);
    combined.set(this.pending); combined.set(new Uint8Array(pcm), this.pending.length);
    let offset = 0;
    for (; offset + 32000 <= combined.length; offset += 32000) send(combined.slice(offset, offset + 32000).buffer);
    this.pending = combined.slice(offset);
  }
  finish(send: (frame: ArrayBuffer) => void): void {
    if (this.pending.length > 0) { const padded = new Uint8Array(32000); padded.set(this.pending); send(padded.buffer); }
    this.pending = new Uint8Array(0);
  }
}
export class MiniCPMSession implements ExperimentSession {
  private readonly playback = new Playback();
  private readonly microphone = new Microphone();
  private readonly frames = new FrameBuffer();
  private readonly cancellation = new AbortController();
  private sender: SessionSender | null = null;
  private live = false;
  private stopping = false;
  private timer: ReturnType<typeof setTimeout> | null = null;
  constructor(private readonly model: MiniCPMModel, private readonly request: () => Promise<StartRequest>, private readonly emit: Emit) {}
  async run(parent: AbortSignal): Promise<Outcome> {
    const signal = AbortSignal.any([parent, this.cancellation.signal]);
    try {
      await this.playback.open(signal);
      const request = await this.request(); signal.throwIfAborted();
      this.live = request.mode === "full_duplex";
      if (this.live) await this.microphone.open(this.playback.context, signal);
      const outcome = await streamSession({ path: this.model.endpoint, request, signal, timeoutMs: 1_800_000,
        decode: raw => decode(raw, request.mode),
        onAudio: pcm => this.playback.play(pcm, 24000, this.live ? 10 : 240),
        onMessage: (message, sender): Outcome | undefined => {
          switch (message.type) {
            case "loading": this.emit({ type: "status", phase: "preparing", message: "Preparing MiniCPM. First use builds the local runtime and downloads weights…" }); return undefined;
            case "ready":
              this.sender = sender;
              if (this.live) {
                this.microphone.start(this.playback.context, pcm => this.frames.push(pcm, frame => sender.audio(frame)), 16000);
                this.timer = setTimeout(() => this.stop(), 119_000);
              }
              this.emit({ type: "status", phase: "running", message: this.live ? "Listening and speaking" : "Generating spoken answer…" }); return undefined;
            case "listen": return undefined;
            case "metrics": this.emit({ type: "measurement", value: message.value }); return undefined;
            case "assistant_text_delta": this.emit({ type: "transcript", speaker: "assistant", text: message.text }); return undefined;
            case "done": this.emit({ type: "measurement", value: message.value }); return { kind: message.complete ? "complete" : "incomplete", message: this.live ? "Capture stopped" : message.complete ? "Spoken answer complete" : "Generation limit reached; output is incomplete" };
            case "error": throw new Error(message.message);
            default: return unreachable(message);
          }
        },
      });
      this.sender = null; this.microphone.close();
      if (this.timer !== null) clearTimeout(this.timer);
      this.emit({ type: "status", phase: "draining", message: "Playing remaining audio…" });
      await this.playback.drain(signal); return outcome;
    } finally {
      if (this.timer !== null) clearTimeout(this.timer);
      this.sender = null; this.microphone.close(); await this.playback.close();
    }
  }
  stop(): void {
    if (!this.live || this.sender === null || this.stopping) { this.cancellation.abort(); return; }
    this.stopping = true; this.microphone.close();
    const sender = this.sender; this.frames.finish(frame => sender.audio(frame)); sender.send({ type: "stop" });
    if (this.timer !== null) clearTimeout(this.timer);
    this.timer = setTimeout(() => this.cancellation.abort(), 15_000);
    this.emit({ type: "status", phase: "stopping", message: "Draining speech… Stop again to cancel." });
  }
}
