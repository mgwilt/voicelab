import { Playback } from "../../audio/playback.js";
import type { Emit, VoiceSession, MingAudioModel, Outcome } from "../../domain/session.js";
import { unreachable } from "../../domain/session.js";
import { streamSession } from "../../transport/socket.js";
import type { ReferenceAudio } from "../synthesis/reference.js";
import type { HistoryTurn } from "./history.js";
import { decode } from "../conversation/protocol.js";

export interface Question {
  readonly file: File | null;
  readonly task: "spoken_dialogue" | "audio_understanding" | "asr" | "text" | "tts";
  readonly seed: number; readonly max_decode_steps: number; readonly cfg_scale: number;
  readonly sigma: number; readonly temperature: number; readonly flow_steps: number;
  readonly prompt: string; readonly voice: string; readonly max_tokens: number;
  readonly reference: () => Promise<ReferenceAudio | null>; readonly reference_text: string | null; readonly trim_reference_silence: boolean;
  readonly thinking: boolean; readonly system_prompt: string | null; readonly history: () => Promise<readonly HistoryTurn[]>;
}

export class MingAudioSession implements VoiceSession {
  private readonly playback = new Playback();
  private readonly cancellation = new AbortController();
  constructor(private readonly model: MingAudioModel, private readonly question: Question, private readonly emit: Emit) {}
  async run(parent: AbortSignal): Promise<Outcome> {
    const signal = AbortSignal.any([parent, this.cancellation.signal]);
    try {
      await this.playback.open(signal);
      let encoded: string | null = null;
      if (this.question.file !== null) {
        const bytes = new Uint8Array(await this.question.file.arrayBuffer());
        let binary = "";
        for (const byte of bytes) binary += String.fromCharCode(byte);
        encoded = btoa(binary);
      }
      const reference = await this.question.reference();
      const history = await this.question.history();
      signal.throwIfAborted();
      const request = { type: "start", task: this.question.task, audio_wav_base64: encoded, prompt: this.question.prompt,
        voice: this.question.voice, max_tokens: this.question.max_tokens, reference_audio: reference,
        seed: this.question.seed, max_decode_steps: this.question.max_decode_steps, cfg_scale: this.question.cfg_scale,
        sigma: this.question.sigma, temperature: this.question.temperature, flow_steps: this.question.flow_steps,
        reference_text: this.question.reference_text, trim_reference_silence: this.question.trim_reference_silence, thinking: this.question.thinking,
        system_prompt: this.question.system_prompt, history };
      this.emit({ type: "status", phase: "preparing", message: "Preparing local inference…" });
      const result = await streamSession({ path: this.model.endpoint, request, signal, decode: raw => decode(raw, "ming_audio"),
        onAudio: pcm => this.playback.play(pcm, this.model.outputSampleRate, 125),
        onMessage: (message): Outcome | undefined => {
          switch (message.type) {
            case "loading": case "ready": return undefined;
            case "assistant_text_delta": this.emit({ type: "transcript", speaker: "assistant", text: message.text }); return undefined;
            case "metrics": this.emit({ type: "measurement", value: message.value }); return undefined;
            case "done":
              this.emit({ type: "measurement", value: message.value });
              return { kind: message.complete ? "complete" : "incomplete", message: message.complete ? "Response complete" : "Response incomplete" };
            case "error": throw new Error(message.message);
            default: return unreachable(message);
          }
        },
      });
      this.emit({ type: "status", phase: "draining", message: "Playing response…" });
      await this.playback.drain(signal);
      return result;
    } finally { await this.playback.close(); }
  }
  stop(): void { this.cancellation.abort(); }
}
