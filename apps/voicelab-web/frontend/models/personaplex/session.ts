import { Microphone } from "../../audio/microphone.js";
import { Playback } from "../../audio/playback.js";
import type { PersonaPlexModel, Emit, ExperimentSession, Outcome } from "../../domain/experiment.js";
import { unreachable } from "../../domain/experiment.js";
import { streamSession, type SessionSender } from "../../transport/socket.js";
import type { ReferenceAudio } from "../synthesis/reference.js";
import { decode } from "../conversation/protocol.js";

export interface StartRequest {
  readonly type: "start"; readonly prompt: string; readonly voice: string;
  readonly reference_audio: ReferenceAudio | null; readonly seed: number;
  readonly text_temperature: number; readonly audio_temperature: number;
  readonly text_top_k: number; readonly audio_top_k: number;
}

export class PersonaPlexSession implements ExperimentSession {
  private readonly playback = new Playback();
  private readonly microphone = new Microphone();
  private readonly cancellation = new AbortController();
  private sender: SessionSender | null = null;
  private stopTimer: ReturnType<typeof setTimeout> | null = null;
  private stopping = false;

  constructor(private readonly model: PersonaPlexModel, private readonly request: () => Promise<StartRequest>, private readonly emit: Emit) {}

  async run(parent: AbortSignal): Promise<Outcome> {
    const signal = AbortSignal.any([parent, this.cancellation.signal]);
    try {
      this.emit({ type: "status", phase: "preparing", message: "Requesting microphone…" });
      await this.playback.open(signal);
      const request = await this.request();
      signal.throwIfAborted();
      await this.microphone.open(this.playback.context, signal);
      const result = await streamSession({
        path: this.model.endpoint, request, signal, decode: raw => decode(raw, "personaplex"),
        onAudio: pcm => this.playback.play(pcm, this.model.outputSampleRate, 3),
        onMessage: (message, sender): Outcome | undefined => {
          switch (message.type) {
            case "loading":
              this.emit({ type: "status", phase: "preparing", message: "Preparing the model…" });
              return undefined;
            case "ready":
              this.sender = sender;
              this.microphone.start(this.playback.context, pcm => sender.audio(pcm), 24000);
              this.emit({ type: "status", phase: "running", message: "Listening and speaking" });
              return undefined;
            case "metrics": this.emit({ type: "measurement", value: message.value }); return undefined;
            case "assistant_text_delta":
              this.emit({ type: "transcript", speaker: "assistant", text: message.text });
              return undefined;
            case "done":
              this.emit({ type: "measurement", value: message.value });
              return { kind: "complete", message: "Session stopped" };
            case "error": throw new Error(message.message);
            default: return unreachable(message);
          }
        },
      });
      this.sender = null;
      this.microphone.close();
      if (this.stopTimer !== null) clearTimeout(this.stopTimer);
      this.emit({ type: "status", phase: "draining", message: "Playing remaining audio…" });
      await this.playback.drain(signal);
      return result;
    } finally {
      if (this.stopTimer !== null) clearTimeout(this.stopTimer);
      this.sender = null;
      this.microphone.close();
      await this.playback.close();
    }
  }
  stop(): void {
    if (this.sender === null || this.stopping) {
      this.cancellation.abort();
      return;
    }
    this.stopping = true;
    this.microphone.close();
    this.sender.send({ type: "stop" });
    this.emit({ type: "status", phase: "stopping", message: "Finishing queued frames…" });
    this.stopTimer = setTimeout(() => this.cancellation.abort(), 10_000);
  }
}
