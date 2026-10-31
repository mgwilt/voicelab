import { Playback } from "../../audio/playback.js";
import type { Emit, ExperimentSession, Outcome, BreezeModel, SesameModel } from "../../domain/experiment.js";
import { unreachable } from "../../domain/experiment.js";
import { streamSession } from "../../transport/socket.js";
import { decode } from "./protocol.js";
import type { SynthesisRequest as SesameRequest } from "../sesame/request.js";
import type { SynthesisRequest as BreezeRequest } from "../breeze/request.js";

type SynthesisInput =
  | { readonly model: SesameModel; readonly request: SesameRequest }
  | { readonly model: BreezeModel; readonly request: BreezeRequest };

export class SynthesisSession implements ExperimentSession {
  private readonly playback = new Playback();
  private readonly cancellation = new AbortController();
  constructor(private readonly input: SynthesisInput, private readonly emit: Emit) {}

  async run(parent: AbortSignal): Promise<Outcome> {
    const signal = AbortSignal.any([parent, this.cancellation.signal]);
    try {
      this.emit({ type: "status", phase: "preparing", message: "Preparing speech…" });
      await this.playback.open(signal);
      const outcome = await streamSession({
        path: this.input.model.endpoint, request: this.input.request, signal, decode,
        onAudio: pcm => {
          this.playback.play(pcm, this.input.model.outputSampleRate, 125);
          this.emit({ type: "status", phase: "running", message: "Generating and playing speech" });
        },
        onMessage: (message): Outcome | undefined => {
          switch (message.type) {
            case "ready": return undefined;
            case "complete": case "incomplete":
              this.emit({ type: "measurement", value: message.value });
              return { kind: message.type, message: message.type === "complete" ? "Speech complete" : "Incomplete: frame limit reached" };
            case "error": throw new Error(message.message);
            default: return unreachable(message);
          }
        },
      });
      this.emit({ type: "status", phase: "draining", message: "Playing remaining audio…" });
      await this.playback.drain(signal);
      return outcome;
    } finally {
      await this.playback.close();
    }
  }
  stop(): void { this.cancellation.abort(); }
}
