import { Playback } from "../../audio/playback.js";
import type { Emit, ExperimentSession, Outcome, SynthesisModel } from "../../domain/experiment.js";
import { unreachable } from "../../domain/experiment.js";
import { streamSession } from "../../transport/socket.js";
import { decode } from "./protocol.js";
interface SynthesisInput<Request extends object> {
  readonly model: SynthesisModel;
  readonly request: Request | (() => Request | Promise<Request>);
}

export class SynthesisSession<Request extends object> implements ExperimentSession {
  private readonly playback = new Playback();
  private readonly cancellation = new AbortController();
  constructor(private readonly input: SynthesisInput<Request>, private readonly emit: Emit) {}

  async run(parent: AbortSignal): Promise<Outcome> {
    const signal = AbortSignal.any([parent, this.cancellation.signal]);
    try {
      this.emit({ type: "status", phase: "preparing", message: "Preparing speech…" });
      await this.playback.open(signal);
      const request = typeof this.input.request === "function" ? await this.input.request() : this.input.request;
      signal.throwIfAborted();
      const outcome = await streamSession({
        path: this.input.model.endpoint, request, signal, decode: raw => decode(raw, this.input.model.outputSampleRate),
        onAudio: pcm => {
          this.playback.play(pcm, this.input.model.outputSampleRate, 125);
          this.emit({ type: "status", phase: "running", message: "Generating and playing speech" });
        },
        onMessage: (message): Outcome | undefined => {
          switch (message.type) {
            case "ready": return undefined;
            case "complete": case "incomplete": case "unverified":
              this.emit({ type: "measurement", value: message.value });
              return { kind: message.type, message: message.type === "complete" ? "Speech complete" : message.type === "incomplete" ? "Incomplete generation" : "Audio received; upstream completion is unverified" };
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
