import type { Measurement } from "../../domain/experiment.js";
import { boolean, finite, nullableNumber, parse, record, text } from "../../transport/decode.js";

export type Message =
  | { readonly type: "ready"; readonly sampleRate: number }
  | { readonly type: "complete" | "incomplete" | "unverified"; readonly value: Measurement }
  | { readonly type: "error"; readonly message: string };
export function decode(raw: string, sampleRate = 24000): Message {
  const data = parse(raw);
  const type = data["type"];
  switch (type) {
    case "ready":
      if (data["sample_rate"] !== sampleRate) throw new Error("Unexpected synthesis sample rate");
      return { type, sampleRate };
    case "complete": case "incomplete": case "unverified": {
      const metrics = record(data["metrics"]);
      const eos = metrics["eos"] === null ? null : boolean(metrics["eos"]);
      const completion = metrics["completion"];
      if (eos === false && completion === "complete") throw new Error("Completion contradicts the stop signal");
      if (eos === true && completion === "incomplete") throw new Error("Completion contradicts the stop signal");
      const expected = completion === undefined || completion === null
        ? eos === null ? "unverified" : eos ? "complete" : "incomplete" : completion;
      if (expected !== type) throw new Error("Inconsistent completion state");
      const first = nullableNumber(metrics["first_pcm_seconds"]);
      return { type, value: { basis: "output", rtf: nullableNumber(metrics["output_rtf"]),
        audioSeconds: finite(metrics["audio_seconds"]), latencyLabel: "First PCM",
        latencyMs: first === null ? null : first * 1000, backlogSeconds: null } };
    }
    case "error": return { type, message: text(data["message"]) };
    default: throw new Error(`Unknown synthesis event: ${String(type)}`);
  }
}
