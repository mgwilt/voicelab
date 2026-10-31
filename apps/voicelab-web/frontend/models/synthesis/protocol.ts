import type { Measurement } from "../../domain/experiment.js";
import { boolean, finite, nullableNumber, parse, record, text } from "../../transport/decode.js";

export type Message =
  | { readonly type: "ready"; readonly sampleRate: 24000 }
  | { readonly type: "complete" | "incomplete"; readonly value: Measurement }
  | { readonly type: "error"; readonly message: string };
export function decode(raw: string): Message {
  const data = parse(raw);
  const type = data["type"];
  switch (type) {
    case "ready":
      if (data["sample_rate"] !== 24000) throw new Error("Unexpected synthesis sample rate");
      return { type, sampleRate: 24000 };
    case "complete": case "incomplete": {
      const metrics = record(data["metrics"]);
      if (boolean(metrics["eos"]) !== (type === "complete")) throw new Error("Inconsistent completion state");
      const first = nullableNumber(metrics["first_pcm_seconds"]);
      return { type, value: { basis: "output", rtf: nullableNumber(metrics["output_rtf"]),
        audioSeconds: finite(metrics["audio_seconds"]), latencyLabel: "First PCM",
        latencyMs: first === null ? null : first * 1000, backlogSeconds: null } };
    }
    case "error": return { type, message: text(data["message"]) };
    default: throw new Error(`Unknown synthesis event: ${String(type)}`);
  }
}
