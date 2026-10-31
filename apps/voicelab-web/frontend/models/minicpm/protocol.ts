import type { Measurement } from "../../domain/experiment.js";
import { boolean, finite, nullableNumber, parse, record, text } from "../../transport/decode.js";
export type Mode = "full_duplex" | "turn_based";
export type Message =
  | { readonly type: "loading" | "ready" | "listen" }
  | { readonly type: "assistant_text_delta"; readonly text: string }
  | { readonly type: "metrics"; readonly value: Measurement }
  | { readonly type: "done"; readonly value: Measurement; readonly complete: boolean }
  | { readonly type: "error"; readonly message: string };
function measurement(value: unknown, mode: Mode): Measurement {
  const data = record(value); const first = nullableNumber(data["first_pcm_seconds"]);
  const basis = mode === "full_duplex" ? "input" : "output";
  return { basis, rtf: nullableNumber(data["rtf"]), audioSeconds: finite(data[`${basis}_seconds`]),
    latencyLabel: basis === "input" ? "Compute to first PCM" : "First PCM",
    latencyMs: first === null ? null : first * 1000, backlogSeconds: null };
}
export function decode(raw: string, mode: Mode): Message {
  const data = parse(raw);
  switch (data["type"]) {
    case "loading": return { type: "loading" };
    case "listen": return { type: "listen" };
    case "ready":
      if (data["input_sample_rate"] !== 16000 || data["output_sample_rate"] !== 24000) throw new Error("Unexpected MiniCPM sample rates.");
      return { type: "ready" };
    case "assistant_text_delta": return { type: "assistant_text_delta", text: text(data["text"]) };
    case "metrics": return { type: "metrics", value: measurement(data, mode) };
    case "done": {
      const metrics = record(data["metrics"]);
      return { type: "done", value: measurement(metrics, mode), complete: mode === "full_duplex" || boolean(metrics["completed"]) };
    }
    case "error": return { type: "error", message: text(data["message"]) };
    default: throw new Error("Unknown MiniCPM event.");
  }
}
