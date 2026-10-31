import type { Measurement } from "../../domain/session.js";
import { boolean, finite, nullableNumber, parse, record, text } from "../../transport/decode.js";

export type Message =
  | { readonly type: "loading" | "ready" }
  | { readonly type: "assistant_text_delta"; readonly text: string }
  | { readonly type: "metrics"; readonly value: Measurement }
  | { readonly type: "done"; readonly value: Measurement; readonly complete: boolean }
  | { readonly type: "error"; readonly message: string };

function measurement(value: unknown, basis: "input" | "output"): Measurement {
  const data = record(value);
  const first = nullableNumber(data["first_pcm_seconds"]);
  return { basis, rtf: nullableNumber(data["rtf"]),
    audioSeconds: finite(data[basis === "input" ? "input_seconds" : "output_seconds"]),
    latencyLabel: basis === "input" ? "Compute to first PCM" : "First PCM", latencyMs: first === null ? null : first * 1000, backlogSeconds: null };
}
export function decode(raw: string, workflow: "personaplex" | "ming_audio"): Message {
  const data = parse(raw);
  const basis = workflow === "personaplex" ? "input" : "output";
  switch (data["type"]) {
    case "loading": return { type: "loading" };
    case "ready":
      if (data["input_sample_rate"] !== (workflow === "personaplex" ? 24000 : 16000) ||
          data["output_sample_rate"] !== (workflow === "personaplex" ? 24000 : 44100)) throw new Error("Unexpected conversation sample rates");
      return { type: "ready" };
    case "assistant_text_delta": return { type: "assistant_text_delta", text: text(data["text"]) };
    case "metrics": return { type: "metrics", value: measurement(data, basis) };
    case "done": {
      const metrics = record(data["metrics"]);
      return { type: "done", value: measurement(metrics, basis), complete: workflow === "personaplex" || boolean(metrics["completed"]) };
    }
    case "error": return { type: "error", message: text(data["message"]) };
    default: throw new Error("Unknown conversation event");
  }
}
